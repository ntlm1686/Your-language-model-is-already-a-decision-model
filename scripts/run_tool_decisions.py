"""Run paired MetaTool, When2Call, or BFCL tool decisions with Qwen, CLM, or Jev.

MetaTool uses all 1,040 Task1 test queries. When2Call uses 200 eligible MCQ
questions from each of its three labeled classes, selected with seed 42.
The official When2Call MCQ test has no direct-answer gold labels.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import random
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LETTERS = "ABCD"
META_OPTIONS = {
    "yes": "Use an external tool because the query requires live or external data, a specialized input or output, or an action beyond a language model's own capabilities.",
    "no": "Answer directly without an external tool because the query can be addressed from general knowledge or reasoning alone.",
}
WHEN_SYSTEM = ("You are a helpful AI assistant. You have access to the following tools described in <tool></tool> "
               "which you can use to answer the user's questions. Only use a tool if it directly answers the user's question.\n"
               "To use a tool, return JSON in the following format: "
               '{"name": "tool_name", "arguments": {"argument1": "value1"}}')


def sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def task_prompt(state: str, options: list[str], suite: str) -> str:
    instruction = ("Should the assistant use an external tool for this query?" if suite == "metatool" else
                   "Which function should be called for the user's request?" if suite == "bfcl" else
                   "Which candidate is the best next assistant response to the user's question?")
    return (instruction + "\n\n" + state + "\n\nCandidates:\n" +
            "\n".join(f"{LETTERS[i]}. {text}" for i, text in enumerate(options)) +
            "\n\nReply with one letter.\nAnswer:")


def qwen_chat(tok, state: str, options: list[str], suite: str) -> list[int]:
    chat = tok.apply_chat_template([{"role": "user", "content": task_prompt(state, options, suite)}],
                                   tokenize=False, add_generation_prompt=True, enable_thinking=False) + "Answer: "
    return tok.encode(chat, add_special_tokens=False)


def load_tasks(suite: str, tok, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    if suite == "metatool":
        path = ROOT / "external/MetaTool/dataset/tmp_dataset/Task1.json"
        rows = json.loads(path.read_text())
        assert len(rows) == 1040
        tasks = []
        for i, row in enumerate(rows):
            assert row["label"] in ("positive", "negative")
            labels = list(META_OPTIONS)
            rng.shuffle(labels)
            options = [META_OPTIONS[x] for x in labels]
            state = "User query: " + row["query"]
            gold = labels.index("yes" if row["label"] == "positive" else "no")
            tasks.append({"id": f"metatool_{i:04d}", "source": row["label"], "state": state,
                          "options": options, "gold": gold})
    elif suite == "when2call":
        path = ROOT / "external/When2Call/data/test/when2call_test_mcq.jsonl"
        rows = [json.loads(x) for x in path.read_text().splitlines() if x]
        assert len(rows) == 3652
        groups = collections.defaultdict(list)
        for row in rows:
            groups[row["correct_answer"]].append(row)
        assert set(groups) == {"tool_call", "request_for_info", "cannot_answer"}
        tasks = []
        for name in sorted(groups):
            group = groups[name]
            rng.shuffle(group)
            taken = 0
            for row in group:
                answer_names = list(row["answers"])
                assert set(answer_names) == {"direct", "tool_call", "request_for_info", "cannot_answer"}
                rng.shuffle(answer_names)
                options = [row["answers"][x] for x in answer_names]
                state = (WHEN_SYSTEM + "\n\n" + "\n\n".join(f"<tool>{t}</tool>" for t in row["tools"]) +
                         "\n\nUser question: " + row["question"])
                if len(qwen_chat(tok, state, options, suite)) > 8192:
                    continue
                tasks.append({"id": row["uuid"], "source": name, "state": state,
                              "options": options, "gold": answer_names.index(name)})
                taken += 1
                if taken == 200:
                    break
            assert taken == 200, (name, taken)
        rng.shuffle(tasks)
    else:
        data_path = ROOT / "data/bfcl/BFCL_v4_multiple.json"
        answer_path = ROOT / "data/bfcl/possible_answer/BFCL_v4_multiple.json"
        rows = [json.loads(x) for x in data_path.read_text().splitlines() if x]
        answers = {x["id"]: x for x in (json.loads(y) for y in answer_path.read_text().splitlines() if y)}
        assert len(rows) == len(answers) == 200
        tasks = []
        for row in rows:
            functions = row["function"].copy()
            names = {x["name"] for x in functions}
            gold_name = next(iter(answers[row["id"]]["ground_truth"][0]))
            assert len(answers[row["id"]]["ground_truth"]) == 1 and gold_name in names
            rng.shuffle(functions)
            options = [f["name"] + ": " + f["description"] + "\nParameters: " +
                       json.dumps(f["parameters"], ensure_ascii=False, sort_keys=True) for f in functions]
            assert len(row["question"]) == 1 and len(row["question"][0]) == 1
            state = "User request: " + row["question"][0][0]["content"]
            tasks.append({"id": row["id"], "source": f"{len(functions)}_tools", "state": state,
                          "options": options, "gold": [f["name"] for f in functions].index(gold_name)})
    assert len({t["id"] for t in tasks}) == len(tasks)
    assert all(len(qwen_chat(tok, t["state"], t["options"], suite)) <= 8192 for t in tasks)
    return tasks


def run_gpu(tasks: list[dict], suite: str, output: Path, model_path: str, head_path: str, device: str, tok) -> None:
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM

    sys.path.insert(0, str(ROOT / "external/CLM/src"))
    from clm.heads import HeadPair

    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16,
                                                  attn_implementation="sdpa", low_cpu_mem_usage=True).eval().to(device)
    head = HeadPair("clm", head_path, device=device).ensure()
    letter_ids = [tok.encode(x, add_special_tokens=False) for x in LETTERS]
    assert all(len(x) == 1 for x in letter_ids)
    letter_ids = [x[0] for x in letter_ids]
    done = {json.loads(x)["id"] for x in output.read_text().splitlines()} if output.exists() else set()
    output.parent.mkdir(parents=True, exist_ok=True)

    def embed(text: str, keep: str):
        ids = tok.encode(text, add_special_tokens=False)
        count = len(ids)
        ids = ids[:2047] if keep == "head" else ids[-2047:]
        hidden = model.model(input_ids=torch.tensor([ids], device=device), use_cache=False,
                             return_dict=True).last_hidden_state[0, -1]
        return F.normalize(hidden.float(), dim=-1), count

    start = time.monotonic()
    with output.open("a") as f, torch.inference_mode():
        for n, task in enumerate(tasks, 1):
            if task["id"] in done:
                continue
            options = task["options"]
            t0 = time.monotonic()
            sums = [0.0] * len(options)
            rotations = []
            for shift in range(len(options)):
                rotated = options[shift:] + options[:shift]
                ids = qwen_chat(tok, task["state"], rotated, suite)
                hidden = model.model(input_ids=torch.tensor([ids], device=device), use_cache=False,
                                     return_dict=True).last_hidden_state[0, -1]
                lp = torch.log_softmax(model.lm_head(hidden)[letter_ids[:len(options)]].float(), dim=0).cpu().tolist()
                for i in range(len(rotated)):
                    sums[(shift + i) % len(options)] += lp[i]
                rotations.append({"shift": shift, "prompt_tokens": len(ids), "logprobs": lp})
            torch.cuda.synchronize(device)
            qwen_s = time.monotonic()-t0
            t0 = time.monotonic()
            question = ("Should the assistant use an external tool?" if suite == "metatool" else
                        "Which function should be called?" if suite == "bfcl" else
                        "Which next response best handles the user question?")
            state_vec, state_tokens = embed(task["state"] + "\n\n" + question, "tail")
            action_vecs = [embed(x, "head") for x in options]
            sp = F.normalize(head.state_head(state_vec), dim=-1)
            ap = F.normalize(head.action_head(torch.stack([x[0] for x in action_vecs])), dim=-1)
            clm_logits = (sp @ ap.T).float() * head.scale
            clm_probs = torch.softmax(clm_logits, dim=0).cpu().tolist()
            torch.cuda.synchronize(device)
            rec = {"id": task["id"], "source": task["source"], "gold": task["gold"],
                   "input_sha256": sha(json.dumps({"state": task["state"], "options": options}, ensure_ascii=False)),
                   "qwen_pred": max(range(len(options)), key=lambda i: sums[i]),
                   "clm_pred": max(range(len(options)), key=lambda i: clm_probs[i]),
                   "qwen_logprob_sums": sums, "qwen_rotations": rotations,
                   "clm_probs": clm_probs, "clm_state_tokens": state_tokens,
                   "clm_max_tokens": 2048, "qwen_seconds": qwen_s, "clm_seconds": time.monotonic()-t0}
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if n % 50 == 0 or n == len(tasks):
                print(f"{suite} GPU {n}/{len(tasks)} elapsed={time.monotonic()-start:.0f}s", flush=True)


def run_jev(tasks: list[dict], suite: str, output: Path, model: str) -> None:
    key = os.environ.get("TYPESAFE_API_KEY")
    if not key:
        raise SystemExit("TYPESAFE_API_KEY is required")
    done = {json.loads(x)["id"] for x in output.read_text().splitlines()} if output.exists() else set()
    output.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with output.open("a") as f:
        for n, task in enumerate(tasks, 1):
            if task["id"] in done:
                continue
            criteria = {LETTERS[i]: x for i, x in enumerate(task["options"])}
            question = ("Should the assistant use an external tool for this query?" if suite == "metatool" else
                        "Which function should be called for the user's request?" if suite == "bfcl" else
                        "Which candidate is the best next assistant response to the user's question?")
            body = {"model": model, "state": task["state"], "questions": {"decision": {
                "type": "choice", "instructions": question, "criteria": criteria}}}
            payload = json.dumps(body, ensure_ascii=False).encode()
            req = urllib.request.Request("https://api.typesafe.ai/v1/systemone", data=payload,
                                         headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
            t0 = time.monotonic()
            try:
                with urllib.request.urlopen(req, timeout=120) as response:
                    result = json.load(response)
            except Exception as exc:
                print(f"{suite} Jev stopped at {task['id']}: {type(exc).__name__}: {exc}", flush=True)
                raise
            answer = result.get("answers", {}).get("decision", {})
            probs = answer.get("probabilities", {})
            if answer.get("type") != "choice" or set(probs) != set(criteria):
                raise ValueError(f"Invalid Jev response for {task['id']}: {answer}")
            pred = LETTERS.index(max(probs, key=probs.get))
            rec = {"id": task["id"], "source": task["source"], "gold": task["gold"],
                   "input_sha256": sha(json.dumps({"state": task["state"], "options": task["options"]}, ensure_ascii=False)),
                   "prediction": pred, "probabilities": probs, "reported_choice": answer.get("choice"),
                   "model": result.get("model"), "usage": result.get("usage"),
                   "latency_s": time.monotonic()-t0, "request_sha256": hashlib.sha256(payload).hexdigest()}
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if n % 50 == 0 or n == len(tasks):
                print(f"{suite} Jev {n}/{len(tasks)} elapsed={time.monotonic()-start:.0f}s", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("suite", choices=["metatool", "when2call", "bfcl"])
    p.add_argument("mode", choices=["gpu", "jev"])
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--model", default=os.environ.get("MODEL_PATH", str(ROOT / "models/qwen3_8b")))
    p.add_argument("--head", default=os.environ.get("CLM_REFERENCE_HEAD", str(ROOT / "models/reference_head/CLM_v0.1-8B.pt")))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--jev-model", default="jev-1.13.0")
    p.add_argument("--limit", type=int)
    a = p.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    tasks = load_tasks(a.suite, tok)
    if a.limit is not None:
        tasks = tasks[:a.limit]
    print(f"Selected {len(tasks)} {a.suite} tasks", flush=True)
    if a.mode == "gpu":
        run_gpu(tasks, a.suite, a.output, a.model, a.head, a.device, tok)
    else:
        run_jev(tasks, a.suite, a.output, a.jev_model)


if __name__ == "__main__":
    main()
