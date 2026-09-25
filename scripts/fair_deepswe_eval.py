"""Fresh CLM and Qwen letter scoring on the same raw DeepSWE candidates.

Both methods use the same twelve normalized final agent turns per candidate.
This is a new candidate set; it cannot reproduce the published Opus-5 31/38.
"""
from __future__ import annotations

import argparse
import gzip
import json
import random
import sys
import time
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
CLM_REPO = Path(os.environ.get("CLM_REPO", ROOT / "external/CLM"))
sys.path.insert(0, str(CLM_REPO))
from evaluation.bon_eval import load_heads  # noqa: E402


RAW = ROOT / "data/deepswe/runs"
RUNS = [
    "qwen3.8-27b-mini-swe-xhigh",
    "qwen3.8-27b-claude-code-xhigh",
    "qwen3.8-27b-pi-medium",
    "qwen3.8-27b-pi-xhigh",
]
LETTERS = "ABCD"


def shorten(s, limit: int) -> str:
    if not isinstance(s, str):
        s = json.dumps(s, ensure_ascii=False)
    if len(s) <= limit:
        return s
    a = limit * 3 // 5
    return s[:a] + "\n[…clipped…]\n" + s[-(limit-a):]


def content_text(content) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content or "")
    parts = []
    for block in content:
        if not isinstance(block, dict):
            continue
        typ = block.get("type", "content")
        v = block.get("text") or block.get("arguments") or block
        parts.append(f"{typ}: {shorten(v, 2400)}")
    return "\n".join(parts)


def extract(task: str, run: str, tail: int | None = 12):
    path = RAW / run / "tasks" / f"datacurve__{task}" / "trajectory.json.gz"
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        steps = json.load(handle)["steps"]
    user = next(s for s in steps if s.get("source") == "user" or s.get("role") == "user")
    instruction = user.get("message") or content_text(user.get("content"))
    turns = []
    if run.endswith("mini-swe-xhigh") or "claude-code" in run:
        for s in steps:
            if s.get("source") != "agent":
                continue
            action = "\n".join(
                f"{name}: {content_text(s[name]) if name == 'content' else shorten(s[name], 10000)}"
                for name in ("message", "reasoning_content", "tool_calls") if s.get(name)
            )
            observation = s.get("observation")
            turns.append({"action": action or "(empty agent turn)",
                          "observation": "" if not observation else shorten(observation, 10000)})
    else:
        for s in steps:
            role = s.get("role")
            if role == "assistant":
                turns.append({"action": content_text(s.get("content")) or "(empty agent turn)",
                              "observation": ""})
            elif role == "tool" and turns:
                value = content_text(s.get("content"))
                turns[-1]["observation"] += ("\n" if turns[-1]["observation"] else "") + value
    return instruction, turns[-tail:] if tail else turns


def normalize_turns(turns: list[dict], cap: int) -> list[dict]:
    return [{"action": shorten(t["action"], cap),
             "observation": shorten(t["observation"], cap)} for t in turns]


def make_prompt(task: str, instruction: str, order: list[str], turns_by_run: dict) -> str:
    out = [
        "Which completed coding-agent attempt is most likely to pass the hidden tests?",
        "Use the task and the final 12 action/observation turns of each attempt.",
        "The attempts are anonymous. Return exactly one letter: A, B, C, or D.",
        "", f"Task: {task}", instruction, "", "Attempts:",
    ]
    for letter, run in zip(LETTERS, order):
        lines = [f"{letter}."]
        for i, turn in enumerate(turns_by_run[run], 1):
            lines.append(f"Turn {i} action:\n{turn['action']}")
            if turn["observation"]:
                lines.append(f"Turn {i} observed:\n{turn['observation']}")
        out.append("\n".join(lines))
    out.append("Answer:")
    return "\n\n".join(out)


def prepare():
    tok = AutoTokenizer.from_pretrained(os.environ.get("MODEL_PATH", str(ROOT / "models/qwen3_8b")))
    tasks = json.loads((ROOT / "data/deepswe/heldout_tasks.json").read_text())
    records = []
    for task in tasks:
        extracted = {run: extract(task, run) for run in RUNS}
        instruction = shorten(extracted[RUNS[0]][0], 6000)
        rewards = {}
        for run in RUNS:
            p = RAW / run / "tasks" / f"datacurve__{task}" / "result.json"
            x = json.loads(p.read_text())
            rewards[run] = int((x.get("verifier") or {}).get("reward") == 1)
        base = RUNS.copy()
        random.Random(f"{task}-42").shuffle(base)
        for cap in [700, 600, 500, 400, 300]:
            turns = {run: normalize_turns(extracted[run][1], cap) for run in RUNS}
            orders = [base[i:] + base[:i] for i in range(4)]
            prompts = [make_prompt(task, instruction, order, turns) for order in orders]
            n = []
            for prompt in prompts:
                rendered = tok.apply_chat_template(
                    [{"role": "user", "content": prompt}], tokenize=False,
                    add_generation_prompt=True, enable_thinking=False,
                ) + "Answer: "
                n.append(len(tok.encode(rendered, add_special_tokens=False)))
            if max(n) <= 16384:
                break
        else:
            raise ValueError(f"Prompt exceeds 16K at smallest cap: {task}")
        records.append({"task": task, "instruction": instruction, "turns": turns,
                        "orders": orders, "rewards": rewards, "cap": cap, "qwen_tokens": n})
    out = ROOT / "data/deepswe" / "fair_inputs.jsonl"
    with out.open("w") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    print("prepared",len(records),"tasks", "qwen token range",min(min(r["qwen_tokens"]) for r in records),max(max(r["qwen_tokens"]) for r in records),"out",out)


def score(shard: int, nshards: int):
    inp = ROOT / "data/deepswe" / "fair_inputs.jsonl"
    all_rows = [json.loads(line) for line in inp.open()]
    rows = [x for i, x in enumerate(all_rows) if i % nshards == shard]
    model_path = os.environ.get("MODEL_PATH", str(ROOT / "models/qwen3_8b"))
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16,
        attn_implementation="sdpa", low_cpu_mem_usage=True,
    ).eval().to("cuda:0")
    head_path = os.environ.get("CLM_DEEPSWE_HEAD", str(ROOT / "models/deepswe_head/best_head.pt"))
    sh, ah, cfg = load_heads(head_path, torch.device("cuda:0"))
    letter_ids = [tok.encode(c, add_special_tokens=False)[0] for c in LETTERS]
    assert all(len(tok.encode(c, add_special_tokens=False)) == 1 for c in LETTERS)
    action_cache = {}

    @torch.inference_mode()
    def hidden(ids: list[int], normalize: bool = True):
        x = torch.tensor([ids], device="cuda:0")
        h = model.model(input_ids=x, use_cache=False, return_dict=True).last_hidden_state[0, -1].float()
        return F.normalize(h, dim=-1) if normalize else h

    def state_ids(instruction: str, prior: list[dict]) -> list[int]:
        messages = [{"role": "user", "content": instruction}]
        for t in prior:
            messages.append({"role": "assistant", "content": t["action"]})
            if t["observation"]:
                messages.append({"role": "user", "content": t["observation"]})
        ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
        return list(ids[-8191:])

    out = ROOT / "results/deepswe" / f"fair_scores_shard{shard}.jsonl"
    start = time.monotonic()
    with out.open("w") as handle, torch.inference_mode():
        for i, row in enumerate(rows, 1):
            torch.cuda.synchronize()
            clm_start = time.monotonic()
            clm = {}
            for run in RUNS:
                turns = row["turns"][run]
                scores = []
                for j, turn in enumerate(turns):
                    q = hidden(state_ids(row["instruction"], turns[:j]))
                    action = turn["action"]
                    if action not in action_cache:
                        ids = tok.encode(action, add_special_tokens=False)[:8191]
                        action_cache[action] = hidden(ids)
                    a = action_cache[action]
                    zq = F.normalize(sh(q), dim=-1)
                    za = F.normalize(ah(a), dim=-1)
                    scores.append(float((zq * za).sum()))
                clm[run] = {"score": sum(scores) / len(scores), "steps": len(scores),
                            "step_scores": scores}
            torch.cuda.synchronize()
            clm_seconds = time.monotonic() - clm_start
            qwen_start = time.monotonic()
            qwen = []
            for order in row["orders"]:
                prompt = make_prompt(row["task"], row["instruction"], order, row["turns"])
                rendered = tok.apply_chat_template(
                    [{"role": "user", "content": prompt}], tokenize=False,
                    add_generation_prompt=True, enable_thinking=False,
                ) + "Answer: "
                ids = tok.encode(rendered, add_special_tokens=False)
                last = hidden(ids, normalize=False)
                logits = model.lm_head(last.to(model.lm_head.weight.dtype))[letter_ids].float()
                probs = F.softmax(logits, dim=0).cpu().tolist()
                qwen.append({"order": order, "probs": probs, "tokens": len(ids)})
            torch.cuda.synchronize()
            qwen_seconds = time.monotonic() - qwen_start
            result = {"task": row["task"], "rewards": row["rewards"],
                      "clm": clm, "qwen": qwen, "cap": row["cap"],
                      "timing": {"clm_seconds": clm_seconds,
                                 "qwen_seconds": qwen_seconds}}
            handle.write(json.dumps(result) + "\n")
            handle.flush()
            cpick = max(clm, key=lambda k: clm[k]["score"])
            print(f"shard {shard} {i}/{len(rows)} {row['task']} clm_pass={row['rewards'][cpick]} "
                  f"elapsed={time.monotonic()-start:.0f}s", flush=True)
    print("DONE",out,flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["prepare", "score"])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args()
    if args.command == "prepare":
        prepare()
    else:
        score(args.shard, args.nshards)


if __name__ == "__main__":
    main()
