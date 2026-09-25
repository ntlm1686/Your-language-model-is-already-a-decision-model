"""Evaluate untrained Qwen3.5-9B with the archived Qwen3-8B letter prompts.

The text backbone and original LM head are taken from the multimodal checkpoint;
no decision-specific weights or images are used. Outputs are append-only and
shardable. Archived Qwen3-8B tokenizer selects the exact original benchmark IDs.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

from compare_qwen_clm_webprm import ROOT, build_prompt
from fair_deepswe_eval import make_prompt as deep_prompt
from qwen3_8b_letters import prompt as jev_prompt, tasks as jev_tasks
from run_jev_webprm import prepare as web_prepare
from run_phishing_comparison import CRITERIA, QUESTION, digest, select_emails
from run_tool_decisions import load_tasks, sha

MODEL = "Qwen/Qwen3.5-9B"
REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
DEFAULT_OLD_TOKENIZER = ROOT / "models/qwen3_8b_tokenizer"
if not DEFAULT_OLD_TOKENIZER.exists():
    DEFAULT_OLD_TOKENIZER = ROOT / "models/qwen3_8b"
OLD_TOKENIZER = Path(os.environ.get("QWEN3_8B_TOKENIZER", str(DEFAULT_OLD_TOKENIZER)))
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


class LetterModel:
    def __init__(self, device: str):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
        full = AutoModelForImageTextToText.from_pretrained(
            MODEL, revision=REVISION, local_files_only=True, dtype=torch.bfloat16,
            attn_implementation="sdpa", device_map={"": device}).eval()
        self.backbone = full.model.language_model
        self.head = full.get_output_embeddings()
        del full
        self.letter_ids = [self._one(x) for x in LETTERS]

    def _one(self, token: str) -> int:
        ids = self.tokenizer.encode(token, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"{token!r} is not one token: {ids}")
        return ids[0]

    @torch.inference_mode()
    def score(self, content: str, n: int, *, max_length: int = 16384) -> tuple[list[float], int, float]:
        chat = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False) + "Answer: "
        ids = self.tokenizer.encode(chat, add_special_tokens=False)
        if len(ids) > max_length:
            raise ValueError(f"prompt has {len(ids)} tokens; cap {max_length}")
        torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        h = self.backbone(input_ids=torch.tensor([ids], device=self.device),
                          use_cache=False, return_dict=True).last_hidden_state[0, -1]
        logits = self.head(h)[self.letter_ids[:n]].float()
        lp = torch.log_softmax(logits, dim=0).cpu().tolist()
        torch.cuda.synchronize(self.device)
        return lp, len(ids), time.perf_counter() - start


def archived(path: Path) -> dict:
    return {r.get("id", (r.get("source"), r.get("state_idx"))): r
            for r in (json.loads(s) for s in path.read_text().splitlines())}


def cases(suite: str) -> list[dict]:
    if suite == "jevbench":
        return [{"id": r["id"], "gold": str(r["expected"]), "row": r} for r in jev_tasks()]
    if suite in ("metatool", "when2call", "bfcl"):
        old = AutoTokenizer.from_pretrained(OLD_TOKENIZER, local_files_only=True)
        items = load_tasks(suite, old)
        baseline = archived(ROOT / f"results/{suite}/qwen_clm_clm2k.jsonl")
        assert set(baseline) == {x["id"] for x in items}
        for x in items:
            assert baseline[x["id"]]["gold"] == x["gold"]
            assert baseline[x["id"]]["input_sha256"] == sha(json.dumps(
                {"state": x["state"], "options": x["options"]}, ensure_ascii=False))
        return items
    if suite == "phishing":
        items = select_emails(ROOT / "external/jev-phishing-bench/data/emails.jsonl", 2000, 42)
        baseline = archived(ROOT / "results/phishing/qwen_clm_2000_clm2k.jsonl")
        for x in items:
            assert baseline[x["id"]]["email_sha256"] == digest(x)
            x["order"] = baseline[x["id"]]["option_order"]
        return items
    if suite in ("webprm120", "webprm1141"):
        small = suite == "webprm120"
        name = "qwen_clm_120_clm2k.jsonl" if small else "qwen_clm_1141_clm2k.jsonl"
        return web_prepare(ROOT / "data/webprm/test-00000-of-00001.jsonl",
                           str(OLD_TOKENIZER), 20 if small else 999999,
                           ROOT / "results/webprm" / name)
    if suite == "deepswe":
        items = [json.loads(x) for x in (ROOT / "data/deepswe/fair_inputs.jsonl").read_text().splitlines()]
        baseline = [json.loads(x) for x in (ROOT / "results/deepswe/fair_results.jsonl").read_text().splitlines()]
        assert {x["task"] for x in items} == {x["task"] for x in baseline}
        return items
    raise ValueError(suite)


def evaluate(suite: str, row: dict, model: LetterModel) -> dict:
    if suite == "jevbench":
        source = row["row"]
        labels = [str(x) for x in source["labels"]]
        scores = {x: 0.0 for x in labels}
        details = []
        for shift in range(len(labels)):
            order = labels[shift:] + labels[:shift]
            lp, tokens, seconds = model.score(jev_prompt(source, order), len(order), max_length=32768)
            for name, value in zip(order, lp):
                scores[name] += value
            details.append({"order": order, "logprobs": lp, "tokens": tokens, "seconds": seconds})
        pred = max(scores, key=scores.get)
        return {"id": row["id"], "gold": row["gold"], "prediction": pred,
                "correct": pred == row["gold"], "scores": scores, "rotations": details,
                "latency_s": sum(x["seconds"] for x in details)}
    if suite in ("metatool", "when2call", "bfcl"):
        options = row["options"]
        scores = [0.0] * len(options)
        details = []
        for shift in range(len(options)):
            rotated = options[shift:] + options[:shift]
            from run_tool_decisions import task_prompt
            lp, tokens, seconds = model.score(task_prompt(row["state"], rotated, suite),
                                               len(options), max_length=32768)
            for i, value in enumerate(lp):
                scores[(shift + i) % len(options)] += value
            details.append({"shift": shift, "logprobs": lp, "tokens": tokens, "seconds": seconds})
        pred = max(range(len(scores)), key=scores.__getitem__)
        return {"id": row["id"], "gold": row["gold"], "prediction": pred,
                "correct": pred == row["gold"], "scores": scores,
                "input_sha256": sha(json.dumps({"state": row["state"], "options": options}, ensure_ascii=False)),
                "rotations": details, "latency_s": sum(x["seconds"] for x in details)}
    if suite == "phishing":
        email = json.dumps(row["email"], ensure_ascii=False)
        orders = [row["order"], list(reversed(row["order"]))]
        scores = {x: 0.0 for x in CRITERIA}
        details = []
        for order in orders:
            prompt = (f"{QUESTION}\n\nEmail:\n{email}\n\nOptions:\n" +
                      "\n".join(f"{letter}. {name}: {CRITERIA[name]}" for letter, name in zip("AB", order)) +
                      "\n\nReply with one letter.\nAnswer:")
            lp, tokens, seconds = model.score(prompt, 2, max_length=32768)
            for name, value in zip(order, lp):
                scores[name] += value
            details.append({"order": order, "logprobs": lp, "tokens": tokens, "seconds": seconds})
        pred = max(scores, key=scores.get)
        return {"id": row["id"], "gold": "phishing" if row["y"] else "legitimate",
                "prediction": pred, "correct": pred == ("phishing" if row["y"] else "legitimate"),
                "email_sha256": digest(row), "scores": scores, "rotations": details,
                "latency_s": sum(x["seconds"] for x in details)}
    if suite in ("webprm120", "webprm1141"):
        lp, tokens, seconds = model.score(build_prompt(row["state"], row["options"]), 5,
                                          max_length=32768)
        pred = max(range(5), key=lp.__getitem__)
        return {"source": row["source"], "state_idx": row["state_idx"],
                "gold": row["gold"], "prediction": pred, "correct": pred == row["gold"],
                "logprobs": lp, "tokens": tokens, "latency_s": seconds}
    if suite == "deepswe":
        scores = {run: 0.0 for run in row["rewards"]}
        details = []
        for order in row["orders"]:
            lp, tokens, seconds = model.score(
                deep_prompt(row["task"], row["instruction"], order, row["turns"]),
                4, max_length=32768)
            for run, value in zip(order, lp):
                scores[run] += value
            details.append({"order": order, "logprobs": lp, "tokens": tokens, "seconds": seconds})
        pred = max(scores, key=scores.get)
        return {"task": row["task"], "gold": row["rewards"], "prediction": pred,
                "correct": bool(row["rewards"][pred]), "scores": scores,
                "rotations": details, "latency_s": sum(x["seconds"] for x in details)}
    raise ValueError(suite)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", choices=["jevbench", "metatool", "when2call", "bfcl",
                                           "phishing", "webprm120", "webprm1141", "deepswe"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--nshards", type=int, default=1)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    items = cases(args.suite)
    if args.limit is not None:
        items = items[:args.limit]
    items = [x for i, x in enumerate(items) if i % args.nshards == args.shard]
    output = ROOT / "results/qwen3_5_9b" / f"{args.suite}_shard{args.shard}of{args.nshards}.jsonl"
    output.parent.mkdir(exist_ok=True)
    key = lambda x: x.get("id", x.get("task", (x.get("source"), x.get("state_idx"))))
    done = {key(json.loads(line)) for line in output.read_text().splitlines()} if output.exists() else set()
    print(f"{args.suite}: {len(items)} selected, {len(done)} already complete", flush=True)
    model = LetterModel(args.device)
    start = time.monotonic()
    with output.open("a") as handle:
        for n, row in enumerate(items, 1):
            if key(row) in done:
                continue
            result = evaluate(args.suite, row, model)
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            if n % 25 == 0 or n == len(items):
                print(f"{args.suite} {n}/{len(items)} elapsed={time.monotonic()-start:.0f}s", flush=True)
    records = [json.loads(x) for x in output.read_text().splitlines()]
    print(json.dumps({"suite": args.suite, "count": len(records), "correct": sum(x["correct"] for x in records),
                      "p50_ms": round(statistics.median(x["latency_s"] for x in records)*1000)}), flush=True)


if __name__ == "__main__":
    main()
