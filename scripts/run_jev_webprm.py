"""Evaluate Jev on exactly the 120 WebPRM states used by Qwen and CLM.

Selection and candidate order are reconstructed from the original seed and
checked against the committed Qwen/CLM records before any API call.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
import urllib.error
import urllib.request
from pathlib import Path

from transformers import AutoTokenizer

from compare_qwen_clm_webprm import ROOT, build_prompt, load_states

LETTERS = "ABCDE"
INSTRUCTIONS = ("Choose the next assistant response that best advances the user's web task. "
                "Use the page state and previous actions.")


def prepare(data: Path, model_path: str, per_source: int, reference: Path) -> list[dict]:
    tok = AutoTokenizer.from_pretrained(model_path)
    groups: dict[str, list[dict]] = {}
    for state in load_states(data):
        groups.setdefault(state["source"], []).append(state)
    rng = random.Random(42)
    selected = []
    for source, states in sorted(groups.items()):
        rng.shuffle(states)
        taken = 0
        for state in states:
            options = [state["positive"], *state["negatives"]]
            rng.shuffle(options)
            prompt = build_prompt(state["state"], options)
            chat = tok.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False,
                add_generation_prompt=True, enable_thinking=False,
            ) + "Answer: "
            if len(tok.encode(chat, add_special_tokens=False)) > 8192:
                continue
            selected.append({"source": source, "state_idx": state["state_idx"],
                             "gold": options.index(state["positive"]),
                             "state": state["state"], "options": options})
            taken += 1
            if taken == per_source:
                break
    archived = [json.loads(line) for line in
                reference.read_text().splitlines()]
    expected = {(r["source"], r["state_idx"]): r["gold"] for r in archived}
    actual = {(r["source"], r["state_idx"]): r["gold"] for r in selected}
    if len(archived) != len(selected) or actual != expected:
        raise ValueError("WebPRM state IDs or candidate order differ from archived baseline")
    return selected


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=ROOT / "data/webprm/test-00000-of-00001.jsonl")
    p.add_argument("--model-path", default=os.environ.get("MODEL_PATH", str(ROOT / "models/qwen3_8b")))
    p.add_argument("--jev-model", default="jev-1.13.0")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--limit", type=int, default=120)
    p.add_argument("--per-source", type=int, default=20)
    p.add_argument("--reference", type=Path, default=ROOT / "results/webprm/qwen_clm_120_clm2k.jsonl")
    args = p.parse_args()
    key = os.environ.get("TYPESAFE_API_KEY")
    if not key:
        raise SystemExit("TYPESAFE_API_KEY is required")
    selected = prepare(args.data, args.model_path, args.per_source, args.reference)
    if not 1 <= args.limit <= len(selected):
        raise SystemExit(f"limit must be between 1 and {len(selected)}")
    done = { (r["source"], r["state_idx"]) for r in
             (json.loads(line) for line in args.output.read_text().splitlines()) } if args.output.exists() else set()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a") as handle:
        for i, row in enumerate(selected[:args.limit], 1):
            if (row["source"], row["state_idx"]) in done:
                continue
            question = {"type": "choice", "instructions": INSTRUCTIONS,
                        "criteria": dict(zip(LETTERS, row["options"]))}
            body = {"model": args.jev_model, "state": row["state"],
                    "questions": {"decision": question}}
            payload = json.dumps(body, ensure_ascii=False).encode()
            req = urllib.request.Request(
                "https://api.typesafe.ai/v1/systemone", data=payload, method="POST",
                headers={"Authorization": f"Bearer {key}",
                         "Content-Type": "application/json"},
            )
            start = time.perf_counter()
            try:
                with urllib.request.urlopen(req, timeout=120) as response:
                    status = response.status
                    result = json.load(response)
            except urllib.error.HTTPError as exc:
                status = exc.code
                result = {"error": exc.read().decode(errors="replace")[:300]}
            latency = time.perf_counter() - start
            answer = result.get("answers", {}).get("decision", {})
            probs = answer.get("probabilities", {})
            valid = (status == 200 and answer.get("type") == "choice" and
                     set(probs) == set(LETTERS) and
                     answer.get("choice") in LETTERS)
            prediction = LETTERS.index(max(probs, key=probs.get)) if valid else None
            record = {"source": row["source"], "state_idx": row["state_idx"],
                      "gold": row["gold"], "prediction": prediction,
                      "correct": prediction == row["gold"], "valid": valid,
                      "probabilities": probs if valid else None,
                      "reported_choice": answer.get("choice"),
                      "model": result.get("model"), "usage": result.get("usage"),
                      "latency_s": latency, "status_code": status,
                      "request_sha256": hashlib.sha256(payload).hexdigest()}
            if not valid:
                record["error"] = result.get("error") or "invalid answer"
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
            handle.flush()
            if i % 10 == 0 or not valid or i == args.limit:
                print(f"{i}/{args.limit} valid={valid}", flush=True)
            if not valid:
                raise SystemExit(f"stopped on invalid response at item {i}")


if __name__ == "__main__":
    main()
