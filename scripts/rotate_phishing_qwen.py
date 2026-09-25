"""Counterbalance A/B labels for the saved Qwen phishing evaluation."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from run_phishing_comparison import ROOT, QUESTION, CRITERIA, digest, select_emails


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=ROOT / "external/jev-phishing-bench/data/emails.jsonl")
    p.add_argument("--first", type=Path, default=ROOT / "results/phishing/qwen_clm_2000_clm2k.jsonl")
    p.add_argument("--output", type=Path, default=ROOT / "results/phishing/qwen_rotated_2000.jsonl")
    p.add_argument("--model", default=str(ROOT / "models/qwen3_8b"))
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()
    data = select_emails(a.data, 2000, 42)
    first = {r["id"]: r for r in (json.loads(x) for x in a.first.read_text().splitlines())}
    assert len(first) == 2000 and set(first) == {r["id"] for r in data}
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.bfloat16, attn_implementation="sdpa", low_cpu_mem_usage=True
    ).eval().to(a.device)
    letter_ids = [tok.encode(x, add_special_tokens=False) for x in "AB"]
    assert all(len(x) == 1 for x in letter_ids)
    letter_ids = [x[0] for x in letter_ids]
    done = {json.loads(x)["id"] for x in a.output.read_text().splitlines()} if a.output.exists() else set()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with a.output.open("a") as f, torch.inference_mode():
        for i, row in enumerate(data, 1):
            if row["id"] in done:
                continue
            old = first[row["id"]]
            assert old["email_sha256"] == digest(row) and old["y"] == row["y"]
            choices = list(reversed(old["option_order"]))
            email = json.dumps(row["email"], ensure_ascii=False)
            prompt = (f"{QUESTION}\n\nEmail:\n{email}\n\nOptions:\n"
                      + "\n".join(f"{letter}. {name}: {CRITERIA[name]}" for letter, name in zip("AB", choices))
                      + "\n\nReply with one letter.\nAnswer:")
            chat = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                           add_generation_prompt=True, enable_thinking=False) + "Answer: "
            ids = tok.encode(chat, add_special_tokens=False)
            t0 = time.monotonic()
            hidden = model.model(input_ids=torch.tensor([ids], device=a.device), use_cache=False,
                                 return_dict=True).last_hidden_state[0, -1]
            probs2 = torch.softmax(model.lm_head(hidden)[letter_ids].float(), dim=0).cpu().tolist()
            torch.cuda.synchronize(a.device)
            scores = {name: math.log(max(old["qwen_probs"][old["option_order"].index(name)], 1e-30))
                      + math.log(max(probs2[choices.index(name)], 1e-30)) for name in CRITERIA}
            rec = {"id": row["id"], "y": row["y"], "email_sha256": digest(row),
                   "reverse_order": choices, "reverse_probs": probs2,
                   "logprob_sum": scores, "prediction": max(scores, key=scores.get),
                   "reverse_prompt_tokens": len(ids), "reverse_seconds": time.monotonic()-t0}
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if i % 100 == 0 or i == len(data):
                print(f"Qwen rotation {i}/{len(data)} elapsed={time.monotonic()-start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
