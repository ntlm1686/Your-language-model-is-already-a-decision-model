"""Fresh Qwen3-8B letter-logprob evaluation on JevBench's 231 public tasks.

The answer key and provenance are never included in prompts. Cyclic rotations
put every candidate in every letter position once, then log probabilities are
summed by candidate. Run with one shard per GPU.
"""
import argparse
import glob
import json
import math
import time
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "external/jevbench"
MODEL = os.environ.get("MODEL_PATH", str(ROOT / "models/qwen3_8b"))


def tasks():
    rows = [json.loads(line) for p in glob.glob(str(BENCH / "datasets/public/*.jsonl"))
            for line in open(p)]
    rows.sort(key=lambda r: r["id"])
    assert len(rows) == 231 and len({r["id"] for r in rows}) == 231
    return rows


def description(row, key):
    q = row["question"]
    if q["type"] == "score":
        return q["criteria"][int(key)]
    if q["type"] == "noul":
        return q["criteria"]["true" if key == "yes" else "false"]
    return q["criteria"][key]


def prompt(row, order):
    state = row["state"] if isinstance(row["state"], str) else json.dumps(row["state"], ensure_ascii=False)
    options = "\n".join(f"{chr(65+i)}. {description(row, key)}" for i, key in enumerate(order))
    return ("Read the state and answer the question. Select the best option. "
            "Return one letter only.\n\nState:\n" + state +
            "\n\nQuestion:\n" + row["question"]["instructions"] +
            "\n\nOptions:\n" + options + "\n\nAnswer:")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--nshards", type=int, required=True)
    args = parser.parse_args()
    rows = [r for i, r in enumerate(tasks()) if i % args.nshards == args.shard]
    tok = AutoTokenizer.from_pretrained(str(MODEL))
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL), dtype=torch.bfloat16, attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).eval().to("cuda:0")
    letters = [tok.encode(chr(65+i), add_special_tokens=False) for i in range(6)]
    assert all(len(ids) == 1 for ids in letters)
    letter_ids = [ids[0] for ids in letters]
    output = ROOT / "results/jevbench" / f"qwen3_8b_letters_shard{args.shard}.jsonl"
    start = time.monotonic()
    with output.open("w") as handle, torch.inference_mode():
        for i, row in enumerate(rows, 1):
            labels = [str(x) for x in row["labels"]]
            n = len(labels)
            scores = {key: 0.0 for key in labels}
            rotations = []
            for rotation in range(n):
                order = labels[rotation:] + labels[:rotation]
                rendered = tok.apply_chat_template(
                    [{"role": "user", "content": prompt(row, order)}],
                    tokenize=False, add_generation_prompt=True,
                    enable_thinking=False,
                ) + "Answer: "
                ids = tok.encode(rendered, add_special_tokens=False)
                x = torch.tensor([ids], device="cuda:0")
                hidden = model.model(input_ids=x, use_cache=False,
                                     return_dict=True).last_hidden_state[0, -1]
                logits = model.lm_head(hidden)[letter_ids[:n]].float()
                logprobs = torch.log_softmax(logits, dim=0).cpu().tolist()
                for key, lp in zip(order, logprobs):
                    scores[key] += lp
                rotations.append({"order": order, "logprobs": logprobs,
                                  "input_tokens": len(ids)})
            pred = max(scores, key=scores.get)
            expected = str(row["expected"])
            result = {"id": row["id"], "tier": row["id"].split("-")[0],
                      "type": row["question"]["type"], "expected": expected,
                      "prediction": pred, "correct": pred == expected,
                      "scores": scores, "rotations": rotations}
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"shard {args.shard} {i}/{len(rows)} correct={result['correct']} "
                  f"elapsed={time.monotonic()-start:.0f}s", flush=True)
    print("DONE", output, flush=True)


if __name__ == "__main__":
    main()
