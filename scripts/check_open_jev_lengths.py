"""Audit 4,096-token eligibility with Open-Jev's own candidate renderer/tokenizer."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer
from jev.api import candidate_prompts, compile_request

from run_open_jev import LOADERS, ROOT

BASES = {
    "2b": ("Qwen/Qwen3.5-2B", "15852e8c16360a2fea060d615a32b45270f8a8fc"),
    "9b": ("Qwen/Qwen3.5-9B", "c202236235762e1c871ad0ccb60c8ee5ba337b9a"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=BASES, required=True)
    parser.add_argument("--suite", choices=LOADERS, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    model_id, revision = BASES[args.model]
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    tok.padding_side = "right"
    tasks = LOADERS[args.suite]()
    records = []
    for index, task in enumerate(tasks, 1):
        body = task["body"]
        compiled = compile_request(body["state"], body["questions"])
        rendered = [tok.apply_chat_template([{"role": "user", "content": text}],
                                            tokenize=False, add_generation_prompt=True,
                                            enable_thinking=False)
                    for record in compiled for text in candidate_prompts(record)]
        lengths = [len(ids) for ids in tok(rendered, add_special_tokens=False)["input_ids"]]
        records.append({"id": task["id"], "max_candidate_tokens": max(lengths),
                        "eligible": max(lengths) <= 4096})
        if index % 500 == 0:
            print(f"{args.suite} {index}/{len(tasks)}", flush=True)
    report = {"model": model_id, "revision": revision, "suite": args.suite,
              "total": len(records), "eligible": sum(r["eligible"] for r in records),
              "max_candidate_tokens": max(r["max_candidate_tokens"] for r in records),
              "items": records}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "items"}, indent=2))


if __name__ == "__main__":
    main()
