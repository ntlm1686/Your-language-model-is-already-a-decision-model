"""Check and summarize committed Qwen3.5-9B per-item results."""
from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {"jevbench": 231, "metatool": 1040, "when2call": 600,
            "bfcl": 200, "phishing": 2000, "webprm120": 120,
            "webprm1141": 1141, "deepswe": 38}


def summarize() -> dict:
    result = {}
    for suite, count in EXPECTED.items():
        paths = sorted((ROOT / "results/qwen3_5_9b").glob(f"{suite}_shard*of*.jsonl"))
        rows = [json.loads(line) for path in paths for line in path.read_text().splitlines()]
        key = lambda r: r.get("id", r.get("task", (r.get("source"), r.get("state_idx"))))
        if len(rows) != count or len({key(r) for r in rows}) != count:
            raise ValueError(f"{suite}: expected {count} unique rows, found {len(rows)}")
        latencies = [r["latency_s"] * 1000 for r in rows]
        token_counts = [d["tokens"] for row in rows for d in row.get("rotations", [])]
        token_counts.extend(row["tokens"] for row in rows if "tokens" in row)
        result[suite] = {"n": count, "correct": sum(bool(r["correct"]) for r in rows),
                         "p50_ms": round(statistics.median(latencies)),
                         "p95_ms": round(sorted(latencies)[int(.95*(len(latencies)-1))]),
                         "max_prompt_tokens": max(token_counts),
                         "mean_prompt_tokens": round(statistics.mean(token_counts))}
        shared = ROOT / "results/open_jev_9b" / f"{suite}.jsonl"
        if shared.exists() and suite != "deepswe":
            eligible = {r["id"] for r in (json.loads(x) for x in shared.read_text().splitlines())
                        if r["outcome"] == "ok"}
            def shared_key(row):
                return (f"{row['source']}:{row['state_idx']}" if suite.startswith("webprm")
                        else row["id"])
            indexed = {shared_key(row): row for row in rows}
            if not eligible <= set(indexed):
                raise ValueError(f"{suite}: Open-Jev eligible IDs missing from Qwen3.5 output")
            result[suite]["open_jev_eligible_n"] = len(eligible)
            result[suite]["correct_on_open_jev_eligible"] = sum(indexed[k]["correct"] for k in eligible)

    direct_rows = [json.loads(line) for line in (ROOT / "results/qwen3_5_9b/jevbench_shard0of1.jsonl").read_text().splitlines()]
    direct = {row["id"]: row["prediction"] for row in direct_rows}
    api_files = {"serial_torch": "jevbench_api.jsonl",
                 "batch_torch": "jevbench_api_batch_safe.jsonl",
                 "batch_fla": "jevbench_api_batched.jsonl"}
    result["jevbench_api"] = {}
    for name, filename in api_files.items():
        rows = [json.loads(line) for line in (ROOT / "results/qwen3_5_9b" / filename).read_text().splitlines()]
        indexed = {row["id"]: row for row in rows}
        if len(rows) != 231 or set(indexed) != set(direct):
            raise ValueError(f"{filename}: expected the same 231 unique JevBench IDs")
        result["jevbench_api"][name] = {
            "correct": sum(row["correct"] for row in rows),
            "prediction_agreement_with_direct": sum(indexed[k]["prediction"] == direct[k] for k in direct),
            "p50_ms": round(statistics.median(row["latency_s"] for row in rows) * 1000),
            "p95_ms": round(sorted(row["latency_s"] for row in rows)[int(.95 * 230)] * 1000),
        }
    return result


if __name__ == "__main__":
    output = summarize()
    path = ROOT / "results/qwen3_5_9b/summary.json"
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))
