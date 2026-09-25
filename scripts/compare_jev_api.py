"""Compare a fresh JevBench TypeSafe run with the committed Qwen/CLM records.

Rejects partial or failed runs so a missing API answer cannot be counted as a
wrong decision. The input run stays under private/; only aggregate metrics print.
"""
import argparse
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("jev_results", type=Path)
    args = p.parse_args()
    jev = read_jsonl(args.jev_results)
    qwen = read_jsonl(ROOT / "results/jevbench/qwen3_8b_letters_results.jsonl")
    clm = read_jsonl(ROOT / "results/jevbench/clm_public_2048_results.jsonl")
    ids = {r["id"] for r in qwen}
    if len(jev) != 231 or {r["task_id"] for r in jev} != ids:
        raise SystemExit("Jev run is incomplete or uses different task IDs")
    if any(not r["ok"] or not r["valid"] for r in jev):
        raise SystemExit("Jev run has failed/invalid requests; inspect raw records before scoring")
    models = {r["model"] for r in jev}
    if len(models) != 1:
        raise SystemExit(f"multiple resolved Jev versions in one run: {models}")
    by_id = {r["task_id"]: r for r in jev}
    if len(by_id) != 231:
        raise SystemExit("duplicate task IDs")
    qwen_by_id = {r["id"]: r for r in qwen}
    clm_by_id = {r["id"]: r for r in clm}
    latency = [r["latency_s"] for r in jev]
    input_tokens = [r.get("usage", {}).get("input_tokens") for r in jev]
    output = {
        "model": next(iter(models)), "n": 231,
        "correct": {
            "jev_fresh": sum(bool(r["correct"]) for r in jev),
            "qwen_local": sum(bool(r["correct"]) for r in qwen),
            "clm_local": sum(bool(r["correct"]) for r in clm),
        },
        "paired_jev_vs_qwen": {
            "jev_only": sum(by_id[k]["correct"] and not qwen_by_id[k]["correct"] for k in ids),
            "qwen_only": sum(qwen_by_id[k]["correct"] and not by_id[k]["correct"] for k in ids),
        },
        "paired_jev_vs_clm": {
            "jev_only": sum(by_id[k]["correct"] and not clm_by_id[k]["correct"] for k in ids),
            "clm_only": sum(clm_by_id[k]["correct"] and not by_id[k]["correct"] for k in ids),
        },
        "jev_latency_s": {"mean": statistics.mean(latency), "median": statistics.median(latency)},
        "jev_input_tokens": sum(input_tokens) if all(isinstance(x, (int, float)) for x in input_tokens) else None,
        "jev_reported_cost_usd": sum(r["cost_usd"] for r in jev) if all(isinstance(r.get("cost_usd"), (int, float)) for r in jev) else None,
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
