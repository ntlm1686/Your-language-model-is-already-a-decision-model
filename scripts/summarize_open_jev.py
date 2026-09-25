"""Check Open-Jev per-item records and summarize coverage, accuracy, and latency."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {"jevbench": 231, "metatool": 1040, "when2call": 600, "bfcl": 200,
            "phishing": 2000, "webprm120": 120, "webprm1141": 1141, "deepswe": 152}


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def baseline_counts(suite: str, ids: set[str]) -> dict[str, int]:
    """Recount existing baselines on exactly the Open-Jev eligible IDs."""
    if suite == "jevbench":
        files = {"qwen": "qwen3_8b_letters_results.jsonl",
                 "clm": "clm_public_2048_results.jsonl",
                 "jev": "jev_1_13_0_fresh_results.jsonl"}
        root = ROOT / "results/jevbench"
        records = {name: {r.get("task_id", r.get("id")): bool(r["correct"])
                          for r in read(root / filename)} for name, filename in files.items()}
    elif suite in ("metatool", "when2call", "bfcl"):
        root = ROOT / f"results/{suite}"
        gpu = {r["id"]: r for r in read(root / "qwen_clm_clm2k.jsonl")}
        jev = {r["id"]: r for r in read(root / "jev.jsonl")}
        records = {"qwen": {k: r["qwen_pred"] == r["gold"] for k, r in gpu.items()},
                   "clm": {k: r["clm_pred"] == r["gold"] for k, r in gpu.items()},
                   "jev": {k: r["prediction"] == r["gold"] for k, r in jev.items()}}
    elif suite == "phishing":
        root = ROOT / "results/phishing"
        gpu = {r["id"]: r for r in read(root / "qwen_clm_2000_clm2k.jsonl")}
        rotated = {r["id"]: r for r in read(root / "qwen_rotated_2000.jsonl")}
        jev = {r["id"]: r for r in read(root / "jev_2000.jsonl")}
        gold = lambda r: "phishing" if r["y"] else "legitimate"
        records = {"qwen": {k: r["prediction"] == gold(r) for k, r in rotated.items()},
                   "clm": {k: r["clm_pred"] == gold(r) for k, r in gpu.items()},
                   "jev": {k: r["prediction"] == gold(r) for k, r in jev.items()}}
    elif suite in ("webprm120", "webprm1141"):
        root = ROOT / "results/webprm"
        gpu_file = "qwen_clm_120_clm2k.jsonl" if suite == "webprm120" else "qwen_clm_1141_clm2k.jsonl"
        jev_file = "jev_1_13_0_fresh_results.jsonl" if suite == "webprm120" else "jev_1141.jsonl"
        key = lambda r: f"{r['source']}:{r['state_idx']}"
        gpu = {key(r): r for r in read(root / gpu_file)}
        jev = {key(r): r for r in read(root / jev_file)}
        records = {"qwen": {k: r["qwen_pred"] == r["gold"] for k, r in gpu.items()},
                   "clm": {k: r["clm_pred"] == r["gold"] for k, r in gpu.items()},
                   "jev": {k: r["prediction"] == r["gold"] for k, r in jev.items()}}
    else:
        return {}
    assert all(ids <= set(values) for values in records.values())
    return {name: sum(values[k] for k in ids) for name, values in records.items()}


def percentile(values: list[float], p: float) -> float:
    values = sorted(values)
    return values[math.ceil(p * len(values)) - 1]


def summarize(path: Path, suite: str) -> dict:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    assert len(rows) == EXPECTED[suite], (path, len(rows), EXPECTED[suite])
    assert len({r["id"] for r in rows}) == len(rows)
    assert {r["outcome"] for r in rows} <= {"ok", "too_long"}
    ok = [r for r in rows if r["outcome"] == "ok"]
    assert all(r["status"] == 200 and r["model"] in ("Qwen/Qwen3.5-2B", "Qwen/Qwen3.5-9B") for r in ok)
    values = [r["latency_s"] for r in ok]
    inference = [r["inference_seconds"] for r in ok]
    assert all(v is not None and v >= 0 for v in inference)
    result = {"total": len(rows), "eligible": len(ok), "too_long": len(rows)-len(ok),
              "p50_ms": round(1000 * statistics.median(values)) if values else None,
              "p95_ms": round(1000 * percentile(values, .95)) if values else None,
              "inference_p50_ms": round(1000 * statistics.median(inference)) if inference else None,
              "model": next(iter({r["model"] for r in ok})) if ok else None,
              "checkpoint_sha256": next(iter({r["checkpoint_sha256"] for r in ok})) if ok else None}
    if suite == "deepswe":
        by_task = defaultdict(list)
        for row in rows:
            by_task[row["task"]].append(row)
        assert len(by_task) == 38 and all(len(group) == 4 for group in by_task.values())
        scores, successful = {}, 0
        for task, group in by_task.items():
            if any(row["outcome"] != "ok" for row in group):
                continue
            accum = {name: 0.0 for name in group[0]["rewards"]}
            for row in group:
                for letter, run in zip("ABCD", row["order"]):
                    accum[run] += math.log(max(row["probabilities"][letter], 1e-30))
            selected = max(accum, key=accum.get)
            successful += bool(group[0]["rewards"][selected])
            scores[task] = {"selected": selected, "success": bool(group[0]["rewards"][selected])}
        result.update(tasks_eligible=len(scores), tasks_correct=successful)
    else:
        result["correct"] = sum(bool(r["correct"]) for r in ok)
        result["baseline_correct_on_eligible"] = baseline_counts(suite, {r["id"] for r in ok})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "results")
    parser.add_argument("--output", type=Path, default=ROOT / "results/open_jev_summary.json")
    args = parser.parse_args()
    output = {}
    for size in ("2b", "9b"):
        directory = args.root / f"open_jev_{size}"
        output[size] = {}
        for suite in EXPECTED:
            path = directory / f"{suite}.jsonl"
            if path.exists():
                output[size][suite] = summarize(path, suite)
        extended = directory / "deepswe_8k.jsonl"
        if extended.exists():
            output[size]["deepswe_8k_exploratory"] = summarize(extended, "deepswe")
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
