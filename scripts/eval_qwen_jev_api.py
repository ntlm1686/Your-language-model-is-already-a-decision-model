"""Run the public JevBench 231 through the local Qwen Jev-format API.

Choice criteria are sent in the archived Qwen evaluator's `labels` order;
JevBench JSON object order differs on 119 items. Results are append-only and
resumable. No answer key is transmitted to the model service.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def tasks() -> list[dict]:
    base = ROOT / "external/jevbench/datasets/public"
    rows = [json.loads(line) for name in ("easy", "original", "hard")
            for line in (base / f"{name}.jsonl").read_text().splitlines() if line]
    rows.sort(key=lambda row: row["id"])
    if len(rows) != 231 or len({row["id"] for row in rows}) != 231:
        raise ValueError("expected 231 unique public JevBench rows; run setup_sources.py repos")
    return rows


def body_for(row: dict) -> dict:
    question = copy.deepcopy(row["question"])
    if question["type"] == "choice":
        criteria = question["criteria"]
        if set(criteria) != set(row["labels"]):
            raise ValueError(f"candidate labels differ from criteria on {row['id']}")
        question["criteria"] = {key: criteria[key] for key in row["labels"]}
    return {"model": "qwen3-8b", "state": row["state"], "questions": {"decision": question}}


def post(url: str, body: dict, timeout: float) -> tuple[int, dict, float]:
    data = json.dumps(body, ensure_ascii=False, allow_nan=False).encode()
    request = urllib.request.Request(url.rstrip("/") + "/v1/systemone", data=data,
                                     headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.load(response), time.perf_counter() - start
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read()), time.perf_counter() - start


def evaluate(row: dict, url: str, timeout: float) -> dict:
    body = body_for(row)
    request_sha = hashlib.sha256(json.dumps(body, ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    status, response, latency = post(url, body, timeout)
    result = {"id": row["id"], "type": row["question"]["type"],
              "expected": str(row["expected"]), "request_sha256": request_sha,
              "status": status, "latency_s": latency}
    if status == 422 and "exceeds max_length" in str(response.get("error", "")):
        result.update(outcome="too_long", reason=response["error"])
        return result
    if status != 200:
        raise RuntimeError(f"{row['id']}: HTTP {status}: {response}")
    if response.get("model") != "qwen3-8b":
        raise ValueError(f"{row['id']}: unexpected model {response.get('model')}")
    answers = response.get("answers", {})
    if set(answers) != {"decision"}:
        raise ValueError(f"{row['id']}: invalid answer object")
    answer = answers["decision"]
    kind = row["question"]["type"]
    if answer.get("type") != kind:
        raise ValueError(f"{row['id']}: answer type mismatch")
    if kind == "noul":
        p = answer.get("noul")
        if isinstance(p, bool) or not isinstance(p, (int, float)) or not math.isfinite(p) or not 0 <= p <= 1:
            raise ValueError(f"{row['id']}: invalid Noul probability")
        probs, pred = {"no": 1-p, "yes": p}, "yes" if p > 0.5 else "no"
    else:
        probs = answer.get("probabilities", {})
        expected_keys = (set(row["labels"]) if kind == "choice" else
                         {str(i) for i in range(len(row["question"]["criteria"]))})
        if not isinstance(probs, dict) or set(probs) != expected_keys or any(
                isinstance(p, bool) or not isinstance(p, (int, float)) or
                not math.isfinite(p) or not 0 <= p <= 1 for p in probs.values()):
            raise ValueError(f"{row['id']}: invalid probability map")
        if not math.isclose(sum(probs.values()), 1.0, abs_tol=1e-6, rel_tol=1e-6):
            raise ValueError(f"{row['id']}: probabilities do not sum to one")
        pred = max(probs, key=probs.get)
        if kind == "choice" and answer.get("choice") != pred:
            raise ValueError(f"{row['id']}: reported Choice disagrees with probabilities")
    usage = response.get("usage", {})
    if not isinstance(usage.get("input_tokens"), int) or usage["input_tokens"] <= 0:
        raise ValueError(f"{row['id']}: invalid token usage")
    metadata = response.get("metadata", {})
    result.update(outcome="ok", prediction=pred, correct=pred == str(row["expected"]),
                  probabilities=probs, usage=usage, method=metadata.get("method"),
                  inference_seconds=metadata.get("inference_seconds"),
                  forward_passes=metadata.get("forward_passes"),
                  model_max_length=metadata.get("max_length"))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8795")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--timeout", type=float, default=180)
    args = parser.parse_args()
    rows = tasks()
    if args.limit is not None:
        if not 1 <= args.limit <= len(rows):
            raise ValueError("limit must be between 1 and 231")
        rows = rows[:args.limit]
    existing = [json.loads(line) for line in args.output.read_text().splitlines() if line] if args.output.exists() else []
    by_id = {row["id"]: row for row in rows}
    if len({record["id"] for record in existing}) != len(existing):
        raise ValueError("output contains duplicate IDs")
    for record in existing:
        row = by_id.get(record["id"])
        if row is None:
            raise ValueError(f"output contains unexpected ID {record['id']}")
        digest = hashlib.sha256(json.dumps(body_for(row), ensure_ascii=False, allow_nan=False).encode()).hexdigest()
        if record["request_sha256"] != digest:
            raise ValueError(f"request changed for resumed item {row['id']}")
    done = {record["id"] for record in existing}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a") as handle:
        for i, row in enumerate(rows, 1):
            if row["id"] in done:
                continue
            record = evaluate(row, args.url, args.timeout)
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
            if i % 10 == 0 or i == len(rows):
                print(f"{i}/{len(rows)} {row['id']} {record['outcome']}", flush=True)
    final = [json.loads(line) for line in args.output.read_text().splitlines() if line]
    eligible = [record for record in final if record["outcome"] == "ok"]
    summary = {"count": len(final), "eligible": len(eligible),
               "correct": sum(record["correct"] for record in eligible),
               "p50_latency_ms": round(1000 * statistics.median(record["latency_s"] for record in eligible), 2)
               if eligible else None,
               "input_tokens": sum(record["usage"]["input_tokens"] for record in eligible)}
    baseline_path = ROOT / "results/jevbench/qwen3_8b_letters_results.jsonl"
    if baseline_path.exists():
        baseline = {record["id"]: record for record in
                    (json.loads(line) for line in baseline_path.read_text().splitlines() if line)}
        if set(record["id"] for record in eligible) <= set(baseline):
            disagreements = [record["id"] for record in eligible
                             if record["prediction"] != baseline[record["id"]]["prediction"]]
            summary["archived_qwen_comparison_on_eligible_ids"] = {
                "baseline_correct": sum(baseline[record["id"]]["correct"] for record in eligible),
                "prediction_agreement": len(eligible) - len(disagreements),
                "prediction_disagreement_ids": disagreements,
            }
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
