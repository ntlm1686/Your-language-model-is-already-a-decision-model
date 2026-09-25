"""Evaluate a locally served Open-Jev checkpoint on the archived decision sets.

Start the upstream `python -m jev.server` first. This client reconstructs the
same states/options as our Jev runs, checks archived item IDs and input hashes,
and writes one resumable JSONL record per item. 422 length rejections are
recorded as ineligible; they are never scored as wrong or silently truncated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
QWEN_MODEL = Path(os.environ.get("MODEL_PATH", ROOT / "models/qwen3_8b"))


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False).encode()).hexdigest()


def choice(state: object, instructions: str, criteria: dict[str, str]) -> dict:
    return {"state": state, "questions": {"decision": {
        "type": "choice", "instructions": instructions, "criteria": criteria}}}


def load_jevbench() -> list[dict]:
    base = ROOT / "external/jevbench/datasets/public"
    rows = [r for name in ("easy", "original", "hard") for r in jsonl(base / f"{name}.jsonl")]
    reference = {r["task_id"] for r in jsonl(ROOT / "results/jevbench/jev_1_13_0_fresh_results.jsonl")}
    assert len(rows) == len(reference) == 231 and {r["id"] for r in rows} == reference
    tasks = []
    for row in rows:
        body = {"state": row["state"], "questions": {"decision": row["question"]}}
        tasks.append({"id": row["id"], "gold": row["expected"], "body": body,
                      "family": row["family"], "question_type": row["question"]["type"]})
    return tasks


def load_tool_suite(suite: str) -> list[dict]:
    from transformers import AutoTokenizer
    from run_tool_decisions import load_tasks

    tok = AutoTokenizer.from_pretrained(QWEN_MODEL)
    tasks = load_tasks(suite, tok)
    reference = {r["id"]: r for r in jsonl(ROOT / f"results/{suite}/qwen_clm_clm2k.jsonl")}
    assert len(tasks) == len(reference)
    result = []
    for task in tasks:
        old = reference[task["id"]]
        assert task["gold"] == old["gold"]
        assert digest({"state": task["state"], "options": task["options"]}) == old["input_sha256"]
        question = ("Should the assistant use an external tool for this query?" if suite == "metatool" else
                    "Which function should be called for the user's request?" if suite == "bfcl" else
                    "Which candidate is the best next assistant response to the user's question?")
        keys = "ABCD"[:len(task["options"])]
        body = choice(task["state"], question, dict(zip(keys, task["options"])))
        result.append({"id": task["id"], "gold": keys[task["gold"]], "body": body,
                       "source": task["source"], "input_sha256": old["input_sha256"]})
    return result


def load_phishing() -> list[dict]:
    from run_phishing_comparison import CRITERIA, QUESTION, digest as email_digest, select_emails

    rows = select_emails(ROOT / "external/jev-phishing-bench/data/emails.jsonl", 2000, 42)
    reference = {r["id"]: r for r in jsonl(ROOT / "results/phishing/jev_2000.jsonl")}
    assert len(rows) == len(reference) == 2000
    tasks = []
    for row in rows:
        assert email_digest(row) == reference[row["id"]]["email_sha256"]
        tasks.append({"id": row["id"], "gold": "phishing" if row["y"] else "legitimate",
                      "body": choice(row["email"], QUESTION, CRITERIA),
                      "email_sha256": reference[row["id"]]["email_sha256"]})
    return tasks


def load_webprm(full: bool) -> list[dict]:
    from run_jev_webprm import INSTRUCTIONS, LETTERS, prepare

    reference = ROOT / ("results/webprm/qwen_clm_1141_clm2k.jsonl" if full else
                        "results/webprm/qwen_clm_120_clm2k.jsonl")
    rows = prepare(ROOT / "data/webprm/test-00000-of-00001.jsonl",
                   str(QWEN_MODEL), 999999 if full else 20, reference)
    return [{"id": f"{r['source']}:{r['state_idx']}", "gold": LETTERS[r["gold"]],
             "body": choice(r["state"], INSTRUCTIONS, dict(zip(LETTERS, r["options"]))),
             "source": r["source"], "state_idx": r["state_idx"]} for r in rows]


def load_deepswe() -> list[dict]:
    from run_jev_deepswe import INSTRUCTIONS, LETTERS, candidate_text

    rows = jsonl(ROOT / "data/deepswe/fair_inputs.jsonl")
    assert len(rows) == 38
    tasks = []
    for row in rows:
        for i, order in enumerate(row["orders"]):
            criteria = {letter: candidate_text(row["turns"][run]) for letter, run in zip(LETTERS, order)}
            body = choice(f"Task: {row['task']}\n{row['instruction']}", INSTRUCTIONS, criteria)
            tasks.append({"id": f"{row['task']}:{i}", "task": row["task"], "rotation": i,
                          "order": order, "rewards": row["rewards"], "body": body})
    return tasks


LOADERS = {"jevbench": load_jevbench, "metatool": lambda: load_tool_suite("metatool"),
           "when2call": lambda: load_tool_suite("when2call"),
           "bfcl": lambda: load_tool_suite("bfcl"), "phishing": load_phishing,
           "webprm120": lambda: load_webprm(False),
           "webprm1141": lambda: load_webprm(True), "deepswe": load_deepswe}


def fetch(url: str, body: dict, timeout: float) -> tuple[int, dict, float]:
    payload = json.dumps(body, ensure_ascii=False, allow_nan=False).encode()
    request = urllib.request.Request(url.rstrip("/") + "/v1/systemone", data=payload,
                                     headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.load(response), time.perf_counter() - start
    except urllib.error.HTTPError as error:
        data = error.read()
        return error.code, json.loads(data), time.perf_counter() - start


def evaluate(task: dict, url: str, timeout: float) -> dict:
    body = task["body"]
    status, response, elapsed = fetch(url, body, timeout)
    result = {k: v for k, v in task.items() if k != "body"}
    result.update(request_sha256=digest(body), status=status, latency_s=elapsed)
    if status == 422 and "exceeds max_length" in str(response.get("error", "")):
        result.update(outcome="too_long", reason=response["error"])
        return result
    if status != 200:
        raise RuntimeError(f"{task['id']}: HTTP {status}: {response}")
    answer = response.get("answers", {}).get("decision", {})
    kind = body["questions"]["decision"]["type"]
    if answer.get("type") != kind:
        raise ValueError(f"{task['id']}: answer type mismatch: {answer}")
    if kind == "choice":
        keys = set(body["questions"]["decision"]["criteria"])
        probs = answer.get("probabilities", {})
        if set(probs) != keys or not all(isinstance(p, (int, float)) and math.isfinite(p) and 0 <= p <= 1 for p in probs.values()):
            raise ValueError(f"{task['id']}: invalid choice probabilities: {answer}")
        prediction = max(probs, key=probs.get)
        assert prediction == answer["choice"]
    elif kind == "noul":
        p = answer.get("noul")
        if not isinstance(p, (int, float)) or not math.isfinite(p) or not 0 <= p <= 1:
            raise ValueError(f"{task['id']}: invalid noul: {answer}")
        probs, prediction = {"yes": p, "no": 1-p}, "yes" if p >= 0.5 else "no"
    elif kind == "score":
        probs = answer.get("probabilities", {})
        levels = body["questions"]["decision"].get("criteria") or []
        if set(probs) != {str(i) for i in range(len(levels))}:
            raise ValueError(f"{task['id']}: invalid score probabilities: {answer}")
        prediction = max(probs, key=probs.get)
    else:
        raise ValueError(kind)
    usage, metadata = response.get("usage", {}), response.get("metadata", {})
    result.update(outcome="ok", prediction=prediction, probabilities=probs,
                  model=response.get("model"), usage=usage,
                  inference_seconds=metadata.get("inference_seconds"),
                  model_max_length=metadata.get("max_length"),
                  checkpoint_sha256=metadata.get("checkpoint_sha256"),
                  base_revision=metadata.get("base_revision"))
    if "gold" in task:
        result["correct"] = prediction == str(task["gold"])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", choices=LOADERS)
    parser.add_argument("--url", required=True, help="local Open-Jev server base URL")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--timeout", type=float, default=180)
    args = parser.parse_args()
    tasks = LOADERS[args.suite]()
    if args.limit is not None:
        assert 1 <= args.limit <= len(tasks)
        tasks = tasks[:args.limit]
    existing = jsonl(args.output) if args.output.exists() else []
    done = {r["id"] for r in existing}
    assert len(done) == len(existing)
    by_id = {t["id"]: t for t in tasks}
    assert done <= set(by_id)
    for row in existing:
        task = by_id[row["id"]]
        assert row["request_sha256"] == digest(task["body"]), row["id"]
        if "gold" in task:
            assert row["gold"] == task["gold"], row["id"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with args.output.open("a") as handle:
        for index, task in enumerate(tasks, 1):
            if task["id"] in done:
                continue
            row = evaluate(task, args.url, args.timeout)
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
            if index % 50 == 0 or index == len(tasks):
                print(f"{args.suite} {index}/{len(tasks)} ok={row['outcome']} elapsed={time.monotonic()-started:.0f}s", flush=True)
    records = jsonl(args.output)
    eligible = [r for r in records if r["outcome"] == "ok"]
    summary = {"suite": args.suite, "total": len(records), "eligible": len(eligible),
               "too_long": len(records)-len(eligible),
               "latency_median_s": statistics.median(r["latency_s"] for r in eligible) if eligible else None,
               "inference_median_s": statistics.median(r["inference_seconds"] for r in eligible) if eligible else None}
    if args.suite != "deepswe":
        summary["correct"] = sum(bool(r["correct"]) for r in eligible)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
