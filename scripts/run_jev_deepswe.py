"""Score the 38 shared DeepSWE candidate sets with Jev choice probabilities.

Each task uses the same four cyclic candidate orders as the Qwen letter-logit
run. The task and final twelve normalized turns are the only model evidence;
verifier rewards are used only after inference to score selected trajectories.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LETTERS = "ABCD"
INSTRUCTIONS = ("Which completed coding-agent attempt is most likely to pass the hidden tests? "
                "Use the task and the final 12 action/observation turns of each attempt.")


def candidate_text(turns: list[dict]) -> str:
    parts = []
    for i, turn in enumerate(turns, 1):
        parts.append(f"Turn {i} action:\n{turn['action']}")
        if turn["observation"]:
            parts.append(f"Turn {i} observed:\n{turn['observation']}")
    return "\n".join(parts)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--limit", type=int, default=38)
    p.add_argument("--model", default="jev-1.13.0")
    args = p.parse_args()
    key = os.environ.get("TYPESAFE_API_KEY")
    if not key:
        raise SystemExit("TYPESAFE_API_KEY is required")
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    rows = [json.loads(line) for line in
            (ROOT / "data/deepswe/fair_inputs.jsonl").read_text().splitlines()]
    if len(rows) != 38 or not 1 <= args.limit <= 38:
        raise SystemExit("expected 38 tasks and a limit between 1 and 38")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        for n, row in enumerate(rows[:args.limit], 1):
            scores = {run: 0.0 for run in row["rewards"]}
            rotations = []
            for order in row["orders"]:
                question = {"type": "choice", "instructions": INSTRUCTIONS,
                            "criteria": {letter: candidate_text(row["turns"][run])
                                         for letter, run in zip(LETTERS, order)}}
                body = {"model": args.model,
                        "state": f"Task: {row['task']}\n{row['instruction']}",
                        "questions": {"decision": question}}
                payload = json.dumps(body, ensure_ascii=False).encode()
                request = urllib.request.Request(
                    "https://api.typesafe.ai/v1/systemone", data=payload, method="POST",
                    headers={"Authorization": f"Bearer {key}",
                             "Content-Type": "application/json"},
                )
                start = time.perf_counter()
                try:
                    with urllib.request.urlopen(request, timeout=120) as response:
                        status = response.status
                        result = json.load(response)
                except urllib.error.HTTPError as exc:
                    status = exc.code
                    result = {"error": exc.read().decode(errors="replace")[:300]}
                latency = time.perf_counter() - start
                answer = result.get("answers", {}).get("decision", {})
                probs = answer.get("probabilities", {})
                valid = (status == 200 and answer.get("type") == "choice" and
                         set(probs) == set(LETTERS) and answer.get("choice") in LETTERS and
                         all(isinstance(x, (float, int)) and 0 <= x <= 1 for x in probs.values()))
                if not valid:
                    raise SystemExit(f"invalid API response for {row['task']}: HTTP {status}, {result.get('error', answer)}")
                for letter, run in zip(LETTERS, order):
                    scores[run] += math.log(max(probs[letter], 1e-30))
                rotations.append({"order": order, "probabilities": probs,
                                  "reported_choice": answer["choice"],
                                  "model": result.get("model"), "usage": result.get("usage"),
                                  "latency_s": latency,
                                  "request_sha256": hashlib.sha256(payload).hexdigest()})
            pick = max(scores, key=scores.get)
            record = {"task": row["task"], "rewards": row["rewards"],
                      "scores": scores, "pick": pick, "success": row["rewards"][pick],
                      "rotations": rotations}
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
            handle.flush()
            if n % 5 == 0 or n == args.limit:
                print(f"{n}/{args.limit} successful_picks={record['success']}", flush=True)


if __name__ == "__main__":
    main()
