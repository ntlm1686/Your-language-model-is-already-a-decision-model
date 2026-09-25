"""Recount Jev 1.13.0 saved outcomes on JevBench's 231 public items."""
import glob
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "external/jevbench"
published = json.loads((BENCH / "results/v1.2/wip/jevbench-v1.2-wip-per-task.json").read_text())
tasks = [json.loads(line) for path in glob.glob(str(BENCH / "datasets/public/*.jsonl"))
         for line in Path(path).read_text().splitlines() if line]
ids = {task["id"] for task in tasks}
saved = published["systems"]["jev-1.13.0"]["public_tasks"]
assert len(tasks) == len(ids) == len(saved) == 231 and ids == set(saved)
counts = Counter()
for task in tasks:
    outcome = saved[task["id"]][0]
    counts[(task["id"].split("-")[0], outcome)] += 1
    counts[("all", outcome)] += 1
assert counts[("all", "c")] == 200
print(json.dumps({"correct": counts[("all", "c")], "total": len(tasks),
                  "by_tier": {tier: counts[(tier, "c")] for tier in ("easy", "original", "hard")}}, indent=2))
