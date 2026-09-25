"""Summarize a fresh JevBench or DeepSWE run without expected-result constants."""
import argparse
import collections
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("file", type=Path)
    p.add_argument("--benchmark", choices=("jevbench", "deepswe_full"), required=True)
    args = p.parse_args()
    data = [json.loads(line) for line in args.file.read_text().splitlines() if line]
    key = "id" if args.benchmark == "jevbench" else "task"
    if len(data) != len({r[key] for r in data}):
        raise SystemExit("duplicate item ids")
    by = collections.defaultdict(lambda: [0, 0])
    for row in data:
        if args.benchmark == "jevbench":
            correct = int(str(row["prediction"]) == str(row["expected"]))
            for group in ("all", f"tier:{row['tier']}", f"type:{row['type']}"):
                by[group][0] += correct
                by[group][1] += 1
        else:
            chosen = max(row["clm"], key=lambda run: row["clm"][run]["score"])
            by["all"][0] += int(row["rewards"][chosen])
            by["all"][1] += 1
    print(json.dumps(dict(sorted(by.items())), indent=2))


if __name__ == "__main__":
    main()
