"""Merge shard JSONL records, rejecting duplicate or missing items."""
import argparse
import glob
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--glob", required=True, help="quoted glob pattern for input shards")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--key", choices=("id", "task"), required=True)
    p.add_argument("--expected", type=int, required=True)
    args = p.parse_args()
    files = sorted(glob.glob(args.glob))
    if not files:
        raise SystemExit(f"no shards matched {args.glob}")
    rows = [json.loads(line) for file in files for line in Path(file).read_text().splitlines() if line]
    if len(rows) != len({row[args.key] for row in rows}) or len(rows) != args.expected:
        raise SystemExit(f"expected {args.expected} unique {args.key} values; got {len(rows)} rows")
    rows.sort(key=lambda row: row[args.key])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    print(f"merged {len(files)} shards, {len(rows)} rows -> {args.output}")


if __name__ == "__main__":
    main()
