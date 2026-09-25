"""Validate paired full WebPRM next-action scores and write aggregate counts."""
from __future__ import annotations

import argparse
import collections
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpu", type=Path, default=ROOT / "results/webprm/qwen_clm_1141_clm2k.jsonl")
    p.add_argument("--jev", type=Path, default=ROOT / "results/webprm/jev_1141.jsonl")
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    gpu, jev = read(a.gpu), read(a.jev)
    key = lambda r: (r["source"], r["state_idx"])
    assert len(gpu) == len(jev) == 1141
    g, j = ({key(r): r for r in rows} for rows in (gpu, jev))
    assert len(g) == len(j) == 1141 and set(g) == set(j)
    assert all(r["clm_max_tokens"] == 2048 for r in gpu)
    assert all(r["valid"] and r["model"] == "jev-1.13.0" and
               r["gold"] == g[key(r)]["gold"] for r in jev)
    out = {"n": 1141, "correct": {}, "by_source": {}, "paired": {},
           "latency_s": {"qwen_median": statistics.median(r["qwen_seconds"] for r in gpu),
                         "clm_median": statistics.median(r["clm_seconds"] for r in gpu),
                         "jev_median": statistics.median(r["latency_s"] for r in jev)},
           "jev_input_tokens": sum(r["usage"].get("input_tokens", 0) for r in jev)}
    preds = {"qwen": {key(r): r["qwen_pred"] for r in gpu},
             "clm_2k": {key(r): r["clm_pred"] for r in gpu},
             "jev": {key(r): r["prediction"] for r in jev}}
    for name, pp in preds.items():
        out["correct"][name] = sum(pp[k] == g[k]["gold"] for k in g)
    by_source = collections.defaultdict(list)
    for k in g:
        by_source[k[0]].append(k)
    for source, keys in sorted(by_source.items()):
        out["by_source"][source] = {"n": len(keys), **{
            name: sum(pp[k] == g[k]["gold"] for k in keys) for name, pp in preds.items()}}
    for a_name, b_name in (("jev", "qwen"), ("jev", "clm_2k"), ("qwen", "clm_2k")):
        out["paired"][f"{a_name}_vs_{b_name}"] = {
            f"{a_name}_only": sum(preds[a_name][k] == g[k]["gold"] and preds[b_name][k] != g[k]["gold"] for k in g),
            f"{b_name}_only": sum(preds[b_name][k] == g[k]["gold"] and preds[a_name][k] != g[k]["gold"] for k in g)}
    s = json.dumps(out, indent=2)
    if a.output:
        a.output.write_text(s + "\n")
    print(s)


if __name__ == "__main__":
    main()
