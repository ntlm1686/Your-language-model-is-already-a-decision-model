"""Validate and summarize paired MetaTool or When2Call model decisions."""
from __future__ import annotations

import argparse
import collections
import json
import statistics
from pathlib import Path

from run_tool_decisions import ROOT, load_tasks, sha


def read(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("suite", choices=["metatool", "when2call", "bfcl"])
    p.add_argument("--gpu", type=Path)
    p.add_argument("--jev", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--model", default=str(ROOT / "models/qwen3_8b"))
    a = p.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    tasks = load_tasks(a.suite, tok)
    base = ROOT / "results" / a.suite
    gpu = read(a.gpu or base / "qwen_clm_clm2k.jsonl")
    jev = read(a.jev or base / "jev.jsonl")
    g, j = ({r["id"]: r for r in rows} for rows in (gpu, jev))
    assert len(tasks) == len(gpu) == len(jev) == len(g) == len(j)
    assert {t["id"] for t in tasks} == set(g) == set(j)
    for t in tasks:
        h = sha(json.dumps({"state": t["state"], "options": t["options"]}, ensure_ascii=False))
        x, y = g[t["id"]], j[t["id"]]
        assert x["input_sha256"] == y["input_sha256"] == h
        assert x["gold"] == y["gold"] == t["gold"]
        assert x["clm_max_tokens"] == 2048
        assert len(x["qwen_rotations"]) == len(t["options"])
        assert y["model"] == "jev-1.13.0" and set(y["probabilities"]) == set("ABCD"[:len(t["options"])])
    pred = {"qwen": {r["id"]: r["qwen_pred"] for r in gpu},
            "clm_2k": {r["id"]: r["clm_pred"] for r in gpu},
            "jev": {r["id"]: r["prediction"] for r in jev}}
    out = {"n": len(tasks), "correct": {}, "by_source": {}, "paired": {},
           "clm_truncated_states": sum(r["clm_state_tokens"] > 2047 for r in gpu),
           "latency_s": {"qwen_median": statistics.median(r["qwen_seconds"] for r in gpu),
                         "clm_median": statistics.median(r["clm_seconds"] for r in gpu),
                         "jev_median": statistics.median(r["latency_s"] for r in jev)},
           "jev_input_tokens": sum(r["usage"].get("input_tokens", 0) for r in jev)}
    for name, pp in pred.items():
        out["correct"][name] = sum(pp[t["id"]] == t["gold"] for t in tasks)
    by_source = collections.defaultdict(list)
    for t in tasks:
        by_source[t["source"]].append(t)
    for source, subset in sorted(by_source.items()):
        out["by_source"][source] = {"n": len(subset), **{
            name: sum(pp[t["id"]] == t["gold"] for t in subset) for name, pp in pred.items()}}
    for a_name, b_name in (("jev", "qwen"), ("jev", "clm_2k"), ("qwen", "clm_2k")):
        out["paired"][f"{a_name}_vs_{b_name}"] = {
            f"{a_name}_only": sum(pred[a_name][t["id"]] == t["gold"] and pred[b_name][t["id"]] != t["gold"] for t in tasks),
            f"{b_name}_only": sum(pred[b_name][t["id"]] == t["gold"] and pred[a_name][t["id"]] != t["gold"] for t in tasks)}
    s = json.dumps(out, indent=2)
    (a.output or base / "summary.json").write_text(s + "\n")
    print(s)


if __name__ == "__main__":
    main()
