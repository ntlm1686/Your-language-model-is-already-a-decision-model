"""Validate and summarize the paired 2,000-email phishing comparison."""
from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x]


def metrics(rows: list[dict], predictions: dict[str, str]) -> dict:
    n = len(rows)
    correct = sum(predictions[r["id"]] == ("phishing" if r["y"] else "legitimate") for r in rows)
    tp = sum(r["y"] == 1 and predictions[r["id"]] == "phishing" for r in rows)
    fp = sum(r["y"] == 0 and predictions[r["id"]] == "phishing" for r in rows)
    return {"correct": correct, "n": n, "accuracy": correct / n,
            "phishing_recall": tp / sum(r["y"] == 1 for r in rows),
            "false_positive_rate": fp / sum(r["y"] == 0 for r in rows)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=ROOT / "external/jev-phishing-bench/data/emails.jsonl")
    p.add_argument("--gpu", type=Path, default=ROOT / "results/phishing/qwen_clm_2000_clm2k.jsonl")
    p.add_argument("--jev", type=Path, default=ROOT / "results/phishing/jev_2000.jsonl")
    p.add_argument("--rotated", type=Path, default=ROOT / "results/phishing/qwen_rotated_2000.jsonl")
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    data, gpu, jev, rotated = read(a.data), read(a.gpu), read(a.jev), read(a.rotated)
    assert len(data) == len(gpu) == len(jev) == len(rotated) == 2000
    for source in (data, gpu, jev, rotated):
        assert len({r["id"] for r in source}) == 2000
    d, g, j, q = ({r["id"]: r for r in source} for source in (data, gpu, jev, rotated))
    assert set(d) == set(g) == set(j) == set(q)
    for id_, row in d.items():
        assert row["y"] == g[id_]["y"] == j[id_]["y"]
        assert g[id_]["email_sha256"] == j[id_]["email_sha256"] == q[id_]["email_sha256"]
        assert q[id_]["reverse_order"] == list(reversed(g[id_]["option_order"]))
        assert g[id_]["clm_max_tokens"] == 2048
        assert len(g[id_]["qwen_probs"]) == len(g[id_]["clm_probs"]) == 2
        assert set(j[id_]["probabilities"]) == {"phishing", "legitimate"}
    pred = {"jev": {k: v["prediction"] for k, v in j.items()},
            "qwen": {k: v["qwen_pred"] for k, v in g.items()},
            "qwen_rotated": {k: v["prediction"] for k, v in q.items()},
            "clm_2k": {k: v["clm_pred"] for k, v in g.items()}}
    sys.path.insert(0, str(ROOT / "external/jev-phishing-bench"))
    from bench.heuristics import features
    pred["url_host_rule"] = {r["id"]: "phishing" if features(r["email"])["hosting_or_shortener"] else "legitimate" for r in data}
    out = {"n": 2000, "metrics": {name: metrics(data, val) for name, val in pred.items()},
           "paired": {}, "by_url_category": {}, "model_versions": sorted({v["model"] for v in jev}),
           "latency_s": {"jev_median": statistics.median(v["latency_s"] for v in jev),
                         "qwen_median": statistics.median(v["qwen_seconds"] for v in gpu),
                         "qwen_rotated_median": statistics.median(g[v["id"]]["qwen_seconds"] + v["reverse_seconds"] for v in rotated),
                         "clm_median": statistics.median(v["clm_seconds"] for v in gpu)},
           "jev_input_tokens": sum(v["usage"].get("input_tokens", 0) for v in jev),
           "clm_truncated_states": sum(v["clm_state_tokens"] > 2047 for v in gpu)}
    by_category = collections.defaultdict(list)
    for row in data:
        by_category[row["url_category"]].append(row)
    for category, subset in sorted(by_category.items()):
        out["by_url_category"][category] = {"n": len(subset), "phishing": sum(r["y"] for r in subset),
                                            "correct": {name: sum(pp[r["id"]] == ("phishing" if r["y"] else "legitimate")
                                                            for r in subset) for name, pp in pred.items()}}
    for a_name, b_name in (("jev", "qwen_rotated"), ("jev", "clm_2k"), ("qwen_rotated", "clm_2k")):
        a_only = b_only = 0
        for row in data:
            gold = "phishing" if row["y"] else "legitimate"
            a_ok, b_ok = pred[a_name][row["id"]] == gold, pred[b_name][row["id"]] == gold
            a_only += a_ok and not b_ok
            b_only += b_ok and not a_ok
        out["paired"][f"{a_name}_vs_{b_name}"] = {f"{a_name}_only": a_only, f"{b_name}_only": b_only}
    s = json.dumps(out, indent=2)
    if a.output:
        a.output.write_text(s + "\n")
    print(s)


if __name__ == "__main__":
    main()
