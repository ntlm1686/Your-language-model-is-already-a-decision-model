"""Recompute the reported counts from committed per-item inference records."""
from __future__ import annotations

import collections
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def rows(relative: str) -> list[dict]:
    path = ROOT / relative
    result = [json.loads(line) for line in path.read_text().splitlines() if line]
    if not result:
        raise ValueError(f"empty result: {path}")
    return result


def check_jevbench() -> dict:
    files = {
        "qwen_letters": ("qwen3_8b_letters_results.jsonl", 163),
        "clm_2048": ("clm_public_2048_results.jsonl", 95),
    }
    counts = {}
    ids = None
    for label, (name, target) in files.items():
        data = rows(f"results/jevbench/{name}")
        current_ids = {r["id"] for r in data}
        assert len(data) == len(current_ids) == 231, label
        if ids is None:
            ids = current_ids
        assert ids == current_ids, label
        assert all(r["correct"] == (str(r["prediction"]) == str(r["expected"])) for r in data)
        value = sum(r["correct"] for r in data)
        assert value == target, (label, value)
        counts[label] = f"{value}/231"
    jev = rows("results/jevbench/jev_1_13_0_fresh_results.jsonl")
    assert len(jev) == 231 and {r["task_id"] for r in jev} == ids
    assert all(r["ok"] and r["valid"] and r["model"] == "jev-1.13.0" for r in jev)
    correct = sum(bool(r["correct"]) for r in jev)
    assert correct == 199
    counts["jev_fresh"] = f"{correct}/231"
    return counts


def check_deepswe() -> dict:
    original = rows("results/deepswe/fair_results.jsonl")
    full = rows("results/deepswe/clm_full_history_scores.jsonl")
    inputs = rows("data/deepswe/fair_inputs.jsonl")
    assert len({r["task"] for r in original}) == len(original) == 38
    assert {r["task"] for r in original} == {r["task"] for r in full} == {r["task"] for r in inputs}
    old_clm = sum(r["rewards"][r["clm_pick"]] for r in original)
    qwen = sum(r["rewards"][r["qwen_pick"]] for r in original)
    full_clm = sum(r["rewards"][max(r["clm"], key=lambda k: r["clm"][k]["score"])] for r in full)
    oracle = sum(any(r["rewards"].values()) for r in original)
    random_expected = sum(sum(r["rewards"].values()) / 4 for r in original)
    by_run = collections.Counter()
    for r in original:
        for run, reward in r["rewards"].items():
            by_run[run] += reward
    best_fixed = max(by_run.values())
    assert (old_clm, qwen, full_clm, oracle, random_expected, best_fixed) == (21, 18, 22, 29, 19.25, 22)
    jev = rows("results/deepswe/jev_1_13_0_fresh_results.jsonl")
    assert len(jev) == 38 and {r["task"] for r in jev} == {r["task"] for r in original}
    assert all(len(r["rotations"]) == 4 and all(x["model"] == "jev-1.13.0" for x in r["rotations"])
               and r["success"] == r["rewards"][r["pick"]] for r in jev)
    assert sum(r["success"] for r in jev) == 20
    return {"clm_full_history": "22/38", "clm_window_only": "21/38",
            "qwen_letters": "18/38", "jev_fresh": "20/38",
            "oracle": "29/38", "best_fixed": "22/38"}


def check_webprm() -> dict:
    data = rows("results/webprm/qwen_clm_120_clm2k.jsonl")
    assert len(data) == len({(r["source"], r["state_idx"]) for r in data}) == 120
    qwen = sum(r["qwen_pred"] == r["gold"] for r in data)
    clm = sum(r["clm_pred"] == r["gold"] for r in data)
    assert (qwen, clm) == (65, 24)
    assert all(r["clm_max_tokens"] == 2048 for r in data)
    jev = rows("results/webprm/jev_1_13_0_fresh_results.jsonl")
    by_id = {(r["source"], r["state_idx"]): r for r in data}
    assert len(jev) == 120 and {(r["source"], r["state_idx"]) for r in jev} == set(by_id)
    assert all(r["valid"] and r["model"] == "jev-1.13.0" and
               r["gold"] == by_id[r["source"], r["state_idx"]]["gold"] for r in jev)
    assert sum(r["correct"] for r in jev) == 65
    return {"qwen_letters": "65/120", "clm_reference_default_2k": "24/120", "jev_fresh": "65/120"}


def check_phishing() -> dict:
    gpu = rows("results/phishing/qwen_clm_2000_clm2k.jsonl")
    rotated = rows("results/phishing/qwen_rotated_2000.jsonl")
    jev = rows("results/phishing/jev_2000.jsonl")
    by_id = {r["id"]: r for r in gpu}
    assert len(gpu) == len(by_id) == 2000
    assert len(rotated) == len(jev) == 2000
    assert {r["id"] for r in rotated} == {r["id"] for r in jev} == set(by_id)
    assert all(r["clm_max_tokens"] == 2048 for r in gpu)
    assert all(r["model"] == "jev-1.13.0" for r in jev)
    for source in (rotated, jev):
        assert all(r["y"] == by_id[r["id"]]["y"] and
                   r["email_sha256"] == by_id[r["id"]]["email_sha256"] for r in source)
    gold = lambda r: "phishing" if r["y"] else "legitimate"
    counts = {"qwen_single": sum(r["qwen_pred"] == gold(r) for r in gpu),
              "qwen_rotated": sum(r["prediction"] == gold(r) for r in rotated),
              "clm_default_2k": sum(r["clm_pred"] == gold(r) for r in gpu),
              "jev_fresh": sum(r["prediction"] == gold(r) for r in jev)}
    assert counts == {"qwen_single": 1320, "qwen_rotated": 1330,
                      "clm_default_2k": 1212, "jev_fresh": 1251}
    return {k: f"{v}/2000" for k, v in counts.items()}


def check_webprm_full() -> dict:
    gpu = rows("results/webprm/qwen_clm_1141_clm2k.jsonl")
    jev = rows("results/webprm/jev_1141.jsonl")
    key = lambda r: (r["source"], r["state_idx"])
    by_id = {key(r): r for r in gpu}
    assert len(gpu) == len(jev) == len(by_id) == 1141
    assert {key(r) for r in jev} == set(by_id)
    assert all(r["clm_max_tokens"] == 2048 for r in gpu)
    assert all(r["valid"] and r["model"] == "jev-1.13.0" and
               r["gold"] == by_id[key(r)]["gold"] for r in jev)
    counts = {"qwen_letters": sum(r["qwen_pred"] == r["gold"] for r in gpu),
              "clm_default_2k": sum(r["clm_pred"] == r["gold"] for r in gpu),
              "jev_fresh": sum(r["correct"] for r in jev)}
    assert counts == {"qwen_letters": 679, "clm_default_2k": 209, "jev_fresh": 760}
    return {k: f"{v}/1141" for k, v in counts.items()}


def check_tool_suite(name: str, expected_n: int, expected: dict[str, int]) -> dict:
    gpu = rows(f"results/{name}/qwen_clm_clm2k.jsonl")
    jev = rows(f"results/{name}/jev.jsonl")
    by_id = {r["id"]: r for r in gpu}
    assert len(gpu) == len(jev) == len(by_id) == expected_n
    assert {r["id"] for r in jev} == set(by_id)
    assert all(r["clm_max_tokens"] == 2048 for r in gpu)
    assert all(r["model"] == "jev-1.13.0" and r["input_sha256"] == by_id[r["id"]]["input_sha256"] and
               r["source"] == by_id[r["id"]]["source"] and r["gold"] == by_id[r["id"]]["gold"] for r in jev)
    counts = {"qwen": sum(r["qwen_pred"] == r["gold"] for r in gpu),
              "clm_default_2k": sum(r["clm_pred"] == r["gold"] for r in gpu),
              "jev": sum(r["prediction"] == r["gold"] for r in jev)}
    assert counts == expected
    return {k: f"{v}/{expected_n}" for k, v in counts.items()}


if __name__ == "__main__":
    print(json.dumps({"jevbench": check_jevbench(), "deepswe": check_deepswe(),
                      "webprm_pilot": check_webprm(), "webprm_full": check_webprm_full(),
                      "phishing": check_phishing(),
                      "metatool": check_tool_suite("metatool", 1040, {"qwen": 842, "clm_default_2k": 518, "jev": 802}),
                      "when2call": check_tool_suite("when2call", 600, {"qwen": 307, "clm_default_2k": 201, "jev": 440}),
                      "bfcl": check_tool_suite("bfcl", 200, {"qwen": 198, "clm_default_2k": 169, "jev": 198})}, indent=2))
