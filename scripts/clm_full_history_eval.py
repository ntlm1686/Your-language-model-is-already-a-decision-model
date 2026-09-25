"""Score the shared Qwen trajectories with CLM using full preceding history.

The released DeepSWE trace builder is unavailable; this uses the same
normalized raw turns as fair_deepswe_eval.py, but includes every preceding
turn in each state before the official Recipe applies its 8K tail cap.
"""
import argparse
import json
import sys
import time
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

ROOT = Path(__file__).resolve().parents[1]
CLM_REPO = Path(os.environ.get("CLM_REPO", ROOT / "external/CLM"))
sys.path.insert(0, str(CLM_REPO / "train"))
from embed_utils import Recipe
from fair_deepswe_eval import ROOT, RUNS, extract, normalize_turns
sys.path.insert(0, str(CLM_REPO))
from evaluation.bon_eval import load_heads


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shard", type=int, required=True)
    p.add_argument("--nshards", type=int, required=True)
    args = p.parse_args()
    rows = [json.loads(s) for s in (ROOT / "data/deepswe/fair_inputs.jsonl").open()]
    rows = [r for i, r in enumerate(rows) if i % args.nshards == args.shard]
    model_path = os.environ.get("MODEL_PATH", str(ROOT / "models/qwen3_8b"))
    recipe = Recipe(model_path, max_len=8192)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16,
        attn_implementation="sdpa", low_cpu_mem_usage=True,
    ).eval().to("cuda:0")
    head_path = os.environ.get("CLM_DEEPSWE_HEAD", str(ROOT / "models/deepswe_head/best_head.pt"))
    sh, ah, _ = load_heads(head_path, torch.device("cuda:0"))
    action_cache = {}

    @torch.inference_mode()
    def embed(ids):
        x = torch.tensor([ids], device="cuda:0")
        h = model.model(input_ids=x, use_cache=False, return_dict=True).last_hidden_state[0, -1].float()
        return F.normalize(h, dim=-1)

    out = ROOT / "results/deepswe" / f"clm_full_history_shard{args.shard}.jsonl"
    start = time.monotonic()
    with out.open("w") as handle, torch.inference_mode():
        for i, row in enumerate(rows, 1):
            results = {}
            for run in RUNS:
                _, original = extract(row["task"], run, tail=None)
                turns = normalize_turns(original, row["cap"])
                assert turns[-len(row["turns"][run]):] == row["turns"][run]
                scores = []
                for j in range(len(turns) - len(row["turns"][run]), len(turns)):
                    messages = [{"role": "user", "content": row["instruction"]}]
                    for t in turns[:j]:
                        messages.append({"role": "assistant", "content": t["action"]})
                        if t["observation"]:
                            messages.append({"role": "user", "content": t["observation"]})
                    q = embed(recipe.state_ids(messages))
                    action = turns[j]["action"]
                    if action not in action_cache:
                        action_cache[action] = embed(recipe.text_ids(action, keep="head"))
                    a = action_cache[action]
                    scores.append(float((F.normalize(sh(q), dim=-1) *
                                         F.normalize(ah(a), dim=-1)).sum()))
                results[run] = {"score": sum(scores) / len(scores), "step_scores": scores}
            handle.write(json.dumps({"task": row["task"], "rewards": row["rewards"],
                                     "clm": results}) + "\n")
            handle.flush()
            print(f"shard {args.shard} {i}/{len(rows)} elapsed={time.monotonic()-start:.0f}s", flush=True)
    print("DONE", out, flush=True)


if __name__ == "__main__":
    main()
