"""Fresh official-style CLM-v0.1 scoring on JevBench's 231 public items.

Uses CLM's build_pairs, Qwen3-8B last-token pooling, the released generic
state/action heads, and the CLM server's default 2048-token tail cap. Run one
shard per GPU; no JevBench answer key or provenance enters model inputs.
"""
import argparse
import glob
import json
import math
import sys
import time
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
CLM_REPO = Path(os.environ.get("CLM_REPO", ROOT / "external/CLM"))
sys.path.insert(0, str(CLM_REPO / "src"))
from clm.schema import build_pairs
sys.path.insert(0, str(CLM_REPO))
from evaluation.bon_eval import load_heads

MODEL = os.environ.get("MODEL_PATH", str(ROOT / "models/qwen3_8b"))
HEAD = Path(os.environ.get("CLM_REFERENCE_HEAD", ROOT / "models/reference_head/CLM_v0.1-8B.pt"))


def tasks():
    rows = [json.loads(line) for p in glob.glob(str(ROOT / "external/jevbench/datasets/public/*.jsonl"))
            for line in open(p)]
    rows.sort(key=lambda r: r["id"])
    assert len(rows) == 231 and len({r["id"] for r in rows}) == 231
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--nshards", type=int, required=True)
    parser.add_argument("--max-tokens", type=int, default=2048)
    args = parser.parse_args()
    rows = [r for i, r in enumerate(tasks()) if i % args.nshards == args.shard]
    tok = AutoTokenizer.from_pretrained(str(MODEL))
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL), dtype=torch.bfloat16, attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).eval().to("cuda:0")
    sh, ah, _ = load_heads(str(HEAD), torch.device("cuda:0"))
    ck = torch.load(HEAD, map_location="cpu", weights_only=False)
    scale = float(torch.as_tensor(ck["logit_scale"]).float().exp().clamp(max=100.0))
    cache = {}

    @torch.inference_mode()
    def embed(text):
        if text in cache:
            return cache[text]
        ids = tok.encode(text, add_special_tokens=False)
        if not ids:
            ids = tok.encode(" ", add_special_tokens=False)
        ids = ids[-args.max_tokens:]
        x = torch.tensor([ids], device="cuda:0")
        h = model.model(input_ids=x, use_cache=False,
                        return_dict=True).last_hidden_state[0, -1].float()
        h = F.normalize(h, dim=-1)
        cache[text] = (h, len(ids))
        return cache[text]

    out = ROOT / "results/jevbench" / f"clm_public_{args.max_tokens}_shard{args.shard}.jsonl"
    start = time.monotonic()
    with out.open("w") as handle, torch.inference_mode():
        for i, row in enumerate(rows, 1):
            state, keys, options = build_pairs(row["state"], {"q": row["question"]})["q"]
            q, qtokens = embed(state)
            zq = F.normalize(sh(q), dim=-1)
            scores = []
            option_tokens = []
            for option in options:
                a, atokens = embed(option)
                za = F.normalize(ah(a), dim=-1)
                scores.append(float((zq * za).sum()))
                option_tokens.append(atokens)
            logits = torch.tensor(scores) * scale
            probs = torch.softmax(logits, dim=0).tolist()
            best = keys[max(range(len(keys)), key=lambda j: scores[j])]
            pred = {"true": "yes", "false": "no"}.get(best, best) if row["question"]["type"] == "noul" else best
            expected = str(row["expected"])
            result = {"id": row["id"], "tier": row["id"].split("-")[0],
                      "type": row["question"]["type"], "expected": expected,
                      "prediction": pred, "correct": pred == expected,
                      "keys": keys, "scores": scores, "probabilities": probs,
                      "state_tokens": qtokens, "option_tokens": option_tokens}
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"shard {args.shard} {i}/{len(rows)} correct={result['correct']} "
                  f"elapsed={time.monotonic()-start:.0f}s", flush=True)
    print("DONE", out, flush=True)


if __name__ == "__main__":
    main()
