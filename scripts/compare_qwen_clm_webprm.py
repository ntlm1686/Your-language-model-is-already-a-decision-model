"""Pilot: Qwen3-8B letter logits versus the released CLM head on WebPRMBench.

This is a closed-set, offline action-selection comparison, not an online
browser-completion benchmark. It uses the same Qwen3-8B weights for both paths.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import re
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
from clm.heads import HeadPair  # noqa: E402


PATTERN = re.compile(
    r"\[The Begin of Response 1\]\s*(.*?)\s*\[The End of Response 1\].*?"
    r"\[The Begin of Response 2\]\s*(.*?)\s*\[The End of Response 2\]",
    re.S,
)
LETTERS = "ABCDE"


def load_states(path: Path):
    groups = collections.defaultdict(list)
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            groups[(row["source_name"], row["state_idx"])].append(row)
    for key, rows in groups.items():
        assert len(rows) == 4, key
        positives, negatives = [], []
        for row in rows:
            match = PATTERN.search(row["prompt"])
            assert match, key
            choices = match.groups()
            positives.append(choices[row["expected"] - 1])
            negatives.append(choices[2 - row["expected"]])
        assert len(set(positives)) == 1 and len(set(negatives)) == 4, key
        context = rows[0]["prompt"].split("#### Assistant Responses ####", 1)[0]
        context = context[context.index("#### Intent ####") :].strip()
        yield {"source": key[0], "state_idx": key[1], "state": context,
               "positive": positives[0], "negatives": negatives}


def build_prompt(state: str, options: list[str]) -> str:
    lines = ["Choose the next assistant response that best advances the user's web task.",
             "Use the page state and previous actions. Reply with one letter.", "", state,
             "", "Candidate responses:"]
    lines.extend(f"{letter}. {option}" for letter, option in zip(LETTERS, options))
    lines.extend(["", "Answer:"])
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "data/webprm/test-00000-of-00001.jsonl")
    ap.add_argument("--model", default=os.environ.get("MODEL_PATH", str(ROOT / "models/qwen3_8b")))
    ap.add_argument("--head", default=os.environ.get("CLM_REFERENCE_HEAD", str(ROOT / "models/reference_head/CLM_v0.1-8B.pt")))
    ap.add_argument("--output", type=Path, default=ROOT / "results/webprm/qwen_clm_120_clm2k.jsonl")
    ap.add_argument("--per-source", type=int, default=10)
    ap.add_argument("--max-length", type=int, default=8192)
    ap.add_argument("--clm-max-length", type=int, default=2048,
                    help="CLM state/action cap; 2048 matches the default CLM server")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model)
    groups = collections.defaultdict(list)
    for state in load_states(args.data):
        groups[state["source"]].append(state)
    selected = []
    for source, states in sorted(groups.items()):
        rng.shuffle(states)
        taken = 0
        for state in states:
            options = [state["positive"], *state["negatives"]]
            rng.shuffle(options)
            prompt = build_prompt(state["state"], options)
            chat = tok.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False,
                add_generation_prompt=True, enable_thinking=False,
            ) + "Answer: "
            ids = tok.encode(chat, add_special_tokens=False)
            if len(ids) > args.max_length:
                continue
            state.update(options=options, gold=options.index(state["positive"]),
                         chat_ids=ids, prompt_tokens=len(ids))
            selected.append(state)
            taken += 1
            if taken == args.per_source:
                break
        print(f"selected {source}: {taken}/{args.per_source}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).eval().to(args.device)
    head = HeadPair("clm", args.head, device=args.device).ensure()
    letter_ids = [tok.encode(c, add_special_tokens=False)[0] for c in LETTERS]
    assert all(len(tok.encode(c, add_special_tokens=False)) == 1 for c in LETTERS)
    completed = set()
    if args.output.exists():
        with args.output.open() as handle:
            for line in handle:
                row = json.loads(line)
                completed.add((row["source"], row["state_idx"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def embed(text: str, keep: str) -> torch.Tensor:
        ids = tok.encode(text, add_special_tokens=False)
        ids = ids[:args.clm_max_length - 1] if keep == "head" else ids[-args.clm_max_length + 1:]
        inp = torch.tensor([ids], device=args.device)
        hidden = model.model(input_ids=inp, use_cache=False, return_dict=True).last_hidden_state[0, -1]
        return F.normalize(hidden.float(), dim=-1)

    start = time.monotonic()
    with args.output.open("a") as handle, torch.inference_mode():
        for n, state in enumerate(selected, start=1):
            key = (state["source"], state["state_idx"])
            if key in completed:
                continue
            inp = torch.tensor([state["chat_ids"]], device=args.device)
            torch.cuda.synchronize(args.device)
            qwen_start = time.monotonic()
            hidden = model.model(input_ids=inp, use_cache=False, return_dict=True).last_hidden_state[0, -1]
            logits = model.lm_head(hidden)[letter_ids].float()
            llm_probs = torch.softmax(logits, dim=0).cpu().tolist()
            torch.cuda.synchronize(args.device)
            qwen_seconds = time.monotonic() - qwen_start

            clm_start = time.monotonic()
            clm_state = state["state"] + "\n\nWhich response best advances the user's web task?"
            state_embedding = embed(clm_state, "tail")
            state_proj = F.normalize(head.state_head(state_embedding), dim=-1)
            action_embeddings = torch.stack([embed(x, "head") for x in state["options"]])
            action_proj = F.normalize(head.action_head(action_embeddings), dim=-1)
            clm_logits = (state_proj @ action_proj.T).float() * head.scale
            clm_probs = torch.softmax(clm_logits, dim=0).cpu().tolist()
            torch.cuda.synchronize(args.device)
            clm_seconds = time.monotonic() - clm_start
            row = {"source": state["source"], "state_idx": state["state_idx"],
                   "gold": state["gold"], "qwen_pred": max(range(5), key=lambda i: llm_probs[i]),
                   "clm_pred": max(range(5), key=lambda i: clm_probs[i]),
                   "qwen_probs": llm_probs, "clm_probs": clm_probs,
                   "prompt_tokens": state["prompt_tokens"],
                   "clm_max_tokens": args.clm_max_length,
                   "state_chars": len(state["state"]),
                   "action_chars": [len(x) for x in state["options"]],
                   "qwen_seconds": qwen_seconds, "clm_seconds": clm_seconds}
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            print(f"{n}/{len(selected)} {state['source']} qwen={row['qwen_pred']==row['gold']} "
                  f"clm={row['clm_pred']==row['gold']} elapsed={time.monotonic()-start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
