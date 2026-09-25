"""Evaluate Qwen3-8B, the released CLM head, and Jev on the same emails.

Input is the verified emails.jsonl produced by jev-phishing-bench/prepare_data.py.
All methods see only the email object and use the same two verdict descriptions.
Results are append-only and resumable. No email text or API key is written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUESTION = "This email arrived in the user's inbox and contains a link (`link_url`). Is it a phishing attempt, or a legitimate message whose link is safe to click?"
CRITERIA = {
    "phishing": "The email or its link is a phishing attempt or otherwise malicious. The user should not click the link.",
    "legitimate": "A normal, legitimate message. The link is safe to click.",
}


def select_emails(path: Path, limit: int, seed: int) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    assert len(rows) == 2000 and len({r["id"] for r in rows}) == 2000
    assert sum(r["y"] for r in rows) == 1000
    if limit < len(rows):
        assert limit % 2 == 0
        rng = random.Random(seed)
        rows = rng.sample([r for r in rows if r["y"] == 1], limit // 2) + rng.sample(
            [r for r in rows if r["y"] == 0], limit // 2)
        rng.shuffle(rows)
    return rows


def digest(row: dict) -> str:
    return hashlib.sha256(json.dumps(row["email"], ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def run_gpu(rows: list[dict], output: Path, model_path: str, head_path: str) -> None:
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer

    sys.path.insert(0, str(ROOT / "external/CLM/src"))
    from clm.heads import HeadPair

    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, attn_implementation="sdpa", low_cpu_mem_usage=True
    ).eval().to("cuda:0")
    head = HeadPair("clm", head_path, device="cuda:0").ensure()
    letter_ids = [tok.encode(x, add_special_tokens=False) for x in "AB"]
    assert all(len(x) == 1 for x in letter_ids)
    letter_ids = [x[0] for x in letter_ids]
    done = {json.loads(line)["id"] for line in output.read_text().splitlines()} if output.exists() else set()
    output.parent.mkdir(parents=True, exist_ok=True)

    def embed(s: str, keep: str) -> tuple[torch.Tensor, int]:
        ids = tok.encode(s, add_special_tokens=False)
        count = len(ids)
        ids = ids[:2047] if keep == "head" else ids[-2047:]
        inp = torch.tensor([ids], device="cuda:0")
        hidden = model.model(input_ids=inp, use_cache=False, return_dict=True).last_hidden_state[0, -1]
        return F.normalize(hidden.float(), dim=-1), count

    start = time.monotonic()
    with output.open("a") as f, torch.inference_mode():
        for i, row in enumerate(rows, 1):
            if row["id"] in done:
                continue
            email = json.dumps(row["email"], ensure_ascii=False)
            # Rotate the labels deterministically to avoid a fixed-letter prior.
            choices = ["phishing", "legitimate"] if int(hashlib.sha256(row["id"].encode()).hexdigest(), 16) % 2 else ["legitimate", "phishing"]
            prompt = (f"{QUESTION}\n\nEmail:\n{email}\n\nOptions:\n"
                      + "\n".join(f"{letter}. {name}: {CRITERIA[name]}" for letter, name in zip("AB", choices))
                      + "\n\nReply with one letter.\nAnswer:")
            chat = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                           add_generation_prompt=True, enable_thinking=False) + "Answer: "
            ids = tok.encode(chat, add_special_tokens=False)
            if len(ids) > 8192:
                raise ValueError(f"Qwen prompt exceeds 8192 tokens: {row['id']}")
            t0 = time.monotonic()
            hidden = model.model(input_ids=torch.tensor([ids], device="cuda:0"), use_cache=False,
                                 return_dict=True).last_hidden_state[0, -1]
            probs = torch.softmax(model.lm_head(hidden)[letter_ids].float(), dim=0).cpu().tolist()
            torch.cuda.synchronize()
            qwen_s = time.monotonic() - t0
            t0 = time.monotonic()
            state_vec, state_tokens = embed(email + "\n\n" + QUESTION, "tail")
            action_vecs = [embed(name + ": " + CRITERIA[name], "head") for name in choices]
            state_proj = F.normalize(head.state_head(state_vec), dim=-1)
            action_proj = F.normalize(head.action_head(torch.stack([x[0] for x in action_vecs])), dim=-1)
            clm_probs = torch.softmax((state_proj @ action_proj.T).float() * head.scale, dim=0).cpu().tolist()
            torch.cuda.synchronize()
            clm_s = time.monotonic() - t0
            rec = {"id": row["id"], "y": row["y"], "email_sha256": digest(row),
                   "option_order": choices, "qwen_probs": probs, "clm_probs": clm_probs,
                   "qwen_pred": choices[probs.index(max(probs))],
                   "clm_pred": choices[clm_probs.index(max(clm_probs))],
                   "qwen_prompt_tokens": len(ids), "clm_state_tokens": state_tokens,
                   "clm_max_tokens": 2048, "qwen_seconds": qwen_s, "clm_seconds": clm_s}
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if i % 50 == 0 or i == len(rows):
                print(f"GPU {i}/{len(rows)} elapsed={time.monotonic()-start:.0f}s", flush=True)


def run_jev(rows: list[dict], output: Path, jev_model: str) -> None:
    key = os.environ.get("TYPESAFE_API_KEY")
    if not key:
        raise SystemExit("TYPESAFE_API_KEY is required")
    done = {json.loads(line)["id"] for line in output.read_text().splitlines()} if output.exists() else set()
    output.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with output.open("a") as f:
        for i, row in enumerate(rows, 1):
            if row["id"] in done:
                continue
            payload = json.dumps({"model": jev_model, "state": row["email"], "questions": {
                "verdict": {"type": "choice", "instructions": QUESTION, "criteria": CRITERIA}}},
                ensure_ascii=False).encode()
            req = urllib.request.Request("https://api.typesafe.ai/v1/systemone", data=payload,
                                         headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
            t0 = time.monotonic()
            try:
                with urllib.request.urlopen(req, timeout=120) as response:
                    result = json.load(response)
            except Exception as exc:
                print(f"Jev stopped at {row['id']}: {type(exc).__name__}: {exc}", flush=True)
                raise
            answer = result.get("answers", {}).get("verdict", {})
            probs = answer.get("probabilities", {})
            if answer.get("type") != "choice" or set(probs) != set(CRITERIA):
                raise ValueError(f"Invalid Jev answer at {row['id']}: {answer}")
            rec = {"id": row["id"], "y": row["y"], "email_sha256": digest(row),
                   "prediction": max(probs, key=probs.get), "probabilities": probs,
                   "reported_choice": answer.get("choice"), "model": result.get("model"),
                   "usage": result.get("usage"), "latency_s": time.monotonic()-t0,
                   "request_sha256": hashlib.sha256(payload).hexdigest()}
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if i % 50 == 0 or i == len(rows):
                print(f"Jev {i}/{len(rows)} elapsed={time.monotonic()-start:.0f}s", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["gpu", "jev"])
    p.add_argument("--data", type=Path, default=ROOT / "external/jev-phishing-bench/data/emails.jsonl")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--limit", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model", default=os.environ.get("MODEL_PATH", str(ROOT / "models/qwen3_8b")))
    p.add_argument("--head", default=os.environ.get("CLM_REFERENCE_HEAD", str(ROOT / "models/reference_head/CLM_v0.1-8B.pt")))
    p.add_argument("--jev-model", default="jev-1.13.0")
    a = p.parse_args()
    rows = select_emails(a.data, a.limit, a.seed)
    if a.mode == "gpu":
        run_gpu(rows, a.output, a.model, a.head)
    else:
        run_jev(rows, a.output, a.jev_model)


if __name__ == "__main__":
    main()
