"""Serve untrained Qwen3.5-9B letter logits through the Jev-format Choice API."""
from __future__ import annotations

import argparse
import json
import os

# Reproduce the original PyTorch DeltaNet path even if optional FLA kernels
# are installed. This must run before the Qwen3.5 modeling module is imported.
if os.environ.get("QWEN35_USE_FLA") != "1":
    import transformers.utils.import_utils as _import_utils
    _import_utils.is_flash_linear_attention_available = lambda: False

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

from eval_qwen35_letters import MODEL, REVISION
from qwen_jev_api import LETTERS, Predictor, render, softmax
from serve_qwen_jev import QwenScorer, make_server


class Qwen35Scorer(QwenScorer):
    def __init__(self, device: str = "cuda:0", max_length: int = 8192,
                 batch_rotations: bool = True, fallback_margin: float = 0.1):
        self.torch, self.device, self.max_length = torch, device, max_length
        self.batch_rotations = batch_rotations
        self.fallback_margin = fallback_margin
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
        full = AutoModelForImageTextToText.from_pretrained(
            MODEL, revision=REVISION, local_files_only=True, dtype=torch.bfloat16,
            attn_implementation="sdpa", device_map={"": device}).eval()
        self.backbone = full.model.language_model
        self.head = full.get_output_embeddings()
        del full
        self.letter_ids = self._single_token_ids(LETTERS)
        self.yes_no_ids = self._single_token_ids("YN", tokens=("Yes", "No"))

    def _restricted_logits(self, ids: list[int], allowed: list[int]) -> list[float]:
        with torch.inference_mode():
            tensor = torch.tensor([ids], device=self.device)
            hidden = self.backbone(input_ids=tensor, use_cache=False,
                                   return_dict=True).last_hidden_state[0, -1]
            return self.head(hidden)[allowed].float().cpu().tolist()

    def score(self, record: dict) -> tuple[list[float], int, int, str]:
        n = len(record["options"])
        if n > len(LETTERS) and self.batch_rotations:
            state, question = render(record["state"]), render(record["question"])
            encoded = [self._encode(
                "Context:\n" + state + "\n\nQuestion: " + question + "\n"
                "Proposed answer: " + option + "\n"
                "Is this proposed answer correct? Answer Yes or No.\nAnswer:")
                for option in record["options"]]
            odds = []
            # Bound memory while scoring all candidates with the same Yes/No
            # prompt and log-odds rule used by the serial API fallback.
            for start in range(0, n, 8):
                chunk = encoded[start:start + 8]
                lengths = [len(ids) for ids in chunk]
                inputs = torch.full((len(chunk), max(lengths)), self.tokenizer.pad_token_id,
                                    dtype=torch.long, device=self.device)
                mask = torch.zeros_like(inputs)
                for i, ids in enumerate(chunk):
                    inputs[i, :len(ids)] = torch.tensor(ids, device=self.device)
                    mask[i, :len(ids)] = 1
                with torch.inference_mode():
                    hidden = self.backbone(input_ids=inputs, attention_mask=mask,
                                           use_cache=False, return_dict=True).last_hidden_state
                    last = hidden[torch.arange(len(chunk), device=self.device),
                                  torch.tensor(lengths, device=self.device) - 1]
                    logits = self.head(last)[:, self.yes_no_ids].float()
                    odds.extend((logits[:, 0] - logits[:, 1]).cpu().tolist())
            torch.cuda.synchronize(self.device)
            return softmax(odds), sum(map(len, encoded)), n, "batched_candidate_yes_no_log_odds"
        if not self.batch_rotations:
            return super().score(record)
        state, question = render(record["state"]), render(record["question"])
        rotations = []
        for shift in range(n):
            order = list(range(shift, n)) + list(range(shift))
            options = "\n".join(f"{LETTERS[i]}. {record['options'][index]}"
                                for i, index in enumerate(order))
            content = ("Read the state and answer the question. Select the best option. "
                       "Return one letter only.\n\nState:\n" + state +
                       "\n\nQuestion:\n" + question + "\n\nOptions:\n" + options +
                       "\n\nAnswer:")
            rotations.append((order, self._encode(content)))

        lengths = [len(ids) for _, ids in rotations]
        inputs = torch.full((n, max(lengths)), self.tokenizer.pad_token_id,
                            dtype=torch.long, device=self.device)
        mask = torch.zeros_like(inputs)
        for i, (_, ids) in enumerate(rotations):
            inputs[i, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=self.device)
            mask[i, :len(ids)] = 1
        with torch.inference_mode():
            hidden = self.backbone(input_ids=inputs, attention_mask=mask,
                                   use_cache=False, return_dict=True).last_hidden_state
            last = hidden[torch.arange(n, device=self.device),
                          torch.tensor(lengths, device=self.device) - 1]
            logits = self.head(last)[:, self.letter_ids[:n]].float()
            logprobs = torch.log_softmax(logits, dim=-1).cpu().tolist()
        torch.cuda.synchronize(self.device)
        sums = [0.0] * n
        for (order, _), values in zip(rotations, logprobs):
            for index, value in zip(order, values):
                sums[index] += value
        probabilities = softmax(sums)
        ranked = sorted(probabilities, reverse=True)
        if n > 1 and ranked[0] - ranked[1] < self.fallback_margin:
            probs, tokens, passes, _ = super().score(record)
            return probs, tokens + sum(lengths), passes + n, "batched_with_sequential_margin_fallback"
        return probabilities, sum(lengths), n, "batched_letter_logprob_rotations"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-length", type=int, default=8192)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8796)
    p.add_argument("--batch-rotations", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--fallback-margin", type=float, default=0.1)
    a = p.parse_args()
    if not 0 <= a.fallback_margin < 1:
        p.error("fallback-margin must be between 0 and 1")
    scorer = Qwen35Scorer(device=a.device, max_length=a.max_length,
                          batch_rotations=a.batch_rotations, fallback_margin=a.fallback_margin)
    predictor = Predictor(scorer, model_name="qwen3.5-9b")
    server = make_server(predictor, host=a.host, port=a.port)
    print(json.dumps({"url": f"http://{a.host}:{a.port}", "model": predictor.model_name,
                      "batch_rotations": a.batch_rotations}), flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
