"""Serve Qwen3-8B next-token decisions using Jev's /v1/systemone JSON API.

The model emits no text. Up to 26 candidates use position-rotated letter
log probabilities. Larger Choice requests use independent Yes/No log odds so
the Jev protocol's full 255-candidate range remains available.
"""

from __future__ import annotations

import argparse
import json
import math
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from qwen_jev_api import LETTERS, Predictor, render, softmax, strict_json


ROOT = Path(__file__).resolve().parents[1]


class QwenScorer:
    def __init__(self, model_path: str, *, device: str = "cuda:0", max_length: int = 8192):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if max_length < 1:
            raise ValueError("max_length must be positive")
        self.torch, self.device, self.max_length = torch, device, max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
            attn_implementation="sdpa", low_cpu_mem_usage=True).eval().to(device)
        self.letter_ids = self._single_token_ids(LETTERS)
        self.yes_no_ids = self._single_token_ids("YN", tokens=("Yes", "No"))

    def _single_token_ids(self, symbols: str, *, tokens: tuple[str, ...] | None = None) -> list[int]:
        values = tokens or tuple(symbols)
        encoded = [self.tokenizer.encode(value, add_special_tokens=False) for value in values]
        if any(len(ids) != 1 for ids in encoded):
            raise ValueError(f"Qwen tokenizer must encode {values} as single tokens")
        return [ids[0] for ids in encoded]

    def _encode(self, content: str) -> list[int]:
        prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False) + "Answer: "
        ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if len(ids) > self.max_length:
            raise ValueError(f"input length {len(ids)} exceeds max_length {self.max_length}")
        return ids

    def _restricted_logits(self, ids: list[int], allowed: list[int]) -> list[float]:
        torch = self.torch
        with torch.inference_mode():
            tensor = torch.tensor([ids], device=self.device)
            hidden = self.model.model(input_ids=tensor, use_cache=False,
                                      return_dict=True).last_hidden_state[0, -1]
            logits = self.model.lm_head(hidden)[allowed].float()
            return logits.cpu().tolist()

    def score(self, record: dict) -> tuple[list[float], int, int, str]:
        n = len(record["options"])
        state = render(record["state"])
        question = render(record["question"])
        if n <= len(LETTERS):
            # Match scripts/qwen3_8b_letters.py: one forward pass per cyclic
            # rotation, restricted log-softmax across the displayed letters,
            # summed by original candidate, then softmax across candidates.
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
            sums = [0.0] * n
            for order, ids in rotations:
                logits = self._restricted_logits(ids, self.letter_ids[:n])
                maximum = max(logits)
                denominator = maximum + math.log(sum(math.exp(x - maximum) for x in logits))
                for index, logit in zip(order, logits):
                    sums[index] += logit - denominator
            if self.device.startswith("cuda"):
                self.torch.cuda.synchronize(self.device)
            return softmax(sums), sum(len(ids) for _, ids in rotations), n, "letter_logprob_rotations"

        # The TypeSafe Choice schema accepts up to 255 candidates, beyond the
        # single-letter alphabet. Score each candidate independently with
        # constrained Yes/No logits; this path is explicit in metadata.
        encoded = []
        for option in record["options"]:
            content = ("Context:\n" + state + "\n\nQuestion: " + question + "\n"
                       "Proposed answer: " + option + "\n"
                       "Is this proposed answer correct? Answer Yes or No.\nAnswer:")
            encoded.append(self._encode(content))
        odds = []
        for ids in encoded:
            yes, no = self._restricted_logits(ids, self.yes_no_ids)
            odds.append(yes - no)
        if self.device.startswith("cuda"):
            self.torch.cuda.synchronize(self.device)
        return softmax(odds), sum(len(ids) for ids in encoded), n, "candidate_yes_no_log_odds"


def make_server(predictor: Predictor, *, host: str = "127.0.0.1", port: int = 8795,
                max_body_bytes: int = 4 * 1024 * 1024) -> ThreadingHTTPServer:
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(30)

        def log_message(self, *args):
            pass  # Do not log prompt bodies or request headers.

        def send(self, status: int, data: dict):
            payload = json.dumps(data, ensure_ascii=False, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            path = urlsplit(self.path).path
            if path == "/health":
                return self.send(200, {"status": "ready", "model": predictor.model_name})
            if path == "/v1/models":
                return self.send(200, {"models": [{"id": predictor.model_name,
                                                   "name": predictor.model_name,
                                                   "description": "Local Qwen3-8B next-token decision adapter"}],
                                       "aliases": ["qwen3-8b", "jev-latest", "jev-1.13.0", "open-jev"]})
            return self.send(404, {"error": "not found"})

        def do_POST(self):
            if urlsplit(self.path).path not in ("/v1/systemone", "/v1/inference", "/api/jev"):
                return self.send(404, {"error": "not found"})
            origin = self.headers.get("Origin")
            if origin and urlsplit(origin).netloc != self.headers.get("Host"):
                return self.send(403, {"error": "cross-origin requests are disabled"})
            if self.headers.get_content_type() != "application/json":
                return self.send(415, {"error": "Content-Type must be application/json"})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 1 <= length <= max_body_bytes:
                    return self.send(413, {"error": "request body size is outside server limits"})
                data = self.rfile.read(length)
                if len(data) != length:
                    raise ValueError("incomplete request body")
                request = strict_json(data)
                with lock:
                    result = predictor.predict(request)
                return self.send(200, result)
            except (ValueError, KeyError, TypeError, UnicodeError) as error:
                return self.send(422, {"error": str(error)})
            except Exception as error:
                return self.send(500, {"error": "model inference failed", "error_type": type(error).__name__})

    return ThreadingHTTPServer((host, port), Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(ROOT / "models/qwen3_8b"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8795)
    args = parser.parse_args()
    scorer = QwenScorer(args.model, device=args.device, max_length=args.max_length)
    predictor = Predictor(scorer)
    server = make_server(predictor, host=args.host, port=args.port)
    print(json.dumps({"url": f"http://{args.host}:{server.server_port}",
                      "model": predictor.model_name, "max_length": args.max_length}), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
