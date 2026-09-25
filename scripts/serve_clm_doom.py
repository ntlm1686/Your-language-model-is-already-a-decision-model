"""Serve released CLM-v0.1-8B locally for matched Doom Choice requests.

Uses CLM's official ``build_pairs`` and released projection heads, with the
same Qwen3-8B final-token embedding and default 2048-token tail cap as the
repository's CLM benchmark runs. This small server has no embedding API
dependency and supports the Choice wire path needed by the Doom client.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "external/CLM/src"))
sys.path.insert(0, str(ROOT / "external/CLM"))


class CLMPredictor:
    model_name = "clm-latest"

    def __init__(self, model_path: str, head_path: str, *, max_tokens: int = 2048):
        import torch
        import torch.nn.functional as F
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from evaluation.bon_eval import load_heads

        self.torch, self.F, self.max_tokens = torch, F, max_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.backbone = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.bfloat16, attn_implementation="sdpa",
            low_cpu_mem_usage=True).eval().to("cuda:0")
        self.state_head, self.action_head, _ = load_heads(head_path, torch.device("cuda:0"))
        checkpoint = torch.load(head_path, map_location="cpu", weights_only=False)
        self.scale = float(torch.as_tensor(checkpoint["logit_scale"]).float().exp().clamp(max=100.0))
        self.action_cache: dict[str, tuple[object, int]] = {}

    @property
    def device(self):
        return "cuda:0"

    def embed(self, text: str) -> tuple[object, int]:
        torch, F = self.torch, self.F
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if not ids:
            ids = self.tokenizer.encode(" ", add_special_tokens=False)
        ids = ids[-self.max_tokens:]
        tensor = torch.tensor([ids], device=self.device)
        with torch.inference_mode():
            hidden = self.backbone.model(input_ids=tensor, use_cache=False,
                                         return_dict=True).last_hidden_state[0, -1].float()
        return F.normalize(hidden, dim=-1), len(ids)

    def predict(self, body: object) -> dict:
        from clm.schema import answer_from_logits, build_pairs

        if not isinstance(body, dict) or "state" not in body or not isinstance(body.get("questions"), dict):
            raise ValueError("body must contain state and questions")
        if body.get("model") not in (None, self.model_name):
            raise ValueError("unknown model")
        questions = body["questions"]
        if not questions or any(q.get("type") != "choice" for q in questions.values()):
            raise ValueError("Doom CLM server supports nonempty Choice questions only")
        pairs = build_pairs(body["state"], questions)
        started = time.perf_counter()
        answers, token_count = {}, 0
        with self.torch.inference_mode():
            for question_id, (state, keys, options) in pairs.items():
                state_vec, spent = self.embed(state)
                token_count += spent
                projected_state = self.F.normalize(self.state_head(state_vec), dim=-1)
                logits = []
                for option in options:
                    if option not in self.action_cache:
                        self.action_cache[option] = self.embed(option)
                        token_count += self.action_cache[option][1]
                    action_vec = self.action_cache[option][0]
                    projected_action = self.F.normalize(self.action_head(action_vec), dim=-1)
                    logits.append(float(self.scale * (projected_state * projected_action).sum()))
                answers[question_id] = answer_from_logits(questions[question_id], keys, logits)
        self.torch.cuda.synchronize()
        return {"model": self.model_name, "answers": answers,
                "usage": {"input_tokens": token_count, "output_tokens": 0},
                "metadata": {"inference_seconds": time.perf_counter() - started,
                             "max_length": self.max_tokens, "method": "clm_scaled_cosine"}}


def serve(predictor: CLMPredictor, host: str, port: int) -> None:
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, status: int, body: dict):
            payload = json.dumps(body, ensure_ascii=False, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            if self.path == "/health":
                return self.send(200, {"status": "ready", "model": predictor.model_name})
            return self.send(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/v1/systemone":
                return self.send(404, {"error": "not found"})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 1 <= length <= 4 * 1024 * 1024:
                    return self.send(413, {"error": "invalid request size"})
                body = json.loads(self.rfile.read(length))
                with lock:
                    response = predictor.predict(body)
                return self.send(200, response)
            except (ValueError, TypeError, KeyError, AttributeError) as error:
                return self.send(422, {"error": str(error)})
            except Exception as error:
                return self.send(500, {"error": "CLM inference failed", "error_type": type(error).__name__})

    server = ThreadingHTTPServer((host, port), Handler)
    print(json.dumps({"url": f"http://{host}:{port}", "model": predictor.model_name}), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(ROOT / "models/qwen3_8b"))
    parser.add_argument("--head", default=str(ROOT / "models/reference_head/CLM_v0.1-8B.pt"))
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8797)
    args = parser.parse_args()
    serve(CLMPredictor(args.model, args.head, max_tokens=args.max_tokens), args.host, args.port)


if __name__ == "__main__":
    main()
