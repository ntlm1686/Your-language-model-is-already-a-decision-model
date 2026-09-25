# Qwen3-8B behind the Jev System One API

`scripts/serve_qwen_jev.py` is a local, text-only adapter for the Jev JSON
request and response format. It runs Qwen3-8B and returns model probabilities
without generating an answer string. It does **not** load Jev weights or claim
Jev's calibration. This lets an existing Jev client call Qwen through the same
`/v1/systemone` route.

## Start the server

Download the pinned Qwen3-8B weights first with `python scripts/setup_sources.py
model`. The model path can be any local Qwen3-8B checkpoint. For the existing
weights on this machine:

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/serve_qwen_jev.py \
  --model /home/tiger/clm_eval/qwen3_8b \
  --device cuda:0 --host 127.0.0.1 --port 8795 --max-length 8192
```

`--device cuda:0` refers to the first GPU visible to that process. The server
binds to localhost by default and requires no API key. It loads the model once,
then serializes GPU inference across concurrent HTTP requests.

```bash
curl -fsS http://127.0.0.1:8795/health
curl -fsS http://127.0.0.1:8795/v1/models
curl -fsS http://127.0.0.1:8795/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{"model":"jev-1.13.0","state":"Customer asks for a refund.","questions":{"refund":{"type":"noul","instructions":"Is a refund requested?"},"route":{"type":"choice","instructions":"Where should the case go?","criteria":{"returns":"Refunds and exchanges","shipping":"Delivery issues"}},"priority":{"type":"score","instructions":"Rate urgency.","criteria":["Low","Medium","High"]}}}'
```

The response has `model`, `answers` and `usage`. The `answers` object contains
one entry per question ID:

| Question | Answer fields |
| --- | --- |
| `choice` | `type`, `choice`, `probabilities` keyed by candidate name, `confidence` |
| `noul` | `type`, `noul` (probability of Yes) |
| `score` | `type`, `score` (expected zero-based level), `probabilities` keyed by level index, `confidence`, `legend` |

The response also includes `metadata` with scoring method, measured inference
seconds, number of forward passes, context limit and calibration status. The
resolved `model` is always `qwen3-8b`, even if the client supplied
`jev-1.13.0`, `jev-latest` or `open-jev` as a request alias. This prevents a
Qwen result from being mistaken for an actual Jev result. The route aliases
`/v1/inference` and `/api/jev` are accepted too.

For a JevBench TypeSafe adapter, set `TYPESAFE_ENDPOINT=http://127.0.0.1:8795`
and set `TYPESAFE_API_KEY=local` because that adapter requires a nonempty key
environment variable. The local server does not inspect it. Use `--model
jev-1.13.0` in the adapter to exercise its usual request shape; the response
still identifies Qwen3-8B.
We also sent a public JevBench item through the upstream `TypeSafeAdapter.run`
without changing that adapter: it returned HTTP 200, parsed the native
probability map, and identified the response model as `qwen3-8b`.

## Scoring

For a question with at most 26 alternatives, the adapter renders all options
as `A. ...`, `B. ...`, and so on. It applies the Qwen chat template with
thinking disabled and ends the prompt with `Answer: `. It reads the next-token
logits for the displayed letters, normalizes over those letters, and repeats
the prompt once per cyclic option rotation. It sums each candidate's log
probability over rotations, then applies softmax over candidates. This matches
the JevBench Qwen3-8B letter-scoring protocol in
`scripts/qwen3_8b_letters.py`. It is also the default for `noul` and `score`.
For Choice and Noul, criteria descriptions appear as the displayed options;
candidate keys stay outside the prompt and map the result back to the Jev
response format. This reproduces the JevBench prompt construction for these
question types when the request contains the same state, instructions and
criteria.

Jev allows as many as 255 Choice candidates. Above 26, the adapter instead
tests each candidate separately with constrained next-token `Yes` and `No`
logits, then normalizes the resulting log odds over candidates. The response
`metadata.method` identifies which method was used. A one-option Choice returns
probability one.

Inputs longer than `--max-length` receive HTTP 422. The server never silently
truncates. `usage.input_tokens` sums the actual encoded tokens across all
forward passes; `usage.output_tokens` is zero. `metadata.inference_seconds`
measures model work after request parsing, including tokenization and GPU
synchronization. HTTP latency includes transport and request parsing, while
model startup is excluded from both. Qwen probabilities are **uncalibrated**;
the Jev-style `confidence` field follows the wire formula and is not an
estimated chance that the decision is correct.

## Verification

The fast schema tests cover Choice, Noul, Score, multiple questions in one
request, 255-candidate acceptance, JSON validation and HTTP error responses:

```bash
python -m unittest discover -s tests -p 'test_qwen_jev_api.py' -v
```

They do not load weights. For an actual GPU smoke test, start the server and
run the `curl` request above; the response should report `model: qwen3-8b`
and `usage.input_tokens > 0`.

To evaluate the full public JevBench set through this API, first fetch its
pinned source with `python scripts/setup_sources.py repos`, then run:

```bash
python scripts/eval_qwen_jev_api.py \
  --url http://127.0.0.1:8795 \
  --output results/jevbench/qwen3_8b_jev_api.jsonl
```

The client writes one resumable JSONL record per item with request hash,
prediction, probability map, token usage and latency. It sends Choice
criteria in `labels` order: 119 public rows have a different original JSON
object order, whereas the archived Qwen letter evaluator uses `labels` order.
The answer key never enters the API request. Length rejections are reported
as ineligible, not scored as wrong. Compare accuracy only on the exact set
of eligible IDs when a different `--max-length` is used.

Our matched-environment local run on 2026-09-25 used Python 3.11, PyTorch
2.12.0, and Transformers 4.57.6, as in the archived direct Qwen evaluation.
It scored **163/231** with **167 ms median HTTP latency** and all 231 items
eligible. Its predictions agree with the archived direct Qwen run on **231/231**
items. A preliminary run using Transformers 5.10.2 also scored 163/231, but
three individual predictions differed; the matched software environment
resolved that discrepancy.
