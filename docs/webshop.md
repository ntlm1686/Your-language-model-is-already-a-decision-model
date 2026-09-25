# WebShop closed-loop evaluation

We evaluate Qwen3.5-9B next-token probabilities, hosted Jev 1.13.0, and Open-Jev-9B on the **500 official test goal indices (0–499)** with the full product catalog. The official validation split is 500–1499. This is a text-observation experiment, not a screenshot policy.

## Protocol

- Source: `princeton-nlp/WebShop`, revision `64fa2a5c15c7daa698b9ac93f5bb5437b634c9bd`.
- No model is trained for this experiment. Open-Jev retains its released trained adapter/head; Qwen has no decision training.
- All policies share the same initialized environment server, product prices, instructions, available-click extraction, prompt, history rule, and budget. Each policy follows its own trajectory.
- At a search page, use the visible instruction verbatim (lowercased) as the search query. This deterministic shared rule generates no tokens and never reads hidden target ASINs, goal attributes, or goal queries. This is our fixed-search protocol, not a reproduction of the original paper's trained BART search policy.
- All visible clickable actions are offered in environment order. No oracle filtering or candidate pruning. State includes the current page, instruction, and previous 10 actions.
- Maximum 100 environment actions; terminate early on purchase or the third repetition of an identical page/action pair. Unfinished episodes score zero. The repeat rule applies equally to all policies.
- The environment's terminal reward is on 0–1; report average reward ×100 and success rate (reward=1). Invalid/overlength request rejections terminate the episode with zero reward and remain in the denominator. Transient network failures are retried and logged; persistent infrastructure failures stop that worker rather than being silently dropped.
- Hosted Jev returns rounded probabilities, whose sum may be 0.99 or 1.01. Validation allows per-option rounding error. The hosted API occasionally returns a `choice` whose displayed probability is slightly below another option (observed 0.47 versus 0.48). We execute its returned `choice`, preserve the raw response, and count these discrepancies in the summary; we do not substitute our own argmax. Local models must return an argmax choice.
- Qwen uses BF16, SDPA, FLA, batched cyclic option rotations, no near-tie fallback. Up to 26 actions use letter probabilities; larger candidate sets use independent Yes/No next-token log odds, batched in chunks of eight and recorded in response metadata. Three real large-candidate requests are checked against the earlier serial path in `results/webshop/qwen_batch_validation.json`; final results come from a fresh complete 500-task run. No context truncation (Qwen cap 8192; Open-Jev per-candidate cap 4096).
- Open-Jev uses the pinned official loader, released calibration, no prefix cache, candidate batch size 8. It identifies itself as `Qwen/Qwen3.5-9B` in the API response; saved records label it `open_jev` and deployment loads the Open-Jev adapter/head.
- Four H100s are used. Final Qwen results use four independent workers. Open-Jev initially uses two workers and expands to four workers, resuming only unfinished episodes with the same settings. Hosted Jev initially uses one worker and expands to four concurrent workers; each request remains a single decision. Each local service receives one evaluation stream.

## Timing and tokens

Episode time includes state/action preparation, all HTTP calls and retries, environment steps, and stopping checks; excludes environment/model loading, synthetic warmup and result serialization. Three real-request batching checks also precede the final Qwen run; their time is not part of the task measurements. Decision time includes the HTTP round trip. Local and hosted network boundaries differ. Shape-specific compilation remains included.

Token counts are the API's reported input/output usage, accumulated over successful decisions. Qwen counts all scored rotations (or independent candidates), Open-Jev counts its rendered candidate prompts; these are deployment workloads, not identical strings. Tokens from failed requests may be unavailable. The fixed search rule costs no model tokens. No money estimate is inferred from token counts.

## Data and setup

Original Google Drive downloads were inaccessible. Full files were downloaded from the pinned mirror `HongbangYuan/webshop`, revision `0129d4a81dbdb827e76afd20a1e2c38b61098613`. SHA256:

```
2ef591d65df3af89e972ab72468eb82cbf124d876552d9f3678667edd620a6c8  items_shuffle.json
1d36af476bdb8f82a5da62bd8acdabe54cd8de2fa84010d37da5c4890feb447e  items_ins_v2.json
```

The setup script downloads data, checks hashes, selects the upstream full-catalog paths, and builds the upstream Lucene index. Java 11 is required. Use the isolated WebShop environment (Python 3.11, NumPy 1.26, Pyserini 0.17, Gym 0.23.1, spaCy with `en_core_web_sm`) separately from the model environment.

```bash
uv venv models/webshop_py311 --python /usr/bin/python3.11 --system-site-packages
uv pip install --python models/webshop_py311/bin/python \
  'numpy<2' 'pyserini==0.17.0' 'gym==0.23.1' flask beautifulsoup4 cleantext \
  rank_bm25 rich thefuzz spacy gdown faiss-cpu huggingface-hub selenium==4.2.0
models/webshop_py311/bin/python -m spacy download en_core_web_sm
# JAVA_HOME must point to a Java 11 JDK.
models/webshop_py311/bin/python scripts/setup_webshop.py
```

Start model services using the [Qwen setup](qwen35.md) and [Open-Jev setup](open_jev.md). For an initial two-model run, on GPUs 0/1 use `serve_qwen35_jev.py --port 8830/8831 --fallback-margin 0` with `QWEN35_USE_FLA=1`. For GPUs 2/3, use `python -m jev.server` with the 9B checkpoint, `--batch-size 8 --max-length 4096 --no-prefix-cache --port 8832/8833`.

```bash
models/webshop_py311/bin/python scripts/eval_webshop.py --limit 500 \
  --api-key-file private/doom_api_key
python scripts/summarize_webshop.py --require-complete
```

For the final four-GPU Qwen run, start a Qwen server on each port 8830–8833 and pass `--models qwen --qwen-urls http://127.0.0.1:8830 http://127.0.0.1:8831 http://127.0.0.1:8832 http://127.0.0.1:8833`. To run only hosted Jev with four concurrent episodes, pass `--models jev --jev-workers 4`. To resume Open-Jev on four GPUs, start its servers on those four ports and pass `--models open_jev --open-jev-urls` followed by the four URLs.

The runner saves one atomic JSON file per model/task, including full state, candidate descriptions, probabilities, actions, API metadata, tokens and timings. It resumes completed episodes and validates the protocol/instruction hashes. `manifest.json` records all 500 visible instructions, the catalog size and protocol. Summaries before all 500 episodes finish are explicitly partial; do not interpret unequal partial samples as a model ranking.

## Complete run results

All three models completed all 500 test IDs. There were no context-length rejections. Qwen used the letter path on 1,263 decisions and batched Yes/No on 126 decisions. Hosted Jev returned three choices that differed from the argmax of its displayed probabilities; the API choices were executed and all responses remain in the traces.

| Model | Successes | Mean score / 100 | Mean actions | Mean input tokens | Mean reported output tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-9B | 124/500 | 60.52 | 3.87 | 32,540 | 0 |
| Jev 1.13.0 (hosted) | 123/500 | 56.59 | 4.66 | 3,922 | 419 |
| Open-Jev-9B | 83/500 | 48.82 | 15.63 | 54,625 | 0 |

The [summary](../results/webshop/summary.json), [deployment settings](../results/webshop/deployment.json), [environment versions](../results/webshop/environment.json), and [500-task manifest](../results/webshop/manifest.json) are committed. Per-episode files are under `results/webshop/{qwen,jev,open_jev}/`. Batch validation matched the serial decision on all three checked requests, with identical token counts. The final Qwen table uses the complete fresh batch-optimized run; the earlier serial-large-candidate trial is not used.

## Task and decision times

Local models use one NVIDIA H100 80GB per worker; Jev uses its hosted API. Task time and token counts are means over all 500 attempts, while decision time is the median API round trip.

| Model | Success rate | Score / 100 | Task time | Decision time | Input tokens / task |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-9B | 24.8% (124/500) | 60.52 | 1.36 s | 382 ms | 32,540 |
| Jev 1.13.0 (hosted) | 24.6% (123/500) | 56.59 | 1.41 s | 345 ms | 3,922 |
| Open-Jev-9B | 16.6% (83/500) | 48.82 | 3.80 s | 71 ms | 54,625 |

## Recreate the chart

The chart reads the verified summary, starts its bar axis at zero, and labels all three measured success rates.

```bash
uv run --no-project --with matplotlib python scripts/plot_webshop.py
```

Exports: `docs/assets/webshop_success.svg` and `docs/assets/webshop_success.png`.
