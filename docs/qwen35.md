# Untrained Qwen3.5-9B evaluation

The [Qwen3.5-9B checkpoint](https://huggingface.co/Qwen/Qwen3.5-9B) is used as released, revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`. The text backbone and original language-model head provide next-token logits. No decision labels, LoRA, scalar head, or images are used. This is a text-only evaluation of a multimodal checkpoint.

The [runner](../scripts/eval_qwen35_letters.py) reuses the Qwen3-8B prompt builders, option descriptions, and archived candidate orders. For When2Call and WebPRM, the Qwen3-8B tokenizer is used **only to reconstruct the original eligible item IDs**. The scored model uses its own tokenizer and sees the same raw text. Some prompts tokenize to more than 8K or 16K Qwen3.5 tokens; the runner permits up to 32K so that no archived item is truncated. The per-item output includes every prompt's Qwen3.5 token count. The exact same textual input and options are used, while token counts and effective context budget differ between tokenizers.

Scoring follows the existing task-specific rule: all cyclic option rotations for JevBench, MetaTool, When2Call, BFCL, and DeepSWE; both A/B orders for PhishNChips; one saved order for each WebPRM state. Scores sum restricted next-token log probabilities and select the largest sum. DeepSWE success is the selected candidate's recorded verifier reward. In the original serial archive, latency is BF16 model forward time on a local H100, summed over rotations; it excludes loading, prompt tokenization, and HTTP. The Doom client measures full API round-trip latency separately.

## Optimized evaluation used by the main table

Every Qwen3.5-9B time in the main benchmark table now comes from the same [optimized runner](../scripts/eval_qwen35_optimized.py). It uses the pinned checkpoint, BF16, SDPA, and flash-linear-attention 0.5.2, and batches all option rotations for one item into one forward pass. WebPRM retains its original single candidate order. There is no sequential near-tie fallback and no context truncation; the cap remains 32,768 tokens.

Four independent workers run on four H100 80GB GPUs, with one model per GPU and item IDs distributed by index modulo four. Every worker asserts that the FLA chunk kernel is active. A short synthetic warmup precedes measurement. Shape-specific compilation during real items remains included; no timed items or outliers are removed.

**Timing:** the main table uses the median measured time per decision, including prompt construction, tokenization, GPU input preparation, model forward, and score aggregation. It excludes model loading, dataset loading, output-file writing, and HTTP. DeepSWE includes all four candidate rotations in a single batched decision. Jev's saved times include the remote API round trip, so these are deployment measurements with different network boundaries. Raw records also contain the narrower GPU-stage duration as `forward_s`; that value is not used in the main time column.

**Accuracy:** optimized kernels and batching can change near-tied predictions. The main table uses accuracy and timing from the same new run. Earlier predictions remain in the original directory. The [recount script](../scripts/summarize_qwen35_optimized.py) checks complete identical item IDs, labels, prompt token lengths, candidate rotations, and archived input hashes where available. It also recounts the exact Open-Jev-eligible subsets and reports prediction agreement with the earlier serial run.

Install the environment described below and the optional kernel package, then run:

```bash
models/open_jev_py311/bin/pip install flash-linear-attention==0.5.2
for gpu in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$gpu QWEN3_8B_TOKENIZER=models/qwen3_8b_tokenizer \
    models/open_jev_py311/bin/python scripts/eval_qwen35_optimized.py \
    --shard "$gpu" --nshards 4 > "/tmp/qwen35_optimized_${gpu}.log" 2>&1 &
done
wait
python scripts/summarize_qwen35_optimized.py
```

Per-item records and the checked summary are in `results/qwen3_5_9b/optimized/`. For a fresh rerun, choose a new `--output-dir`; by default, completed IDs are resumed. The older API comparison below remains a historical implementation check and is not the source of the main benchmark time column.

## Optimized run results

| Benchmark | Correct / total | Median decision time | Matches original prediction |
| --- | ---: | ---: | ---: |
| JevBench public | 186/231 | 64 ms | 229/231 |
| WebPRMBench sample | 59/120 | 123 ms | 119/120 |
| WebPRMBench full | 673/1141 | 132 ms | 1133/1141 |
| DeepSWE | 19/38 | 2,543 ms | 38/38 |
| PhishNChips | 1478/2000 | 55 ms | 1989/2000 |
| MetaTool Task1 | 867/1040 | 55 ms | 1038/1040 |
| When2Call | 371/600 | 139 ms | 595/600 |
| BFCL v4 Multiple | 199/200 | 58 ms | 200/200 |

## Original serial run (archive)

Install the [Open-Jev Python environment](open_jev.md#serving-setup), which has Transformers 5.10.2 and PyTorch 2.12.0. Fetch the pinned benchmark inputs and tokenizer with `python scripts/setup_sources.py repos qwen_tokenizer webprm phishing when2call metatool bfcl`. The DeepSWE normalized inputs are committed under `data/deepswe/`. Download the pinned Qwen3.5-9B checkpoint into the Hugging Face cache:

```bash
models/open_jev_py311/bin/python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen3.5-9B", revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a")
PY
```

```bash
export QWEN3_8B_TOKENIZER=models/qwen3_8b_tokenizer
CUDA_VISIBLE_DEVICES=0 models/open_jev_py311/bin/python scripts/eval_qwen35_letters.py jevbench
CUDA_VISIBLE_DEVICES=0 models/open_jev_py311/bin/python scripts/eval_qwen35_letters.py webprm120
CUDA_VISIBLE_DEVICES=0 models/open_jev_py311/bin/python scripts/eval_qwen35_letters.py webprm1141
CUDA_VISIBLE_DEVICES=0 models/open_jev_py311/bin/python scripts/eval_qwen35_letters.py deepswe
CUDA_VISIBLE_DEVICES=0 models/open_jev_py311/bin/python scripts/eval_qwen35_letters.py phishing
CUDA_VISIBLE_DEVICES=0 models/open_jev_py311/bin/python scripts/eval_qwen35_letters.py metatool
CUDA_VISIBLE_DEVICES=0 models/open_jev_py311/bin/python scripts/eval_qwen35_letters.py when2call
CUDA_VISIBLE_DEVICES=0 models/open_jev_py311/bin/python scripts/eval_qwen35_letters.py bfcl
python scripts/summarize_qwen35.py
```

Each suite resumes by item ID from `results/qwen3_5_9b/*.jsonl`. The summary script checks expected counts and duplicate IDs, then writes `results/qwen3_5_9b/summary.json`.

The Jev-format server was also tested on all 231 public JevBench items with `scripts/eval_qwen_jev_api.py --model-name qwen3.5-9b --url http://127.0.0.1:8796 --output results/qwen3_5_9b/jevbench_api.jsonl`. The original sequential service agreed with the direct runner on 231/231 items. Its local API round-trip p50 was 541 ms; the direct runner's BF16 forward-only p50 was 514 ms.

The optimized server batches cyclic option orders into one forward with right padding and a padding mask. By default, it uses the original PyTorch DeltaNet implementation and reruns choices sequentially when the two highest batch probabilities are within 0.1. This preserved 231/231 JevBench decisions at 140 ms p50; 10/231 requests used the fallback. Set `QWEN35_USE_FLA=1` and `--fallback-margin 0` to use the optional `flash-linear-attention==0.5.2` chunk kernel: it reached 61 ms p50 but changed two near-tied predictions (229/231 agreement, still 186/231 correct). The causal-conv1d package was not installed, so its convolution remained on the PyTorch path. First-use Triton compilation creates latency outliers; p50 excludes no items.

## Jev-format server for Doom

```bash
CUDA_VISIBLE_DEVICES=0 models/open_jev_py311/bin/python scripts/serve_qwen35_jev.py --port 8796
models/doom_py311/bin/python scripts/eval_doom_qwen.py \
  --policies api_position_none --url http://127.0.0.1:8796 \
  --endpoint-model qwen3.5-9b --expected-model qwen3.5-9b \
  --max-wall-seconds 30 \
  --output results/doom/qwen3_5_9b_30s_batch_safe.json \
  --trace results/doom/qwen3_5_9b_30s_batch_safe_steps.jsonl \
  --video-policy api_position_none --video /tmp/qwen3_5_9b_30s_batch_safe_seed100.mp4
models/doom_py311/bin/python scripts/render_doom_gifs.py \
  --video /tmp/qwen3_5_9b_30s_batch_safe_seed100.mp4 \
  --trace results/doom/qwen3_5_9b_30s_batch_safe_steps.jsonl \
  --model-name Qwen3.5-9B --output results/doom/qwen3_5_9b_30s_batch_safe_seed100.gif
```

`--max-wall-seconds` starts a monotonic timer after each episode begins. Inference, HTTP, rendering, and game stepping count against the budget. The client completes any in-flight decision, then stops before starting another one if the deadline passed. The result records real elapsed seconds and whether the environment had ended before the stop.

To reproduce the fastest 30-second run and its GIF, install [flash-linear-attention](https://github.com/fla-org/flash-linear-attention) at version `0.5.2`, start the server with `QWEN35_USE_FLA=1 CUDA_VISIBLE_DEVICES=0 models/open_jev_py311/bin/python scripts/serve_qwen35_jev.py --port 8796 --fallback-margin 0`, and use output names ending in `_fast`. The faster FLA path is a numerical variant; the default PyTorch path preserves the archived JevBench choices exactly.
