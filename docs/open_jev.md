# Open-Jev 2B and 9B benchmark protocol

We evaluate the released [Open-Jev-2B](https://huggingface.co/ZefanCai/Open-Jev-2B) and [Open-Jev-9B](https://huggingface.co/ZefanCai/Open-Jev-9B) LoRA adapters and scalar heads. These are **not** standalone language models: the official loader combines each checkpoint with its pinned Qwen3.5 backbone. We use the author's non-generative `Choice`/`Noul`/`Score` interface, saved calibration temperatures, and local `/v1/systemone` server. We do not use generic `AutoPeftModel.generate`.

## Model structure

For a `Choice` question, the official [prompt renderer](https://github.com/Zefan-Cai/Open-Jev/blob/3308a15ccd7eea1df7a37d6ddc39b023b801ba16/jev/api.py) creates **one text prompt per candidate**. Each prompt contains the context (state), question, and one proposed answer (action), followed by a yes/no correctness question. The [model](https://github.com/Zefan-Cai/Open-Jev/blob/3308a15ccd7eea1df7a37d6ddc39b023b801ba16/jev/model.py) scores the final non-padding hidden state with a trained one-output linear head. Saved temperature calibration converts the candidate logits to probabilities; the highest score wins.

```text
state + action_i → Qwen3.5 language backbone + LoRA → final hidden state
                 → trained linear head → scalar score_i
all candidate scores → calibrated softmax → selected action
```

This is joint state/action encoding with one forward per candidate (or a batch of those forwards). It does not use next-letter token probabilities or independent state/action embeddings. `Noul` uses a single yes/no score; `Score` scores each ordinal level and combines their probabilities. Although the original Qwen3.5 checkpoint has a vision component, the Open-Jev loader retains `full.model.language_model` and discards the vision component. The released inference path is text-only.

## Pinned sources

| Component | Revision |
| --- | --- |
| `Zefan-Cai/Open-Jev` source | `3308a15ccd7eea1df7a37d6ddc39b023b801ba16` |
| `ZefanCai/Open-Jev-2B` package | `0c7aa498b1627be8da4acf34c863ff0ee0a92785` |
| `ZefanCai/Open-Jev-9B` package | `47e966881e489511c0c7f5633a9e1960a676a551` |
| 2B base: `Qwen/Qwen3.5-2B` | `15852e8c16360a2fea060d615a32b45270f8a8fc` |
| 9B base: `Qwen/Qwen3.5-9B` | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` |

Download the source and both adapters with `python scripts/setup_sources.py open_jev`. The official loader downloads the pinned base weights on first use. The model packages and upstream source stay under ignored `models/` and `external/`; weights are not committed.

## Serving setup

Use a PyTorch build compatible with the host NVIDIA driver. This machine uses Python 3.11, PyTorch 2.12.0 with CUDA 12.9, Transformers 5.10.2, PEFT 0.19.1, and Accelerate 1.13.0. An attempted PyTorch 2.14 CUDA 13 environment could not initialize CUDA on this host; no benchmark result came from it. The working local environment was created as follows, using the machine's already installed driver-compatible PyTorch:

```bash
uv venv models/open_jev_py311 --python /usr/bin/python3.11 --system-site-packages
uv pip install --python models/open_jev_py311/bin/python --no-deps \
  'transformers==5.10.2' 'peft==0.19.1' 'accelerate==1.13.0' \
  'safetensors' 'huggingface-hub==1.33.0'
uv pip install --python models/open_jev_py311/bin/python --no-deps -e external/Open-Jev
```

In two separate terminals, start the upstream servers (one model per GPU):

```bash
CUDA_VISIBLE_DEVICES=1 models/open_jev_py311/bin/python -m jev.server \
  --checkpoint models/open_jev_2b/package/checkpoint --device cuda:0 \
  --max-length 4096 --batch-size 1 --no-prefix-cache --host 127.0.0.1 --port 8791

CUDA_VISIBLE_DEVICES=2 models/open_jev_py311/bin/python -m jev.server \
  --checkpoint models/open_jev_9b/package/checkpoint --device cuda:0 \
  --max-length 4096 --batch-size 1 --no-prefix-cache --host 127.0.0.1 --port 8792
```

`--no-prefix-cache` follows the model cards' recommended reproducible configuration. The 4,096-token cap applies to **each independently rendered candidate**. Oversize inputs receive HTTP 422 and are recorded as `too_long`, never silently truncated or counted as errors.
The optional `flash-linear-attention` and `causal-conv1d` kernels are absent in this environment, so the Qwen3.5 backbone uses the upstream PyTorch fallback. Latency results apply to that specific serving stack.

## Benchmark calls

`run_open_jev.py` rebuilds the exact candidate sets from the pinned data and checks IDs, gold labels, or input hashes against the archived Qwen/Jev records before calling a local server. It saves one JSONL row per input, including outcome, prediction, probabilities, latency, backend inference time, model revision, and request hash. It supports resume and fails on unexpected HTTP or malformed responses. The old Qwen3-8B tokenizer is needed only to reproduce the old sample/option selection; the extra baseline weights are unnecessary.

```bash
export MODEL_PATH="$PWD/models/qwen3_8b_tokenizer"  # used only to reconstruct old candidate sets
for suite in jevbench metatool when2call bfcl phishing webprm120 webprm1141 deepswe; do
  python scripts/run_open_jev.py "$suite" --url http://127.0.0.1:8791 \
    --output "results/open_jev_2b/${suite}.jsonl"
  python scripts/run_open_jev.py "$suite" --url http://127.0.0.1:8792 \
    --output "results/open_jev_9b/${suite}.jsonl"
done
python scripts/summarize_open_jev.py
```

Prepare older inputs with `python scripts/setup_sources.py repos qwen_tokenizer webprm phishing when2call metatool bfcl`; the DeepSWE normalized input is already committed in this repository. See the [main protocol](benchmark_protocol.md). `check_open_jev_lengths.py` uses the official renderer and pinned Qwen3.5 tokenizer for an independent coverage audit, for example:

```bash
MODEL_PATH="$PWD/models/qwen3_8b_tokenizer" models/open_jev_py311/bin/python \
  scripts/check_open_jev_lengths.py --model 2b --suite deepswe \
  --output results/open_jev_2b/lengths_deepswe.json
```

The Open-Jev latency is a localhost HTTP round trip on a single H100 with batch size 1. Jev 1.13.0 in our original table is a remote API call, while Qwen3-8B and CLM were measured by local Python forwards. The times describe these deployments; they do not isolate model architecture speed. For accuracy, compare only the same eligible item IDs, especially on long WebPRM and DeepSWE states.

### Long DeepSWE trajectories

The released checkpoint configuration is capped at 4,096 tokens per candidate. The official renderer/tokenizer length audit finds **0 of 152** DeepSWE rotated requests eligible under that cap for either model. A separate exploratory run uses the official server's `--max-length 8192` override, without changing or truncating the task and candidate text:

```bash
CUDA_VISIBLE_DEVICES=3 models/open_jev_py311/bin/python -m jev.server \
  --checkpoint models/open_jev_2b/package/checkpoint --device cuda:0 \
  --max-length 8192 --batch-size 1 --no-prefix-cache --host 127.0.0.1 --port 8793
MODEL_PATH="$PWD/models/qwen3_8b_tokenizer" python scripts/run_open_jev.py deepswe \
  --url http://127.0.0.1:8793 --output results/open_jev_2b/deepswe_8k.jsonl
```

Repeat with the 9B checkpoint on another GPU and port if desired. The extended-context scores are labelled **exploratory** because 8,192 tokens exceed the checkpoint's published 4,096-token evaluation configuration. They are never pooled with the default-cap results.

## Measured results

Both released checkpoints were evaluated on every suite above. The [machine-readable summary](../results/open_jev_summary.json) is recomputed from the committed per-item JSONL files by `scripts/summarize_open_jev.py`. The Qwen/CLM/Jev baseline counts below are recounted on exactly the Open-Jev-eligible IDs for each row. Latency is the median localhost round trip for an eligible item, rounded to milliseconds.

| Benchmark | Eligible / total | Qwen3-8B | CLM default 2K | Jev 1.13.0 | Open-Jev-2B correct | 2B p50 | Open-Jev-9B correct | 9B p50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| JevBench | 231/231 | 163 | 95 | 199 | 150 | 444 ms | 179 | 602 ms |
| MetaTool Task1 | 1,040/1,040 | 842 | 518 | 802 | 692 | 222 ms | 792 | 293 ms |
| When2Call | 599/600 | 306 | 200 | 439 | 235 | 600 ms | 451 | 790 ms |
| BFCL v4 Multiple | 200/200 | 198 | 169 | 198 | 196 | 328 ms | 197 | 447 ms |
| PhishNChips | 2,000/2,000 | 1,330 | 1,212 | 1,251 | 1,000 | 240 ms | 1,453 | 315 ms |
| WebPRM sample | 93/120 | 53 | 19 | 54 | 43 | 1,153 ms | 47 | 1,527 ms |
| WebPRM full | 842/1,141 | 528 | 152 | 580 | 421 | 1,182 ms | 472 | 1,527 ms |

At the official 4K cap, all 152 DeepSWE rotations for both checkpoints returned `too_long` (0/38 complete tasks eligible). At 8K, both evaluated all 38 tasks: 2B selected a successful trajectory for **24/38** tasks, p50 **1,530 ms per rotation**; 9B scored **23/38**, p50 **2,073 ms per rotation**. These 8K results are separate exploratory measurements. The Qwen, CLM, and Jev DeepSWE task scores in the main table use their own evaluation configurations and are not part of the 4K same-eligible comparison above.
