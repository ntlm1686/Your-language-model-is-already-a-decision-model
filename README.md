# Your Language Model Is Already a Decision Model

Research question: given fixed candidate actions or trajectories, does a specialized decision model improve accuracy, latency, or context use over direct option log probabilities from a general language model?

This repository contains experiments completed on **2026-09-25**, per-item outputs, and reproducible code. Qwen3-8B is scored using `Answer: X` next-token probabilities; CLM uses released heads; Jev 1.13.0 was queried through TypeSafe's API. **No complete browser-agent evaluation has been performed yet.**

The [article](docs/article.md) presents the no-training Qwen decision method, diagram, Jev-format serving API, same-eligible-model comparison, discussion, and reproduction commands.

## Results

| Benchmark | Metric | Qwen3-8B (score · p50 latency) | CLM (score · p50 latency) | Fresh Jev 1.13.0 (score · p50 latency) | Reference |
| --- | --- | ---: | ---: | ---: | ---: |
| [JevBench public 231](docs/benchmark_protocol.md#jevbench) | Correct typed decisions | 163/231 · 167 ms | 95/231 · — | **199/231** · 336 ms | Historical Jev: 200/231 |
| [WebPRMBench sample of 120 states](docs/benchmark_protocol.md#webprmbench) | Correct choice among five next actions | **65/120** · 139 ms | 24/120 · 269 ms | **65/120** · 371 ms | Random: 24/120 |
| [WebPRMBench full eligible 1,141 states](docs/webprm.md) | Correct choice among five next actions | 679/1,141 · 153 ms | 209/1,141 · 293 ms | **760/1,141** · 409 ms | Random: 228/1,141 |
| [DeepSWE, 38 tasks with identical candidates](docs/benchmark_protocol.md#deepswe) | Successful trajectory selected from four | 18/38 · — | 21/38 · — | 20/38 · — | Best fixed agent: 22/38 |
| [PhishNChips, 2,000 emails](docs/phishing.md) | Correct phishing/legitimate decisions | **1,330/2,000** · 67 ms | 1,212/2,000 · 101 ms | 1,251/2,000 · 336 ms | URL rule: **1,833/2,000** |
| [MetaTool Task1, 1,040 queries](docs/tool_decisions.md) | Correct use-tool decision | **842/1,040** · 75 ms | 518/1,040 · 109 ms | 802/1,040 · 370 ms | Balanced chance: 520/1,040 |
| [When2Call, 600-item balanced sample](docs/tool_decisions.md) | Correct next response | 307/600 · 188 ms | 201/600 · 180 ms | **440/600** · 371 ms | Four-way random: 150/600 |
| [BFCL v4 Multiple, 200 items](docs/tool_decisions.md) | Correct function name selected | **198/200** · 110 ms | 169/200 · 142 ms | **198/200** · 365 ms | Random expectation: 77/200 |

Generic CLM uses its default 2K context. DeepSWE uses a separate task-specific head; the table shows its final-12-history result. See the [protocol](docs/benchmark_protocol.md) for full methods and limits. WebPRM is offline action selection, and the phishing set is largely separable by a URL rule.

The released Open-Jev-2B and Open-Jev-9B adapters were also run locally. Their [full table](docs/open_jev.md#measured-results) includes exact shared-eligible Qwen/CLM/Jev baselines, context coverage, and per-item latency. On JevBench they scored 150/231 and 179/231; on the 842 WebPRM states eligible under their official 4K limit, 421/842 and 472/842, compared with Qwen's 528/842 and Jev's 580/842.

Latency is the median per decision, rounded to the nearest millisecond; `—` means no matching per-item timing was recorded. Qwen and CLM times are local H100 measurements, while Jev times include the remote API round trip. The JevBench Qwen latency comes from the [local Jev-format API](docs/qwen_jev_api.md), which reproduced all 231 archived Qwen predictions in the same software environment. Qwen uses the scoring protocol for each row: all option rotations on JevBench and the tool-decision tasks, one fixed order on WebPRM, and both A/B orders on PhishNChips. These deployment measurements do not isolate model inference speed.

## Decision model architectures

The [discussion](docs/decision_model_paradigms.md) compares four ways to score actions from a state: next-token letter probabilities, a jointly encoded state/action score head, CLM-style contrastive embeddings, and RAM-inspired action queries with cross-attention. Released Open-Jev is an example of the score-head approach: it independently scores each state/action pair using a Qwen3.5 language backbone with LoRA and a scalar head; see its [architecture and local benchmark protocol](docs/open_jev.md). RAM itself is an image-tagging model; using its query-decoder pattern for actions would require new training. For the exact Qwen scoring rule, see the [log-probability diagram](docs/logprob_method.md).

![Four decision-model scoring architectures](docs/assets/decision_model_paradigms.svg)

## Reproduce the saved results

```bash
git clone --filter=blob:none https://github.com/ntlm1686/Your-language-model-is-already-a-decision-model.git
cd Your-language-model-is-already-a-decision-model
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/verify_results.py
```

`verify_results.py` uses only the Python standard library to recompute counts from committed per-item records. It does not download weights. To rerun GPU inference, follow the [protocol](docs/benchmark_protocol.md) to fetch pinned upstream code, data, and weights. `setup_sources.py` fetches pinned Git commits using `--depth=1 --filter=blob:none`; large weights and raw trajectories are excluded from Git.

The [Jev API procedure](docs/jev_api.md) records how to repeat the 231-item run. `compare_jev_api.py` checks item IDs and failed requests before comparing Jev against Qwen and CLM.

The [Qwen3-8B Jev-format API](docs/qwen_jev_api.md) serves the same `Choice`/`Noul`/`Score` JSON interface locally using next-letter log probabilities. Its [JevBench API run](results/jevbench/qwen3_8b_jev_api.jsonl) scored **163/231** with **167 ms** median local HTTP latency. It agrees with the archived direct Qwen run on **231/231** individual predictions in the same PyTorch/Transformers environment. These probabilities are uncalibrated and the response identifies the model as `qwen3-8b`.

## Layout

```text
scripts/                 Pinned downloads, GPU inference, and offline checks
data/deepswe/            The 38-task list and normalized comparison inputs
results/jevbench/        Per-item Jev/Qwen/CLM outputs for 231 tasks
results/open_jev_2b/     Local Open-Jev-2B per-item outputs and length audit
results/open_jev_9b/     Local Open-Jev-9B per-item outputs and length audit
results/webprm/          Per-item Jev/Qwen/CLM outputs for 120 and 1,141 states
results/deepswe/         Jev/Qwen/CLM trajectory selections and scores
results/phishing/         Per-email Jev/Qwen/CLM outputs and summary
results/metatool/         Tool-use awareness decisions and summary
results/when2call/        Next-response decisions and summary
results/bfcl/             BFCL v4 function-name selections and summary
docs/                    Benchmark procedures, limits, and analyses
workflows/               Scripts for fetching TypeSafe's public workflow examples
```

Third-party data and weights remain with their publishers: [CLM](https://github.com/Contrastive-LM/CLM), [JevBench](https://github.com/fstandhartinger/jevbench), [WebPRMBench](https://huggingface.co/datasets/ZYao720/WEBPRMBENCH), [DeepSWE trajectories](https://huggingface.co/datasets/kaitchup/DeepSWE1.1-trajectories-Qwen3.8-27B), [PhishNChips](https://huggingface.co/datasets/AreLit/PhishNChips), [MetaTool](https://github.com/HowieHwong/MetaTool), [When2Call](https://github.com/NVIDIA/When2Call), and [BFCL](https://github.com/ShishirPatil/gorilla).

## Next experiment

To test whether a decision model helps during web execution, run Qwen, CLM, and Jev on identical candidate actions **inside complete browser tasks**. Measure action accuracy, final task success, total tokens, end-to-end latency, and cost. On the current offline WebPRM sample, Jev and Qwen tie in accuracy. DeepSWE's 38-task sample is too small to establish a general model ranking.
