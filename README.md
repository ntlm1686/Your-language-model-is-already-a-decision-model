# Your Language Model Is Already a Decision Model

Research question: given fixed candidate actions or trajectories, does a specialized decision model improve accuracy, latency, or context use over direct option log probabilities from a general language model?

This repository contains experiments completed on **2026-09-25**, per-item outputs, and reproducible code. Qwen3-8B is scored using `Answer: X` next-token probabilities; CLM uses released heads; Jev 1.13.0 was queried through TypeSafe's API. **No complete browser-agent evaluation has been performed yet.**

## Results

| Benchmark | Metric | Qwen3-8B letter log probabilities | CLM | Fresh Jev 1.13.0 | Reference |
| --- | --- | ---: | ---: | ---: | ---: |
| [JevBench public 231](docs/benchmark_protocol.md#jevbench) | Correct typed decisions | 163/231 | 95/231 | **199/231** | Historical Jev: 200/231 |
| [WebPRMBench sample of 120 states](docs/benchmark_protocol.md#webprmbench) | Correct choice among five next actions | **65/120** | 24/120 | **65/120** | Random: 24/120 |
| [WebPRMBench full eligible 1,141 states](docs/webprm.md) | Correct choice among five next actions | 679/1,141 | 209/1,141 | **760/1,141** | Random: 228/1,141 |
| [DeepSWE, 38 tasks with identical candidates](docs/benchmark_protocol.md#deepswe) | Successful trajectory selected from four | 18/38 | 21/38 | 20/38 | Best fixed agent: 22/38 |
| [PhishNChips, 2,000 emails](docs/phishing.md) | Correct phishing/legitimate decisions | **1,330/2,000** | 1,212/2,000 | 1,251/2,000 | URL rule: **1,833/2,000** |
| [MetaTool Task1, 1,040 queries](docs/tool_decisions.md) | Correct use-tool decision | **842/1,040** | 518/1,040 | 802/1,040 | Balanced chance: 520/1,040 |
| [When2Call, 600-item balanced sample](docs/tool_decisions.md) | Correct next response | 307/600 | 201/600 | **440/600** | Four-way random: 150/600 |
| [BFCL v4 Multiple, 200 items](docs/tool_decisions.md) | Correct function name selected | **198/200** | 169/200 | **198/200** | Random expectation: 77/200 |

Generic CLM uses its default 2K context. DeepSWE uses a separate task-specific head; the table shows its final-12-history result. See the [protocol](docs/benchmark_protocol.md) for full methods and limits. WebPRM is offline action selection, and the phishing set is largely separable by a URL rule.

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

## Layout

```text
scripts/                 Pinned downloads, GPU inference, and offline checks
data/deepswe/            The 38-task list and normalized comparison inputs
results/jevbench/        Per-item Jev/Qwen/CLM outputs for 231 tasks
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
