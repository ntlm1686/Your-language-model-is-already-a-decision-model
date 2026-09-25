# Your Language Model Is Already a Decision Model

A language model can already be called as a decision model. Give it the state and candidate actions, then choose an action from its next-token option probabilities. **No decision-specific training is required.**

## Results

**Across the benchmarks below, Qwen3.5-9B achieves broadly comparable results to Jev with no additional fine-tuning.** We use the [released Qwen3.5-9B checkpoint](https://huggingface.co/Qwen/Qwen3.5-9B) as-is: **no fine-tuning, no LoRA, and no trained decision head in our experiments**. Actions are selected using next-token probabilities.

The stronger model depends on the task: Jev leads on JevBench, WebPRM, and When2Call; Qwen leads on phishing and MetaTool. BFCL and WebShop results are nearly identical, with WebShop success at **24.8% for Qwen versus 24.6% for Jev**.

Both models are tested on the same questions or tasks. Qwen runs on **one NVIDIA H100 80GB GPU**; Jev is accessed through its hosted API. Time measures one decision; for DeepSWE, it measures selecting a trajectory for one task. [Measurement details](docs/qwen35.md).

| Benchmark | Metric | Qwen3.5-9B | Qwen time | Jev 1.13.0 | Jev time |
| --- | --- | ---: | ---: | ---: | ---: |
| [JevBench public](docs/benchmark_protocol.md#jevbench) | Accuracy | 80.5% (186/231) | 64 ms | **86.1% (199/231)** | 336 ms |
| [WebPRMBench sample](docs/benchmark_protocol.md#webprmbench) | Accuracy | 49.2% (59/120) | 123 ms | **54.2% (65/120)** | 371 ms |
| [WebPRMBench full](docs/webprm.md) | Accuracy | 59.0% (673/1,141) | 132 ms | **66.6% (760/1,141)** | 409 ms |
| [DeepSWE](docs/benchmark_protocol.md#deepswe) | Pass rate | 50.0% (19/38) | 2,543 ms | **52.6% (20/38)** | 2,280 ms |
| [PhishNChips](docs/phishing.md) | Accuracy | **73.9% (1,478/2,000)** | 55 ms | 62.5% (1,251/2,000) | 336 ms |
| [MetaTool Task1](docs/tool_decisions.md) | Accuracy | **83.4% (867/1,040)** | 55 ms | 77.1% (802/1,040) | 370 ms |
| [When2Call](docs/tool_decisions.md) | Accuracy | 61.8% (371/600) | 139 ms | **73.3% (440/600)** | 371 ms |
| [BFCL v4 Multiple](docs/tool_decisions.md) | Accuracy | **99.5% (199/200)** | 58 ms | 99.0% (198/200) | 365 ms |
| [WebShop full](docs/webshop.md) | Success rate | **24.8% (124/500)** | 382 ms | 24.6% (123/500) | 345 ms |

Accuracy counts decisions matching the benchmark answer. DeepSWE pass rate counts tasks where the selected trajectory passed its recorded verifier. WebShop success requires full task reward. Parentheses show successful items / evaluated items.

![Qwen3.5-9B and hosted Jev 1.13.0 Doom replays side by side, showing step, kills, model input, action scores, and latency](results/doom/qwen3_5_9b_vs_jev_15s_seed100.gif)

Qwen3.5-9B (left) and Jev 1.13.0 (right), both with a **15-second budget** on seed 100. The replay follows real elapsed time and finishes **12 : 3**. Both panels show **TIMEOUT** when the 15-second evaluation window ends. If an episode ends earlier, its final score is held until the window closes. [Protocol and traces](docs/doom.md).

## Calibration

On six matched, labeled benchmarks (5,212 decisions), Qwen3.5-9B can **keep the decisions in the main table** and report the selected action's mean next-token probability as confidence. This uses the same forward passes. Its mean 10-bin ECE is **0.055**, versus **0.099** for hosted Jev; mean multiclass Brier scores are nearly equal at **0.323** and **0.322**. Qwen has **no additional fine-tuning or fitted calibration**. Using the current aggregated scores as confidence instead raises Qwen's mean ECE to **0.110**. Results vary by task: Jev has lower ECE on JevBench, and its Brier score is better on WebPRM and When2Call. The models use different prompts and inference budgets, so this compares their saved decision probabilities on matched tasks rather than isolating model architecture. [Per-benchmark results and reproducible code](docs/calibration.md).

**Qwen3.5-9B — current action choice, mean token probability as confidence**

![Qwen3.5-9B confidence versus accuracy across six benchmarks](docs/assets/calibration_qwen.svg)

**Jev 1.13.0 — API probabilities**

![Jev confidence versus accuracy across the same six benchmarks](docs/assets/calibration_jev.svg)

In both figures, the horizontal axis groups stated confidence into 10 ranges. Bar height is actual accuracy; the dashed straight line marks perfect calibration. [Download Qwen PNG](docs/assets/calibration_qwen.png) · [Download Jev PNG](docs/assets/calibration_jev.png).

![Qwen3.5-9B next-token log-probability decision rule](docs/assets/qwen_logprob.svg)

## Serve Qwen through a Jev-format API

The [Qwen3.5-9B local adapter](scripts/serve_qwen35_jev.py) accepts Jev's `POST /v1/systemone` JSON interface with `state` and typed `questions`. It returns the expected `answers` for `Choice`, `Noul`, and `Score`, while identifying the actual model as `qwen3.5-9b`. Its probabilities have no fitted calibration. Its JevBench API evaluation scored 186/231. The [Qwen3-8B adapter](scripts/serve_qwen_jev.py) uses the same schema and also matched all 231 direct predictions.

```bash
CUDA_VISIBLE_DEVICES=0 QWEN35_USE_FLA=1 models/open_jev_py311/bin/python scripts/serve_qwen35_jev.py \
  --device cuda:0 --port 8796 --max-length 8192 --fallback-margin 0
```

```bash
curl -sS http://127.0.0.1:8796/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.5-9b","state":"Customer asks for a refund.","questions":{"route":{"type":"choice","instructions":"Where should the case go?","criteria":{"returns":"Refunds and exchanges","shipping":"Delivery issues"}}}}'
```

For `Choice` requests with up to 26 options, the adapter uses rotated letter scores. Larger sets use separately scored Yes/No logits per candidate; the response metadata identifies the method. See the [API protocol](docs/qwen_jev_api.md) and [protocol tests](tests/test_qwen_jev_api.py).

## Full model comparison

Accuracy is shown below; DeepSWE uses pass rate and WebShop uses task success rate. **Qwen3.5-9B uses the released checkpoint with no additional fine-tuning, LoRA, or trained decision head.** Qwen3-8B also receives no additional training. Each benchmark uses the same questions or tasks for the models being compared.

| Benchmark (items) | [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) | [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) | [CLM](https://github.com/Contrastive-LM/CLM) | [Jev 1.13.0](https://docs.typesafe.ai/models) | [Open-Jev-2B](https://huggingface.co/ZefanCai/Open-Jev-2B) | [Open-Jev-9B](https://huggingface.co/ZefanCai/Open-Jev-9B) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| JevBench public (231) | 80.5% | 70.6% | 41.1% | **86.1%** | 64.9% | 77.5% |
| WebPRMBench sample (93) | 53.8% | 57.0% | 20.4% | **58.1%** | 46.2% | 50.5% |
| WebPRMBench full (842) | 61.6% | 62.7% | 18.1% | **68.9%** | 50.0% | 56.1% |
| DeepSWE (38) | 50.0% | 47.4% | 55.3% | 52.6% | **63.2%**† | 60.5%† |
| PhishNChips (2,000) | **73.9%** | 66.5% | 60.6% | 62.5% | 50.0% | 72.7% |
| MetaTool Task1 (1,040) | **83.4%** | 81.0% | 49.8% | 77.1% | 66.5% | 76.2% |
| When2Call (599) | 61.8% | 51.1% | 33.4% | 73.3% | 39.2% | **75.3%** |
| BFCL v4 Multiple (200) | **99.5%** | 99.0% | 84.5% | 99.0% | 98.0% | 98.5% |
| [WebShop full (500)](docs/webshop.md) | **24.8%** | — | — | 24.6% | — | 16.6% |

— means not evaluated. WebPRM and When2Call use the shared subset that fits Open-Jev's default 4K context. † DeepSWE uses the exploratory Open-Jev 8K runs; neither checkpoint could fit these inputs at 4K. CLM uses its default 2K context except for the separate DeepSWE-specific head. [Per-item results and evaluation details](docs/open_jev.md) include the original counts and context limits.

Qwen3.5-9B beats Qwen3-8B on six of the eight complete benchmark sets. Its WebPRM full-set accuracy is six decisions lower (673 versus 679 on the same 1,141 states). One fixed candidate order was tested on WebPRM, so this small difference does not establish that 8B is a better web decision model.

All 500 test tasks use the full 1,181,430-product catalog and the same search rule and budget. Qwen and Jev differ by **one successful task**. [Scores, task times, token usage, and traces](docs/webshop.md#complete-run-results).

## Closed-loop Doom experiment

Both models receive the same text observations and three actions in ViZDoom Defend the Center, with **15 seconds per episode** and seeds 100–104.

| Model | Kills, seeds 100–104 | Mean kills | Typical decision time |
| --- | --- | ---: | ---: |
| Qwen3.5-9B | 12, 10, 6, 9, 14 | **10.2** | **55 ms** |
| Jev 1.13.0 | 3, 3, 3, 3, 2 | 2.8 | 336 ms |

Qwen runs on one NVIDIA H100 80GB GPU per worker. Time includes its local API request and the remote API request for Jev. Each episode is warmed up before its clock starts. The [Doom protocol](docs/doom.md) includes per-step inputs, scores, actual episode endings, and earlier runs with the other models.

## Decision making without extra model training

We present the state and candidate actions in one prompt, stop at `Answer: `, and read Qwen3.5-9B's next-token probabilities for the option letters. We rotate the options so each action appears at every letter position, add its log probabilities across rotations, and choose the highest-scoring action. Neither Qwen3.5-9B nor Qwen3-8B receives **decision-specific fine-tuning**. The procedure generates no explanation tokens and needs no embedding model.

With N candidates, this rule uses N forward passes, each containing the state and all options. That makes it easy to inspect the per-option scores, but cost grows with context length and candidate count. The [exact prompt and scoring rule](docs/logprob_method.md), [Qwen3.5-9B runner](scripts/eval_qwen35_letters.py), and [reproduction notes](docs/qwen35.md) are available.

## Possible decision model architectures

**Decision model architectures mainly differ in how state and action are connected and interact within the network.**

Three designs can score a candidate action given a state. Qwen's next-token method puts the state and every action in one prompt and reads restricted token probabilities, with no new weights. CLM independently embeds states and actions with released contrastive heads, allowing action embeddings to be cached. A proposed RAM-inspired design would feed action tokens into a Transformer. State-encoder features enter its cross-attention layers as keys and values, while the action tokens supply the queries. The Transformer updates the action tokens and outputs a score for each action; it would need training and has **not** been evaluated here. The [architecture note](docs/decision_model_paradigms.md) expands the comparison.

![Three decision-model scoring architectures](docs/assets/decision_model_paradigms.svg)

The interface avoids collecting decision labels and training a head. Specialized models can still do better: Jev leads on JevBench and eligible WebPRM states, while Open-Jev-9B leads on the selected When2Call subset. Qwen3.5-9B leads on MetaTool, phishing, and BFCL. Rotating N options requires scoring N prompt orders; the server batches them into one forward. CLM can reuse action embeddings. None of these scores alone establishes the best architecture for a browser agent.

The restricted-letter softmax is a distribution over the supplied options, **not a calibrated probability that an action is correct**. The released Qwen, CLM, and Open-Jev paths tested in the main table consume text. The state/action formulation could accept images if the encoders, input pipeline, and alignment data support them; the pixel-input Doom run is a separate vision-model experiment, not evidence that these text paths already process images. WebShop adds simulated shopping task success, task time, and token usage; real-site browser workflows remain untested.

## Reproduce the saved results

```bash
git clone --filter=blob:none https://github.com/ntlm1686/Your-language-model-is-already-a-decision-model.git
cd Your-language-model-is-already-a-decision-model
python scripts/verify_results.py
python scripts/summarize_open_jev.py
python scripts/summarize_qwen35.py
python scripts/summarize_qwen35_optimized.py
python scripts/verify_doom_timed.py
python scripts/summarize_webshop.py --require-complete
python scripts/eval_calibration.py
```

These standard-library scripts recount the committed records without downloading weights. GPU and API rerun procedures are in the [benchmark protocol](docs/benchmark_protocol.md), [Qwen Jev-format API notes](docs/qwen_jev_api.md), [hosted Jev procedure](docs/jev_api.md), [Open-Jev protocol](docs/open_jev.md), and [Doom protocol](docs/doom.md). `scripts/setup_sources.py` fetches pinned upstream revisions; large weights and raw trajectories are excluded from Git.

## Layout

```text
scripts/                 Pinned downloads, GPU inference, and offline checks
data/deepswe/            The 38-task list and normalized comparison inputs
results/jevbench/        Per-item Jev/Qwen/CLM outputs for 231 tasks
results/open_jev_2b/     Local Open-Jev-2B per-item outputs and length audit
results/open_jev_9b/     Local Open-Jev-9B per-item outputs and length audit
results/qwen3_5_9b/      Original scores and API checks; optimized/ holds the main results
results/webprm/          Per-item Jev/Qwen/CLM outputs for 120 and 1,141 states
results/deepswe/         Jev/Qwen/CLM trajectory selections and scores
results/phishing/         Per-email Jev/Qwen/CLM outputs and summary
results/metatool/         Tool-use awareness decisions and summary
results/when2call/        Next-response decisions and summary
results/bfcl/             BFCL v4 function-name selections and summary
results/doom/             Five-seed game summaries, decision traces, and GIF replays
results/webshop/          Complete 500-task episodes for three models, prompts, scores, and timings
results/calibration/      Per-decision probabilities and six-benchmark calibration summary
docs/                    Benchmark procedures, limits, and analyses
workflows/               Scripts for fetching TypeSafe's public workflow examples
```

## References

Model releases, benchmark sources, and methods used in this repository. Reported scores come from our runs; exact revisions and evaluation settings are recorded in the reproduction notes.

1. **Qwen models.** [Qwen3.5-9B model card](https://huggingface.co/Qwen/Qwen3.5-9B) and [Qwen3-8B model card](https://huggingface.co/Qwen/Qwen3-8B). Released language-model checkpoints used without additional fine-tuning.
2. **TypeSafe AI, Jev.** [Official model documentation](https://docs.typesafe.ai/models). Hosted Jev 1.13.0, accessed through the typed decision API.
3. **Open-Jev.** [Official implementation](https://github.com/Zefan-Cai/Open-Jev), [Open-Jev-2B](https://huggingface.co/ZefanCai/Open-Jev-2B), and [Open-Jev-9B](https://huggingface.co/ZefanCai/Open-Jev-9B). Released decision-model adapters and heads.
4. **CLM.** [Official repository](https://github.com/Contrastive-LM/CLM). Contrastive state/action scoring and the released evaluation code.
5. **JevBench.** [Benchmark repository](https://github.com/fstandhartinger/jevbench). Source of the 231 public decision tasks.
6. **WebPRMBench.** [WEBPRMBENCH dataset](https://huggingface.co/datasets/ZYao720/WEBPRMBENCH). Source of the web action-selection inputs; our state regrouping and subsets are described in the [protocol](docs/webprm.md).
7. **DeepSWE trajectories.** [DeepSWE1.1-trajectories-Qwen3.8-27B](https://huggingface.co/datasets/kaitchup/DeepSWE1.1-trajectories-Qwen3.8-27B). Source trajectories and recorded verifier outcomes for our 38-task selection experiment.
8. **PhishNChips.** [Dataset](https://huggingface.co/datasets/AreLit/PhishNChips) and [Jev phishing benchmark preparation](https://github.com/anisselbd/jev-phishing-bench). Source of the 2,000-email comparison.
9. **MetaTool.** [MetaTool Benchmark for Large Language Models: Deciding Whether to Use Tools and Which to Use](https://github.com/HowieHwong/MetaTool). We evaluate Task1, tool-use awareness.
10. **NVIDIA, When2Call.** [Official dataset and evaluation repository](https://github.com/NVIDIA/When2Call). Decisions about when to call a function, answer, or ask for clarification.
11. **Berkeley Function Calling Leaderboard (BFCL).** [Official Gorilla/BFCL repository](https://github.com/ShishirPatil/gorilla). We evaluate the v4 Multiple function-selection subset.
12. **Yao, Chen, Yang, and Narasimhan (2022).** [WebShop: Towards Scalable Real-World Web Interaction with Grounded Language Agents](https://arxiv.org/abs/2207.01206). [Official environment and code](https://github.com/princeton-nlp/WebShop). We use all 500 test tasks and the full product catalog, with the shared search rule documented in our [protocol](docs/webshop.md).
13. **ViZDoom.** [Official environment](https://github.com/Farama-Foundation/ViZDoom) and [Defend the Center scenario](https://vizdoom.farama.org/main/environments/default/). Environment for the timed Doom comparison; our setup also follows the [AlexWortega/openjev example](https://huggingface.co/AlexWortega/openjev).
14. **Recognize Anything (RAM).** [Recognize Anything: A Strong Image Tagging Model](https://github.com/xinyu1205/recognize-anything). Inspiration for the proposed action-token Transformer with state features entering through cross-attention; this proposal is not an evaluated model here.
