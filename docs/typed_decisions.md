# Typed Decisions: full test split

We ran the [LocalLLaMA/typed-decisions benchmark](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) on 2026-09-29, using its complete **400-case test split**: 100 cases in each of four workflows, with five `choice`, `noul`, or `score` questions per case (**2,000 decisions**). Both models receive each case's full `state`, question instructions, candidate keys, and criteria descriptions. We did not train or tune either model on this dataset. Qwen3.5-9B uses its released checkpoint; hosted Jev resolves to `jev-1.13.0`.

The dataset's gold distribution is the average of three teacher-model samples. Agreement with it is the benchmark's target, not independent proof that a decision is correct. We use the pinned dataset revision `f7a2487edd7a043a5441a5e9ccc7fe5ddbd9ebe8`; the downloaded test Parquet has SHA-256 `4f294f218ea1da27f3efef936359389c62ea4d3973a41457732990f1d31b647c`.

## Complete paired results

| Model and probability rule | Accuracy ↑ | Soft-label Brier ↓ | TV ↓ | KL ↓ | ECE ↓ | Typical case time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-9B, current aggregated scores | 62.1% (1,242/2,000) | 0.370 | 0.405 | 1.616 | 0.247 | 316 ms |
| Qwen3.5-9B, same decisions with mean token probabilities | 62.1% (1,242/2,000) | 0.168 | 0.275 | **0.292** | 0.067 | 316 ms |
| Jev 1.13.0, hosted API | **73.9% (1,478/2,000)** | **0.148** | **0.250** | 1.525 | **0.032** | **196 ms** |

Jev wins on accuracy, Brier, total variation, and ECE. Qwen's mean token probabilities have lower KL under our clipping rule: Jev often returns zero for labels the soft teacher gives some probability. Both Qwen rows use **the same action choices and GPU forward passes**; the second row reports the selected action's mean probability across option rotations. If that mean probability also chooses the action, Qwen gets 61.4% accuracy with Brier 0.168.

A paired bootstrap over the 400 cases (1,000 resamples, seed 42) gives a 95% interval of **+9.7 to +14.0 percentage points** for Jev's accuracy advantage and **−0.032 to −0.007** for Jev minus Qwen Brier. These intervals describe this test set's case variation, not performance on a new workflow.

Accuracy is against the dataset's discrete label. Brier is the mean sum over options of `(model probability - soft gold probability)²`; TV is half the absolute-distance sum. KL is `KL(gold || model)` with predicted probabilities clipped at `1e-12`. ECE uses ten equal-width bins of selected-action probability versus discrete-label accuracy. We renormalize Jev's rounded probability vectors before scoring. The same code computes every model's metrics.

As a scoring check, our uniform-probability Brier is **0.238**, matching the dataset card's uniform reference.

| Question type | Decisions | Qwen accuracy | Jev accuracy | Qwen Brier | Jev Brier |
| --- | ---: | ---: | ---: | ---: | ---: |
| Choice | 600 | 61.5% | **73.7%** | 0.146 | **0.125** |
| Noul | 600 | 71.8% | **79.2%** | 0.174 | **0.070** |
| Score | 800 | 55.2% | **70.1%** | **0.179** | 0.224 |

Jev is more accurate for all three question types. Qwen's softer score distributions give it a lower Brier score on the ordered `score` questions despite lower accuracy. [Machine-readable results](../results/typed_decisions/summary.json) include each workflow's metrics.

The dataset card reports an earlier Jev 1.13.0 run at 72.7% accuracy, 0.148 Brier, and 710 ms per case. Our fresh run is 73.9%, 0.148, and 196 ms. The two runs use the same named model and test split but different dates and client/network conditions. Our ECE and KL have explicit definitions above; they should only be compared with the Qwen numbers computed here, not directly with the card's ECE and KL values.

Timing is a median per **five-question case**, after model loading. Qwen executes locally on one H100 80GB GPU per worker; we used four independent workers on four GPUs to finish the test split. Qwen's 316 ms includes local prompt construction and model inference but no HTTP. Jev's 196 ms includes the remote HTTP round trip. Qwen scores 7,100 rotated prompts across the test set in 2,000 batched GPU forwards; Jev answers five questions in each of 400 API requests. These are different latency and token-consumption boundaries. Jev reported 378,236 input tokens; Qwen processed 2,460,688 input tokens across its rotated prompts.

## Reproduce

```bash
uv run --no-project --with pyarrow python scripts/setup_typed_decisions.py

for gpu in 0 1 2 3; do
  QWEN35_USE_FLA=1 CUDA_VISIBLE_DEVICES=$gpu models/open_jev_py311/bin/python \
    scripts/eval_typed_decisions.py qwen --device cuda:0 --shard "$gpu" --nshards 4 &
done
wait

read -rs TYPESAFE_API_KEY
export TYPESAFE_API_KEY
python scripts/eval_typed_decisions.py jev
python scripts/summarize_typed_decisions.py
```

The [dataset setup script](../scripts/setup_typed_decisions.py) verifies the source hash and stores the downloaded data under ignored `private/`. The [runner](../scripts/eval_typed_decisions.py) resumes completed cases, checks that Qwen's optimized FLA kernel is active, and saves only case IDs, input hashes, probabilities, usage, and timings to `results/typed_decisions/`. API keys and the raw states are never written to the committed results. The [summarizer](../scripts/summarize_typed_decisions.py) checks all 400 paired case IDs and request hashes before scoring. The original Qwen3.5-9B setup is documented in [Qwen serving notes](qwen35.md).
