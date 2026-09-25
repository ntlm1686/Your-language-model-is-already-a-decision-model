# Qwen3-8B letter logits vs. CLM on WebPRMBench

Fresh Jev 1.13.0 results on the same 120 states are documented in [the Jev API report](jev_fresh_eval.md).

## Expanded eligible test set

The complete test file has 1,150 states. Nine exceed the 8,192-token Qwen prompt cap, leaving 1,141 states for the expanded paired comparison. Candidate order is randomized once under seed 42. Its randomization stream differs from the 120-state pilot after the first source split, so these are separate runs. Qwen and CLM run locally; Jev 1.13.0 receives the same state and five candidate descriptions through its API. CLM uses its default 2,048-token cap.

| Source | N | Jev | Qwen3-8B | CLM 2K |
| --- | ---: | ---: | ---: | ---: |
| AssistantBench | 30 | 20 | 17 | 1 |
| Mind2Web cross-domain | 416 | 270 | 240 | 80 |
| Mind2Web cross-task | 141 | 102 | 83 | 25 |
| Mind2Web cross-website | 146 | 90 | 87 | 29 |
| WebArena | 201 | 148 | 132 | 35 |
| WorkArena | 207 | 130 | 120 | 39 |
| **Total** | **1,141** | **760 (66.6%)** | **679 (59.5%)** | **209 (18.3%)** |

Five-way random choice has an expected 228.2 correct. Jev beats Qwen on this fixed-order protocol by 81 cases. In the paired outcomes, Jev alone is correct on 171 states and Qwen alone on 90. This is stronger evidence than the 120-state pilot, where both scored 65, but option-order sensitivity remains unmeasured. The generic CLM head scores below random expectation on this task; it was not trained specifically for WebPRM.

Median measured time per state is 0.409 s for Jev's remote API, 0.153 s for Qwen's local forward, and 0.293 s for CLM's six local forwards. Hardware and network differ, and the CLM server could cache candidate embeddings, so these are method-specific measurements rather than a direct deployment speed ranking. Jev reported 4,362,678 input tokens.

The [per-item Qwen/CLM scores](../results/webprm/qwen_clm_1141_clm2k.jsonl), [Jev answers](../results/webprm/jev_1141.jsonl), and [summary](../results/webprm/summary_1141.json) are committed. The [protocol](benchmark_protocol.md#expanded-1141-state-run) gives the commands to reproduce them.

## Setup

- Data: `ZYao720/WEBPRMBENCH`, downloaded as `data/test-00000-of-00001.jsonl`.
  The 4,600 pairwise rows were regrouped into 1,150 states, each with one
  environment-verified preferred response and four rejected responses.
- Sample: fixed seed 42, 20 states from each of six source splits, 120 total.
  States whose Qwen prompt exceeded 8,192 tokens were skipped before sampling.
- Model: `Qwen/Qwen3-8B`, revision
  `b968826d9c46dd6066d109eabc6255188de91218`, bfloat16 on one H100.
- Qwen scoring: chat template with thinking disabled, options shuffled and
  labelled A-E, forced assistant prefix `Answer: `. Softmax was applied to the
  next-token logits of the five one-token letters. No answer was generated.
- CLM scoring: same Qwen3-8B last-token hidden states, L2 normalization,
  released `CLM_v0.1-8B.pt` state/action projection heads, scaled cosine and
  softmax. The state includes the question instruction after the page context;
  state and candidate embeddings use CLM's default 2,048-token cap. This is a manual local implementation of
  the published scoring path, not the vLLM server.

## Results

| Source | Qwen letter logits | CLM head | Total |
| --- | ---: | ---: | ---: |
| AssistantBench | 9 | 1 | 20 |
| Mind2Web cross-domain | 7 | 1 | 20 |
| Mind2Web cross-task | 13 | 5 | 20 |
| Mind2Web cross-website | 10 | 4 | 20 |
| WebArena | 15 | 10 | 20 |
| WorkArena | 11 | 3 | 20 |
| **Total** | **65/120 (54.2%)** | **24/120 (20.0%)** | **120** |

Five-way random choice would average 20%. The two methods agreed on 12 correct
and 43 incorrect states. Qwen alone was correct on 53 states; CLM alone on 12.

Qwen prompt lengths were 958-7,107 tokens, median 2,791; 379,198 input tokens
in all. On this unbatched Transformers implementation, model-forward time
summed to 20.75 s for Qwen and 31.30 s for CLM; model loading, preprocessing,
and output writing are excluded. CLM required one state plus five action
forwards per case; the released server can cache action embeddings, so these
times are not a serving-speed comparison.

Qwen's conditional probability is not calibrated confidence: among 46 cases
where its maximum A-E probability was at least 0.99, 35 were correct (76.1%).

## Checks and limits

- The manual CLM path reproduced the README tides example: it assigned the
  correct candidate probability 0.993, compared with 0.997 shown in the README.
- This is offline next-action selection from supplied candidates. It does not
  measure full browser-task success, recovery, or total operational cost.
- The CLM reference head was not fine-tuned on WebPRMBench; the five-way task
  was reconstructed from the dataset's original pairwise format. These results
  do not establish a general ranking of models across web benchmarks.
- One candidate-letter ordering was tested. A stronger study should repeat
  with several permutations and prompt formats, then evaluate complete tasks.

## Reproduce

```bash
python scripts/compare_qwen_clm_webprm.py \
  --per-source 20 --clm-max-length 2048 \
  --output results/webprm/qwen_clm_120_clm2k.jsonl
```
