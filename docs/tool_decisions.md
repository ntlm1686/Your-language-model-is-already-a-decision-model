# Tool decision benchmarks

These experiments provide fixed candidate decisions to all three systems. They test decision selection, not free-form tool calls or completed agent tasks. The [protocol](benchmark_protocol.md#metatool-tool-use-awareness) fixes source commits, prompts, and commands. The table cells below are correct decisions over the evaluated items.

## MetaTool Task1: use a tool?

The public Task1 file contains 520 positive and 520 negative user queries. Only the query enters the state. Qwen3-8B scores both A/B option orders; Jev gets one typed choice question; the generic CLM head uses its default 2K cap.

| Label | N | Qwen | Jev | CLM 2K |
| --- | ---: | ---: | ---: | ---: |
| Use a tool | 520 | 373 | 290 | 175 |
| Do not use a tool | 520 | 469 | 512 | 343 |
| **Total** | **1,040** | **842 (81.0%)** | **802 (77.1%)** | **518 (49.8%)** |

Jev accepts almost all no-tool cases, but misses 230 of 520 cases that do need a tool. Qwen handles more positive cases. The 1,040 inputs are balanced, so a fixed all-yes or all-no rule gets 520 correct. No CLM state was truncated.

The [per-item GPU outputs](../results/metatool/qwen_clm_clm2k.jsonl), [Jev outputs](../results/metatool/jev.jsonl), and [validated summary](../results/metatool/summary.json) are saved.

## When2Call: select the next response

We sampled 200 eligible items from each correct category in the official MCQ test, for 600 total. Each question supplies four concrete candidate responses: a direct answer, a tool call, a request for missing information, and an inability statement. The official test has **no item with `direct` as the correct category**; its presence as a candidate is a dataset limitation. Qwen sums A-D log probabilities across all four cyclic option positions.

| Correct category | N | Jev | Qwen | CLM 2K |
| --- | ---: | ---: | ---: | ---: |
| Cannot answer | 200 | 102 | 58 | 25 |
| Request more information | 200 | 155 | 54 | 17 |
| Call a tool | 200 | 183 | 195 | 159 |
| **Total** | **600** | **440 (73.3%)** | **307 (51.2%)** | **201 (33.5%)** |

Jev beats Qwen on 160 paired items where Qwen is wrong; Qwen beats Jev on 27. Qwen identifies tool calls well but rarely selects the other two correct behaviors. A fixed always-call rule gets 200/600; uniform choice among the four responses has an expected 150/600. CLM's 2K state cap truncated 14 of the 600 tool contexts.

The [per-item GPU outputs](../results/when2call/qwen_clm_clm2k.jsonl), [Jev outputs](../results/when2call/jev.jsonl), and [validated summary](../results/when2call/summary.json) are saved. This balanced 600-item selection is not the official full-test score, and the experiment does not test generated tool arguments or execution.

## BFCL v4 Multiple: select a function name

The 200 official multiple-function rows each provide two to four candidate functions and one labeled correct function name. We give the same name, description, and parameter schema to every method. Qwen rotates the A-D positions as needed; Jev gets one choice question; generic CLM uses its default 2K cap.

| Candidates | N | Jev | Qwen | CLM 2K |
| --- | ---: | ---: | ---: | ---: |
| Two functions | 79 | 77 | 77 | 68 |
| Three functions | 85 | 85 | 85 | 65 |
| Four functions | 36 | 36 | 36 | 36 |
| **Total** | **200** | **198 (99.0%)** | **198 (99.0%)** | **169 (84.5%)** |

Jev and Qwen are correct on 197 of the same 200 items, with one unique success each. This name-only subtask is nearly saturated for both and provides little evidence for choosing one over the other. The experiment omits argument generation, execution, and BFCL's official overall scoring. No CLM state was truncated.

The [per-item GPU outputs](../results/bfcl/qwen_clm_clm2k.jsonl), [Jev outputs](../results/bfcl/jev.jsonl), and [validated summary](../results/bfcl/summary.json) are saved. The [protocol](benchmark_protocol.md#bfcl-v4-multiple-function-selection) identifies the pinned source and checksums.
