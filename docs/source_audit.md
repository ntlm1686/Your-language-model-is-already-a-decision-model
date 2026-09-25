# Jev-related benchmarks with reproducible public inputs

This audit predates our fresh Jev API runs. Their methods and results are recorded in [the fresh Jev report](jev_fresh_eval.md).

Checked 2026-09-25. Repositories were shallow cloned with HTTP/1.1 and
`--filter=blob:none` to keep their public commits fixed locally.

## JevBench public subset

Repository: https://github.com/fstandhartinger/jevbench at
`26eb72d4e0e60d8ace0adfc77a384063442561cd`.

The public task files contain 48 easy, 72 original, and 111 hard labeled
decisions: **231 complete input/answer pairs**. The published per-task artifact
contains Jev 1.13.0 outcomes for those exact 231 ids. I checked that both id
sets match exactly, then recomputed the saved Jev public result: **200/231**
(easy 48/48, original 71/72, hard 81/111). This is an offline replay of
published Jev decisions, not a fresh Jev API call. Its CLI can make fresh
Jev calls with `TYPESAFE_API_KEY` and compare any model with the same tasks.
The complete current JevBench leaderboard cannot be reproduced because its
other tasks, including fresh sealed items, are private.

## Jev phishing benchmark

Repository: https://github.com/anisselbd/jev-phishing-bench at
`1d56e8c64d029a9554a0874e2ef2901ed196e230`.

`uv run prepare_data.py` completed locally, verified all four upstream SHA-256
checksums, and wrote **2,000 emails** (1,000 phishing, 1,000 legitimate) to
`data/emails.jsonl`. With the repository's `bench.heuristics.features`, I
recomputed the published no-model hosting/shortener control as **1,833/2,000
= 91.65%** (835 true positives, 2 false positives). The README reports this
rounded to 91.6%.

The full Jev-vs-Haiku experiment has code and complete labeled input data,
but new Jev and Haiku inference requires API credentials and cost. Raw model
responses are not committed, so the published model metrics cannot be
independently recalculated offline from the repository alone.

## Browser automation capability suite

https://github.com/laihenyi/pi-Jev-browser/tree/main/benchmarks provides 22
scenarios. Its `npm run benchmark` local tier is offline and tests the browser
execution layer; its model tier calls real Jev on local fixture pages and
requires a TypeSafe credential. This is closer to an interactive web agent,
but it is not an independent model-quality benchmark: most scenarios test
the combined agent and its guards, and the third-party live tier may change.

## Recommended first comparison

Use JevBench's 231 public tasks for a fixed typed-decision comparison of
CLM, Qwen logprob, and Jev; report accuracy by tier, latency, and calibration
separately. Then use the phishing set for a fully labeled, longer 2,000-case
test of calibration and cost. For the user's web-agent goal, add the local
browser suite as a separate end-to-end test rather than mixing its success
rate with JevBench accuracy.

## Fresh local Qwen3-8B run (2026-09-25)

`qwen3_8b_letters.py` scored all 231 JevBench public tasks with local
`Qwen/Qwen3-8B` revision `b968826d9c46dd6066d109eabc6255188de91218` in
BF16. The prompt includes only `state` and the typed question, not `expected`
or provenance. It disables thinking, forces `Answer: `, reads A-F next-token
logits, and cyclically rotates all options before summing candidate log
probabilities. It ran 849 prompt forwards over four H100 GPUs.

Result: **163/231** overall; easy **48/48**, original **59/72**, hard **56/111**.
By type: choice **97/139**, noul **52/74**, score **14/18**. The published
Jev result for these same public tasks is **200/231**. This is a fresh Qwen
run versus published Jev outputs, not a fresh matched-hardware Jev latency
comparison. Raw records are in `qwen3_8b_letters_results.jsonl` and the
machine-readable summary in `qwen3_8b_letters_summary.json`.

## Fresh local CLM-v0.1 run (2026-09-25)

`clm_public_eval.py` used the same public 231 tasks, CLM's official
`clm.schema.build_pairs` question rendering, the same Qwen3-8B encoder,
last-token pooling, and the released `CLM_v0.1-8B.pt` reference head. The
default CLM server cap of 2,048 tokens scored **95/231** (easy 30/48,
original 25/72, hard 40/111). This is the only generic CLM JevBench result
reported in this repository.
The default-cap run is saved in `clm_public_2048_results.jsonl` and
`clm_public_2048_summary.json`. I spot-checked three cases through the
released `clm.Engine` with the same local encoder; its predictions matched
the standalone scoring code.

These figures compare correct picks on the same tasks, not identical model
prompt strings or matched serving latency. Jev's 200/231 is a saved historical
run; Qwen and CLM results are fresh local runs. The head here is CLM's
general-purpose reference head, not its DeepSWE-specific fine-tuned head.
