# Benchmark protocol and comparison limits

This document records the local evaluation protocol used on 2026-09-25. Recompute all reported local counts from committed per-item records with:

```bash
python scripts/verify_results.py
```

## Common environment

- Backbone: [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B), revision `b968826d9c46dd6066d109eabc6255188de91218`, BF16, Transformers `4.57.6`, PyTorch `2.12.0`.
- CLM source: [Contrastive-LM/CLM](https://github.com/Contrastive-LM/CLM), commit `d43894a1c97c83d2bb1b5cfcaac5759f72ece8a6`.
- Generic CLM head: [CLM-v0.1-8B](https://huggingface.co/Contrastive-LM/CLM-v0.1-8B), revision `e939398d4556fcd9400c76fa8c5a513202f42b0a`.
- DeepSWE-specific head: [deepswe-clm-heads-8k](https://huggingface.co/Contrastive-LM/deepswe-clm-heads-8k), revision `c60876f3fdf7a75dc58d33b776e469b7e903d0ee`.
- The original runs split items by index across four H100 GPUs, with one visible GPU per process. Inference used individual Transformers forwards, without server batching or optimized candidate embedding caches.

Install dependencies and download pinned sources:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/setup_sources.py repos heads model webprm deepswe
```

The `model` component downloads approximately 16 GB. Checking committed results needs no download. The `deepswe` component fetches only the 38 selected tasks across four runs: 304 trajectory and result files. Existing copies can be supplied with `MODEL_PATH`, `CLM_REPO`, `CLM_REFERENCE_HEAD`, and `CLM_DEEPSWE_HEAD`.

```bash
python scripts/audit_published_jev.py
```

This command needs only `setup_sources.py repos`; it recounts Jev's historical 200/231 from JevBench's published per-item records.

### Sharding

Each command using `--shard 0 --nshards 1` runs all items on one GPU. For four GPUs, launch shards 0 through 3 independently, setting each process's `CUDA_VISIBLE_DEVICES` to a different physical GPU. Each process addresses its visible GPU as `cuda:0`. `merge_shards.py` checks counts and duplicate IDs.

## JevBench

Source: [JevBench](https://github.com/fstandhartinger/jevbench), commit `26eb72d4e0e60d8ace0adfc77a384063442561cd`, `datasets/public/`: 48 easy, 72 original, and 111 hard items, or 231 total. Sealed tasks are excluded.

Qwen receives each item's `state`, question instructions, and candidate descriptions. Thinking is disabled. After the forced `Answer: ` prefix, the script reads A–F next-token logits. Cyclic rotations put each candidate in every letter position; log probabilities are summed by candidate. Ground-truth labels, `expected`, and provenance never enter the prompt.

CLM uses the official `clm.schema.build_pairs` state and candidate rendering. The same Qwen backbone provides last-token hidden states, projected by the released generic state/action heads before cosine scoring. The reported comparison uses CLM's **default 2,048-token context**; state text keeps its tail and candidate embeddings are cached. This is a local Transformers implementation. Predictions matched the official `clm.Engine` on three spot-checked items.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/qwen3_8b_letters.py --shard 0 --nshards 1
python scripts/merge_shards.py --glob 'results/jevbench/qwen3_8b_letters_shard*.jsonl' --output results/jevbench/qwen3_8b_letters_results.jsonl --key id --expected 231
python scripts/summarize_run.py results/jevbench/qwen3_8b_letters_results.jsonl --benchmark jevbench

CUDA_VISIBLE_DEVICES=0 python scripts/clm_public_eval.py --shard 0 --nshards 1 --max-tokens 2048
python scripts/merge_shards.py --glob 'results/jevbench/clm_public_2048_shard*.jsonl' --output results/jevbench/clm_public_2048_results.jsonl --key id --expected 231
python scripts/summarize_run.py results/jevbench/clm_public_2048_results.jsonl --benchmark jevbench
```

Results: Qwen 163/231; CLM at its default 2K context 95/231. Fresh Jev 1.13.0 scored 199/231 via the [API procedure](jev_api.md). The authors' saved historical Jev result was 200/231; see the [fresh Jev report](jev_fresh_eval.md) and [source audit](source_audit.md).

## WebPRMBench

Source: [ZYao720/WEBPRMBENCH](https://huggingface.co/datasets/ZYao720/WEBPRMBENCH), `test` JSONL at revision `badefe7a5372e73b0736186b598e299292d4dd59`. We regrouped 4,600 pairwise rows into 1,150 web states, each with one environment-verified preferred response and four rejected responses. With seed 42, we sampled 20 eligible states from each of six source splits, for 120 states total. Eligibility required a Qwen prompt of at most 8,192 tokens.

Qwen reads next-token logits for five options A–E. The generic CLM head scores the same state and candidates individually using its **default 2,048-token cap**. The Qwen prompt still has an 8,192-token eligibility cap, so the sample is unchanged. Candidate order is randomized once rather than fully rotated as in JevBench. This is an **exploratory offline next-action test**; order sensitivity remains to be measured.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/compare_qwen_clm_webprm.py --per-source 20 \
  --clm-max-length 2048 --output results/webprm/qwen_clm_120_clm2k.jsonl
```

Qwen and fresh Jev each scored 65/120; default-2K CLM scored 24/120. These are not browser task completion rates. See the [WebPRM analysis](webprm.md) and [fresh Jev report](jev_fresh_eval.md).

### Expanded 1,141-state run

The full test file has 1,150 states; nine exceed the 8,192-token Qwen prompt cap. The expanded run includes all 1,141 eligible states. It uses the same five candidate responses per state, with one fixed random ordering under seed 42. Because random draws continue past the first 20 states of each source split, the candidate order in this run is independent of the earlier 120-state pilot. The generic CLM head remains at its default 2,048-token cap.

```bash
python scripts/compare_qwen_clm_webprm.py --per-source 999999 \
  --clm-max-length 2048 --output results/webprm/qwen_clm_1141_clm2k.jsonl
python scripts/run_jev_webprm.py \
  --per-source 999999 --limit 1141 \
  --reference results/webprm/qwen_clm_1141_clm2k.jsonl \
  --output results/webprm/jev_1141.jsonl
python scripts/summarize_webprm_full.py --output results/webprm/summary_1141.json
```

The API key is supplied through the environment and is never saved in the repository. The Jev runner reconstructs state IDs and candidate order and checks them against the GPU records before calling the API.

## PhishNChips email decision

Source: [jev-phishing-bench](https://github.com/anisselbd/jev-phishing-bench), commit `1d56e8c64d029a9554a0874e2ef2901ed196e230`, which prepares 2,000 emails from [PhishNChips v5.2](https://huggingface.co/datasets/AreLit/PhishNChips). Its preparation script verifies the four source files against published SHA-256 hashes. The resulting `emails.jsonl` has SHA-256 `5158f333b9f813dd93ee0ead8bf26e9798679f048285acba3d20d9ecca12a7ba`.

All three models receive the six-field email object, the same phishing-versus-legitimate question, and the same two option descriptions. Only `email` enters the model input; `y`, URL category, source, and strategy are excluded. Jev gets one `choice` question per email; the upstream author's run bundled nine questions into each call, so our Jev result is a new, controlled comparison rather than a rerun of their published 62.6%. Qwen3-8B uses A/B next-token probabilities with a deterministic, roughly balanced label order. A second run reverses A/B and sums the two log probabilities for each semantic option; the rotated score is the main Qwen readout. The generic CLM head scores the question and email against the two descriptions with its default 2,048-token cap. None of the 2,000 CLM states reaches that cap.

```bash
python scripts/setup_sources.py phishing
python scripts/run_phishing_comparison.py gpu \
  --output results/phishing/qwen_clm_2000_clm2k.jsonl
python scripts/rotate_phishing_qwen.py
python scripts/run_phishing_comparison.py jev \
  --output results/phishing/jev_2000.jsonl
python scripts/summarize_phishing.py --output results/phishing/summary.json
```

The simple URL-host rule from the pinned upstream repository scores 1,833/2,000 (91.65%). It uses a list of common URL shorteners and free-hosting services. The dataset is largely separable by URL construction; this rule is an essential control for interpreting model accuracy. Emails are synthetic, and URL reputation rather than human review supplies the labels. The experiment measures offline classification, not actual agent clicks or account compromise.

## MetaTool tool-use awareness

Source: [MetaTool](https://github.com/HowieHwong/MetaTool), commit `35e81bb7576826e980c80fed8f8c0a2b4a1e6fbb`, `dataset/tmp_dataset/Task1.json`. It has 1,040 distinct user queries, 520 labeled positive (use a tool) and 520 negative. All are evaluated. The model state contains only the user query. The two options explain when to use an external tool versus answering directly. Qwen sums A/B log probabilities over both option orders; Jev receives one typed choice question; CLM scores the same state and two option descriptions under its default 2K cap. This is a decision-only reformulation of Task1, not the upstream free-form explanation scoring protocol.

```bash
python scripts/setup_sources.py metatool
python scripts/run_tool_decisions.py metatool gpu --output results/metatool/qwen_clm_clm2k.jsonl
python scripts/run_tool_decisions.py metatool jev --output results/metatool/jev.jsonl
python scripts/summarize_tool_decisions.py metatool
```

The Jev command requires `TYPESAFE_API_KEY` in the environment. The key is not written to results.

## When2Call next-response decision

Source: [NVIDIA/When2Call](https://github.com/NVIDIA/When2Call), commit `ecc8d42388e91ab37e7e737d48e16e8ecea3d1dc`, `data/test/when2call_test_mcq.jsonl`. The official MCQ test has 3,652 items and four answer candidates per item. Its `correct_answer` field contains only three classes: `tool_call` (1,295), `request_for_info` (1,062), and `cannot_answer` (1,295). `direct` is present as a candidate but never as the correct answer. We sample 200 Qwen-eligible items from each labeled class with seed 42; eligibility requires an 8,192-token Qwen prompt. We report this as a balanced **600-item sample**, not the official full-test score.

All methods see the user question, the supplied tool definitions, and the same four concrete candidate responses. Qwen sums log probabilities over four cyclic A-D option rotations. Jev receives one typed choice question with the four responses as criteria. The generic CLM head uses its default 2K state cap; the report counts how often it truncates. This measures choosing among supplied responses. It does not measure generating tool-call arguments, executing tools, or the official When2Call length-normalized metric.

```bash
python scripts/setup_sources.py when2call
python scripts/run_tool_decisions.py when2call gpu --output results/when2call/qwen_clm_clm2k.jsonl
python scripts/run_tool_decisions.py when2call jev --output results/when2call/jev.jsonl
python scripts/summarize_tool_decisions.py when2call
```

## BFCL v4 multiple-function selection

Source: [Berkeley Function Calling Leaderboard](https://github.com/ShishirPatil/gorilla/tree/6ea57973c7a6097fd7c5915698c54c17c5b1b6c8/berkeley-function-call-leaderboard/bfcl_eval/data), commit `6ea57973c7a6097fd7c5915698c54c17c5b1b6c8`. We fetch `BFCL_v4_multiple.json` and its `possible_answer` file by pinned raw GitHub URLs. SHA-256 values are embedded in `setup_sources.py` and checked on download. All 200 rows have exactly one ground-truth function name present among the 2–4 supplied functions: 79 rows have two candidates, 85 have three, and 36 have four.

All models see the user request and the same candidate names, descriptions, and parameter schemas. Qwen sums letter log probabilities over all cyclic candidate rotations. Jev receives one typed choice question, and CLM uses the generic head at its default 2K cap. The scoring checks **function name selection only**; it does not evaluate generated arguments or the official BFCL overall function-calling score.

```bash
python scripts/setup_sources.py bfcl
python scripts/run_tool_decisions.py bfcl gpu --output results/bfcl/qwen_clm_clm2k.jsonl
python scripts/run_tool_decisions.py bfcl jev --output results/bfcl/jev.jsonl
python scripts/summarize_tool_decisions.py bfcl
```

## DeepSWE

Candidates come from [kaitchup/DeepSWE1.1-trajectories-Qwen3.8-27B](https://huggingface.co/datasets/kaitchup/DeepSWE1.1-trajectories-Qwen3.8-27B), revision `e58b83310346104971fd17c42f9955e7cd5e3b25`. The 38 task names follow CLM's published held-out list. Each task has four **new public trajectories** from different agent configurations. They are not the paper's Opus 5 trajectories, so the paper's 31/38 cannot be compared directly.

`data/deepswe/fair_inputs.jsonl` stores the exact normalized comparison inputs: task instructions, each candidate's final 12 action/observation turns, candidate orders, and verifier outcomes. Labels are used for evaluation only. Qwen reads four anonymous trajectories in one prompt and sums A–D log probabilities over four cyclic rotations: 152 forwards and 2,355,652 input tokens. The DeepSWE-specific CLM head scores each of the final 12 turns, then selects the trajectory with the highest mean. `fair_scores_shard*.jsonl` contains step scores, Qwen probabilities, and measured inference timing.

Rebuild the normalized inputs and rerun the within-window history comparison:

```bash
python scripts/fair_deepswe_eval.py prepare
CUDA_VISIBLE_DEVICES=0 python scripts/fair_deepswe_eval.py score --shard 0 --nshards 1
python scripts/summarize_fair_deepswe.py
```

The corrected **full prior-history approximation** also needs raw trajectories, fetched by `setup_sources.py deepswe`. It applies the official 8K truncation `Recipe`, but the authors' original trace-construction code is unavailable, so it is not an exact reconstruction of their inputs:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/clm_full_history_eval.py --shard 0 --nshards 1
python scripts/merge_shards.py --glob 'results/deepswe/clm_full_history_shard*.jsonl' --output results/deepswe/clm_full_history_scores.jsonl --key task --expected 38
python scripts/summarize_run.py results/deepswe/clm_full_history_scores.jsonl --benchmark deepswe_full
```

Within-window CLM selected 21/38, the corrected full-history approximation selected 22/38, Qwen selected 18/38, and fresh Jev selected 20/38. The best fixed agent configuration also selected 22/38. A separate timing run of the **earlier within-window implementation** measured 348.53 GPU seconds for CLM and 250.40 GPU seconds for Qwen. Those timings do not apply to the corrected full-history method or the official vLLM service. See the [DeepSWE analysis](deepswe.md) and [fresh Jev report](jev_fresh_eval.md).

## Missing experiments

We now have fresh Jev inference on one offline web-action candidate set, but not closed-loop browser task success or measured end-to-end token and cost savings from a hybrid agent. To test that application, fix browser states, candidate generation, and budget across systems. Measure action accuracy, final task success, per-step and whole-task latency, input/output tokens, and API/GPU cost. Offline per-item accuracy alone does not establish speed or savings.
