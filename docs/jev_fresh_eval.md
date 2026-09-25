# Fresh Jev 1.13.0 evaluation on three public candidate sets

Run date: 2026-09-25. Every API response resolved to `jev-1.13.0`; all calls used `POST https://api.typesafe.ai/v1/systemone`. API credentials were supplied through `TYPESAFE_API_KEY` and are not present in the repository. Committed JSONL files contain per-item predictions, probabilities, usage, and measured request latency, without authorization headers or raw input text.

| Dataset | Jev | Qwen3-8B letter logits | CLM | Jev calls | Jev input tokens | Jev API cost at $0.042/M input tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| JevBench public typed decisions | **199/231** | 163/231 | Generic head, default 2K: 95/231 | 231 | 211,309 | $0.008875 |
| WebPRMBench offline next action | **65/120** | **65/120** | Generic head, default 2K: 24/120 | 120 | 425,440 | $0.017868 |
| DeepSWE trajectory choice | 20/38 | 18/38 | DeepSWE head: 21/38 (final-12 history), 22/38 (full-history approximation) | 152 | 2,526,668 | $0.106120 |

The API costs above come from returned `input_tokens` and the [published Jev 1.13.0 tariff](https://docs.typesafe.ai/models). They exclude local Qwen/CLM GPU costs and do not represent a matched cost comparison. Across these runs, Jev used 3,163,417 input tokens and approximately $0.13286 of listed API input charges, excluding smoke tests.

## JevBench: same 231 public item IDs

The upstream JevBench TypeSafe adapter sent each item's original `state` and typed question without its answer key. All 231 responses were valid. Fresh Jev scored 48/48 easy, 71/72 original, and 80/111 hard; by question type it scored 123/139 choice, 62/74 noul, and 14/18 score. Compared with Qwen on the same item IDs, Jev alone was correct on 39 items and Qwen alone on 3. The mean API request latency was 0.346 seconds, with a 0.336-second median; these are remote request times, not local model forward times.

The JevBench authors' saved historical `jev-1.13.0` result was 200/231. Three hard-item outcomes changed in the fresh run: `hard-opus-b-multi_hop-03` became correct, while `hard-sol-b-long_policy-06` and `hard-sol-c-judge_hard-13` became incorrect. The net change is minus one correct answer. The returned model version matched, but the saved data do not establish why individual outcomes changed.

The committed run is [`results/jevbench/jev_1_13_0_fresh_results.jsonl`](../results/jevbench/jev_1_13_0_fresh_results.jsonl). To repeat it, follow [the JevBench API procedure](jev_api.md). The comparison script rejects partial runs, failed answers, mismatched IDs, and mixed model versions.

## WebPRMBench: same 120 states and candidate order

`scripts/run_jev_webprm.py` reconstructs the 120 Qwen/CLM examples from the original [WebPRMBench](https://huggingface.co/datasets/ZYao720/WEBPRMBENCH) data, seed 42, per-source sample sizes, Qwen token cap, and option shuffling. It checks every selected `(source, state_idx, gold)` against the committed baseline before calling the API. Jev receives the same page state and five candidate responses as a typed choice question. It makes one call per state; Qwen also used one candidate ordering in this pilot.

All 120 Jev responses were valid. Jev and Qwen each scored 65/120, but they disagreed on individual items: Jev alone was correct on 11, and Qwen alone on 11. The generic CLM head scored 24/120 with its default 2K context on the same states and candidate order. Qwen predictions were identical to the earlier run. Jev's mean API request latency was 0.389 seconds (median 0.371 seconds). This is an offline next-action test, not a browser task-completion or token-savings test. Per-item records are in [`results/webprm/jev_1_13_0_fresh_results.jsonl`](../results/webprm/jev_1_13_0_fresh_results.jsonl).

After downloading the pinned WebPRM data and Qwen tokenizer with `setup_sources.py webprm model`, a fresh run is:

```bash
python scripts/run_jev_webprm.py --limit 3 --output private/jev/webprm_smoke.jsonl
python scripts/run_jev_webprm.py --limit 120 --output private/jev/webprm_full.jsonl
```

The script requires `TYPESAFE_API_KEY` in the environment and refuses to overwrite an output file. Use new output paths for later runs.

## DeepSWE: identical four candidate trajectories

`scripts/run_jev_deepswe.py` reads the committed `data/deepswe/fair_inputs.jsonl`. Jev receives the task and the same bounded final-12 action/observation turns that Qwen saw, framed as four typed choice criteria. It makes four calls per task using the **same cyclic candidate orders** as Qwen, sums the log probabilities for each underlying trajectory, and selects the maximum. Verifier rewards are read only after the calls to score the choice.

All 152 Jev responses resolved to the same model. Jev selected a successful trajectory on 20/38 tasks, including 12/21 tasks where candidate outcomes differed. Qwen scored 18/38 and the within-window CLM reconstruction scored 21/38. The corrected CLM full-history approximation scored 22/38, but its state includes prior turns beyond Jev's and Qwen's final-12 excerpts. The best fixed agent configuration also scored 22/38, and the candidate-set oracle scored 29/38. Jev alone succeeded on five tasks that Qwen missed, while Qwen alone succeeded on three that Jev missed.

Jev's mean API request latency was 0.570 seconds (median 0.563 seconds), for 86.70 seconds summed over 152 calls. Summing the four sequential request latencies for each task gives a **2.280-second median per task**; Qwen3.5-9B's four serial GPU forwards took 6.592 seconds per task at the median. These are different measurement boundaries: Jev includes the remote round trip, while Qwen excludes prompt tokenization and HTTP. Each Qwen rotation had a median of 15,929 input tokens, which explains why its task latency is much higher than on the shorter benchmarks. Two reported `choice` labels differed from the largest **rounded** returned probability; selection consistently uses the returned probabilities, not the reported sample label. Per-task records are in [`results/deepswe/jev_1_13_0_fresh_results.jsonl`](../results/deepswe/jev_1_13_0_fresh_results.jsonl).

```bash
python scripts/run_jev_deepswe.py --limit 1 --output private/jev/deepswe_smoke.jsonl
python scripts/run_jev_deepswe.py --limit 38 --output private/jev/deepswe_full.jsonl
```

The DeepSWE head was trained for trajectory selection and the generic CLM head was not. The 38 tasks use publicly available Qwen3.8-27B agent trajectories, **not** the Opus 5 trajectories behind CLM's published 31/38 result. This sample does not support a broad ranking of decision architectures.

## Interpretation

On JevBench's typed decisions, fresh Jev substantially outperformed the direct Qwen letter readout. On this WebPRM next-action sample, Jev and Qwen tied. On this DeepSWE candidate set, Jev fell between Qwen and the specialized CLM head. The three benchmarks differ in task framing, candidate length, CLM training, and available history. None measures the outcome of putting a decision model inside a live browser-agent loop. API latency cannot be compared directly with local forward-only GPU timings from the earlier experiments.
