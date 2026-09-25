# Doom: Defend the Center

This is a closed-loop game experiment: a model chooses one of `turn_left`, `turn_right`, or `attack` after every observation, and that choice changes the next observation. The game is [ViZDoom's Defend the Center](https://vizdoom.farama.org/main/environments/default/). The protocol follows the [AlexWortega/openjev Doom code](https://huggingface.co/AlexWortega/openjev/tree/058a6c24911b46d908fbe23541390f8af3df3e4d/code), pinned to revision `058a6c24911b46d908fbe23541390f8af3df3e4d`.

## Main comparison: 15-second budget

The README now uses fresh Qwen3.5-9B and hosted Jev 1.13.0 runs on the same five seeds, observations, and actions. Both receive a 15-second wall-clock budget. One initial-state request warms each episode before its clock starts; no warmup action is applied. Qwen runs across four independent H100 servers, with seeds 100/104 on GPU 0 and 101–103 on GPUs 1–3. Jev uses the hosted API.

| Model | Kills, seeds 100–104 | Mean kills | Median decision time | Episodes stopped by deadline |
| --- | --- | ---: | ---: | ---: |
| Qwen3.5-9B | 12, 10, 6, 9, 14 | 10.2 | 55 ms | 1/5 |
| Jev 1.13.0 | 3, 3, 3, 3, 2 | 2.8 | 336 ms | 5/5 |

The timer includes inference, game stepping, trace writing, and video writing. The seed-100 episode is recorded for both models. The client caps its HTTP timeout by remaining budget and discards responses returned after the deadline. `elapsed_before_s` and `elapsed_after_s` in each trace record the decision's position on the episode clock. Every recorded decision returned before 15 seconds. Summary `wall_seconds` also includes video finalization, which can make it slightly exceed 15 seconds without adding decisions.

The [main GIF](../results/doom/qwen3_5_9b_vs_jev_15s_seed100.gif) is aligned by **real elapsed time**, not action index. Both shown seed-100 episodes actually reached the deadline, finishing 12 : 3. At 15 seconds, both panels display `TIMEOUT` and their final kills for two more seconds. For an episode ending naturally before the deadline, the renderer holds its final score through the rest of the evaluation window; `TIMEOUT` then labels the window ending, while the summary retains the actual `timed_out` flag. Four other Qwen seeds ended naturally. Per-step times include localhost HTTP for Qwen and a network round trip for Jev.

Records: [Qwen summary](../results/doom/qwen3_5_9b_15s.json), [Qwen trace](../results/doom/qwen3_5_9b_15s_steps.jsonl), [Jev summary](../results/doom/jev_1_13_15s.json), [Jev trace](../results/doom/jev_1_13_15s_steps.jsonl).

### Reproduce the 15-second comparison

Start four optimized Qwen servers, one per GPU, on ports 8810–8813. Wait for each server to print its URL before running the clients:

```bash
for gpu in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$gpu QWEN35_USE_FLA=1 \
    models/open_jev_py311/bin/python scripts/serve_qwen35_jev.py \
    --port "$((8810 + gpu))" --fallback-margin 0 > "/tmp/doom15_server${gpu}.log" 2>&1 &
done
```

```bash
models/doom_py311/bin/python scripts/run_doom_qwen_timed.py
models/doom_py311/bin/python scripts/eval_doom_qwen.py \
  --policies api_position_none --url https://api.typesafe.ai \
  --endpoint-model jev-1.13.0 --expected-model jev-1.13.0 \
  --episodes 5 --seed-base 100 --max-wall-seconds 15 --warmup --hard-deadline \
  --api-key-file private/doom_api_key \
  --output results/doom/jev_1_13_15s.json --trace results/doom/jev_1_13_15s_steps.jsonl \
  --video-policy api_position_none --video /tmp/jev_1_13_15s_seed100.mp4
models/doom_py311/bin/python scripts/render_doom_timed_comparison.py \
  --qwen-video /tmp/qwen3_5_9b_15s_seed100.mp4 \
  --qwen-trace results/doom/qwen3_5_9b_15s_steps.jsonl \
  --qwen-summary results/doom/qwen3_5_9b_15s.json \
  --jev-video /tmp/jev_1_13_15s_seed100.mp4 \
  --jev-trace results/doom/jev_1_13_15s_steps.jsonl \
  --jev-summary results/doom/jev_1_13_15s.json \
  --seconds 15 --output results/doom/qwen3_5_9b_vs_jev_15s_seed100.gif
```

Run `python scripts/verify_doom_timed.py` to check the saved seeds, step counts, decision timestamps, probabilities, kills, and aggregate times without a GPU. The token file is local and Git-ignored. MP4 files are temporary rendering inputs; only GIFs are published. Stop the four servers after the experiment.

## Protocol

- ViZDoom 1.2.4, `defend_the_center.cfg`, 640 × 480 screen, episode timeout 2,100 tics, four game tics per action, seeds 100–104, five episodes per policy.
- The text policies see the **same labels-buffer-derived text**, including visible enemy names and screen offsets, ammunition, and health. They do not receive pixels. We use the author's `position_none` action statements verbatim: nearest enemy left or no enemy visible; nearest enemy right; nearest enemy on the crosshair.
- The Qwen3-8B and Qwen3.5-9B servers map those three options to next-letter probabilities with cyclic rotations and **no decision-model training**. CLM uses the released contrastive head and its default 2,048-token cap. ZefanCai Open-Jev-2B/9B use their released LoRA and scalar heads at their official 4,096-token cap. Hosted Jev 1.13.0 uses the TypeSafe API. All six use the same client and JSON Choice request. Model-specific scoring remains each model's published method.
- The AlexWortega 4B NLI model is a **different project from ZefanCai Open-Jev**. We run its original `doom.py` with the same `position_none` hypotheses. Its pixel experiment uses the original `doom_vision.py` with the published `pixels` hypothesis set. The pixel result is separated because its input is a screenshot rather than labels-derived text.
- Kills are the game `KILLCOUNT`. Random and labels-based oracle are controls. The oracle uses information already present in the text observations; it is a simple rule, not a trained model.

Each Choice run records one JSON row per decision with the selected action, option probabilities, enemies, ammunition, kills before action, input-token count, and client-observed API latency. Its summary gives the five episode scores and decision latency p50/p95. The seed-100 GIFs show the game, step, kills, ammunition, health, latency, shared Choice input, and all three action scores. The displayed input is the JSON request content; Qwen internally renders and rotates letter-labeled prompts, while the specialized models have their own prompt renderers. The author's original code exports its **best** episode with a probability chart and reports a mean per-episode decision latency, so its video seed is not identified in the output. The game advances four tics after inference, and its 114 ms of game time per decision is not an enforced wall-clock timeout.

## Earlier results without a wall-clock limit

Local model results are from this host with one NVIDIA H100 per serving process. The Jev 1.13.0 result uses the hosted TypeSafe API. Choice-client timing excludes model load and game stepping; it includes a localhost HTTP round trip for Qwen/CLM/Open-Jev and a network round trip for Jev. The author's direct Python timing excludes HTTP. We ran the local servers on separate GPUs at the same time; GPU memory and CPU/host contention may affect latency. The controls in this ViZDoom installation scored random 1.0 and oracle 16.6 mean kills; these differ from the author's archived oracle score under their installation, so compare the **local** rows here rather than mixing in published scores.

| Policy | Input | Kills, seeds 100–104 | Mean kills | Decision p50 | Decision p95 | Decision mean |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Random | None | 1, 2, 1, 0, 1 | 1.0 | — | — | — |
| Labels oracle | Labels | 19, 14, 15, 18, 17 | 16.6 | — | — | — |
| Qwen3.5-9B, serial PyTorch | Text | 13, 10, 6, 9, 14 | 10.4 | 396 ms | 446 ms | 408 ms |
| Qwen3-8B, next-letter probabilities | Text | 7, 7, 6, 9, 13 | 8.4 | 98 ms | 102 ms | 99 ms |
| CLM-v0.1-8B, default 2K | Text | 0, 0, 0, 0, 0 | 0.0 | 34 ms | 36 ms | 34 ms |
| Jev 1.13.0, hosted API | Text | 9, 11, 19, 9, 12 | 12.0 | 339 ms | 405 ms | 351 ms |
| ZefanCai Open-Jev-2B | Text | 11, 8, 6, 9, 19 | 10.6 | 331 ms | 343 ms | 325 ms |
| ZefanCai Open-Jev-9B | Text | 13, 12, 7, 9, 20 | **12.2** | 448 ms | 462 ms | 439 ms |
| AlexWortega 4B NLI | Text | Per-episode counts unavailable from original summary | 10.4 | — | — | 145 ms |

**Separate pixel-input run:** The same AlexWortega 4B NLI checkpoint, using its original `doom_vision.py` zero-shot `pixels` formulation, scored **9.4 mean kills** over the five seeds (maximum 14) at **174 ms mean decision latency**. It receives raw frames rather than the labels-derived text above, so the result is not a matched-observation comparison. Its local random/oracle controls were again 1.0/16.6.

The AlexWortega row uses the author's original Python timing and five-episode mean; its script does not save per-episode kills or p50/p95. Its summary reports a maximum of 14 kills. For the six models run through our Choice client, the adjacent p50/p95 columns are the HTTP round-trip time per decision across all five episodes. Qwen3-8B and CLM fit the nominal 114 ms game-tic interval in median latency here; the game loop still waits for slower decisions, so scores are not from a hard real-time evaluation. CLM chose `turn_right` on every recorded step, which explains its zero kills in this particular formulation. Open-Jev-9B leads this five-seed score, but five episodes are too few to make a general model ranking claim.

### 30-second real-time budget

We also stopped each episode after 30 real elapsed seconds. The timer includes API inference, HTTP, trace/video writing, and game stepping; an in-flight action completes before the stop check. All local policies saw the same labels-derived state and three action descriptions. Hosted Jev was not rerun under this wall-clock limit, so its earlier 12.0-kill result is separate.

| Policy | Kills, seeds 100–104 | Mean kills | Decision p50 | Timed out |
| --- | --- | ---: | ---: | ---: |
| Qwen3.5-9B, batched FLA | 13, 10, 6, 9, 14 | **10.4** | **58 ms** | 0/5 |
| Qwen3.5-9B, batched PyTorch with near-tie fallback | 9, 9, 6, 9, 13 | 9.2 | 139 ms | 3/5 |
| Qwen3.5-9B, serial PyTorch | 3, 4, 3, 5, 5 | 4.0 | 420 ms | 5/5 |
| Qwen3-8B, serial PyTorch | 7, 7, 6, 9, 13 | 8.4 | 100 ms | 0/5 |
| CLM-v0.1-8B | 0, 0, 0, 0, 0 | 0.0 | 34 ms | 0/5 |
| Open-Jev-2B | 4, 6, 3, 6, 7 | 5.2 | 327 ms | 5/5 |
| Open-Jev-9B | 3, 6, 3, 4, 5 | 4.2 | 454 ms | 5/5 |

The serial 9B path took only 71–74 actions before the deadline. The batched FLA path took 121–249 and ended naturally in every episode. Its Doom run has the same five kills and episode lengths as the unlimited serial run, although some individual actions differ. The fast kernel changed two of 231 separate JevBench predictions relative to the original PyTorch path; the batched PyTorch mode with near-tie fallback preserved all 231. See the [9B speed and reproduction note](qwen35.md).

The [Qwen3.5-9B versus hosted Jev side-by-side GIF](../results/doom/qwen3_5_9b_vs_jev_seed100.gif) is an archived comparison with different wall-clock limits. Both panels use seed 100 and advance by recorded game step; a finished panel displays an **EPISODE ENDED** banner and its final kills. The renderer displays **TIMEOUT** only when the episode summary records `timed_out: true`; neither featured episode timed out. The top scoreboard shows each model's kills from the matching trace, then its final episode-summary count (13 for Qwen and 9 for Jev). The final score remains visible for two extra seconds. Qwen used the 30-second wall-clock protocol but ended naturally after 16.8 seconds; the earlier hosted Jev run had no wall-clock limit. The [individual Qwen3.5-9B optimized replay](../results/doom/qwen3_5_9b_30s_fast_seed100.gif), [conservative batched replay](../results/doom/qwen3_5_9b_30s_batch_safe_seed100.gif), [original serial replay](../results/doom/qwen3_5_9b_30s_seed100.gif), and [Qwen3-8B timed replay](../results/doom/qwen3_8b_30s_seed100.gif) show the same seed at different inference speeds. Raw summaries and traces have matching filename stems in `results/doom/` (`qwen3_5_9b_30s_fast`, `qwen3_5_9b_30s_batch_safe`, `qwen3_5_9b_30s`, `qwen3_5_9b_unlimited`, and the other policy names ending in `_30s`). Temporary MP4s were used only to render the committed GIFs.

## Videos and raw records

The [older Qwen3-8B versus hosted Jev side-by-side GIF](../results/doom/qwen_vs_jev_seed100.gif) remains available. `scripts/compare_qwen_jev_gif.py` recreates either comparison from the two individual GIFs; for the earlier 9B comparison, use `--qwen results/doom/qwen3_5_9b_30s_fast_seed100.gif --jev results/doom/jev_1_13_seed100.gif --output results/doom/qwen3_5_9b_vs_jev_seed100.gif --qwen-trace results/doom/qwen3_5_9b_30s_fast_steps.jsonl --jev-trace results/doom/jev_1_13_steps.jsonl --qwen-summary results/doom/qwen3_5_9b_30s_fast.json --jev-summary results/doom/jev_1_13.json`.

Watch the [four-policy side-by-side GIF](../results/doom/text_models_preview.gif) for an eight-second overview of the common seed 100. `scripts/montage_doom.py` recreates it from the four full GIFs. Open an individual GIF below to read the full prompt, action scores, game state, and latency.

| Policy | GIF replay | Summary | Per-decision trace |
| --- | --- | --- | --- |
| Qwen3-8B | [Seed 100](../results/doom/qwen3_8b_seed100.gif) | [Five episodes](../results/doom/qwen_prompt_variants.json) | [JSONL](../results/doom/qwen_prompt_variants_steps.jsonl) |
| CLM-v0.1-8B | [Seed 100](../results/doom/clm_seed100.gif) | [Five episodes](../results/doom/clm.json) | [JSONL](../results/doom/clm_steps.jsonl) |
| Jev 1.13.0 | [Seed 100](../results/doom/jev_1_13_seed100.gif) | [Five episodes](../results/doom/jev_1_13.json) | [JSONL](../results/doom/jev_1_13_steps.jsonl) |
| Open-Jev-2B | [Seed 100](../results/doom/open_jev_2b_seed100.gif) | [Five episodes](../results/doom/open_jev_2b.json) | [JSONL](../results/doom/open_jev_2b_steps.jsonl) |
| Open-Jev-9B | [Seed 100](../results/doom/open_jev_9b_seed100.gif) | [Five episodes](../results/doom/open_jev_9b.json) | [JSONL](../results/doom/open_jev_9b_steps.jsonl) |
| AlexWortega 4B NLI, text | [Best episode](../results/doom/alex_openjev_nli4b.gif) | [Official JSON](../results/doom/alex_openjev_nli4b.json) | Not exported by upstream script |
| AlexWortega 4B NLI, pixels | [Best episode](../results/doom/alex_openjev_pixel4b.gif) | [Official JSON](../results/doom/alex_openjev_pixel4b.json) | Not exported by upstream script |

We also tested Qwen with two other wordings in [this exploratory result](../results/doom/qwen_text.json): the author's `position` wording without a no-enemy clause scored 1.8 mean kills, and direct action descriptions scored 1.0. The `position_none` wording in the main table was chosen from the author's provided hypotheses and applied unchanged to all four Choice-client models. This prompt sensitivity is a limit on interpreting any single Doom score.

## Reproduce

Download pinned weights and source. The third-party checkpoints and repositories stay outside Git:

```bash
python scripts/setup_sources.py repos heads model open_jev alex_doom
uv venv models/doom_py311 --python /usr/bin/python3.11 --system-site-packages
uv pip install --python models/doom_py311/bin/python -r requirements-doom.txt
```

Start the [Qwen Jev-format server](qwen_jev_api.md) at `127.0.0.1:8795` and the [two official Open-Jev servers](open_jev.md#serving-setup) at ports `8791` and `8792`. For the CLM Choice server:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/serve_clm_doom.py \
  --model models/qwen3_8b --head models/reference_head/CLM_v0.1-8B.pt \
  --port 8797 --max-tokens 2048
```

The fixed five-episode text run for any of those endpoints is:

```bash
models/doom_py311/bin/python scripts/eval_doom_qwen.py \
  --policies api_position_none --url http://127.0.0.1:8797 \
  --endpoint-model clm-latest --expected-model clm-latest \
  --output results/doom/clm.json --trace results/doom/clm_steps.jsonl \
  --video-policy api_position_none --video /tmp/clm_seed100.mp4

models/doom_py311/bin/python scripts/render_doom_gifs.py \
  --video /tmp/clm_seed100.mp4 --trace results/doom/clm_steps.jsonl \
  --model-name CLM-v0.1-8B --output results/doom/clm_seed100.gif
```

Substitute the URL, model fields, and output paths for each server. The Open-Jev servers omit the request model (`--endpoint-model ''`), and their response models are `Qwen/Qwen3.5-2B` or `Qwen/Qwen3.5-9B`. For the Qwen server, use policy `qwen_position_none` and the default model fields. For hosted Jev, use `--url https://api.typesafe.ai --endpoint-model jev-1.13.0 --expected-model jev-1.13.0 --api-key-file private/typesafe_api_key`, with the token kept outside Git. Render its temporary seed-100 video with `scripts/render_doom_gifs.py` as above.

The AlexWortega source needs its own Transformers 5.10.2 environment plus `vizdoom`, `datasets`, `scikit-learn`, `matplotlib`, `imageio`, and `imageio-ffmpeg`. We used the [Open-Jev Python 3.11 environment](open_jev.md#serving-setup), adding the game packages with:

```bash
uv pip install --python models/open_jev_py311/bin/python \
  vizdoom==1.2.4 imageio==2.37.4 imageio-ffmpeg==0.6.0 \
  datasets scikit-learn matplotlib
```

From `external/alex_openjev/code`, run these commands with that environment's Python:

```bash
../../../models/open_jev_py311/bin/python doom.py --ckpt ../qwen3.5-4b-nli-v2 --episodes 5 \
  --hyp position_none --zero-shot-only \
  --out ../../../results/doom/alex_openjev_nli4b.json \
  --video-nli /tmp/alex_openjev_nli4b.mp4

../../../models/open_jev_py311/bin/python doom_vision.py --ckpt ../qwen3.5-4b-nli-v2 --episodes 5 \
  --mode zeroshot --variants pixels \
  --out ../../../results/doom/alex_openjev_pixel4b.json \
  --video /tmp/alex_openjev_pixel4b.mp4
```

Back at the repository root, convert those temporary upstream videos with
`models/doom_py311/bin/python scripts/convert_doom_video_to_gif.py /tmp/alex_openjev_nli4b.mp4 results/doom/alex_openjev_nli4b.gif` and the analogous pixel command. Only GIFs are committed. Recreate the four-model preview with `models/doom_py311/bin/python scripts/montage_doom.py`.

The source revision and weights are pinned in `scripts/setup_sources.py`. These experiments evaluate a small, deterministic game scenario. They do not establish browser-task success or performance across other games.
