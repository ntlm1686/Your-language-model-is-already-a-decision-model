# Your Language Model Is Already a Decision Model

## 1. Decision making without extra model training

A language model can choose among candidate actions without a separately trained decision head. We present the state and all candidate actions in one prompt, stop at `Answer: `, and read the model's next-token probabilities for the option letters. For each option, its score is the log probability of its current letter. We cyclically rotate the options so each action appears at every letter position, add its log probabilities across rotations, and choose the highest-scoring action. The Qwen3-8B weights receive **no decision-specific fine-tuning** in this experiment. This procedure generates no explanation tokens and does not require an embedding model.

![Qwen3-8B next-letter log-probability decision rule](assets/qwen_logprob.svg)

For N candidates, the evaluated rule uses N forward passes, each containing the state and all candidates. This makes the choice auditable: we can save the option order, restricted next-token probabilities, and final score for every decision. It also means the cost grows with both context length and candidate count. The [method note](logprob_method.md) and [inference code](../scripts/qwen3_8b_letters.py) specify the exact prompt and scoring rule.

### Serve it through a Jev-format API

The same decision rule can be placed behind Jev's `POST /v1/systemone` JSON interface. A client sends a `state` and one or more typed `questions`; the local adapter maps each question to Qwen scores and returns the expected `answers` object for `Choice`, `Noul`, and `Score`. `Choice` returns a selected key and a probability map, `Noul` returns the probability of Yes, and `Score` returns level probabilities and their expected ordinal value. The response names the actual model `qwen3-8b`. Its probabilities have **not** been calibrated on a decision dataset.

```bash
python scripts/setup_sources.py model
CUDA_VISIBLE_DEVICES=0 python scripts/serve_qwen_jev.py \
  --model models/qwen3_8b --device cuda:0 --port 8795 --max-length 8192
```

```bash
curl -sS http://127.0.0.1:8795/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-8b","state":"Customer asks for a refund.","questions":{"route":{"type":"choice","instructions":"Where should the case go?","criteria":{"returns":"Refunds and exchanges","shipping":"Delivery issues"}}}}'
```

The adapter is [implemented here](../scripts/serve_qwen_jev.py), with [protocol tests](../tests/test_qwen_jev_api.py), [full usage notes](qwen_jev_api.md), and a [resumable JevBench API evaluation client](../scripts/eval_qwen_jev_api.py). We confirmed that the upstream JevBench `TypeSafeAdapter` parses its response without modification. For up to 26 options it uses the rotated letter method above; for larger `Choice` sets it switches to separately scored Yes/No logits per candidate and exposes that method in response metadata.

## 2. Results on fixed-candidate decisions

We compare the **untrained decision interface** on Qwen3-8B with Jev 1.13.0, released CLM with its default 2K context, and the released Open-Jev-2B/9B adapters. Jev, CLM, and Open-Jev are specialized decision models or adapters; Qwen3-8B is the off-the-shelf base model. The candidate actions and gold labels are fixed per benchmark. Every row below uses the **same eligible item IDs across the five methods**; this matters where Open-Jev's published 4,096-token per-candidate limit excludes long states. Values are correct decisions / eligible decisions.

| Benchmark (shared eligible items) | Qwen3-8B, **no decision training** | CLM 2K | Jev 1.13.0 | Open-Jev-2B | Open-Jev-9B |
| --- | ---: | ---: | ---: | ---: | ---: |
| JevBench public (231) | 163/231 | 95/231 | **199/231** | 150/231 | 179/231 |
| MetaTool Task1 (1,040) | **842/1,040** | 518/1,040 | 802/1,040 | 692/1,040 | 792/1,040 |
| When2Call (599 of 600) | 306/599 | 200/599 | 439/599 | 235/599 | **451/599** |
| BFCL v4 Multiple (200) | **198/200** | 169/200 | **198/200** | 196/200 | 197/200 |
| PhishNChips (2,000) | 1,330/2,000 | 1,212/2,000 | 1,251/2,000 | 1,000/2,000 | **1,453/2,000** |
| WebPRM sample (93 of 120) | 53/93 | 19/93 | **54/93** | 43/93 | 47/93 |
| WebPRM full (842 of 1,141) | 528/842 | 152/842 | **580/842** | 421/842 | 472/842 |

The complete archived per-item outputs and [Open-Jev protocol](open_jev.md) permit recounting every row. The two Open-Jev checkpoints were run locally with the official model code and saved calibration settings; the Jev 1.13.0 results came from its hosted API. The Qwen API independently scored **163/231** on JevBench, with all **231/231** individual predictions matching the archived direct Qwen run in the same PyTorch/Transformers environment. The API run validates the serving layer; the table continues to use the archived direct Qwen results.

### Latency and context coverage

On JevBench, the local Qwen3-8B Jev-format API took **167 ms median HTTP time per decision** on our H100 setup, including cyclic rotations. The corresponding medians were 444 ms for local Open-Jev-2B, 602 ms for local Open-Jev-9B, and 336 ms for the remote Jev API. These are measurements of different deployments and runtimes; they are useful as observed end-to-end times, not as a controlled architecture speed ranking. The generic CLM JevBench run did not record comparable per-item timing. Other per-item latency results are in the [main table](../README.md#results).

At the official 4K Open-Jev cap, **all 152 DeepSWE rotated requests were too long** for both checkpoints, so we report no default-cap accuracy there. An explicitly exploratory 8K configuration selected successful trajectories for 24/38 tasks with 2B and 23/38 with 9B. The original Qwen, CLM, and hosted Jev DeepSWE figures are 18/38, 21/38, and 20/38 on their respective protocols; the 8K Open-Jev numbers should not be merged into the default-cap table.

The results support a narrow claim: **an off-the-shelf language model can provide a working decision API without decision-specific training, with competitive accuracy on several fixed-candidate tasks and usable local latency**. They do not show that it always matches a trained decision model. Jev leads on JevBench and eligible WebPRM states; Open-Jev-9B leads on the selected When2Call and phishing subsets. These are offline decisions, not end-to-end browser task outcomes. The [architecture discussion](decision_model_paradigms.md) describes how score heads, contrastive state/action embeddings, and RAM-inspired action queries differ from next-letter scoring.

## 3. Discussion

![Four decision-model scoring architectures](assets/decision_model_paradigms.svg)

**What the untrained baseline buys.** The same Qwen3-8B weights can answer typed decision requests through a small adapter: no collection of decision labels, fine-tuning run, adapter weights, or specialized head are required. Reusing a general model also makes the decision interface easy to update when the underlying LM improves. The exact 231/231 prediction agreement between direct Qwen scoring and the Jev-format API shows that the serving layer can preserve the decision rule.

**Where specialized models help.** Jev wins the public JevBench and eligible WebPRM comparisons, and Open-Jev-9B has the best observed accuracy on the When2Call subset and PhishNChips. A trained scalar head can fit a decision objective directly; CLM's separate state/action encoders can cache action embeddings; a proposed RAM-inspired action-query decoder could encode the state once while letting candidate queries attend to state features. The latter is a research direction, not a result of this repository. All comparisons must keep the candidate set, answer labels, context allowance, and hardware/accounting rules visible.

**Cost, calibration, and modality.** Rotating N options requires N full Qwen forwards over the state and all options. A single fixed order is faster but may be sensitive to position; an action-embedding model may be cheaper when the same action catalog is reused. Our restricted-letter softmax gives a distribution over the supplied candidates, but it is not a calibrated probability that an action is correct. The current released Qwen/CLM/Open-Jev evaluation paths consume text. The state/action formulation could accept images or other modalities if the model encoders, input pipeline, and training or alignment data support them; the current numbers do not test that possibility.

**Limit of the evidence.** All tasks here select from supplied options. WebPRM and DeepSWE use offline states and saved trajectories. They do not measure whether a browser agent reaches its goal, how many observation or reasoning tokens it spends, or whether using a decision API reduces total workflow cost. That is the next controlled experiment: hold the browser tasks and candidate generator fixed, then compare direct LM decisions, a trained score head, CLM-style retrieval, and an action-query model on task success, decision accuracy, end-to-end time, total tokens, and cost.

## 4. Reproducing the evaluation

The repository pins upstream Git revisions and model checkpoints. The committed per-item JSONL records allow the published counts to be checked without downloading model weights:

```bash
git clone --filter=blob:none https://github.com/ntlm1686/Your-language-model-is-already-a-decision-model.git
cd Your-language-model-is-already-a-decision-model
python scripts/verify_results.py
python scripts/summarize_open_jev.py
python -m unittest discover -s tests -p 'test_qwen_jev_api.py' -v
```

To **rerun Qwen3-8B inference**, use Python 3.11, PyTorch 2.12.0, and Transformers 4.57.6 (the versions of the matching direct/API JevBench runs), install `requirements.txt`, fetch the pinned Qwen checkpoint and JevBench public data, and start the local API:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/setup_sources.py repos model
CUDA_VISIBLE_DEVICES=0 python scripts/serve_qwen_jev.py \
  --model models/qwen3_8b --device cuda:0 --port 8795 --max-length 8192
```

In a second terminal in the same environment, run the 231 public decisions and inspect the archived comparison printed by the client:

```bash
python scripts/eval_qwen_jev_api.py \
  --url http://127.0.0.1:8795 \
  --output results/jevbench/qwen3_8b_jev_api.jsonl
```

The client verifies response shape and input IDs, records one item per line, and resumes an interrupted run. Remove or choose a new output file when deliberately starting a fresh run. The [Qwen API protocol](qwen_jev_api.md) gives a manual request and the 26-option fallback rule. The [main benchmark protocol](benchmark_protocol.md) gives the original Qwen/CLM/Jev task preparation and prompt rules; the [Open-Jev protocol](open_jev.md) gives its separate Transformers 5.10.2 environment, pinned Qwen3.5 backbones, official 4K servers, exact suite commands, and 8K DeepSWE exploratory procedure. `scripts/summarize_open_jev.py` recomputes every Open-Jev coverage count and the Qwen/CLM/Jev scores on the **same eligible IDs**. The released checkpoints and large raw datasets remain with their publishers; the repository stores source-fetch commands and result records, not duplicated weights.

To rerun the Open-Jev columns, download their pinned source/checkpoints and the input sources, create the separate official-code environment described in the [Open-Jev protocol](open_jev.md#serving-setup), and start one official 4K server per checkpoint. Then run the same suites against each local endpoint:

```bash
python scripts/setup_sources.py repos qwen_tokenizer webprm phishing when2call metatool bfcl open_jev
export MODEL_PATH="$PWD/models/qwen3_8b_tokenizer"
for suite in jevbench metatool when2call bfcl phishing webprm120 webprm1141 deepswe; do
  python scripts/run_open_jev.py "$suite" --url http://127.0.0.1:8791 \
    --output "results/open_jev_2b/${suite}.jsonl"
  python scripts/run_open_jev.py "$suite" --url http://127.0.0.1:8792 \
    --output "results/open_jev_9b/${suite}.jsonl"
done
python scripts/summarize_open_jev.py
python scripts/verify_results.py
```

The runner verifies candidate IDs and input hashes against the archived Qwen/Jev inputs, records HTTP 422 length exclusions, and never truncates a request. The 8K DeepSWE run uses a different server cap and output file; its explicit command is in the [Open-Jev protocol](open_jev.md#long-deepswe-trajectories). Repeating the hosted Jev columns requires a Jev API key and the [API procedure](jev_api.md); no key appears in these files.
