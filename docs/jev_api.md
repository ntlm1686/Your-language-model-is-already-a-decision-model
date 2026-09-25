# Run Jev on JevBench's 231 public items

The JevBench authors' saved historical Jev figure is 200/231. Our fresh Jev 1.13.0 run scored 199/231; see the [fresh Jev report](jev_fresh_eval.md). This procedure repeats **fresh Jev calls on the same 231 items** and compares them item by item with the committed Qwen and CLM results. The API's resolved model, probabilities, latency, and token usage are initially written under local `private/`, which Git ignores. The reviewed per-item records from our completed run are committed under `results/`.

First run `python scripts/setup_sources.py repos` as described in the [main protocol](benchmark_protocol.md#common-environment). The [TypeSafe model documentation](https://docs.typesafe.ai/models) currently lists `jev-1.13.0`. Pin this version instead of the moving `jev-latest` alias. The documented price is $0.042 per million input tokens, with free output tokens; verify actual charges and model access in your account.

Set the key in your terminal. Do not put it in the repository or chat:

```bash
read -rs TYPESAFE_API_KEY
export TYPESAFE_API_KEY
curl -fsS https://api.typesafe.ai/v1/models -H "Authorization: Bearer $TYPESAFE_API_KEY"
```

Run three items first to check authentication and response shape. Execute from the repository root:

```bash
mkdir -p private/jev/smoke private/jev/full
export PYTHONPATH="$PWD/external/jevbench"
export JEV_TASKS='external/jevbench/datasets/public/easy.jsonl,external/jevbench/datasets/public/original.jsonl,external/jevbench/datasets/public/hard.jsonl'

python -m jevbench.cli run \
  --tasks "$JEV_TASKS" --adapter typesafe --model jev-1.13.0 \
  --key-env TYPESAFE_API_KEY --limit 3 \
  --price-in-per-m 0.042 --price-out-per-m 0 --cap-usd 1 \
  --results private/jev/smoke/results.jsonl \
  --raw-dir private/jev/smoke/raw \
  --ledger private/jev/ledger.jsonl \
  --manifest private/jev/smoke/manifest.json
```

After confirming that all three smoke-test records have `ok=true`, run the full set:

```bash
python -m jevbench.cli run \
  --tasks "$JEV_TASKS" --adapter typesafe --model jev-1.13.0 \
  --key-env TYPESAFE_API_KEY \
  --price-in-per-m 0.042 --price-out-per-m 0 --cap-usd 1 \
  --results private/jev/full/results.jsonl \
  --raw-dir private/jev/full/raw \
  --ledger private/jev/ledger.jsonl \
  --manifest private/jev/full/manifest.json

python scripts/compare_jev_api.py private/jev/full/results.jsonl
```

`compare_jev_api.py` reports results only if all 231 IDs match, every response is valid, and all responses resolve to the same Jev version. It reports Jev/Qwen/CLM accuracy, paired unique correct items, and Jev latency, input tokens, and reported cost. The JevBench runner stops on 401, 403, or 429 responses or repeated infrastructure errors. If that happens, inspect the run and use **new output paths** for another attempt; the harness does not overwrite raw evidence.

Latency has a different measurement boundary across systems: Jev is a remote API request, while the earlier local Qwen/CLM timings exclude model loading and preprocessing. A claim about hybrid web-agent cost still requires a separate evaluation on identical candidate actions and closed-loop tasks.
