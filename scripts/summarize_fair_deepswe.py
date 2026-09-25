"""Aggregate fresh same-candidate CLM/Qwen DeepSWE shard results."""
import collections
import glob
import json
import math
from pathlib import Path

root = Path(__file__).resolve().parents[1] / 'results/deepswe'
rows = [json.loads(line) for p in glob.glob(str(root / 'fair_scores_shard*.jsonl'))
        for line in open(p)]
rows.sort(key=lambda x: x['task'])
assert len(rows) == 38 and len({r['task'] for r in rows}) == 38
counts = collections.Counter()
merged = []
for r in rows:
    rewards = r['rewards']
    clm_pick = max(r['clm'], key=lambda run: r['clm'][run]['score'])
    qwen_scores = {run: 0.0 for run in rewards}
    qwen_rotation_picks = []
    for round_ in r['qwen']:
        probs, order = round_['probs'], round_['order']
        qwen_rotation_picks.append(order[max(range(4), key=probs.__getitem__)])
        for i, run in enumerate(order):
            qwen_scores[run] += math.log(max(probs[i], 1e-30))
    qwen_pick = max(qwen_scores, key=qwen_scores.get)
    mixed = 0 < sum(rewards.values()) < 4
    counts['clm'] += rewards[clm_pick]
    counts['qwen'] += rewards[qwen_pick]
    counts['random_expected'] += sum(rewards.values()) / 4
    counts['oracle'] += int(any(rewards.values()))
    counts['mixed'] += int(mixed)
    counts['clm_mixed'] += int(mixed) * rewards[clm_pick]
    counts['qwen_mixed'] += int(mixed) * rewards[qwen_pick]
    counts['qwen_same_run_all_orders'] += int(len(set(qwen_rotation_picks)) == 1)
    counts['qwen_input_tokens'] += sum(rd['tokens'] for rd in r['qwen'])
    merged.append({'task': r['task'], 'rewards': rewards, 'clm_pick': clm_pick,
                   'clm_reward': rewards[clm_pick], 'clm_scores': {k: v['score'] for k,v in r['clm'].items()},
                   'qwen_pick': qwen_pick, 'qwen_reward': rewards[qwen_pick],
                   'qwen_logprob_scores': qwen_scores,
                   'qwen_rotation_picks': qwen_rotation_picks,
                   'qwen_tokens': [rd['tokens'] for rd in r['qwen']]})
counts['best_fixed_agent'] = max(sum(r['rewards'][run] for r in rows) for run in rows[0]['rewards'])
counts['paired_clm_only'] = sum(x['clm_reward'] == 1 and x['qwen_reward'] == 0 for x in merged)
counts['paired_qwen_only'] = sum(x['clm_reward'] == 0 and x['qwen_reward'] == 1 for x in merged)
summary = {'tasks': len(rows), **dict(counts)}
(root / 'fair_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
with (root / 'fair_results.jsonl').open('w') as f:
    for row in merged:
        f.write(json.dumps(row) + '\n')
print(json.dumps(summary, indent=2))
