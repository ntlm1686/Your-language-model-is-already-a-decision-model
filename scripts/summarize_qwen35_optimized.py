"""Validate and recount the complete optimized Qwen3.5-9B evaluation."""
from __future__ import annotations
import json
import statistics
from pathlib import Path
from summarize_qwen35 import EXPECTED, ROOT

def key(row):
    return row.get('id', row.get('task', (row.get('source'), row.get('state_idx'))))

def summarize():
    root = ROOT/'results/qwen3_5_9b'
    output = {}
    for suite, expected in EXPECTED.items():
        rows = [json.loads(line) for p in sorted((root/'optimized').glob(f'{suite}_shard*of*.jsonl'))
                for line in p.read_text().splitlines()]
        baseline = [json.loads(line) for p in root.glob(f'{suite}_shard*of*.jsonl')
                    for line in p.read_text().splitlines()]
        old = {key(r): r for r in baseline}
        indexed = {key(r): r for r in rows}
        if len(rows) != expected or len(indexed) != expected or set(indexed) != set(old):
            raise ValueError(f'{suite}: expected {expected} identical unique IDs; found {len(rows)}')
        for k, r in indexed.items():
            assert r['gold'] == old[k]['gold']
            assert r['scoring_mode'] == 'batched_rotations_fla'
            assert 0 < r['forward_s'] <= r['latency_s']
            if 'rotations' in r:
                for newrot, oldrot in zip(r['rotations'], old[k]['rotations'], strict=True):
                    assert newrot['tokens'] == oldrot['tokens']
                    for name in ['order', 'shift']:
                        if name in newrot: assert newrot[name] == oldrot[name]
                assert len(r['prompt_sha256']) == len(r['rotations'])
            else:
                assert r['tokens'] == old[k]['tokens']
            for name in ['input_sha256','email_sha256']:
                if name in r: assert r[name] == old[k][name]
        times = sorted(r['latency_s']*1000 for r in rows)
        result = {'n': len(rows), 'correct': sum(r['correct'] for r in rows),
                  'accuracy_pct': round(100*sum(r['correct'] for r in rows)/len(rows), 2),
                  'p50_ms': round(statistics.median(times)),
                  'p95_ms': round(times[int(.95*(len(times)-1))]),
                  'forward_p50_ms': round(statistics.median(r['forward_s'] for r in rows)*1000),
                  'prediction_agreement_with_serial': sum(r['prediction'] == old[k]['prediction'] for k,r in indexed.items()),
                  'timing_scope': 'prompt_tokenization_forward_aggregation_no_http'}
        shared = ROOT/'results/open_jev_9b'/f'{suite}.jsonl'
        if shared.exists() and suite != 'deepswe':
            eligible = {r['id'] for r in map(json.loads, shared.read_text().splitlines()) if r['outcome']=='ok'}
            shared_rows = {(f"{r['source']}:{r['state_idx']}" if suite.startswith('webprm') else r['id']):r for r in rows}
            assert eligible <= set(shared_rows)
            result['open_jev_eligible_n'] = len(eligible)
            result['correct_on_open_jev_eligible'] = sum(shared_rows[k]['correct'] for k in eligible)
        output[suite] = result
    return output

if __name__ == '__main__':
    result=summarize()
    (ROOT/'results/qwen3_5_9b/optimized/summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
