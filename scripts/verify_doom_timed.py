"""Recount the saved, matched 15-second Doom comparison without GPU access."""
import json
import math
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[1]/'results/doom'

def verify(stem):
    summary=json.loads((ROOT/f'{stem}.json').read_text())
    rows=list(map(json.loads,(ROOT/f'{stem}_steps.jsonl').read_text().splitlines()))
    assert summary['max_wall_seconds']==15
    assert summary['warmup'] and summary['hard_deadline']
    result=summary['results']['api_position_none']
    episodes=result['episodes']
    assert [e['seed'] for e in episodes]==list(range(100,105))
    assert {r['episode'] for r in rows}==set(range(5))
    for i,episode in enumerate(episodes):
        selected=[r for r in rows if r['episode']==i]
        assert len(selected)==episode['steps']
        assert all(r['step']==j and r['seed']==episode['seed'] for j,r in enumerate(selected))
        for row in selected:
            assert 0<=row['elapsed_before_s']<=row['elapsed_after_s']<15
            assert row['policy']=='api_position_none'
            assert set(row['probabilities'])=={'turn_left','turn_right','attack'}
            assert math.isclose(sum(row['probabilities'].values()),1,abs_tol=1e-6)
            assert row['action']==max(row['probabilities'],key=row['probabilities'].get)
            assert row['latency_s']>0
        assert episode['kills']>=selected[-1]['kills_before']
    assert result['mean_kills']==statistics.mean(e['kills'] for e in episodes)
    assert result['p50_latency_ms']==round(statistics.median(r['latency_s'] for r in rows)*1000,2)
    return {'kills':[e['kills'] for e in episodes], 'mean_kills':result['mean_kills'],
            'p50_latency_ms':result['p50_latency_ms'],
            'episodes_stopped_by_deadline':sum(e['timed_out'] for e in episodes)}

if __name__=='__main__':
    print(json.dumps({stem:verify(stem) for stem in ['qwen3_5_9b_15s','jev_1_13_15s']},indent=2))
