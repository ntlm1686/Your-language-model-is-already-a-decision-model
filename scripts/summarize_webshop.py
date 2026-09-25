"""Validate saved WebShop episodes and summarize complete or ongoing runs."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics
from eval_webshop import PROTOCOL, digest, write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('results/webshop'))
    p.add_argument('--require-complete',action='store_true')
    a=p.parse_args()
    manifest=json.loads((a.root/'manifest.json').read_text())
    assert manifest['protocol']==PROTOCOL
    goals={g['id']:g for g in manifest['goals']}
    summary={}
    for model in ('qwen','jev','open_jev'):
        rows=[json.loads(f.read_text()) for f in sorted((a.root/model).glob('[0-9][0-9][0-9].json'))]
        ids=[r['task_id'] for r in rows]
        assert len(set(ids))==len(ids) and set(ids)<=set(range(500))
        if a.require_complete:
            assert set(ids)==set(range(500)),(model,len(ids))
        times=[]
        methods=Counter()
        choice_score_disagreements=0
        candidate_counts=[]
        for r in rows:
            assert r['model']==model and r['protocol_sha256']==digest(PROTOCOL)
            assert r['instruction_sha256']==goals[r['task_id']]['sha256']
            assert 0<=r['reward']<=1 and r['score']==100*r['reward']
            assert r['success']==math.isclose(r['reward'],1,abs_tol=1e-9)
            assert r['steps']==sum('action' in t for t in r['trace'])
            for field in ('input_tokens','output_tokens'):
                assert r[field]==sum((t.get('response') or {}).get('usage',{}).get(field,0) for t in r['trace'])
            assert math.isclose(r['decision_seconds'],sum(t['decision_seconds'] for t in r['trace']),abs_tol=1e-6)
            assert r['elapsed_seconds']>=r['decision_seconds']
            if r['termination']=='purchase':
                assert r['trace'][-1]['done'] and r['trace'][-1]['reward']==r['reward']
            else:
                assert r['reward']==0
            for t in r['trace']:
                if 'request' in t:
                    assert t['request_sha256']==digest(t['request'])
                    if t.get('response'):
                        answer=t['response']['answers']['action']
                        chosen=answer['choice']
                        gap=max(answer['probabilities'].values())-answer['probabilities'][chosen]
                        if model!='jev':
                            assert math.isclose(gap,0,abs_tol=1e-9)
                        choice_score_disagreements+=gap>1e-9
                        assert t['action']==t['request']['questions']['action']['criteria'][chosen]
                        times.append(t['decision_seconds'])
                        methods[t['response'].get('metadata',{}).get('method','hosted')]+=1
                        candidate_counts.append(len(t['request']['questions']['action']['criteria']))
        summary[model]={'completed':len(rows),'expected':500,'complete':len(rows)==500,
            'successes':sum(r['success'] for r in rows),
            'success_rate':statistics.mean(r['success'] for r in rows) if rows else None,
            'mean_score':statistics.mean(r['score'] for r in rows) if rows else None,
            'mean_episode_seconds':statistics.mean(r['elapsed_seconds'] for r in rows) if rows else None,
            'median_decision_seconds':statistics.median(times) if times else None,
            'mean_input_tokens':statistics.mean(r['input_tokens'] for r in rows) if rows else None,
            'mean_output_tokens':statistics.mean(r['output_tokens'] for r in rows) if rows else None,
            'mean_steps':statistics.mean(r['steps'] for r in rows) if rows else None,
            'decision_calls':len(times), 'api_choice_score_disagreements':choice_score_disagreements, 'scoring_methods':dict(methods),
            'max_candidates':max(candidate_counts) if candidate_counts else None,
            'terminations':dict(Counter(r['termination'] for r in rows))}
    write_json(a.root/'summary.json',summary)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    main()
