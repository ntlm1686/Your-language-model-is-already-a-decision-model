"""Run five timed Qwen Doom seeds on independent local serving GPUs."""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seconds',type=float,default=15)
    p.add_argument('--ports',type=int,nargs='+',default=[8810,8811,8812,8813])
    p.add_argument('--output-prefix',type=Path,default=ROOT/'results/doom/qwen3_5_9b_15s')
    p.add_argument('--parts',type=Path,default=ROOT/'private/doom_qwen15_parts')
    args=p.parse_args();args.parts.mkdir(parents=True,exist_ok=True)
    policy='api_position_none'
    def group(i):
        for seed in range(100+i,105,len(args.ports)):
            command=[sys.executable,str(ROOT/'scripts/eval_doom_qwen.py'), '--policies',policy,
                     '--url',f'http://127.0.0.1:{args.ports[i]}','--endpoint-model','qwen3.5-9b',
                     '--expected-model','qwen3.5-9b','--episodes','1','--seed-base',str(seed),
                     '--max-wall-seconds',str(args.seconds),'--warmup','--hard-deadline',
                     '--output',str(args.parts/f'{seed}.json'), '--trace',str(args.parts/f'{seed}_steps.jsonl')]
            if seed==100:
                command+=['--video-policy',policy,'--video',f'/tmp/{args.output_prefix.name}_seed100.mp4']
            subprocess.run(command,check=True,cwd=ROOT)
    with ThreadPoolExecutor(max_workers=len(args.ports)) as pool:
        list(pool.map(group,range(len(args.ports))))
    all_rows=[];episodes=[];summary=None
    for seed in range(100,105):
        part=json.loads((args.parts/f'{seed}.json').read_text())
        if summary is None:summary=part
        episodes.extend(part['results'][policy]['episodes'])
        for line in (args.parts/f'{seed}_steps.jsonl').read_text().splitlines():
            row=json.loads(line);row['episode']=seed-100;all_rows.append(row)
    times=sorted(r['latency_s'] for r in all_rows)
    summary.update(seed_base=100,episodes=5)
    summary['results'][policy]={'episodes':episodes,'mean_kills':statistics.mean(e['kills'] for e in episodes),
        'mean_reward':statistics.mean(e['reward'] for e in episodes),'mean_steps':statistics.mean(e['steps'] for e in episodes),
        'p50_latency_ms':round(1000*statistics.median(times),2),'p95_latency_ms':round(1000*times[math.ceil(.95*len(times))-1],2),
        'mean_latency_ms':round(1000*statistics.mean(times),2),'action_counts':dict(Counter(r['action'] for r in all_rows)),
        'input_tokens':sum(r['input_tokens'] for r in all_rows)}
    args.output_prefix.with_suffix('.json').write_text(json.dumps(summary,indent=2)+'\n')
    Path(str(args.output_prefix)+'_steps.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in all_rows))
    print(json.dumps(summary['results'][policy]),flush=True)

if __name__=='__main__':main()
