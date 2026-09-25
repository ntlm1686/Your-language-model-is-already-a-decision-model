"""Score uncalibrated Qwen option-token probabilities against hosted Jev probabilities.

Uses saved per-option logits/probabilities and labels; fits no temperature or model.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/calibration'
SUITES = {
    'JevBench': ('jevbench', 'results/jevbench/jev_1_13_0_fresh_results.jsonl', 231),
    'WebPRM full': ('webprm1141', 'results/webprm/jev_1141.jsonl', 1141),
    'PhishNChips': ('phishing', 'results/phishing/jev_2000.jsonl', 2000),
    'MetaTool': ('metatool', 'results/metatool/jev.jsonl', 1040),
    'When2Call': ('when2call', 'results/when2call/jev.jsonl', 600),
    'BFCL v4 Multiple': ('bfcl', 'results/bfcl/jev.jsonl', 200),
}
METHODS = ('qwen_token', 'qwen_aggregated', 'qwen_hybrid', 'jev')


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def identity(suite: str, row: dict):
    if suite == 'JevBench':
        return row.get('id', row.get('task_id'))
    if suite == 'WebPRM full':
        return (row['source'], row['state_idx'])
    return row['id']


def softmax(values: list[float]) -> list[float]:
    maximum = max(values)
    weighted = [math.exp(x - maximum) for x in values]
    total = sum(weighted)
    return [x / total for x in weighted]


def qwen_distributions(suite: str, row: dict) -> tuple[list[str], list[float], list[float]]:
    if suite in ('JevBench', 'PhishNChips'):
        labels = list(row['scores'])
    elif suite == 'WebPRM full':
        labels = [str(i) for i in range(len(row['logprobs']))]
    else:
        labels = [str(i) for i in range(len(row['scores']))]
    n = len(labels)
    if suite == 'WebPRM full':
        direct = [math.exp(x) for x in row['logprobs']]
        aggregate = direct.copy()
    else:
        direct = [0.0] * n
        rotations = row['rotations']
        assert len(rotations) >= 1
        for rotation in rotations:
            logprobs = rotation['logprobs']
            assert len(logprobs) == n
            if 'order' in rotation:
                order = [str(x) for x in rotation['order']]
                assert set(order) == set(labels)
                indices = [labels.index(x) for x in order]
            else:
                shift = rotation['shift']
                indices = [(shift + i) % n for i in range(n)]
            assert math.isclose(sum(math.exp(x) for x in logprobs), 1.0, abs_tol=2e-5)
            for index, logp in zip(indices, logprobs):
                direct[index] += math.exp(logp) / len(rotations)
        scores = row['scores']
        aggregate = softmax([scores[x] for x in labels] if isinstance(scores, dict) else scores)
    assert math.isclose(sum(direct), 1.0, abs_tol=2e-5)
    assert math.isclose(sum(aggregate), 1.0, abs_tol=2e-5)
    # The original selection rule uses the aggregated score (or the one WebPRM order).
    old_prediction = str(row['prediction'])
    assert labels[aggregate.index(max(aggregate))] == old_prediction, identity(suite,row)
    assert (old_prediction == str(row['gold'])) == row['correct']
    return labels, direct, aggregate


def jev_distribution(suite: str, row: dict, labels: list[str]) -> list[float]:
    if suite == 'JevBench':
        assert row['status'] == 'ok' and row['valid']
        raw = row['probs']
        selected = str(row['predicted'])
    else:
        assert row.get('valid', True)
        raw = row['probabilities']
        selected = str(row['prediction'])
    if suite in ('JevBench', 'PhishNChips'):
        assert set(raw) == set(labels), (suite, identity(suite,row), raw, labels)
        probs = [raw[x] for x in labels]
    else:
        alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        n = len(labels)
        assert n > 0
        assert set(raw) == set(alphabet[:n])
        probs = [raw[alphabet[i]] for i in range(n)]
    assert selected in labels
    assert all(isinstance(x,(int,float)) and math.isfinite(x) and 0 <= x <= 1 for x in probs)
    # Jev often returns 2-decimal values. Permit its rounding, without re-normalizing.
    assert abs(sum(probs) - 1.0) <= len(probs)*.005 + 1e-8
    return probs


def calibration(rows: list[dict]) -> dict:
    assert rows
    bins = [[] for _ in range(10)]
    for row in rows:
        bins[min(int(row['confidence'] * 10), 9)].append(row)
    summaries=[]
    for i, b in enumerate(bins):
        summaries.append({'lower':i/10, 'upper':(i+1)/10, 'count':len(b),
                          'confidence':sum(x['confidence'] for x in b)/len(b) if b else None,
                          'accuracy':sum(x['correct'] for x in b)/len(b) if b else None})
    n=len(rows)
    confident=[x for x in rows if x['confidence'] >= .9]
    return {'n':n, 'correct':sum(x['correct'] for x in rows),
            'accuracy':sum(x['correct'] for x in rows)/n,
            'mean_confidence':sum(x['confidence'] for x in rows)/n,
            'ece_10':sum(len(b)/n*abs(v['accuracy']-v['confidence']) for b,v in zip(bins,summaries) if b),
            'brier_multiclass':sum(x['brier'] for x in rows)/n,
            'coverage_confidence_90':len(confident)/n,
            'accuracy_confidence_90':sum(x['correct'] for x in confident)/len(confident) if confident else None,
            'bins':summaries}


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,default=OUT)
    p.add_argument('--bootstrap',type=int,default=1000,help='Paired, within-suite bootstrap samples')
    args=p.parse_args()
    data={method:defaultdict(list) for method in METHODS}
    records=[]
    for suite,(prefix,relative,expected) in SUITES.items():
        qrows=[r for shard in sorted((ROOT/'results/qwen3_5_9b/optimized').glob(prefix+'_shard*of4.jsonl'))
                 for r in read_jsonl(shard)]
        jrows=read_jsonl(ROOT/relative)
        qmap={identity(suite,x):x for x in qrows}
        jmap={identity(suite,x):x for x in jrows}
        assert len(qmap)==len(jmap)==expected and set(qmap)==set(jmap),suite
        for item_id,q in sorted(qmap.items(),key=lambda kv:str(kv[0])):
            j=jmap[item_id]
            gold=str(q['gold'])
            if suite == 'JevBench':
                # The published Jev records omit gold, but include correctness.
                assert j['correct'] == (str(j['predicted']) == gold)
            else:
                assert str(j.get('gold',gold)) == gold
                if suite in ('MetaTool','When2Call','BFCL v4 Multiple'):
                    assert q['input_sha256']==j['input_sha256']
                if suite == 'PhishNChips':
                    assert q['email_sha256']==j['email_sha256']
            labels,direct,aggregated=qwen_distributions(suite,q)
            assert gold in labels
            jev=jev_distribution(suite,j,labels)
            for method,probs in [('qwen_token',direct),('qwen_aggregated',aggregated),
                                 ('qwen_hybrid',direct),('jev',jev)]:
                if method == 'jev':
                    selected = str(j['predicted'] if suite=='JevBench' else j['prediction'])
                elif method == 'qwen_hybrid':
                    selected = labels[max(range(len(labels)),key=aggregated.__getitem__)]
                else:
                    selected = labels[max(range(len(labels)),key=probs.__getitem__)]
                index=labels.index(selected)
                record={'suite':suite,'id':str(item_id),'method':method,'gold':gold,
                        'prediction':selected,'correct':selected==gold,
                        'confidence':probs[index], 'gold_probability':probs[labels.index(gold)],
                        'brier':sum((p-(lab==gold))**2 for lab,p in zip(labels,probs)),
                        'probabilities':dict(zip(labels,probs))}
                data[method][suite].append(record)
                records.append(record)
    summary={'definitions':{
       'qwen_token':'Arithmetic mean of position-rotated, option-normalized next-token probabilities; one order for WebPRM.',
       'qwen_aggregated':'Softmax of summed option log probabilities, the archived Qwen decision rule.',
       'qwen_hybrid':'Select with aggregated log probabilities; report the selected option probability from the arithmetic mean of rotated token probabilities.',
       'jev':'Choice/Score probabilities returned by the hosted Jev 1.13.0 API, with its recorded decision.',
       'ece_10':'Ten fixed equal-width confidence bins; confidence is probability of the selected option.',
       'brier_multiclass':'Mean sum over options of (reported probability - one-hot label)^2; lower is better.'},
       'suites':{},'pooled':{}}
    for suite in SUITES:
        summary['suites'][suite]={method:calibration(data[method][suite]) for method in METHODS}
    for method in METHODS:
        summary['pooled'][method]=calibration([x for suite in SUITES for x in data[method][suite]])
    summary['macro']={method:{metric:sum(summary['suites'][suite][method][metric] for suite in SUITES)/len(SUITES)
                              for metric in ('accuracy','mean_confidence','ece_10','brier_multiclass')}
                      for method in METHODS}
    if args.bootstrap:
        rng=random.Random(42)
        draws=defaultdict(list)
        for _ in range(args.bootstrap):
            by_method=defaultdict(list)
            for suite in SUITES:
                indices=[rng.randrange(len(data['jev'][suite])) for _ in data['jev'][suite]]
                for method in METHODS:
                    by_method[method].append(calibration([data[method][suite][i] for i in indices]))
            for method in METHODS:
                for metric in ('ece_10','brier_multiclass'):
                    draws[(method,metric)].append(sum(x[metric] for x in by_method[method])/len(SUITES))
            for metric in ('ece_10','brier_multiclass'):
                for method in ('qwen_token','qwen_aggregated','qwen_hybrid'):
                    draws[(method+'_minus_jev',metric)].append(draws[(method,metric)][-1]-draws[('jev',metric)][-1])
        summary['bootstrap']={'samples':args.bootstrap,'seed':42,'design':'Paired task resampling within each suite; unweighted mean over six suites.',
            'interval_95':{f'{name}:{metric}':[sorted(values)[int(.025*(len(values)-1))],sorted(values)[int(.975*(len(values)-1))]]
                           for (name,metric),values in draws.items()}}
    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    with (args.output_dir/'records.jsonl').open('w') as out:
        for r in records:
            out.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
    for suite in SUITES:
        print(suite, 'n', summary['suites'][suite]['jev']['n'],
              'ECE',*[round(summary['suites'][suite][m]['ece_10'],3) for m in METHODS],
              'Brier',*[round(summary['suites'][suite][m]['brier_multiclass'],3) for m in METHODS])
    print('Macro ECE',*[round(summary['macro'][m]['ece_10'],3) for m in METHODS])
    print('Macro Brier',*[round(summary['macro'][m]['brier_multiclass'],3) for m in METHODS])
    print('Pooled n',summary['pooled']['jev']['n'],'ECE',*[round(summary['pooled'][m]['ece_10'],3) for m in METHODS])
    if args.bootstrap:
        print('Qwen direct - Jev ECE 95% interval',summary['bootstrap']['interval_95']['qwen_token_minus_jev:ece_10'])
        print('Qwen hybrid - Jev ECE 95% interval',summary['bootstrap']['interval_95']['qwen_hybrid_minus_jev:ece_10'])

if __name__=='__main__':
    main()
