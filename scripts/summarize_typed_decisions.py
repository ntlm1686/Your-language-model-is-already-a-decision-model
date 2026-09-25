"""Validate paired typed-decisions runs and score their soft gold distributions."""
from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path
import random
import statistics

from eval_typed_decisions import INPUT, OUT, input_hash, load_cases


METHODS = ('qwen_current', 'qwen_hybrid', 'qwen_direct', 'jev')


def load_run(prefix: str, expected: int) -> dict[str, dict]:
    files = sorted(OUT.glob(f'{prefix}_shard*of*.jsonl'))
    assert files, prefix
    rows = [json.loads(line) for path in files for line in path.read_text().splitlines() if line]
    result = {r['id']: r for r in rows}
    assert len(rows) == len(result) == expected, (prefix, len(rows), len(result))
    return result


def normalize(raw: dict, labels: list[str], *, rounding: bool) -> list[float]:
    assert set(raw) == set(labels), (raw, labels)
    p = [float(raw[k]) for k in labels]
    assert all(math.isfinite(x) and 0 <= x <= 1 for x in p)
    assert abs(sum(p) - 1) <= (.0051 * len(p) if rounding else 2e-5)
    return [x / sum(p) for x in p]


def jev_distribution(answer: dict, labels: list[str], kind: str) -> tuple[list[float], str]:
    assert answer['type'] == kind
    if kind == 'noul':
        true = answer['noul']
        raw = {'false': 1 - true, 'true': true}
        probs = normalize(raw, labels, rounding=True)
        selected = 'true' if true >= .5 else 'false'
    else:
        probs = normalize(answer['probabilities'], labels, rounding=True)
        selected = answer['choice'] if kind == 'choice' else labels[max(range(len(labels)), key=probs.__getitem__)]
    assert selected in labels
    return probs, selected


def question_metrics(records: list[dict]) -> dict:
    n = len(records)
    assert n
    bins = [[] for _ in range(10)]
    for r in records:
        bins[min(int(r['confidence'] * 10), 9)].append(r)
    ece = sum(len(b) / n * abs(statistics.mean(x['confidence'] for x in b) -
                                statistics.mean(x['correct'] for x in b)) for b in bins if b)
    metrics = {'n': n,
               'accuracy': statistics.mean(r['correct'] for r in records),
               'soft_accuracy': statistics.mean(r['soft_accuracy'] for r in records),
               'brier': statistics.mean(r['brier'] for r in records),
               'total_variation': statistics.mean(r['total_variation'] for r in records),
               'kl_gold_to_model_eps_1e_12': statistics.mean(r['kl'] for r in records),
               'ece_10': ece}
    scores = [r for r in records if r['type'] == 'score']
    if scores:
        metrics['score_mae'] = statistics.mean(r['score_error'] for r in scores)
        metrics['within_one_level'] = statistics.mean(r['within_one_level'] for r in scores)
    return metrics


def main() -> None:
    cases = load_cases()
    uniform_brier = statistics.mean(
        sum((1 / len(g['probabilities']) - target) ** 2
            for target in normalize(g['probabilities'], list(g['probabilities']), rounding=True))
        for row in cases for g in row['gold'].values())
    assert round(uniform_brier, 3) == .238, uniform_brier
    qwen = load_run('qwen', len(cases))
    jev = load_run('jev', len(cases))
    assert set(qwen) == set(jev) == {r['id'] for r in cases}
    all_records = defaultdict(list)
    for row in cases:
        ident = row['id']
        q, j = qwen[ident], jev[ident]
        digest = input_hash(row)
        assert q['input_sha256'] == j['input_sha256'] == digest
        assert q['model'] == 'Qwen/Qwen3.5-9B' and j['model'] == 'jev-1.13.0'
        assert set(q['answers']) == set(j['answers']) == set(row['questions'])
        for name, gold in row['gold'].items():
            kind = row['questions'][name]['type']
            assert gold['type'] == kind
            labels = list(gold['probabilities'])
            target = normalize(gold['probabilities'], labels, rounding=True)
            qr = q['answers'][name]
            assert qr['type'] == kind and qr['selected'] in labels
            aggregate = normalize(qr['aggregate'], labels, rounding=False)
            mean_token = normalize(qr['mean_token'], labels, rounding=False)
            direct_selected = labels[max(range(len(labels)), key=mean_token.__getitem__)]
            assert qr['selected'] == labels[max(range(len(labels)), key=aggregate.__getitem__)]
            jev_probs, jev_selected = jev_distribution(j['answers'][name], labels, kind)
            modes = {
                'qwen_current': (aggregate, qr['selected']),
                'qwen_hybrid': (mean_token, qr['selected']),
                'qwen_direct': (mean_token, direct_selected),
                'jev': (jev_probs, jev_selected),
            }
            for method, (probs, selected) in modes.items():
                chosen_index = labels.index(selected)
                entry = {'case_id': ident, 'workflow': row['workflow'], 'question': name, 'type': kind,
                         'correct': selected == gold['label'],
                         'confidence': probs[chosen_index],
                         'soft_accuracy': sum(p * t for p, t in zip(probs, target)),
                         'brier': sum((p - t) ** 2 for p, t in zip(probs, target)),
                         'total_variation': .5 * sum(abs(p - t) for p, t in zip(probs, target)),
                         'kl': sum(t * math.log(t / max(p, 1e-12)) for p, t in zip(probs, target) if t)}
                if kind == 'score':
                    predicted_score = sum(i * p for i, p in enumerate(probs))
                    entry['score_error'] = abs(predicted_score - gold['score'])
                    entry['within_one_level'] = abs(int(selected) - int(gold['label'])) <= 1
                all_records[method].append(entry)
    summary = {'dataset': 'LocalLLaMA/typed-decisions',
               'revision': json.loads((INPUT.parent / 'manifest.json').read_text())['revision'],
               'split': 'all/test', 'cases': len(cases), 'decisions': len(cases) * 5,
               'uniform_reference_brier': uniform_brier,
               'definitions': {'brier': 'Mean per-decision sum over labels of (model probability - soft gold probability)^2',
                               'kl': 'Mean KL(gold || model), clipping model probabilities at 1e-12',
                               'ece_10': '10 equal-width bins of selected-label probability versus hard-label accuracy',
                               'latency': 'Median case time: local Qwen Python scoring or hosted Jev HTTP round trip; neither includes model loading'},
               'models': {}}
    for method, records in all_records.items():
        model_rows = qwen if method.startswith('qwen') else jev
        by_workflow = defaultdict(list)
        by_type = defaultdict(list)
        by_question = defaultdict(list)
        for record in records:
            by_workflow[record['workflow']].append(record)
            by_type[record['type']].append(record)
            by_question[(record['workflow'], record['question'])].append(record)
        summary['models'][method] = {
            'overall': question_metrics(records),
            'by_workflow': {k: question_metrics(v) for k, v in sorted(by_workflow.items())},
            'by_type': {k: question_metrics(v) for k, v in sorted(by_type.items())},
            'ece_10_macro_over_20_questions': statistics.mean(
                question_metrics(v)['ece_10'] for v in by_question.values()),
            'median_case_ms': 1000 * statistics.median(r['latency_s'] for r in model_rows.values()),
            'mean_input_tokens_per_case': statistics.mean(
                r.get('input_tokens', (r.get('usage') or {}).get('input_tokens', 0)) for r in model_rows.values()),
        }
    paired = {}
    for method in ('qwen_hybrid', 'jev'):
        grouped = defaultdict(list)
        for record in all_records[method]:
            grouped[record['case_id']].append(record)
        assert all(len(grouped[row['id']]) == 5 for row in cases)
        paired[method] = {ident: {'accuracy': statistics.mean(r['correct'] for r in group),
                                  'brier': statistics.mean(r['brier'] for r in group)}
                          for ident, group in grouped.items()}
    rng = random.Random(42)
    draws = {'accuracy': [], 'brier': []}
    ids = [row['id'] for row in cases]
    for _ in range(1000):
        sample = [ids[rng.randrange(len(ids))] for _ in ids]
        for metric in draws:
            draws[metric].append(statistics.mean(
                paired['jev'][ident][metric] - paired['qwen_hybrid'][ident][metric]
                for ident in sample))
    summary['paired_bootstrap'] = {
        'description': 'Jev minus Qwen hybrid; 1,000 paired case resamples, seed 42',
        'interval_95': {metric: [sorted(values)[24], sorted(values)[974]]
                        for metric, values in draws.items()}}
    OUT.mkdir(exist_ok=True)
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    for method in METHODS:
        m = summary['models'][method]
        o = m['overall']
        print(f'{method:14} accuracy={o["accuracy"]:.3f} '
              f'Brier={o["brier"]:.3f} KL={o["kl_gold_to_model_eps_1e_12"]:.3f} '
              f'ECE={o["ece_10"]:.3f} p50={m["median_case_ms"]:.0f}ms')


if __name__ == '__main__':
    main()
