"""Run Qwen3.5-9B or hosted Jev on the pinned typed-decisions test set.

The Qwen path uses the released model and no fitted parameters. It saves both
the existing rotated-logprob distribution and the mean token distribution from
the same forward passes. Jev receives the dataset's state and questions as one
typed API request per case. Outputs are resumable JSONL files without API keys.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
import urllib.error
import urllib.request

from qwen_jev_api import LETTERS, compile_request, render, softmax


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / 'private/typed_decisions/test.jsonl'
OUT = ROOT / 'results/typed_decisions'


def load_cases() -> list[dict]:
    rows = [json.loads(line) for line in INPUT.read_text().splitlines() if line]
    assert len(rows) == 400 and len({r['id'] for r in rows}) == 400
    return rows


def input_hash(row: dict) -> str:
    data = json.dumps({'state': row['state'], 'questions': row['questions']},
                      ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(data.encode()).hexdigest()


def qwen_case(row: dict, scorer) -> dict:
    import torch

    questions = compile_request({'state': row['state'], 'questions': row['questions']},
                                model_name='qwen3.5-9b')
    answers = {}
    start = time.perf_counter()
    input_tokens = 0
    rotated_prompts = 0
    for question in questions:
        keys = question['keys']
        n = len(keys)
        assert 2 <= n <= len(LETTERS)
        options = [f'{key}: {description}' for key, description in zip(keys, question['options'])]
        rotations = []
        for shift in range(n):
            order = list(range(shift, n)) + list(range(shift))
            shown = '\n'.join(f'{LETTERS[i]}. {options[j]}' for i, j in enumerate(order))
            content = ('Read the state and answer the question. Select the best option. '
                       'Return one letter only.\n\nState:\n' + render(row['state']) +
                       '\n\nQuestion:\n' + render(question['question']) +
                       '\n\nOptions:\n' + shown + '\n\nAnswer:')
            rotations.append((order, scorer._encode(content)))
        lengths = [len(ids) for _, ids in rotations]
        token_ids = torch.full((n, max(lengths)), scorer.tokenizer.pad_token_id,
                               dtype=torch.long, device=scorer.device)
        mask = torch.zeros_like(token_ids)
        for i, (_, ids) in enumerate(rotations):
            token_ids[i, :len(ids)] = torch.tensor(ids, device=scorer.device)
            mask[i, :len(ids)] = 1
        with torch.inference_mode():
            hidden = scorer.backbone(input_ids=token_ids, attention_mask=mask,
                                     use_cache=False, return_dict=True).last_hidden_state
            last = hidden[torch.arange(n, device=scorer.device),
                          torch.tensor(lengths, device=scorer.device) - 1]
            logits = scorer.head(last)[:, scorer.letter_ids[:n]].float()
            rotation_logprobs = torch.log_softmax(logits, dim=-1).cpu().tolist()
        sums, means = [0.0] * n, [0.0] * n
        for (order, _), logprobs in zip(rotations, rotation_logprobs):
            for index, value in zip(order, logprobs):
                sums[index] += value
                means[index] += math.exp(value) / n
        aggregate = softmax(sums)
        selected = keys[max(range(n), key=aggregate.__getitem__)]
        assert math.isclose(sum(means), 1.0, abs_tol=1e-5)
        answers[question['id']] = {
            'type': question['kind'], 'selected': selected,
            'aggregate': dict(zip(keys, aggregate)),
            'mean_token': dict(zip(keys, means)),
        }
        input_tokens += sum(lengths)
        rotated_prompts += n
    torch.cuda.synchronize(scorer.device)
    return {'id': row['id'], 'workflow': row['workflow'], 'input_sha256': input_hash(row),
            'model': 'Qwen/Qwen3.5-9B', 'answers': answers,
            'latency_s': time.perf_counter() - start, 'input_tokens': input_tokens,
            'rotated_prompts': rotated_prompts, 'gpu_forwards': len(questions)}


def jev_case(row: dict, key: str, model: str) -> dict:
    payload = json.dumps({'model': model, 'state': row['state'],
                          'questions': row['questions']}, ensure_ascii=False).encode()
    request = urllib.request.Request(
        'https://api.typesafe.ai/v1/systemone', data=payload,
        headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'})
    for attempt in range(8):
        start = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                result = json.load(response)
            elapsed = time.perf_counter() - start
            assert result['model'] == model, result.get('model')
            assert set(result['answers']) == set(row['questions'])
            return {'id': row['id'], 'workflow': row['workflow'],
                    'input_sha256': input_hash(row), 'model': result['model'],
                    'answers': result['answers'], 'usage': result.get('usage'),
                    'latency_s': elapsed}
        except urllib.error.HTTPError as exc:
            if exc.code not in (429, 500, 502, 503, 504) or attempt == 7:
                raise
        except (TimeoutError, urllib.error.URLError):
            if attempt == 7:
                raise
        time.sleep(min(2 ** attempt, 30))
    raise AssertionError('unreachable')


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('qwen', 'jev'))
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--nshards', type=int, default=1)
    p.add_argument('--limit', type=int)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--key-file', type=Path)
    p.add_argument('--model', default='jev-1.13.0')
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    assert 0 <= a.shard < a.nshards
    cases = load_cases()
    if a.limit is not None:
        cases = cases[:a.limit]
    cases = [row for i, row in enumerate(cases) if i % a.nshards == a.shard]
    output = a.output or OUT / (f'{a.mode}_shard{a.shard}of{a.nshards}.jsonl')
    output.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(line)['id'] for line in output.read_text().splitlines()} if output.exists() else set()
    assert done <= {r['id'] for r in cases}
    print(f'{a.mode} shard {a.shard}/{a.nshards}: {len(cases)} selected, '
          f'{len(done)} already complete', flush=True)
    if len(done) == len(cases):
        return
    if a.mode == 'qwen':
        from serve_qwen35_jev import Qwen35Scorer
        scorer = Qwen35Scorer(device=a.device, max_length=8192, fallback_margin=0)
        kernels = [m.chunk_gated_delta_rule.__module__ for m in scorer.backbone.modules()
                   if hasattr(m, 'chunk_gated_delta_rule')]
        if not kernels or not all(name.startswith('fla.') for name in kernels):
            raise RuntimeError(f'optimized FLA kernel is inactive: {kernels}')
        key = None
    else:
        if a.key_file:
            key = a.key_file.read_text().strip()
        else:
            key = os.environ.get('TYPESAFE_API_KEY', '').strip()
        if not key:
            p.error('Jev requires --key-file or TYPESAFE_API_KEY')
        scorer = None
    with output.open('a') as handle:
        for i, row in enumerate(cases, 1):
            if row['id'] in done:
                continue
            record = qwen_case(row, scorer) if a.mode == 'qwen' else jev_case(row, key, a.model)
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
            handle.flush()
            if i % 10 == 0 or i == len(cases):
                print(f'{a.mode} shard {a.shard}: {i}/{len(cases)}', flush=True)


if __name__ == '__main__':
    main()
