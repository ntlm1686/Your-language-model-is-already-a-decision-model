"""Download the pinned typed-decisions test split and prepare 400 API requests.

Run with: uv run --no-project --with pyarrow python scripts/setup_typed_decisions.py
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import urllib.request

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'private/typed_decisions'
REVISION = 'f7a2487edd7a043a5441a5e9ccc7fe5ddbd9ebe8'
SOURCE_SHA256 = '4f294f218ea1da27f3efef936359389c62ea4d3973a41457732990f1d31b647c'
URL = (f'https://huggingface.co/datasets/LocalLLaMA/typed-decisions/resolve/'
       f'{REVISION}/all/test-00000-of-00001.parquet')


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    source = OUT / 'test.parquet'
    if not source.exists():
        temporary = source.with_suffix('.parquet.tmp')
        urllib.request.urlretrieve(URL, temporary)
        if hashlib.sha256(temporary.read_bytes()).hexdigest() != SOURCE_SHA256:
            temporary.unlink()
            raise ValueError('downloaded typed-decisions test split failed SHA-256 check')
        temporary.replace(source)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    assert digest == SOURCE_SHA256, (digest, SOURCE_SHA256)
    rows = pq.read_table(source).to_pylist()
    assert len(rows) == 400
    assert Counter(r['workflow'] for r in rows) == dict.fromkeys(
        ('agent_trace_observability', 'customer_service', 'invoice_processing',
         'security_incidents'), 100)
    assert len({r['id'] for r in rows}) == 400
    output = OUT / 'test.jsonl'
    with output.open('w') as handle:
        for row in rows:
            assert row['split'] == 'test' and row['n_questions'] == 5
            state, questions, gold = (json.loads(row[k]) for k in ('state', 'questions', 'gold'))
            assert set(questions) == set(gold) and len(questions) == 5
            for name, q in questions.items():
                g = gold[name]
                assert q['type'] == g['type']
                assert abs(sum(g['probabilities'].values()) - 1) < 2e-5
            handle.write(json.dumps({'id': row['id'], 'workflow': row['workflow'],
                                     'state': state, 'questions': questions, 'gold': gold},
                                    ensure_ascii=False) + '\n')
    manifest = {'dataset': 'LocalLLaMA/typed-decisions', 'revision': REVISION,
                'source': URL, 'source_sha256': digest, 'split': 'all/test',
                'cases': 400, 'decisions': 2000}
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Prepared {len(rows)} cases in {output}; source SHA-256 {digest}')


if __name__ == '__main__':
    main()
