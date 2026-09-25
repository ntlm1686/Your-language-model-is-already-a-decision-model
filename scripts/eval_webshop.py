"""Closed-loop WebShop evaluation with Jev-format decision APIs.

Use full official catalog and test goal indices 0..499. Search uses only the
visible instruction, never the hidden goal query, attributes or target ASIN.
"""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'external/webshop'))
QUESTION = ('Which next action best completes the shopping instruction? Inspect relevant products, '
            'select the requested options, and buy a matching product within the price limit. '
            'Avoid repeating actions that made no progress.')
PROTOCOL = {'version': 1, 'test_ids': [0, 499], 'catalog': 'full',
            'search': 'visible_instruction_verbatim', 'max_steps': 100,
            'repeat_state_action_limit': 3, 'history_actions': 10,
            'observation': 'text', 'truncate': False, 'seed': 233}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def write_json(path, data):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def request(url, body, key, expected):
    headers = {'Content-Type': 'application/json'}
    if key:
        headers['Authorization'] = 'Bearer ' + key
    attempts = []
    for attempt in range(4):
        start = time.perf_counter()
        try:
            req = urllib.request.Request(url.rstrip('/') + '/v1/systemone',
                data=json.dumps(body, ensure_ascii=False).encode(), headers=headers)
            with urllib.request.urlopen(req, timeout=180) as r:
                result = json.load(r)
            attempts.append({'seconds': time.perf_counter()-start, 'status': 200})
            break
        except urllib.error.HTTPError as e:
            attempts.append({'seconds': time.perf_counter()-start, 'status': e.code})
            if e.code in (400, 413, 422):
                return None, attempts, e.read().decode()[:2000]
            if e.code not in (429, 500, 502, 503, 504) or attempt == 3:
                raise
            time.sleep(2**attempt)
        except (TimeoutError, urllib.error.URLError):
            attempts.append({'seconds': time.perf_counter()-start, 'status': 'network_error'})
            if attempt == 3:
                raise
            time.sleep(2**attempt)
    if result.get('model') != expected:
        raise ValueError(('model mismatch', result.get('model'), expected))
    answer = result['answers']['action']
    probs = answer['probabilities']
    criteria = body['questions']['action']['criteria']
    assert set(probs) == set(criteria)
    assert all(math.isfinite(v) and 0 <= v <= 1 for v in probs.values())
    # Hosted Jev rounds each displayed probability to two decimals.
    tolerance = len(probs)*0.005 + 1e-8 if expected == 'jev-1.13.0' else 1e-5
    assert math.isclose(sum(probs.values()), 1, abs_tol=tolerance)
    if expected != 'jev-1.13.0' and not math.isclose(probs[answer['choice']], max(probs.values()), abs_tol=1e-9):
        raise ValueError('API choice disagrees with probabilities: ' + json.dumps(answer))
    return result, attempts, None


def run_episode(env, task_id, model, url, key, expected, output):
    obs, _ = env.reset(task_id)
    instruction = env.get_instruction_text().removeprefix('Instruction: ')
    history, trace, repeats = [], [], Counter()
    start = time.perf_counter()
    reward, reason = 0.0, 'step_limit'
    input_tokens = output_tokens = 0
    for step in range(PROTOCOL['max_steps']):
        available = env.get_available_actions()
        if available['has_search_bar']:
            # Fixed search rule, shared by all policies. No hidden goal fields.
            action = 'search[' + instruction.lower().strip() + ']'
            row = {'step': step, 'observation': obs, 'action': action,
                   'policy': 'fixed_search', 'decision_seconds': 0.0}
        else:
            actions = ['click[' + x + ']' for x in available['clickables'] if x != 'search']
            if not actions:
                reason = 'no_actions'
                break
            criteria = {str(i): action for i, action in enumerate(actions)}
            state = 'Shopping instruction:\n' + instruction + '\n\nCurrent page:\n' + obs
            if history:
                state += '\n\nRecent actions:\n' + '\n'.join(history[-10:])
            body = {'state': state, 'questions': {'action': {
                'type': 'choice', 'instructions': QUESTION, 'criteria': criteria}}}
            if model == 'jev':
                body['model'] = 'jev-1.13.0'
            call_start = time.perf_counter()
            result, attempts, error = request(url, body, key, expected)
            row = {'step': step, 'observation': obs, 'request': body,
                   'request_sha256': digest(body), 'attempts': attempts,
                   'decision_seconds': time.perf_counter()-call_start,
                   'response': result}
            if error:
                row['error'] = error
                trace.append(row)
                reason = 'request_rejected'
                break
            answer = result['answers']['action']
            row['choice_probability_gap'] = max(answer['probabilities'].values()) - answer['probabilities'][answer['choice']]
            action = criteria[answer['choice']]
            row['action'] = action
            input_tokens += result.get('usage', {}).get('input_tokens', 0)
            output_tokens += result.get('usage', {}).get('output_tokens', 0)
        signature = digest([obs, action])
        repeats[signature] += 1
        obs, reward, done, _ = env.step(action)
        row.update(reward=reward, done=done, elapsed_seconds=time.perf_counter()-start)
        history.append(action)
        trace.append(row)
        if done:
            reason = 'purchase'
            break
        if repeats[signature] >= PROTOCOL['repeat_state_action_limit']:
            reason = 'repeated_state_action'
            break
    elapsed = time.perf_counter()-start
    score = float(reward) if reason == 'purchase' else 0.0
    assert 0 <= score <= 1
    record = {'task_id': task_id, 'model': model, 'protocol_sha256': digest(PROTOCOL),
        'instruction': instruction, 'instruction_sha256': digest(instruction),
        'reward': score, 'score': score*100, 'success': math.isclose(score, 1, abs_tol=1e-9),
        'termination': reason, 'steps': len(history), 'elapsed_seconds': elapsed,
        'decision_seconds': sum(x['decision_seconds'] for x in trace),
        'input_tokens': input_tokens, 'output_tokens': output_tokens, 'trace': trace}
    write_json(output / f'{task_id:03d}.json', record)
    print(json.dumps({k: v for k, v in record.items() if k not in ('trace', 'instruction')}), flush=True)


def worker(server, model, worker_id, urls, task_ids, key, output):
    from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv
    env = WebAgentTextEnv(observation_mode='text', server=server,
        session_prefix=f'{model}_{worker_id}_', human_goals=True)
    expected = {'qwen': 'qwen3.5-9b', 'open_jev': 'Qwen/Qwen3.5-9B', 'jev': 'jev-1.13.0'}[model]
    target = output / model
    target.mkdir(parents=True, exist_ok=True)
    # Exclude one synthetic warmup from task timing.
    warm = {'state': 'A shop sells red and blue shirts. The user wants red.',
        'questions': {'action': {'type': 'choice', 'instructions': 'Pick the matching item.',
        'criteria': {'0': 'red shirt', '1': 'blue shirt'}}}}
    if model == 'jev':
        warm['model'] = 'jev-1.13.0'
    request(urls[worker_id], warm, key, expected)
    for task_id in task_ids[worker_id::len(urls)]:
        path = target / f'{task_id:03d}.json'
        if path.exists():
            old = json.loads(path.read_text())
            assert old['protocol_sha256'] == digest(PROTOCOL)
            assert old['instruction_sha256'] == digest(server.goals[task_id]['instruction_text'])
            continue
        run_episode(env, task_id, model, urls[worker_id], key, expected, target)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--limit', type=int, default=500)
    p.add_argument('--start', type=int, default=0)
    p.add_argument('--models', nargs='+', choices=['qwen','open_jev','jev'], default=['qwen','open_jev','jev'])
    p.add_argument('--qwen-urls', nargs='+', default=['http://127.0.0.1:8830','http://127.0.0.1:8831'])
    p.add_argument('--open-jev-urls', nargs='+', default=['http://127.0.0.1:8832','http://127.0.0.1:8833'])
    p.add_argument('--jev-url', default='https://api.typesafe.ai')
    p.add_argument('--jev-workers', type=int, default=1)
    p.add_argument('--api-key-file', type=Path, default=ROOT/'private/doom_api_key')
    p.add_argument('--output', type=Path, default=ROOT/'results/webshop')
    a = p.parse_args()
    assert 0 <= a.start < a.start + a.limit <= 500
    assert 1 <= a.jev_workers <= 4
    random.seed(PROTOCOL['seed'])
    from web_agent_site.envs.web_agent_text_env import SimServer
    from web_agent_site.utils import DEFAULT_FILE_PATH
    server = SimServer('http://127.0.0.1:3000', DEFAULT_FILE_PATH, human_goals=True)
    a.output.mkdir(parents=True, exist_ok=True)
    manifest = {'protocol': PROTOCOL, 'goals': [
        {'id': i, 'instruction': g['instruction_text'], 'sha256': digest(g['instruction_text'])}
        for i, g in enumerate(server.goals[:500])], 'catalog_size': len(server.all_products), 'total_goals': len(server.goals),
        'source_revision': '64fa2a5c15c7daa698b9ac93f5bb5437b634c9bd',
        'data_revision': '0129d4a81dbdb827e76afd20a1e2c38b61098613'}
    manifest_path = a.output/'manifest.json'
    if manifest_path.exists():
        assert json.loads(manifest_path.read_text()) == manifest
    else:
        write_json(manifest_path, manifest)
    urls = {'qwen': a.qwen_urls, 'open_jev': a.open_jev_urls, 'jev': [a.jev_url] * a.jev_workers}
    key = a.api_key_file.read_text().strip() if 'jev' in a.models else None
    with ThreadPoolExecutor(max_workers=sum(len(urls[m]) for m in a.models)) as pool:
        jobs = [pool.submit(worker, server, m, i, urls[m], list(range(a.start,a.start+a.limit)),
                 key if m=='jev' else None, a.output) for m in a.models for i in range(len(urls[m]))]
        errors = []
        for job in as_completed(jobs):
            try:
                job.result()
            except Exception as error:
                print('WORKER ERROR: ' + repr(error), flush=True)
                errors.append(error)
        if errors:
            raise RuntimeError(f'{len(errors)} workers failed; completed episodes are resumable') from errors[0]

if __name__ == '__main__':
    main()
