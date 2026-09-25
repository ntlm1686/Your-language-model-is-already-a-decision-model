"""Re-evaluate every archived task with batched rotations and FLA kernels.

Uses the original prompt builders and scoring rules. A planning pass collects
prompts, one batch scores them, and a replay applies the original aggregation.
Timing covers prompt construction, tokenization, GPU work, and aggregation.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

os.environ['QWEN35_USE_FLA'] = '1'
import torch
from eval_qwen35_letters import LetterModel, ROOT, cases, evaluate

SUITES = ['deepswe', 'jevbench', 'webprm120', 'webprm1141',
          'phishing', 'metatool', 'when2call', 'bfcl']

class Capture:
    def __init__(self):
        self.calls = []
    def score(self, content, n, *, max_length=16384):
        self.calls.append((content, n, max_length))
        return [0.0] * n, 0, 0.0

class Replay:
    def __init__(self, calls, values):
        self.calls, self.values, self.index = calls, values, 0
    def score(self, content, n, *, max_length=16384):
        assert self.calls[self.index] == (content, n, max_length)
        value = self.values[self.index]
        self.index += 1
        return value

class BatchedModel(LetterModel):
    @torch.inference_mode()
    def batch(self, calls):
        sequences = []
        for content, n, cap in calls:
            chat = self.tokenizer.apply_chat_template(
                [{'role': 'user', 'content': content}], tokenize=False,
                add_generation_prompt=True, enable_thinking=False) + 'Answer: '
            ids = self.tokenizer.encode(chat, add_special_tokens=False)
            if len(ids) > cap:
                raise ValueError(f'{len(ids)} tokens exceeds cap {cap}')
            sequences.append(ids)
        torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        lengths = [len(ids) for ids in sequences]
        inputs = torch.full((len(calls), max(lengths)), self.tokenizer.pad_token_id,
                            dtype=torch.long, device=self.device)
        mask = torch.zeros_like(inputs)
        for i, ids in enumerate(sequences):
            inputs[i, :len(ids)] = torch.tensor(ids, device=self.device)
            mask[i, :len(ids)] = 1
        hidden = self.backbone(input_ids=inputs, attention_mask=mask,
                               use_cache=False, return_dict=True).last_hidden_state
        last = hidden[torch.arange(len(calls), device=self.device),
                      torch.tensor(lengths, device=self.device) - 1]
        logits = self.head(last)
        values = [(torch.log_softmax(logits[i, self.letter_ids[:n]].float(), dim=0).cpu().tolist(),
                   lengths[i], 0.0) for i, (_, n, _) in enumerate(calls)]
        torch.cuda.synchronize(self.device)
        return values, time.perf_counter() - start

def key(row):
    return row.get('id', row.get('task', (row.get('source'), row.get('state_idx'))))

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--nshards', type=int, default=1)
    p.add_argument('--suites', nargs='+', default=SUITES, choices=SUITES)
    p.add_argument('--output-dir', type=Path, default=ROOT/'results/qwen3_5_9b/optimized')
    args = p.parse_args()
    assert 0 <= args.shard < args.nshards
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = BatchedModel('cuda:0')
    kernels = [m.chunk_gated_delta_rule.__module__ for m in model.backbone.modules()
               if hasattr(m, 'chunk_gated_delta_rule')]
    if not kernels or not all(x.startswith('fla.') for x in kernels):
        raise RuntimeError(f'FLA kernel not active: {kernels}')
    model.batch([('Choose A or B. A. yes B. no', 2, 32768)])
    print(json.dumps({'shard': args.shard, 'gpu': torch.cuda.get_device_name(),
                      'kernels': sorted(set(kernels))}), flush=True)
    for suite in args.suites:
        rows = [x for i, x in enumerate(cases(suite)) if i % args.nshards == args.shard]
        output = args.output_dir/f'{suite}_shard{args.shard}of{args.nshards}.jsonl'
        done = {key(json.loads(x)) for x in output.read_text().splitlines()} if output.exists() else set()
        with output.open('a') as out:
            for index, row in enumerate(rows):
                if key(row) in done:
                    continue
                torch.cuda.synchronize()
                start = time.perf_counter()
                capture = Capture()
                evaluate(suite, row, capture)
                values, forward_s = model.batch(capture.calls)
                replay = Replay(capture.calls, values)
                result = evaluate(suite, row, replay)
                assert replay.index == len(capture.calls)
                result['latency_s'] = time.perf_counter() - start
                result['forward_s'] = forward_s
                result['scoring_mode'] = 'batched_rotations_fla'
                result['timing_scope'] = 'prompt_tokenization_forward_aggregation_no_http'
                result['prompt_sha256'] = [hashlib.sha256(c.encode()).hexdigest() for c, _, _ in capture.calls]
                for rotation in result.get('rotations', []):
                    rotation.pop('seconds', None)
                out.write(json.dumps(result, ensure_ascii=False)+'\n')
                out.flush()
                if (index+1) % 25 == 0 or index+1 == len(rows):
                    print(f'{suite} shard {args.shard}: {index+1}/{len(rows)}', flush=True)
        print(f'COMPLETE {suite} shard {args.shard}', flush=True)

if __name__ == '__main__':
    main()
