#!/usr/bin/env python3
"""Compare CPU thread counts with the same short prompt; no harness/session writes."""
import argparse
import json
import os
import time
import urllib.request

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--model', default='qwen3.5:2b')
p.add_argument('--threads', nargs='+', type=int, default=[2, 4])
p.add_argument('--repeat', type=int, default=2)
p.add_argument('--tokens', type=int, default=32)
a = p.parse_args()
if min(a.threads) < 1 or a.repeat < 1 or a.tokens < 1:
    p.error('threads, repeat, and tokens must be positive')
host = os.environ.get('OLLAMA_HOST', 'http://127.0.0.1:11434')
if not host.startswith(('http://', 'https://')):
    host = 'http://' + host
rows = []
print('model,threads,run,prompt_tokens,output_tokens,prefill_tok_s,decode_tok_s,total_s,load_s', flush=True)
for thread in a.threads:
    for run in range(1, a.repeat + 1):
        body = {'model': a.model, 'messages': [{'role': 'user', 'content': 'List the integers from 1 to 30, separated by spaces. No commentary.'}],
                'stream': False, 'think': False, 'keep_alive': '10m',
                'options': {'num_ctx': 4096, 'num_thread': thread, 'num_predict': a.tokens, 'temperature': 0, 'seed': 42}}
        req = urllib.request.Request(host.rstrip('/') + '/api/chat', json.dumps(body).encode(), {'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=600) as response:
            d = json.load(response)
        if d.get('error'):
            raise SystemExit(d['error'])
        def rate(count, duration):
            return (d.get(count, 0) * 1e9 / d[duration]) if d.get(duration) else 0
        row = [a.model, thread, run, d.get('prompt_eval_count', 0), d.get('eval_count', 0),
               round(rate('prompt_eval_count', 'prompt_eval_duration'), 2), round(rate('eval_count', 'eval_duration'), 2),
               round(d.get('total_duration', 0) / 1e9, 2), round(d.get('load_duration', 0) / 1e9, 2)]
        print(','.join(map(str, row)), flush=True)
