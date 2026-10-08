#!/usr/bin/env python3
"""Log streamed model/thread comparisons, including first output and cache effects."""
import argparse
import csv
import json
import os
from pathlib import Path
import platform
import sys
import time
import urllib.request

FIELDS = ['model', 'threads', 'workload', 'run', 'phase', 'prompt_tokens', 'output_tokens',
          'first_output_s', 'prefill_s', 'prefill_tok_s', 'decode_tok_s', 'wall_s',
          'total_s', 'load_s', 'available_mb', 'swap_in_pages', 'swap_out_pages', 'load_1m']


def memory():
    try:
        values = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
        return int(values['MemAvailable'].split()[0]) // 1024
    except (OSError, KeyError, ValueError):
        return ''


def swap():
    try:
        values = dict(line.split() for line in Path('/proc/vmstat').read_text().splitlines())
        return int(values['pswpin']), int(values['pswpout'])
    except (OSError, KeyError, ValueError):
        return 0, 0


def request(host, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    return urllib.request.urlopen(urllib.request.Request(host + path, data,
                                   {'Content-Type': 'application/json'}), timeout=900)


def benchmark(host, model, threads, workload, run, args):
    prompt = 'List the integers from 1 to 30, separated by spaces. No commentary.'
    if workload == 'context':
        notes = '\n'.join(f'Note {i}: The project uses local tools, cached pages, concise output, and saved sessions.' for i in range(24))
        prompt = 'Read these project notes, then follow the final instruction.\n' + notes + '\n\n' + prompt
    body = {'model': model, 'messages': [{'role': 'user', 'content': prompt}],
            'stream': True, 'think': False, 'keep_alive': '10m',
            'options': {'num_ctx': args.context, 'num_thread': threads,
                        'num_predict': args.tokens, 'temperature': 0, 'seed': 42}}
    before = swap()
    started = time.monotonic()
    first, stats = None, None
    with request(host, '/api/chat', body) as response:
        for line in response:
            frame = json.loads(line)
            if frame.get('error'):
                raise RuntimeError(frame['error'])
            message = frame.get('message') or {}
            if first is None and any(message.get(key) for key in ('content', 'thinking', 'tool_calls')):
                first = time.monotonic() - started
            if frame.get('done'):
                stats = frame
    if stats is None:
        raise RuntimeError('Model stream ended without completion')
    after = swap()
    def seconds(key):
        return round(stats.get(key, 0) / 1e9, 3)
    def rate(count, duration):
        return round(stats.get(count, 0) * 1e9 / stats[duration], 2) if stats.get(duration) else 0
    return dict(zip(FIELDS, [model, threads, workload, run, 'first' if run == 1 else 'warm',
        stats.get('prompt_eval_count', 0), stats.get('eval_count', 0),
        round(first, 3) if first is not None else '', seconds('prompt_eval_duration'),
        rate('prompt_eval_count', 'prompt_eval_duration'), rate('eval_count', 'eval_duration'),
        round(time.monotonic() - started, 3), seconds('total_duration'), seconds('load_duration'),
        memory(), after[0] - before[0], after[1] - before[1],
        round(os.getloadavg()[0], 2) if hasattr(os, 'getloadavg') else '']))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models', nargs='+', default=['qwen3.5:0.8b', 'qwen3.5:2b'])
    p.add_argument('--model', help='legacy single-model option')
    p.add_argument('--threads', nargs='+', type=int, default=[2, 4])
    p.add_argument('--repeat', type=int, default=3)
    p.add_argument('--tokens', type=int, default=32)
    p.add_argument('--context', type=int, default=4096)
    p.add_argument('--workloads', nargs='+', choices=['short', 'context'], default=['short', 'context'])
    p.add_argument('--output', type=Path, help='write/flush every sample to this CSV')
    p.add_argument('--metadata', type=Path, help='record host, server, model, and benchmark settings as JSON')
    a = p.parse_args()
    if min(a.threads) < 1 or a.repeat < 1 or a.tokens < 1 or a.context < 1:
        p.error('threads, repeat, tokens, and context must be positive')
    models = [a.model] if a.model else a.models
    host = os.environ.get('OLLAMA_HOST', 'http://127.0.0.1:11434').rstrip('/')
    if not host.startswith(('http://', 'https://')):
        host = 'http://' + host
    if a.metadata:
        data = {'recorded_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'machine': platform.machine(),
                'platform': platform.system(), 'logical_cpus': os.cpu_count(), 'available_mb': memory(),
                'settings': {'models': models, 'threads': a.threads, 'repeat': a.repeat,
                             'tokens': a.tokens, 'context': a.context, 'workloads': a.workloads,
                             'think': False, 'temperature': 0, 'seed': 42},
                'notes': 'First means first sample for this workload/settings, not a guaranteed cold load. Warm repetitions may reuse prompt cache. Other clients can add queueing.'}
        with request(host, '/api/version') as response:
            data['server'] = json.load(response)
        data['models'] = {}
        for model in models:
            with request(host, '/api/show', {'model': model}) as response:
                shown = json.load(response)
            data['models'][model] = {'details': shown.get('details'), 'capabilities': shown.get('capabilities'), 'parameters': shown.get('parameters')}
        a.metadata.write_text(json.dumps(data, indent=2) + '\n')
    target = a.output.open('w', newline='') if a.output else None
    try:
        writers = [csv.DictWriter(sys.stdout, fieldnames=FIELDS)]
        if target:
            writers.append(csv.DictWriter(target, fieldnames=FIELDS))
        for writer in writers:
            writer.writeheader()
        sys.stdout.flush()
        for model in models:
            for threads in a.threads:
                for workload in a.workloads:
                    for run in range(1, a.repeat + 1):
                        row = benchmark(host, model, threads, workload, run, a)
                        for writer in writers:
                            writer.writerow(row)
                        sys.stdout.flush()
                        if target:
                            target.flush()
    finally:
        if target:
            target.close()


if __name__ == '__main__':
    main()
