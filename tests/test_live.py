"""Live steering, real cancellable HTTP/process work, and terminal integration."""
import contextlib
import http.server
import io
import json
import os
from pathlib import Path
import select
import shlex
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import ty
import test_ty
from test_ty import Terminal


@contextlib.contextmanager
def control_scope(control):
    old = getattr(ty.IO_CONTEXT, 'control', None)
    ty.IO_CONTEXT.control = control
    try:
        yield
    finally:
        ty.IO_CONTEXT.control = old


class SteeringTest(unittest.TestCase):
    setUp = test_ty.HarnessTest.setUp
    tearDown = test_ty.HarnessTest.tearDown
    def test_batch_runs_all_calls_in_one_model_turn(self):
        calls = [{'function': {'name': 'read_file', 'arguments': {'path': path}}} for path in ('one', 'two')]
        with patch.object(self.agent, 'chat', side_effect=[('', '', calls, {}), ('Done', '', [], {})]), \
                patch.object(self.agent, 'execute', return_value='content') as execute, patch('sys.stderr', io.StringIO()):
            self.agent.run_task('Read both')
        self.assertEqual([c.args[1]['path'] for c in execute.call_args_list], ['one', 'two'])
        results = [m for m in self.agent.session['messages'] if m['role'] == 'tool']
        self.assertEqual(len(results), 2)

    def test_steering_skips_stale_batch_and_replans(self):
        control = self.agent.control = ty.TaskControl()
        old = [{'function': {'name': 'fetch_url', 'arguments': {'url': url}}} for url in ('https://cnn.com', 'https://apnews.com')]
        new = [{'function': {'name': 'fetch_url', 'arguments': {'url': 'https://foxnews.com'}}}]
        observed = []
        def chat(messages, **kwargs):
            observed.append(messages[:])
            if len(observed) == 1:
                control.submit('Use Fox News instead of CNN')
                return '', '', old, {}
            if len(observed) == 2:
                self.assertEqual(messages[-1]['role'], 'user')
                self.assertIn('Fox News', messages[-1]['content'])
                return '', '', new, {}
            return 'Fox report', '', [], {}
        with patch.object(self.agent, 'chat', side_effect=chat), patch.object(self.agent, 'execute', return_value='Fox content') as execute, patch('sys.stderr', io.StringIO()):
            self.assertEqual(self.agent.run_task('CNN news'), 'Fox report')
        self.assertEqual(execute.call_count, 1)
        self.assertEqual(execute.call_args.args[1]['url'], 'https://foxnews.com')
        skipped = [m for m in self.agent.session['messages'] if m['role'] == 'tool' and m['content'].startswith('SKIPPED')]
        self.assertEqual(len(skipped), 2)

    def test_steering_arriving_during_tool_skips_rest_of_batch(self):
        self.agent.control = ty.TaskControl()
        calls = [{'function': {'name': 'read_file', 'arguments': {'path': path}}} for path in ('one', 'two')]
        def execute(*args):
            self.agent.control.submit('Actually only read one')
            return 'first result'
        with patch.object(self.agent, 'chat', side_effect=[('', '', calls, {}), ('Done', '', [], {})]), \
                patch.object(self.agent, 'execute', side_effect=execute) as run, patch('sys.stderr', io.StringIO()):
            self.agent.run_task('Read both')
        self.assertEqual(run.call_count, 1)
        self.assertTrue(any(m['content'].startswith('SKIPPED') for m in self.agent.session['messages'] if m['role'] == 'tool'))

    def test_steering_is_not_lost_when_model_finishes_without_tools(self):
        self.agent.control = ty.TaskControl()
        def first(*args, **kwargs):
            self.agent.control.submit('Also explain why')
            return 'First answer', '', [], {}
        count = 0
        def chat(*args, **kwargs):
            nonlocal count
            count += 1
            return first() if count == 1 else ('Explanation', '', [], {})
        with patch.object(self.agent, 'chat', side_effect=chat):
            result = self.agent.run_task('Question')
        self.assertEqual(result, 'Explanation')
        self.assertTrue(any(m['role'] == 'user' and m['content'] == 'Also explain why' for m in self.agent.session['messages']))

    def test_steering_at_step_limit_gets_a_fresh_tool_budget(self):
        self.agent.control = ty.TaskControl()
        calls = [{'function': {'name': 'read_file', 'arguments': {'path': 'one'}}}]
        count = 0
        def chat(*args, **kwargs):
            nonlocal count
            count += 1
            if count == 1:
                self.agent.control.submit('Use a different source')
                return '', '', calls, {}
            return 'Updated', '', [], {}
        with patch.object(self.agent, 'chat', side_effect=chat), patch.object(self.agent, 'execute') as run, patch('sys.stderr', io.StringIO()):
            self.assertEqual(self.agent.run_task('Task', max_steps=1), 'Updated')
        run.assert_not_called()
        self.assertEqual(count, 2)

    def test_repeated_cancel_does_not_erase_replacement(self):
        control = ty.TaskControl()
        control.cancel('new question')
        control.cancel()
        self.assertEqual(control.replacement, 'new question')

    def test_compaction_stream_reports_progress(self):
        for i in range(3):
            self.agent.session['messages'].extend([{'role': 'user', 'content': 'task ' + str(i) + 'x' * 800},
                                                    {'role': 'assistant', 'content': 'result ' + 'y' * 800}])
        def summary(*args, **kwargs):
            kwargs['progress'](12)
            kwargs['progress'](24)
            return 'Earlier work completed.', '', [], {}
        with patch.object(self.agent, 'chat_raw', side_effect=summary), patch.object(ty, 'Spinner') as spinner, patch('sys.stderr', io.StringIO()) as output:
            self.agent.maybe_compact(force=True)
        self.assertEqual(spinner.return_value.set.call_count, 2)
        spinner.return_value.stop.assert_called_once()
        self.assertIn('context tokens', output.getvalue())
        self.assertEqual(len(self.agent.session['messages']), 4)

    def test_cancelled_compaction_preserves_history(self):
        for i in range(3):
            self.agent.session['messages'].extend([{'role': 'user', 'content': str(i)}, {'role': 'assistant', 'content': 'result'}])
        before = list(self.agent.session['messages'])
        with patch.object(self.agent, 'chat_raw', side_effect=ty.TaskCancelled), patch('sys.stderr', io.StringIO()):
            with self.assertRaises(ty.TaskCancelled):
                self.agent.maybe_compact(force=True)
        self.assertEqual(self.agent.session['messages'], before)
        self.assertEqual(self.agent.session['summary'], '')

    def test_article_links_are_preserved_and_relative_urls_resolved(self):
        raw = '<main><p>Today: <a href="/politics/story">Trump coverage</a>.</p><a href="#top">top</a></main><nav><a href="/ads">ad</a></nav>'
        result = ty._strip_html(raw, 'https://www.example.org/news')
        self.assertIn('[Trump coverage](https://www.example.org/politics/story)', result)
        self.assertNotIn('https://www.example.org/ads', result)
        self.assertNotIn('](https://www.example.org/news#top)', result)

    def test_page_tool_row_shows_source_before_query(self):
        with patch('sys.stderr', io.StringIO()) as out:
            self.ui.tool_row('fetch_url', {'url': 'https://cnn.com/', 'query': 'Trump'}, 'result', 0.1)
        self.assertIn('https://cnn.com/', out.getvalue())

    def test_fetch_rejects_search_keywords(self):
        for value in ('Trump Fox News', 'Trump'):
            with self.assertRaisesRegex(ValueError, 'web_search'):
                ty.normalize_url(value)

    def test_stream_collects_multiple_native_calls(self):
        calls = [{'function': {'name': 'read_file', 'arguments': {'path': path}}}
                 for path in ('one', 'two')]
        frames = [{'message': {'tool_calls': [call]}} for call in calls]
        frames.append({'message': {}, 'done': True})
        response = io.BytesIO(('\n'.join(json.dumps(f) for f in frames) + '\n').encode())
        self.agent.cfg.stream = True
        with patch('urllib.request.urlopen', return_value=response):
            result = self.agent.chat([], think=False, tools=self.agent.active_tools())
        self.assertEqual(result[2], calls)

    def test_raw_stream_progress_does_not_render_summary(self):
        response = io.BytesIO(b'{"message":{"content":"Short "}}\n{"message":{"content":"summary"},"done":true,"eval_count":2}\n')
        progress = []
        with patch('urllib.request.urlopen', return_value=response):
            result = self.agent.chat_raw([], progress=progress.append)
        self.assertEqual(result[0], 'Short summary')
        self.assertEqual(len(progress), 2)
        self.assertNotIn('Short summary', self.ui.stream.getvalue())


class ComposerTest(unittest.TestCase):
    def setUp(self):
        self.control = ty.TaskControl()
        self.ui = ty.UI('no', Terminal())
        self.console = ty.LiveConsole(self.ui, self.control, None)

    def test_enter_queues_and_escape_replaces_with_submitted_text(self):
        self.console.feed('use Fox News\r', now=0)
        self.assertEqual(self.control.queued(), ['use Fox News'])
        self.assertFalse(self.control.cancelled.is_set())
        self.console.feed('\x1b', now=1)
        self.assertFalse(self.control.cancelled.is_set())
        self.console.feed('', now=1.1)
        self.assertTrue(self.control.cancelled.is_set())
        self.assertEqual(self.control.replacement, 'use Fox News')

    def test_escape_submits_an_unsubmitted_draft(self):
        self.console.feed('new question\x1b', now=0)
        self.console.feed('', now=.1)
        self.assertEqual(self.control.replacement, 'new question')

    def test_arrow_sequence_is_not_escape_cancel(self):
        self.console.feed('abc\x1b[', now=0)
        self.console.feed('D', now=.01)
        self.console.feed('X', now=.02)
        self.assertEqual(self.console.draft, 'abXc')
        self.assertFalse(self.control.cancelled.is_set())

    def test_paste_preserves_newlines_and_does_not_submit_each_line(self):
        self.console.feed('\x1b[200~first\n\n**second**\x1b[20', now=0)
        self.console.feed('1~', now=.01)
        self.assertEqual(self.console.draft, 'first\n\n**second**')
        self.assertFalse(self.control.queued())
        self.console.feed('\r', now=.02)
        self.assertEqual(self.control.queued(), ['first\n\n**second**'])

    def test_ctrl_j_adds_line_and_ctrl_c_preserves_draft(self):
        self.console.feed('first\nsecond\x03', now=0)
        self.assertTrue(self.control.cancelled.is_set())
        self.assertEqual(self.console.draft, 'first\nsecond')
        self.assertEqual(self.control.replacement, '')

    def test_clear_queue_and_completion(self):
        self.control.submit('old')
        self.console.feed('/clear-queue\r')
        self.assertFalse(self.control.queued())
        self.console.feed('/ui v\t')
        self.assertEqual(self.console.draft, '/ui verbose ')

    def test_approval_uses_separate_input_and_preserves_draft(self):
        self.console.draft, self.console.cursor = 'draft task', 10
        result = []
        thread = threading.Thread(target=lambda: result.append(self.console.ask('approval › ')))
        thread.start()
        deadline = time.monotonic() + 1
        while self.console.approval is None and time.monotonic() < deadline:
            time.sleep(.01)
        self.console.feed('y\r')
        thread.join(1)
        self.assertFalse(thread.is_alive())
        self.assertEqual(result, ['y'])
        self.assertEqual(self.console.draft, 'draft task')
        self.assertFalse(self.control.queued())

    def test_history_and_delete_word(self):
        self.console.history = ['old task']
        self.console.history_index = 1
        self.console.feed('new\x1b[A')
        self.assertEqual(self.console.draft, 'old task')
        self.console.feed('\x1b[B')
        self.assertEqual(self.console.draft, 'new')
        self.console.feed(' words\x17')
        self.assertEqual(self.console.draft, 'new ')


class CancelIOTest(unittest.TestCase):
    def setUp(self):
        self.started = threading.Event()
        self.release = threading.Event()
        outer = self
        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == '/headers':
                    outer.started.set()
                    outer.release.wait(3)
                self.send_response(200)
                self.end_headers()
                try:
                    self.wfile.write(b'first\n')
                    self.wfile.flush()
                    if self.path == '/body':
                        outer.started.set()
                        outer.release.wait(3)
                    self.wfile.write(b'last\n')
                except OSError:
                    pass
            def log_message(self, *args):
                pass
        self.server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.server_thread = threading.Thread(target=self.server.serve_forever, kwargs={'poll_interval': .02}, daemon=True)
        self.server_thread.start()
        self.url = 'http://127.0.0.1:' + str(self.server.server_port)

    def tearDown(self):
        self.release.set()
        self.server.shutdown()
        self.server.server_close()
        self.server_thread.join(1)

    def run_cancel_case(self, path):
        control = ty.TaskControl()
        errors = []
        def fetch():
            with control_scope(control):
                try:
                    ty.http_get(self.url + path)
                except BaseException as exc:
                    errors.append(exc)
        worker = threading.Thread(target=fetch)
        worker.start()
        self.assertTrue(self.started.wait(2))
        started = time.monotonic()
        control.cancel()
        worker.join(1)
        self.assertFalse(worker.is_alive())
        self.assertLess(time.monotonic() - started, 1)
        self.assertIsInstance(errors[0], ty.TaskCancelled)

    def test_cancel_waiting_for_headers(self):
        self.run_cancel_case('/headers')

    def test_cancel_waiting_for_body(self):
        self.run_cancel_case('/body')

    def test_successful_http_detaches_interrupts(self):
        control = ty.TaskControl()
        with control_scope(control):
            self.assertEqual(ty.http_get(self.url), 'first\nlast\n')
        self.assertFalse(control.callbacks)

    @unittest.skipUnless(os.name == 'posix', 'process groups require POSIX')
    def test_cancel_kills_shell_and_child(self):
        control = ty.TaskControl()
        errors = []
        with tempfile.TemporaryDirectory(prefix='ty-process-test-') as root:
            pidfile = Path(root) / 'pid'
            code = "import subprocess,time; from pathlib import Path; p=subprocess.Popen(['sleep','30']); Path(%r).write_text(str(p.pid)); p.wait()" % str(pidfile)
            command = shlex.quote(sys.executable) + ' -c ' + shlex.quote(code)
            def run():
                with control_scope(control):
                    try:
                        ty.run_process(command, shell=True, timeout=30)
                    except BaseException as exc:
                        errors.append(exc)
            worker = threading.Thread(target=run)
            worker.start()
            deadline = time.monotonic() + 2
            while not pidfile.exists() and time.monotonic() < deadline:
                time.sleep(.01)
            self.assertTrue(pidfile.exists())
            child = int(pidfile.read_text())
            control.cancel()
            worker.join(2)
            self.assertFalse(worker.is_alive())
            self.assertIsInstance(errors[0], ty.TaskCancelled)
            stat = Path('/proc') / str(child) / 'stat'
            if stat.exists():
                self.assertIn(stat.read_text().split()[2], ('Z', 'X'))


@unittest.skipUnless(ty.termios is not None, 'POSIX pseudo-terminal required')
class TerminalIntegrationTest(unittest.TestCase):
    def run_terminal_case(self, replace, compaction=False):
        import pty
        master, slave = pty.openpty()
        before = ty.termios.tcgetattr(slave)
        with tempfile.TemporaryDirectory(prefix='ty-pty-test-') as root:
            fixture = '''
import io,json,sys,time
from types import SimpleNamespace
import ty
cfg=SimpleNamespace(**ty.DEFAULTS)
ui=ty.UI('no')
a=ty.Agent(cfg,sys.argv[1],ui,ty.new_session(sys.argv[1],cfg.model))
count=[0]
def chat(messages,**kwargs):
    count[0]+=1
    if count[0]==1:
        ui.out('WAITING_FOR_INPUT')
        c=ty.task_control()
        end=time.monotonic()+4
        while time.monotonic()<end:
            c.check()
            if not int(sys.argv[2]) and c.queued():
                break
            time.sleep(.01)
        return '', '', [{'function':{'name':'read_file','arguments':{'path':'old.txt'}}}], {}
    latest=next(m['content'] for m in reversed(messages) if m['role']=='user')
    ui.out('NEW_QUERY:'+latest)
    return 'Done','',[],{}
a.chat=chat
a.execute=lambda *args: ui.out('OLD_TOOL_EXECUTED') or 'old'
def compact(force=False):
    if not force:
        return
    count[0]=1
    ui.out('WAITING_FOR_INPUT')
    c=ty.task_control()
    end=time.monotonic()+4
    while time.monotonic()<end:
        c.check()
        time.sleep(.01)
a.maybe_compact=compact if int(sys.argv[3]) else lambda **kwargs: None
if int(sys.argv[3]):
    ty.handle_command(a,ui,'/compact')
else:
    exit_requested,draft=ty.run_interactive_task(a,'initial')
print('TASK_DONE',flush=True)
'''
            env = dict(os.environ, XDG_DATA_HOME=root + '/data', XDG_CONFIG_HOME=root + '/config', XDG_CACHE_HOME=root + '/cache')
            child = subprocess.Popen([sys.executable, '-c', fixture, root, str(int(replace)), str(int(compaction))], stdin=slave, stdout=slave, stderr=slave, env=env)
            capture = b''
            try:
                deadline = time.monotonic() + 5
                while b'WAITING_FOR_INPUT' not in capture and time.monotonic() < deadline:
                    if select.select([master], [], [], .1)[0]:
                        capture += os.read(master, 65536)
                self.assertIn(b'WAITING_FOR_INPUT', capture)
                os.write(master, b'use Fox News\r' + (b'\x1b' if replace else b''))
                deadline = time.monotonic() + 5
                while b'TASK_DONE' not in capture and time.monotonic() < deadline:
                    if select.select([master], [], [], .1)[0]:
                        capture += os.read(master, 65536)
                child.wait(timeout=2)
                self.assertEqual(child.returncode, 0, capture.decode(errors='replace'))
                self.assertIn(b'NEW_QUERY:use Fox News', capture)
                self.assertNotIn(b'OLD_TOOL_EXECUTED', capture)
                self.assertEqual(ty.termios.tcgetattr(slave), before)
            finally:
                if child.poll() is None:
                    child.kill()
                    child.wait()
                os.close(master)
                os.close(slave)

    def test_enter_steers_in_real_terminal(self):
        self.run_terminal_case(False)

    def test_escape_replaces_in_real_terminal(self):
        self.run_terminal_case(True)

    def test_escape_during_manual_compaction_starts_replacement(self):
        self.run_terminal_case(True, compaction=True)


if __name__ == '__main__':
    unittest.main()
