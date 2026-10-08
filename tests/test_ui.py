"""Command discovery, model selection, history replay, stable drawing, and exits."""
import io
import json
import os
import select
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import ty
import test_ty
from test_ty import Terminal


class DiscoveryTest(unittest.TestCase):
    setUp = test_ty.HarnessTest.setUp
    tearDown = test_ty.HarnessTest.tearDown

    def test_every_command_has_a_description(self):
        self.assertEqual(set(ty.COMMANDS), set(ty.COMMAND_DESCRIPTIONS))
        matches = dict(ty.completion_suggestions('/mo'))
        self.assertIn('/mode', matches)
        self.assertIn('model', matches['/mode'])
        self.assertIn('/exit', dict(ty.completion_suggestions('/ex')))

    def test_argument_descriptions_and_permission_choices(self):
        matches = dict(ty.completion_suggestions('/approve '))
        self.assertIn('readonly', matches)
        self.assertIn('deny', matches['readonly'])
        self.assertEqual(ty.completion_options('/mode 0', '0'), ['0.8b'])

    def test_mode_is_model_selection_and_approve_keeps_permissions(self):
        with patch.object(ty, 'choose_model', return_value='qwen3.5:0.8b') as choose:
            ty.handle_command(self.agent, self.ui, '/mode')
        choose.assert_called_once_with(self.agent)
        self.assertEqual(self.cfg.model, 'qwen3.5:0.8b')
        self.assertEqual(self.cfg.mode, 'smart')
        ty.handle_command(self.agent, self.ui, '/approve readonly')
        self.assertEqual(self.cfg.mode, 'readonly')

    def test_direct_model_alias_works(self):
        ty.handle_command(self.agent, self.ui, '/mode 0.8b')
        self.assertEqual(self.cfg.model, 'qwen3.5:0.8b')

    def test_model_picker_failure_keeps_current_model(self):
        with patch.object(ty, 'http_get', side_effect=OSError('offline')), patch('sys.stderr', io.StringIO()) as errors:
            ty.handle_command(self.agent, self.ui, '/mode')
        self.assertEqual(self.cfg.model, 'qwen3.5:2b')
        self.assertIn('Could not list', errors.getvalue())

    def test_replay_restores_full_archived_and_current_text_without_inference(self):
        self.agent.session['archive'] = [
            {'role': 'user', 'content': 'OLD QUESTION\nsecond line'},
            {'role': 'assistant', 'content': '**OLD ANSWER** [link](https://example.org)'}]
        self.agent.session['messages'] = [
            {'role': 'tool', 'tool_name': 'read_file', 'content': 'FULL TOOL RESULT'},
            {'role': 'user', 'content': 'NEW QUESTION'},
            {'role': 'assistant', 'content': 'NEW ANSWER'}]
        before = json.dumps(self.agent.session)
        with patch.object(self.agent, 'chat') as chat:
            ty.replay_session(self.agent)
        out = self.ui.stream.getvalue()
        for text in ('OLD QUESTION\nsecond line', '**OLD ANSWER**', 'https://example.org', 'FULL TOOL RESULT', 'NEW QUESTION', 'NEW ANSWER'):
            self.assertIn(text, out)
        self.assertLess(out.index('OLD QUESTION'), out.index('NEW QUESTION'))
        self.assertEqual(json.dumps(self.agent.session), before)
        chat.assert_not_called()

    def test_compaction_archives_the_original_messages(self):
        messages = [{'role': role, 'content': f'{i}: {role}'} for i in range(4) for role in ('user', 'assistant')]
        self.agent.session['messages'] = messages[:]
        with patch.object(self.agent, 'chat_raw', return_value=('Short summary', '', [], {})), patch('sys.stderr', io.StringIO()):
            self.agent.maybe_compact(force=True)
        self.assertEqual(self.agent.session['archive'] + self.agent.session['messages'], messages)
        self.assertEqual(len(self.agent.session['messages']), 4)
        self.assertNotIn('archive', self.agent.system_prompt())
        ty.handle_command(self.agent, self.ui, '/clear')
        self.assertFalse(self.agent.session['archive'])

    def test_legacy_compacted_session_replays_its_available_summary(self):
        self.agent.session.pop('archive')
        self.agent.session['summary'] = 'Earlier work: file created.'
        ty.replay_session(self.agent)
        self.assertIn('Earlier work', self.ui.stream.getvalue())

    def test_resume_command_replays_saved_session(self):
        saved = ty.new_session(str(self.root), self.cfg.model)
        saved['messages'] = [{'role': 'user', 'content': 'SAVED CONVERSATION'}]
        with patch.object(ty, 'find_session', return_value=saved):
            ty.handle_command(self.agent, self.ui, '/resume 0')
        self.assertIn('SAVED CONVERSATION', self.ui.stream.getvalue())

    def test_export_includes_archived_messages(self):
        self.agent.session['archive'] = [{'role': 'user', 'content': 'EARLY QUESTION'}]
        path = self.root / 'export.md'
        ty.handle_command(self.agent, self.ui, '/export ' + str(path))
        self.assertIn('EARLY QUESTION', path.read_text())

    def test_ctrl_c_at_idle_prompt_exits_repl(self):
        with patch.object(ty, 'read_prompt', side_effect=KeyboardInterrupt) as read, patch.object(self.agent, 'run_task') as run:
            ty.repl(self.agent, self.ui)
        self.assertEqual(read.call_count, 1)
        run.assert_not_called()

    def test_startup_resume_restores_model_but_explicit_flags_override_it(self):
        for flags, expected in (([], 'qwen3.5:0.8b'), (['--model', '2b'], 'qwen3.5:2b'), (['--fast'], 'qwen3.5:0.8b')):
            saved = ty.new_session(str(self.root), 'qwen3.5:0.8b')
            observed = []
            with patch.object(ty, 'load_config', return_value=dict(ty.DEFAULTS)), \
                    patch.object(ty, 'ensure_ollama', return_value=True), \
                    patch.object(ty, 'migrate_legacy_state'), patch.object(ty, 'find_session', return_value=saved), \
                    patch.object(ty, 'readline', None), patch.object(ty, 'UI', return_value=self.ui), \
                    patch.object(ty, 'repl', side_effect=lambda agent, ui: observed.append(agent.cfg.model)):
                self.assertEqual(ty.main(['--continue', '--dir', str(self.root)] + flags), 0)
            self.assertEqual(observed, [expected])

    def test_runtime_stats_separate_models_and_preserve_bounded_log(self):
        stats = {'eval_count': 24, 'prompt_eval_count': 100, 'eval_duration': 2_000_000_000,
                 'prompt_eval_duration': 5_000_000_000, 'load_duration': 1_000_000_000,
                 'total_duration': 8_000_000_000, 'ty_first_output_s': 6.1}
        with patch('sys.stderr', io.StringIO()):
            self.agent._report_usage(1, stats, 'answer')
            self.cfg.model = 'qwen3.5:0.8b'
            self.agent._report_usage(2, stats, 'answer')
        ty.handle_command(self.agent, self.ui, '/stats')
        self.assertIn('qwen3.5:0.8b', self.ui.stream.getvalue())
        self.assertIn('first output 6.1s', ' '.join(self.ui.stream.getvalue().split()))
        self.assertEqual(self.agent.session['performance'][0]['prefill_s'], 5)
        self.agent.session['performance'] *= 60
        with patch('sys.stderr', io.StringIO()):
            self.agent._report_usage(3, stats, 'answer')
        self.assertEqual(len(self.agent.session['performance']), 100)


class StableConsoleTest(unittest.TestCase):
    def setUp(self):
        self.stream = Terminal()
        self.ui = ty.UI('no', self.stream)
        self.control = ty.TaskControl()
        self.console = ty.LiveConsole(self.ui, self.control, None, idle=True)

    def test_arrows_choose_a_described_command_and_enter_submits(self):
        self.console.feed('/mo')
        self.assertIn('Choose an installed model', self.stream.getvalue())
        self.console.feed('\x1b[B\r')
        self.assertEqual(self.console.result, '/model ')
        self.assertTrue(self.console.input_done.is_set())

    def test_arguments_stay_editable_until_chosen(self):
        self.console.feed('/app\r')
        self.assertEqual(self.console.draft, '/approve ')
        self.assertFalse(self.console.input_done.is_set())
        self.console.feed('\x1b[B\r')
        self.assertEqual(self.console.result, '/approve auto ')

    def test_model_picker_arrows_select_without_a_model_request(self):
        self.console.picker = [('small', 'Smaller model'), ('large', 'Larger model')]
        self.console.feed('\x1b[B\r')
        self.assertEqual(self.console.result, 'large')

    def test_model_picker_escape_returns_without_selecting(self):
        self.console.picker = [('small', 'Smaller model')]
        self.console.feed('\x1b', now=0)
        self.console.feed('', now=.1)
        self.assertIsNone(self.console.result)
        self.assertTrue(self.console.input_done.is_set())
        self.assertFalse(self.console.exit_requested)

    def test_spinner_updates_never_rewrite_the_steering_draft(self):
        self.console.idle = False
        self.console.draft, self.console.cursor = 'DRAFT SHOULD STAY', 17
        self.console.draw()
        self.stream.seek(0)
        self.stream.truncate()
        self.console.set_status('reading context · 1s')
        self.console.set_status('reading context · 2s')
        written = self.stream.getvalue()
        self.assertNotIn('DRAFT SHOULD STAY', written)
        self.assertNotIn('steer ›', written)
        self.assertNotIn('\x1b[2K', written)
        self.assertIn('reading context', written)

    def test_only_the_edited_input_line_is_repainted(self):
        self.console.idle = False
        self.console.draft = 'old text'
        self.console.cursor = 8
        self.console.draw()
        self.stream.seek(0)
        self.stream.truncate()
        self.console.feed('!')
        self.assertIn('old text!', self.stream.getvalue())
        self.assertNotIn('working', self.stream.getvalue())

    def test_ctrl_c_exits_even_with_a_draft(self):
        self.console.feed('unfinished\x03')
        self.assertTrue(self.console.exit_requested)
        self.assertTrue(self.control.cancelled.is_set())

    def test_ctrl_x_stops_work_and_keeps_the_draft(self):
        self.console.idle = False
        self.console.feed('keep this draft\x18')
        self.assertTrue(self.control.cancelled.is_set())
        self.assertEqual(self.console.draft, 'keep this draft')
        self.assertFalse(self.console.exit_requested)

    def test_busy_exit_is_a_local_action(self):
        self.console.idle = False
        self.console.feed('/exit\r')
        self.assertTrue(self.console.exit_requested)
        self.assertTrue(self.control.cancelled.is_set())
        self.assertFalse(self.control.queued())

    def test_literal_paste_can_submit_an_unclosed_code_fence(self):
        self.console.feed('/paste\r```python\rprint(1)\r.\r')
        self.assertEqual(self.console.result, '```python\nprint(1)')


@unittest.skipUnless(ty.termios is not None, 'POSIX pseudo-terminal required')
class FullReplTest(unittest.TestCase):
    def test_resume_completion_model_picker_and_ctrl_c_in_a_real_terminal(self):
        import pty
        master, slave = pty.openpty()
        before = ty.termios.tcgetattr(slave)
        fixture = '''
import json,sys
from types import SimpleNamespace
import ty
cfg=SimpleNamespace(**ty.DEFAULTS)
ui=ty.UI('no')
a=ty.Agent(cfg,sys.argv[1],ui,ty.new_session(sys.argv[1],cfg.model))
a.session['archive']=[{'role':'user','content':'ARCHIVED QUESTION'},{'role':'assistant','content':'**ARCHIVED ANSWER**'}]
a.session['messages']=[{'role':'user','content':'RESUMED QUESTION'}]
a.chat=lambda *args,**kwargs: (_ for _ in ()).throw(RuntimeError('No model request expected'))
ty.http_get=lambda *args,**kwargs: json.dumps({'models':[{'name':'qwen3.5:0.8b','size':1000},{'name':'qwen3.5:2b','size':2000}]})
ty.repl(a,ui)
print('REPL_DONE:'+cfg.model,flush=True)
'''
        with tempfile.TemporaryDirectory(prefix='ty-repl-') as root:
            env = dict(os.environ, XDG_DATA_HOME=root+'/data', XDG_CONFIG_HOME=root+'/config', XDG_CACHE_HOME=root+'/cache')
            child = subprocess.Popen([sys.executable, '-c', fixture, root], stdin=slave, stdout=slave, stderr=slave, env=env)
            capture = bytearray()
            def wait_for(text, start=0):
                deadline = time.monotonic() + 5
                while text not in capture[start:] and time.monotonic() < deadline:
                    if select.select([master], [], [], .05)[0]:
                        capture.extend(os.read(master, 65536))
                self.assertIn(text, capture[start:])
            try:
                wait_for(b'ty smart')
                self.assertIn(b'ARCHIVED QUESTION', capture)
                self.assertIn(b'ARCHIVED ANSWER', capture)
                self.assertIn(b'RESUMED QUESTION', capture)
                os.write(master, b'/mo')
                wait_for(b'Choose an installed model')
                os.write(master, b'\x1b[B\r')
                wait_for('Choose a model · ↑↓'.encode())
                start = len(capture)
                os.write(master, b'\x1b[A\r')
                wait_for(b'model        qwen3.5:0.8b', start)
                wait_for(b'ty smart', start)
                os.write(master, b'\x03')
                wait_for(b'REPL_DONE:qwen3.5:0.8b')
                child.wait(timeout=2)
                self.assertEqual(child.returncode,0,capture.decode(errors='replace'))
                self.assertEqual(ty.termios.tcgetattr(slave),before)
            finally:
                if child.poll() is None:
                    child.kill()
                    child.wait()
                os.close(master)
                os.close(slave)


if __name__ == '__main__':
    unittest.main()
