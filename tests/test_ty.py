"""Regression coverage for retrieval, permission boundaries and model-loop behavior."""
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import urllib.error

import ty


class Terminal(io.StringIO):
    def isatty(self):
        return True


class HarnessTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='ty-tests-')
        self.root = Path(self.temp.name)
        self.patches = []
        for key, value in {'DATA_DIR': self.root / 'data' / 'ty', 'CONF_DIR': self.root / 'config' / 'ty',
                           'CACHE_DIR': self.root / 'cache' / 'ty', 'SESS_DIR': self.root / 'data' / 'ty' / 'sessions',
                           'BACKUP_DIR': self.root / 'data' / 'ty' / 'backups', 'ALLOW_FILE': self.root / 'config' / 'ty' / 'allow.json',
                           'CONF_FILE': self.root / 'config' / 'ty' / 'config.json', 'WEB_CACHE': self.root / 'cache' / 'ty' / 'web.json'}.items():
            p = patch.object(ty, key, str(value))
            p.start()
            self.patches.append(p)
        self.old_cache = ty._web_cache
        ty._web_cache = None
        self.ui = ty.UI('no', io.StringIO())
        self.cfg = SimpleNamespace(**ty.DEFAULTS)
        self.agent = ty.Agent(self.cfg, str(self.root), self.ui, ty.new_session(str(self.root), self.cfg.model))
        self.agent.guardian = ty.Guardian(self.cfg, self.ui, self.agent)

    def tearDown(self):
        ty._web_cache = self.old_cache
        for p in reversed(self.patches):
            p.stop()
        self.temp.cleanup()

    def test_html_main_excludes_navigation(self):
        text = ty._strip_html('<html><title>Guide</title><nav>Buy now</nav><main><h1>Install</h1><p>'
                              + 'Use python3 to install the application. ' * 4
                              + '</p><script>tracking()</script></main><footer>Ads</footer></html>')
        self.assertIn('Guide', text)
        self.assertIn('Use python3', text)
        for noise in ('Buy now', 'tracking()', 'Ads'):
            self.assertNotIn(noise, text)

    def test_short_html_fallback_still_cleans_chrome(self):
        text = ty._strip_html('<div>Answer &amp; example</div><nav>Menu</nav>')
        self.assertIn('Answer & example', text)
        self.assertNotIn('Menu', text)

    def test_query_extracts_late_passage(self):
        text = 'Deployment guide\n\n' + '\n\n'.join(f'Unrelated shopping section {i}. ' * 12 for i in range(20))
        text += '\n\nFor certificate renewal, run certbot renew every day. The HTTPS certificate is renewed automatically.'
        result = ty.focus_page(text, 'HTTPS certificate renewal', 500)
        self.assertIn('certbot renew', result)
        self.assertLessEqual(len(result), 500)
        self.assertIn('omitted', result)
        self.assertNotIn('section 19', result)

    def test_dotted_api_terms_match(self):
        text = 'API guide\n\n' + 'Unrelated introductory material. ' * 100
        text += '\n\nPath.read_text() returns the file as a string. Path.write_text() writes it.'
        result = ty.focus_page(text, 'read_text write_text', 400)
        self.assertIn('Path.read_text()', result)
        self.assertIn('Path.write_text()', result)

    def test_no_match_falls_back_honestly(self):
        text = 'Start of source. ' * 200
        result = ty.focus_page(text, 'unfindableword', 400)
        self.assertTrue(result.startswith('Start of source'))
        self.assertIn('excerpt', result)
        self.assertLessEqual(len(result), 400)

    def test_page_cache_is_query_independent(self):
        page = 'Topic index\n\n' + 'Gardening instructions. ' * 40 + '\n\n' + 'Certificate renewal with certbot renew. ' * 10
        with patch.object(ty, 'http_get', return_value=page) as get:
            first = ty.fetch_url('https://example.org/guide', query='gardening', budget=500)
            second = ty.fetch_url('https://example.org/guide', query='certificate', budget=500)
        self.assertEqual(get.call_count, 1)
        self.assertIn('Gardening', first)
        self.assertIn('certbot', second)
        self.assertNotEqual(first, second)

    def test_search_parses_html_and_redirect(self):
        parser = ty.SearchParser()
        parser.feed('<a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.org%2Fdocs">'
                    'Example <b>docs</b></a><a class="result__snippet">Install <b>ty</b> locally.</a>')
        self.assertEqual(parser.results, [{'title': 'Example docs', 'url': 'https://example.org/docs', 'snippet': 'Install ty locally.'}])

    def test_search_parses_lite(self):
        parser = ty.SearchParser()
        parser.feed('<a class="result-link" href="https://example.org">Documentation</a><td class="result-snippet">A useful guide.</td>')
        self.assertEqual(parser.results[0]['snippet'], 'A useful guide.')

    def test_search_fallback_and_success_cache(self):
        good = '<a class="result-link" href="https://example.org">Guide</a><td class="result-snippet">Useful details.</td>'
        with patch.object(ty, 'http_get', side_effect=['<html>challenge</html>', good]) as get:
            result = ty.web_search('guide')
            again = ty.web_search('guide')
        self.assertIn('https://example.org', result)
        self.assertEqual(result, again)
        self.assertEqual(get.call_count, 2)

    def test_search_errors_are_not_cached(self):
        with patch.object(ty, 'http_get', side_effect=OSError('rate limited')):
            self.assertTrue(ty.web_search('guide').startswith('ERROR'))
        self.assertFalse(ty._web_cache)

    def test_fetch_rejects_file_urls(self):
        with self.assertRaises(ValueError):
            ty.fetch_url('file:///etc/passwd')

    def test_session_allowlist_is_scoped_to_tool(self):
        self.agent._remember_allow('run_shell', 'echo *', False)
        self.assertTrue(self.agent._allow_match('run_shell', 'echo hello'))
        self.assertFalse(self.agent._allow_match('write_file', 'echo hello'))

    def test_uppercase_allow_persists(self):
        with patch('sys.stdin.isatty', return_value=True), patch('builtins.input', return_value='A'), patch('sys.stderr', io.StringIO()):
            self.assertTrue(self.agent._ask('run_shell', {'command': 'echo hi'}, 2, 'test'))
        self.assertEqual(ty.load_json(ty.ALLOW_FILE, {})['run_shell'], ['echo*'])

    def test_readonly_cannot_bypass_with_allowlist(self):
        self.cfg.mode = 'readonly'
        self.agent._remember_allow('run_shell', '*', False)
        with patch.object(self.agent, '_run_shell') as execute, patch('sys.stderr', io.StringIO()):
            result = self.agent.execute('run_shell', {'command': 'echo hello'})
        self.assertEqual(result, 'USER DECLINED')
        execute.assert_not_called()

    def test_relative_escape_and_sibling_are_risky(self):
        for path in ('../elsewhere/file', str(self.root) + '-sibling/file'):
            self.assertEqual(self.agent.guardian.judge('write_file', {'path': path})[0], 2)
        self.assertEqual(self.agent.guardian.judge('write_file', {'path': 'inside.txt'})[0], 1)

    def test_symlink_escape_is_risky(self):
        (self.root / 'outside').symlink_to('/tmp', target_is_directory=True)
        self.assertEqual(self.agent.guardian.judge('write_file', {'path': 'outside/file'})[0], 2)

    def test_new_file_and_existing_file_undo(self):
        with patch('sys.stderr', io.StringIO()):
            self.agent._write_file('new.txt', 'first')
            self.agent._write_file('new.txt', 'second')
        self.assertIn('restored', self.agent.undo())
        self.assertEqual((self.root / 'new.txt').read_text(), 'first')
        self.assertIn('removed new file', self.agent.undo())
        self.assertFalse((self.root / 'new.txt').exists())

    def test_repeated_tools_do_not_execute_twice(self):
        call = {'function': {'name': 'run_shell', 'arguments': {'command': 'echo hi'}}}
        stats = {'eval_count': 2, 'total_duration': 1}
        with patch.object(self.agent, 'chat', side_effect=[('', '', [call], stats), ('', '', [call], stats), ('Done', '', [], stats)]), \
                patch.object(self.agent, 'execute', return_value='hi') as execute, patch('sys.stderr', io.StringIO()):
            result = self.agent.run_task('say hello')
        self.assertEqual(result, 'Done')
        self.assertEqual(execute.call_count, 1)
        self.assertEqual(self.agent.session['messages'][-1]['role'], 'assistant')

    def test_invalid_call_does_not_execute(self):
        call = {'function': {'name': 'run_shell', 'arguments': {'command': 123}}}
        self.agent._seen_sigs = set()
        with patch.object(self.agent, 'execute') as execute, patch('sys.stderr', io.StringIO()):
            self.agent._run_call(1, call)
        execute.assert_not_called()
        self.assertIn('must be text', self.agent.last_results[-1][2])

    def test_disabled_tool_is_rejected(self):
        self.cfg.mode = 'readonly'
        self.assertIn('not enabled', ty.validate_call('write_file', {'path': 'x', 'content': ''}, self.agent.active_tools()))

    def test_system_prefix_is_stable(self):
        with patch.object(ty, 'now_stamp', side_effect=['one', 'two']):
            self.assertEqual(self.agent.system_prompt(), self.agent.system_prompt())

    def test_options_have_output_budget_and_physical_threads(self):
        with patch.object(ty, 'physical_cores', return_value=2):
            self.assertEqual(self.agent._options()['num_thread'], 2)
        self.assertEqual(self.agent._options()['num_predict'], 768)
        self.cfg.num_thread = 4
        self.assertEqual(self.agent._options()['num_thread'], 4)

    def test_context_estimate_includes_tools(self):
        total = self.agent.est_tokens()
        self.cfg.mode = 'readonly'
        self.assertLess(self.agent.est_tokens(), total)

    def test_unique_sessions_in_same_second(self):
        self.assertNotEqual(ty.new_session('/tmp', 'm')['id'], ty.new_session('/tmp', 'm')['id'])

    def test_toml_overrides_json(self):
        Path(ty.CONF_DIR).mkdir(parents=True)
        Path(ty.CONF_FILE).write_text('{"model":"json-model","ctx":2048}')
        (Path(ty.CONF_DIR) / 'config.toml').write_text('model = "toml-model"\nui = "verbose"\n')
        cfg = ty.load_config()
        self.assertEqual(cfg['model'], 'toml-model')
        self.assertEqual(cfg['ctx'], 2048)
        self.assertEqual(cfg['ui'], 'verbose')

    def test_migration_preserves_undo_backups(self):
        old = Path(ty.legacy_path(ty.DATA_DIR))
        (old / 'sessions').mkdir(parents=True)
        (old / 'backups').mkdir()
        (old / 'backups' / 'one').write_text('original')
        session = ty.new_session('/tmp', 'm')
        session['undo'] = [{'existed': True, 'path': '/tmp/x', 'backup': str(old / 'backups' / 'one')}]
        (old / 'sessions' / (session['id'] + '.json')).write_text(json.dumps(session))
        ty.migrate_legacy_state()
        got = ty.list_sessions()[0]
        self.assertTrue(Path(got['undo'][0]['backup']).exists())
        self.assertFalse(old.exists())
        ty.migrate_legacy_state()  # idempotent
        self.assertEqual(len(ty.list_sessions()), 1)

    def test_argument_completion(self):
        self.assertEqual(ty.completion_options('/ui v', 'v'), ['verbose'])
        self.assertIn('/context', ty.completion_options('/con', '/con'))

    def test_plain_output_and_terminal_controls(self):
        stream = io.StringIO()
        ui = ty.UI('no', stream)
        renderer = ty.AnswerRenderer(ui)
        renderer.feed('**Hello**\n`code`\x1b[31m!\x1b[0m\x1b]2;title\x07')
        renderer.finish()
        self.assertNotIn('\x1b', stream.getvalue())
        self.assertNotIn('\x07', stream.getvalue())
        self.assertIn('**Hello**', stream.getvalue())

    def test_terminal_markdown_and_readline_prompt(self):
        stream = Terminal()
        with patch.dict(os.environ, {}, clear=True):
            ui = ty.UI('yes', stream)
        renderer = ty.AnswerRenderer(ui)
        renderer.feed('## Heading\n```python\nprint(1)\n```\n')
        renderer.finish()
        self.assertIn('Heading', stream.getvalue())
        self.assertIn('│ print(1)', ty.ANSI_RE.sub('', stream.getvalue()))
        self.assertIn('\x01\x1b', ui.prompt('smart'))
        with patch.dict(os.environ, {'NO_COLOR': ''}):
            self.assertFalse(ty.UI('yes', Terminal()).color)

    def test_multiline_paste_preserves_blank_lines(self):
        with patch('builtins.input', side_effect=['/paste', 'first', '', '**second**', '.']):
            self.assertEqual(ty.read_input(self.ui, '> '), 'first\n\n**second**')

    def test_bracketed_multiline_paste_is_one_task(self):
        text = 'Explain this:\n```python\nprint(1)\n```\nThen suggest improvements.'
        with patch('builtins.input', return_value=text) as get:
            self.assertEqual(ty.read_input(self.ui, '> '), text)
        self.assertEqual(get.call_count, 1)

    def test_fenced_multiline_input(self):
        with patch('builtins.input', side_effect=['```python', 'print(1)', '```']):
            self.assertEqual(ty.read_input(self.ui, '> '), '```python\nprint(1)\n```')

    def test_backslash_multiline_input(self):
        with patch('builtins.input', side_effect=['first \\', 'second']):
            self.assertEqual(ty.read_input(self.ui, '> '), 'first \nsecond')

    def test_markdown_links_and_emphasis(self):
        stream = Terminal()
        with patch.dict(os.environ, {'TERM': 'xterm-256color'}, clear=True):
            ui = ty.UI('yes', stream)
            renderer = ty.AnswerRenderer(ui)
            renderer.feed('**Bold** and *italic* and `code` and [Docs](https://example.org/docs)\n')
            renderer.finish()
        rendered = stream.getvalue()
        self.assertIn('\x1b[1mBold', rendered)
        self.assertIn('\x1b[3mitalic', rendered)
        self.assertIn('\x1b]8;;https://example.org/docs', rendered)
        self.assertIn('Docs', rendered)
        self.assertNotIn('**Bold**', rendered)

    def test_plain_terminal_links_expose_url(self):
        rendered = ty.render_inline(self.ui, '[Docs](https://example.org/docs)')
        self.assertEqual(rendered, 'Docs (https://example.org/docs)')

    def test_markdown_code_is_not_reformatted(self):
        stream = Terminal()
        renderer = ty.AnswerRenderer(ty.UI('no', stream))
        renderer.feed('~~~text\n**literal** [literal](https://example.org)\n~~~\n> quote\n- bullet\n')
        renderer.finish()
        self.assertIn('│ **literal** [literal](https://example.org)', stream.getvalue())
        self.assertIn('│ quote', stream.getvalue())
        self.assertIn('• bullet', stream.getvalue())

    def test_markdown_across_stream_chunks(self):
        stream = Terminal()
        renderer = ty.AnswerRenderer(ty.UI('no', stream))
        for fragment in ['**bo', 'ld** and [Do', 'cs](https://example.org)', '\n']:
            renderer.feed(fragment)
        renderer.finish()
        self.assertIn('bold and Docs (https://example.org)', stream.getvalue())

    def test_tool_interrupt_stops_later_calls(self):
        first = {'function': {'name': 'read_file', 'arguments': {'path': 'one'}}}
        second = {'function': {'name': 'read_file', 'arguments': {'path': 'two'}}}
        with patch.object(self.agent, 'chat', return_value=('', '', [first, second], {})), \
                patch.object(self.agent, 'execute', side_effect=KeyboardInterrupt) as execute, patch('sys.stderr', io.StringIO()):
            self.agent.run_task('read both')
        self.assertEqual(execute.call_count, 1)

    def test_narrow_status_wraps(self):
        stream = io.StringIO()
        ui = ty.UI('no', stream)
        with patch('shutil.get_terminal_size', return_value=os.terminal_size((40, 24))):
            ui.row('model', 'qwen3.5:2b with smart approvals and reasoning disabled')
        self.assertTrue(all(len(line) <= 40 for line in stream.getvalue().splitlines()))

    def test_streaming_error_is_reported(self):
        class Response(io.BytesIO):
            def __enter__(self): return self
            def __exit__(self, *args): self.close()
        with patch('urllib.request.urlopen', return_value=Response(b'{"error":"model failed"}\n')):
            with self.assertRaisesRegex(RuntimeError, 'model failed'):
                self.agent.chat([], False)


if __name__ == '__main__':
    unittest.main()
