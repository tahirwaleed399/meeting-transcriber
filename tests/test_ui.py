import json
from pathlib import Path
import queue
import tempfile
import time
import tkinter as tk
import unittest
from unittest.mock import Mock, patch

from live_transcriber import TranscriberApp
from transcript_core import Entry, Settings
from transcription_engine import TranscriptionEngine


class UITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = tk.Tk()
        self.root.withdraw()
        self.patches = [patch('live_transcriber.Settings.load', return_value=Settings()),
            patch('live_transcriber.Settings.save'), patch.object(TranscriberApp, 'refresh_devices'),
            patch('live_transcriber.GlobalHotkey', return_value=Mock())]
        for item in self.patches:
            item.start()
        self.app = TranscriberApp(root=self.root, session_dir=self.temp.name)
        self.app.hotkey.drain.return_value = 0
        self.app.toast = Mock()
        self.app.copy_text = Mock(return_value=True)

    def tearDown(self):
        self.app.saver.close()
        try:
            for after_id in self.root.tk.call('after', 'info'):
                self.root.after_cancel(after_id)
            self.root.destroy()
        except tk.TclError:
            pass
        for item in reversed(self.patches):
            item.stop()
        self.temp.cleanup()

    def make_engine(self):
        engine = TranscriptionEngine(Settings(), self.app.events, pending_dir=Path(self.temp.name) / 'pending')
        engine.ready.set()
        engine.note('Computer', state='capturing', last_block=time.monotonic())
        self.app.engine = engine
        return engine

    def entry(self, text, *, final=False, key='phrase', epoch=None):
        self.app.handle_event({'type': 'transcript', 'epoch': self.app.transcript.epoch if epoch is None else epoch,
            'entry': Entry(key, 'Computer', 0., 1., text, final, 2 if final else 1),
            'backlog': 0, 'inference': .1})

    def burst(self):
        engine = self.make_engine()
        self.app.dictation_var.set(True)
        self.app.begin_burst()
        self.entry('early draft')
        engine.decoding = True
        self.app.end_burst()
        return engine

    def test_dictation_waits_for_final_decode_and_queued_ui_result(self):
        engine = self.burst()
        self.app.finish_burst()
        self.app.copy_text.assert_not_called()
        self.assertEqual('early draft', self.app.transcript.text())
        self.app.events.put({'type': 'transcript', 'engine': id(engine), 'epoch': 0,
            'entry': Entry('phrase', 'Computer', 0., 3., 'complete final sentence', True, 2),
            'backlog': 0, 'inference': 2.5})
        engine.decoding = False
        self.app.finish_burst()
        self.app.copy_text.assert_called_once_with('complete final sentence')
        self.assertFalse(self.app.transcript.entries)
        self.assertFalse(self.app.burst_finishing)

    def test_new_burst_cannot_start_while_previous_burst_finishes(self):
        engine = self.burst()
        self.app.toggle_burst()
        self.assertFalse(self.app.dictating)
        self.assertFalse(engine.armed.is_set())
        self.assertTrue(self.app.burst_finishing)
        engine.decoding = False
        self.app.finish_burst()
        self.app.begin_burst()
        self.entry('next burst', key='next')
        self.app.finish_burst()
        self.assertEqual('next burst', self.app.transcript.text())

    def test_clear_cancels_pending_burst_completion(self):
        engine = self.burst()
        self.app.clear()
        self.entry('new text', key='new')
        engine.decoding = False
        self.app.finish_burst()
        self.app.copy_text.assert_not_called()
        self.assertEqual('new text', self.app.transcript.text())

    def test_leaving_dictation_cancels_completion_and_resumes_capture(self):
        engine = self.burst()
        self.app.dictation_var.set(False)
        self.app.toggle_dictation_mode()
        self.assertTrue(engine.armed.is_set())
        engine.decoding = False
        self.app.finish_burst()
        self.app.copy_text.assert_not_called()
        self.assertEqual('early draft', self.app.transcript.text())

    def test_pause_cancels_burst_copy_and_keeps_visible_text(self):
        engine = self.burst()
        self.app.toggle_recording()
        self.assertTrue(engine.stop_event.is_set())
        engine.decoding = False
        self.app.finish_burst()
        self.app.copy_text.assert_not_called()
        self.assertEqual('early draft', self.app.transcript.text())

    def test_clipboard_failure_never_clears_a_completed_burst(self):
        engine = self.burst()
        self.app.copy_text.return_value = False
        engine.decoding = False
        self.app.finish_burst()
        self.assertEqual('early draft', self.app.transcript.text())
        self.assertEqual(0, self.app.transcript.epoch)

    def test_failed_phrase_keeps_burst_text_for_retry(self):
        engine = self.burst()
        from tests.test_pipeline import job
        engine.mailbox.fail(job())
        engine.decoding = False
        self.app.finish_burst()
        self.app.copy_text.assert_not_called()
        self.assertEqual('early draft', self.app.transcript.text())

    def test_capture_interruption_prevents_automatic_clear(self):
        engine = self.burst()
        with self.assertLogs(level='ERROR'):
            self.app.handle_event({'type': 'capture_error', 'engine': id(engine),
                'source': 'Computer', 'message': 'Device disconnected'})
        engine.decoding = False
        self.app.finish_burst()
        self.app.copy_text.assert_not_called()
        self.assertEqual('early draft', self.app.transcript.text())

    def test_copy_clear_undo_and_exports_keep_existing_behavior(self):
        self.entry('first sentence', final=True)
        self.app.copy_and_clear()
        self.app.copy_text.assert_called_once_with('first sentence')
        self.entry('new sentence', final=True, key='new')
        self.app.undo_clear()
        text = self.app.transcript.text()
        self.assertIn('first sentence', text)
        self.assertIn('new sentence', text)
        for extension in ('txt', 'md', 'srt', 'json'):
            path = Path(self.temp.name) / ('export.' + extension)
            self.app.save_as(path)
            self.assertIn('first sentence', path.read_text(encoding='utf-8'))
            self.assertIn('new sentence', path.read_text(encoding='utf-8'))

    def test_read_settings_keeps_dictation_and_custom_hotkey(self):
        self.app.dictation_var.set(True)
        self.app.hotkey_var.set('Alt+F9')
        settings = self.app.read_settings()
        self.assertTrue(settings.dictation)
        self.assertEqual('Alt+F9', settings.hotkey)

    def test_later_silence_warns_once_per_source_and_rearms_after_audio(self):
        engine = self.make_engine()
        engine.note('Computer', state='capturing', peak_max=.8, digital_silence=31.)
        self.app.check_silence()
        self.app.check_silence()
        self.assertEqual(1, self.app.toast.call_count)
        engine.note('You', state='capturing', peak_max=.5, digital_silence=31.)
        self.app.check_silence()
        self.assertEqual(2, self.app.toast.call_count)
        engine.note('Computer', digital_silence=0.)
        self.app.check_silence()
        engine.note('Computer', digital_silence=31.)
        self.app.check_silence()
        self.assertEqual(3, self.app.toast.call_count)

    def test_one_gui_event_exception_does_not_stop_future_polls(self):
        self.app.events.put({'type': 'debug', 'message': 'first'})
        self.app.events.put({'type': 'debug', 'message': 'second'})
        original = self.app.handle_event
        calls = []
        def flaky(event):
            calls.append(event)
            if len(calls) == 1:
                raise RuntimeError('simulated UI failure')
            original(event)
        with patch.object(self.app, 'handle_event', side_effect=flaky), self.assertLogs(level='ERROR'):
            self.app.poll()
            self.app.poll()
        self.assertEqual(2, len(calls))
        self.assertTrue(self.root.tk.call('after', 'info'))

    def test_declining_close_after_save_failure_restores_working_saver(self):
        self.app.saver.close()
        self.app.saver.last_error = 'disk full'
        self.app.closing = True
        self.app.save_started = True
        self.app.save_finished.set()
        with patch('live_transcriber.messagebox.askyesno', return_value=False):
            self.app.check_closed()
        self.assertFalse(self.app.closing)
        self.assertFalse(self.app.save_started)
        self.assertTrue(self.app.saver.thread.is_alive())
        self.entry('still usable', final=True)
        self.app.saver.close()
        self.assertEqual('still usable', self.app.saver.path.read_text(encoding='utf-8'))

    def test_normal_close_finishes_the_save_and_destroys_the_window(self):
        self.entry('saved on close', final=True)
        self.app.close()
        self.app._poll_once()
        self.assertTrue(self.app.save_finished.wait(5))
        path = self.app.saver.path
        self.app.check_closed()
        self.assertEqual('saved on close', path.read_text(encoding='utf-8'))

    def test_unsaved_audio_requires_a_close_choice_even_when_text_saved(self):
        from tests.test_pipeline import job
        engine = self.make_engine()
        engine.mailbox.failed['memory-only'] = job('memory-only')
        self.app.saver.close()
        self.app.closing = True
        self.app.save_started = True
        self.app.save_finished.set()
        with patch('live_transcriber.messagebox.askyesno', return_value=False) as ask:
            self.app.check_closed()
        self.assertIn('only in memory', ask.call_args.args[1])
        self.assertFalse(self.app.closing)

    def test_resume_reuses_models_and_preserves_dictation_preferences(self):
        old = self.make_engine()
        models = {'tiny.en': object(), 'small.en': object()}
        old.recognizer.models = models
        old.recognizer.device = 'cuda'
        old.recognizer.compute = 'int8_float16'
        old.finished.set()
        self.app.dictation_var.set(True)
        self.app.hotkey_var.set('Alt+F9')
        with patch('live_transcriber.TranscriptionEngine.start'):
            self.app.toggle_recording()
        self.assertIs(models, self.app.engine.recognizer.models)
        self.assertEqual('cuda', self.app.engine.recognizer.device)
        self.assertEqual('int8_float16', self.app.engine.recognizer.compute)
        self.assertTrue(self.app.settings.dictation)
        self.assertEqual('Alt+F9', self.app.settings.hotkey)
        self.assertFalse(self.app.engine.armed.is_set())

    def test_models_ready_does_not_allow_dictation_before_audio_arrives(self):
        engine = self.make_engine()
        engine.note('Computer', state='opening', last_block=0.)
        engine.disarm()
        self.app.dictation_var.set(True)
        self.app.toggle_burst()
        self.assertFalse(self.app.dictating)
        self.assertFalse(engine.armed.is_set())
        self.assertIn('still connecting', self.app.toast.call_args.args[0])


if __name__ == '__main__':
    unittest.main()
