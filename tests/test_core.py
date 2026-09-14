import concurrent.futures
import json
from pathlib import Path
import queue
import tempfile
import unittest
from unittest.mock import patch

from transcript_core import Autosaver, Entry, Settings, Transcript, atomic_write, srt_text
from tests.test_pipeline import wait_for


class PersistenceTests(unittest.TestCase):
    def test_atomic_writers_do_not_share_temporary_files(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'latest.json'
            snapshots = [json.dumps({'text': str(i) * 5000}) for i in range(16)]
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                list(executor.map(lambda text: atomic_write(path, text), snapshots))
            self.assertIn(path.read_text(encoding='utf-8'), snapshots)
            self.assertEqual([path], list(Path(folder).iterdir()))

    def test_autosaver_retries_without_needing_another_transcript_update(self):
        with tempfile.TemporaryDirectory() as folder:
            calls = []
            def flaky_write(path, text):
                calls.append(path)
                if len(calls) == 1:
                    raise OSError('temporary sharing violation')
                atomic_write(path, text)
            with patch('transcript_core.atomic_write', side_effect=flaky_write):
                saver = Autosaver(queue.Queue(), folder)
                try:
                    saver.submit({'text': 'keep this', 'entries': []})
                    wait_for(lambda: len(calls) >= 3 and saver.last_error is None)
                finally:
                    saver.close()
            self.assertEqual('keep this', saver.path.read_text(encoding='utf-8'))

    def test_autosaver_final_flush_preserves_latest_snapshot(self):
        with tempfile.TemporaryDirectory() as folder:
            saver = Autosaver(queue.Queue(), folder)
            for i in range(100):
                saver.submit({'text': str(i), 'entries': []})
            saver.close()
            self.assertIsNone(saver.last_error)
            self.assertEqual('99', saver.path.read_text(encoding='utf-8'))
            self.assertEqual('99', json.loads((Path(folder) / 'latest.json').read_text())['text'])

    def test_malformed_settings_do_not_break_startup(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'settings.json'
            path.write_text(json.dumps({'profile': [], 'language': {}, 'vocabulary': None,
                'dictation': 'false', 'hotkey': 12, 'output_id': []}))
            settings = Settings.load(path)
            self.assertEqual(Settings(), settings)

    def test_clear_undo_and_late_results_preserve_current_text(self):
        transcript = Transcript()
        transcript.apply(0, Entry('1', 'Computer', 0., 1., 'first', True, 1))
        transcript.clear()
        self.assertFalse(transcript.apply(0, Entry('old', 'Computer', 0., 1., 'late', True, 2)))
        transcript.apply(1, Entry('new', 'You', 2., 3., 'second', False, 1))
        transcript.undo_clear()
        self.assertEqual('first\n\nsecond', transcript.text())
        transcript.apply(1, Entry('new', 'You', 2., 3., 'second final', True, 2))
        self.assertFalse(transcript.apply(1, Entry('new', 'You', 2., 3., 'late draft', False, 3)))
        self.assertIn('second final', srt_text(transcript.ordered()))

    def test_invalid_recovery_snapshot_cannot_poison_existing_transcript(self):
        transcript = Transcript()
        transcript.apply(0, Entry('existing', 'Computer', 0., 1., 'keep me', True))
        with self.assertRaises(ValueError):
            transcript.restore({'entries': [{'key': 'bad', 'source': 'Computer', 'start': 0.,
                'end': 1., 'text': None, 'final': True}]})
        self.assertEqual('keep me', transcript.text())


if __name__ == '__main__':
    unittest.main()
