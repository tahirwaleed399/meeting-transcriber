import queue
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from audio_jobs import AudioJob, Mailbox, StoredJob, read_job
from audio_recovery import recover_files
from transcript_core import Settings
from transcription_engine import Recognizer, SpeechSegmenter, TranscriptionEngine, FRAMES, RATE


def job(key='phrase', final=True, epoch=0, revision=1):
    return AudioJob(key, epoch, 'Computer', 0., np.ones(FRAMES, np.float32) * .1,
                    final, revision, time.monotonic())


def wait_for(predicate, seconds=4):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.01)
    raise AssertionError('Condition was not reached before the timeout')


class FakeRecognizer:
    device = 'cpu'

    def __init__(self):
        self.boundaries = {}
        self.models = {}
        self.load = Mock(return_value='Test CPU')
        self.transcribe = Mock(side_effect=lambda item: 'decoded ' + item.key)


class MailboxTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.box = Mailbox(max_items=2, directory=self.temp.name)

    def test_overflow_preserves_all_finals_and_bounds_audio_ram(self):
        for i in range(100):
            self.box.put(job(str(i)))
        self.assertEqual(100, len(self.box))
        self.assertEqual(2, sum(isinstance(v, AudioJob) for v in self.box.items.values()))
        for i in range(100):
            item = self.box.get(0)
            self.assertEqual(str(i), item.key)
            np.testing.assert_array_equal(item.audio, job().audio)
            self.box.acknowledge(item)
        self.assertFalse(list(Path(self.temp.name).glob('*.npz')))

    def test_drafts_coalesce_and_do_not_displace_final_audio(self):
        self.box.put(job('first', False))
        self.box.put(job('first', False, revision=2))
        self.box.put(job('second'))
        self.assertFalse(self.box.put(job('third', False)))
        self.assertEqual('second', self.box.get(0).key)
        self.box.put(job('first', True, revision=3))
        self.assertFalse(self.box.put(job('first', False, revision=4)))
        self.assertTrue(self.box.get(0).final)

    def test_failed_job_is_recoverable_until_successful_acknowledgment(self):
        original = job()
        self.box.fail(original)
        saved = self.box.failed[original.key].path
        restored, settings = read_job(saved)
        self.assertEqual(original.key, restored.key)
        self.box.retry_failed()
        retried = self.box.get(0)
        self.assertTrue(saved.exists())
        self.box.acknowledge(retried)
        self.assertFalse(saved.exists())
        self.assertEqual(0, self.box.failed_count())

    def test_disk_full_retains_the_rejected_final_in_memory(self):
        self.box.put(job('1'))
        self.box.put(job('2'))
        with patch.object(self.box, '_store', side_effect=OSError('disk full')):
            with self.assertRaises(OSError):
                self.box.put(job('3'))
        self.assertEqual('3', self.box.failed['3'].key)

    def test_clear_removes_queued_failed_and_inflight_spools(self):
        for i in range(4):
            self.box.put(job(str(i)))
        self.box.get(0)
        self.box.get(0)
        leased = self.box.get(0)
        self.box.fail(job('failed'))
        self.box.clear()
        self.assertFalse(list(Path(self.temp.name).glob('*.npz')))
        self.assertEqual(0, len(self.box))
        self.assertEqual(0, self.box.failed_count())
        self.box.acknowledge(leased)

    def test_corrupt_spool_is_quarantined_without_blocking_later_jobs(self):
        self.box.fail(job('bad'))
        self.box.failed['bad'].path.write_bytes(b'not an archive')
        self.box.retry_failed()
        self.box.put(job('good'))
        with self.assertRaises(Exception):
            self.box.get(0)
        self.assertEqual('good', self.box.get(0).key)
        self.assertEqual(1, self.box.failed_count())

    def test_pending_memory_audio_is_persisted_for_recovery_on_shutdown(self):
        self.box.put(job('pending'))
        self.assertEqual(1, self.box.unsaved_count())
        self.box.preserve_pending()
        self.assertEqual(0, self.box.unsaved_count())
        restored, _ = read_job(self.box.items['pending'].path)
        self.assertEqual('pending', restored.key)


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.events = queue.Queue()
        self.engine = TranscriptionEngine(Settings(), self.events, pending_dir=self.temp.name)
        self.engine.recognizer = FakeRecognizer()
        self.addCleanup(self.temp.cleanup)
        self.addCleanup(self.stop)

    def stop(self):
        self.engine.stop()
        if self.engine.thread:
            self.engine.thread.join(30)
            self.assertFalse(self.engine.thread.is_alive())

    def start(self):
        self.engine.start(capture=False)
        self.assertTrue(self.engine.ready.wait(2))

    def transcripts(self):
        with self.events.mutex:
            return [e for e in self.events.queue if e['type'] == 'transcript']

    def test_transient_cpu_error_retries_same_phrase_and_continues(self):
        self.engine.recognizer.transcribe.side_effect = [RuntimeError('transient'), 'first', 'second']
        self.engine.submit(job('first'))
        self.engine.submit(job('second'))
        self.start()
        wait_for(lambda: len(self.transcripts()) == 2)
        self.assertFalse(self.engine.stop_event.is_set())
        self.assertEqual(['first', 'second'], [e['entry'].text for e in self.transcripts()])

    def test_gpu_error_retries_on_cpu(self):
        self.engine.recognizer.device = 'cuda'
        self.engine.recognizer.transcribe.side_effect = [RuntimeError('GPU lost'), 'recovered']
        self.engine.submit(job())
        self.start()
        wait_for(lambda: len(self.transcripts()) == 1)
        self.engine.recognizer.load.assert_any_call(force_cpu=True)
        self.assertFalse(self.engine.stop_event.is_set())

    def test_permanent_failure_keeps_audio_and_later_speech_continues(self):
        def decode(item):
            if item.key == 'bad':
                raise RuntimeError('bad phrase')
            return 'good'
        self.engine.recognizer.transcribe.side_effect = decode
        self.engine.submit(job('bad'))
        self.engine.submit(job('good'))
        self.start()
        wait_for(lambda: len(self.transcripts()) == 1)
        self.assertEqual(1, self.engine.mailbox.failed_count())
        self.assertTrue(list(Path(self.temp.name).glob('*.npz')))
        self.assertFalse(self.engine.stop_event.is_set())
        self.engine.recognizer.transcribe.side_effect = lambda item: 'recovered'
        self.engine.retry_failed()
        wait_for(lambda: len(self.transcripts()) == 2)
        wait_for(lambda: not list(Path(self.temp.name).glob('*.npz')))

    def test_clear_during_decode_rejects_late_text_and_failed_audio(self):
        entered, release = threading.Event(), threading.Event()
        def decode(item):
            entered.set()
            self.assertTrue(release.wait(3))
            raise RuntimeError('late failure')
        self.engine.recognizer.transcribe.side_effect = decode
        self.start()
        self.engine.submit(job())
        self.assertTrue(entered.wait(2))
        self.engine.clear(1)
        release.set()
        wait_for(lambda: not self.engine.decoding)
        self.assertFalse(self.transcripts())
        self.assertEqual(0, self.engine.mailbox.failed_count())

    def test_drain_waits_for_each_source_and_the_inflight_decode(self):
        self.engine.settings.source = 'Computer + microphone'
        self.engine.capture_threads = [Mock()]
        self.engine.disarm()
        self.assertFalse(self.engine.is_drained())
        self.engine.flush_pending.remove('Computer')
        self.assertFalse(self.engine.is_drained())
        self.engine.flush_pending.remove('You')
        self.engine.decoding = True
        self.assertFalse(self.engine.is_drained())
        self.engine.decoding = False
        self.assertTrue(self.engine.is_drained())
        self.engine.capture_threads = []

    def test_repeated_stop_requests_do_not_extend_shutdown_deadline(self):
        self.engine.stop()
        deadline_origin = self.engine.stop_at
        with patch('transcription_engine.time.monotonic', return_value=deadline_origin + 10):
            self.engine.stop()
        self.assertEqual(deadline_origin, self.engine.stop_at)

    def test_large_backlog_drains_on_pause_without_losing_phrases(self):
        self.engine.mailbox.max_items = 2
        for i in range(80):
            self.engine.submit(job(str(i)))
        self.start()
        self.engine.stop()
        self.assertTrue(self.engine.finished.wait(30))
        self.assertEqual(80, len(self.transcripts()))

    def test_cuda_probe_error_falls_back_before_constructing_models(self):
        ct2 = SimpleNamespace(get_cuda_device_count=lambda: 1,
            get_supported_compute_types=Mock(side_effect=RuntimeError('driver query failed')))
        model = SimpleNamespace(transcribe=lambda *a, **k: ([], None))
        constructor = Mock(return_value=model)
        with patch('transcription_engine.configure_runtime'), patch.dict('sys.modules', {
                'ctranslate2': ct2, 'faster_whisper': SimpleNamespace(WhisperModel=constructor)}):
            recognizer = Recognizer(Settings(), Mock())
            recognizer.load()
        self.assertEqual('cpu', recognizer.device)
        self.assertTrue(all(call.kwargs['device'] == 'cpu' for call in constructor.call_args_list))


class SegmenterAndRecoveryTests(unittest.TestCase):
    def test_speech_pause_finalizes_once_with_preroll(self):
        jobs = []
        segmenter = SpeechSegmenter('Computer', 0, jobs.append)
        for i in range(16):
            segmenter.feed(np.ones(FRAMES, np.float32) * .1, i * FRAMES / RATE,
                           .9 if 2 <= i < 8 else 0.)
        segmenter.flush()
        finals = [j for j in jobs if j.final]
        self.assertEqual(1, len(finals))
        self.assertEqual(0., finals[0].start)
        self.assertTrue(any(not j.final for j in jobs))

    def test_continuous_speech_splits_with_overlap_and_flushes_the_tail(self):
        jobs = []
        segmenter = SpeechSegmenter('Computer', 0, jobs.append)
        for i in range(530):
            segmenter.feed(np.ones(FRAMES, np.float32), i * FRAMES / RATE, .9)
        segmenter.flush()
        finals = [j for j in jobs if j.final]
        self.assertEqual(3, len(finals))
        self.assertTrue(finals[0].truncated)
        self.assertEqual(finals[0].key, finals[1].previous_key)
        self.assertEqual(finals[1].key, finals[2].previous_key)
        self.assertFalse(finals[-1].truncated)

    def test_recovery_writes_separate_text_and_keeps_original_audio(self):
        with tempfile.TemporaryDirectory() as folder:
            box = Mailbox(directory=Path(folder) / 'pending')
            box.fail(job('one'))
            original = box.failed['one'].path
            path, count, failures = recover_files([original], folder,
                recognizer_factory=lambda *args: FakeRecognizer())
            self.assertEqual(1, count)
            self.assertFalse(failures)
            self.assertEqual('decoded one', path.read_text(encoding='utf-8'))
            self.assertTrue(original.exists())


if __name__ == '__main__':
    unittest.main()
