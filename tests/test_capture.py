import contextlib
import multiprocessing
import os
import queue
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from audio_capture import CaptureProcess, MicrophoneRecorder, capture_worker
from transcript_core import Settings
from transcription_engine import TranscriptionEngine, SpeechSegmenter, FRAMES, RATE
from tests.test_pipeline import FakeRecognizer, wait_for


def stalled_worker(source, selection, messages, stop, rate, frames):
    messages.send({'type': 'capture_ready', 'message': 'Synthetic stalled device'})
    time.sleep(60)


def crashed_worker(source, selection, messages, stop, rate, frames):
    messages.send({'type': 'capture_ready', 'message': 'Synthetic crashed device'})
    os._exit(7)


class CaptureProcessTests(unittest.TestCase):
    def test_stalled_native_worker_times_out_and_is_terminated(self):
        capture = CaptureProcess('Computer', '', timeout=3., target=stalled_worker)
        with capture:
            wait_for(lambda: capture.receive(.1) is not None, seconds=30)
            capture.timeout = .15
            start = time.monotonic()
            with self.assertRaises(TimeoutError):
                while True:
                    capture.receive(.05)
        self.assertLess(time.monotonic() - start, 2.)
        self.assertFalse(any(p.name == 'audio-Computer' for p in multiprocessing.active_children()))

    def test_native_crash_reports_eof_without_hanging_the_reader(self):
        capture = CaptureProcess('Computer', '', timeout=3., target=crashed_worker)
        with capture:
            wait_for(lambda: capture.receive(.1) is not None, seconds=30)
            start = time.monotonic()
            with self.assertRaises(RuntimeError):
                while True:
                    capture.receive(.05)
        self.assertLess(time.monotonic() - start, 2.)

    def test_worker_reconnects_when_default_device_changes(self):
        old = SimpleNamespace(id='old', name='Old output')
        new = SimpleNamespace(id='new', name='New output')
        sc = SimpleNamespace(default_speaker=Mock(side_effect=[old, new]),
                             get_microphone=lambda identifier, **kwargs: SimpleNamespace(id=identifier, name=identifier))
        recorder = Mock()
        recorder.__enter__ = Mock(return_value=recorder)
        recorder.__exit__ = Mock(return_value=False)
        recorder.record.return_value = np.zeros((FRAMES, 2), np.float32)
        messages = Mock()
        with patch('audio_capture.com_apartment', contextlib.nullcontext), patch.dict('sys.modules', {'soundcard': sc}), \
                patch('audio_capture.open_recorder', return_value=recorder), \
                patch('audio_capture.time.monotonic', side_effect=[0., 2., 2., 2.]):
            capture_worker('Computer', '', messages, threading.Event(), RATE, FRAMES)
        kinds = [call.args[0]['type'] for call in messages.send.call_args_list]
        self.assertEqual(['device', 'capture_ready', 'audio', 'device_changed'], kinds)

    def test_microphone_stream_closes_if_start_fails(self):
        stream = Mock()
        stream.start.side_effect = RuntimeError('unplugged')
        sd = SimpleNamespace(query_hostapis=lambda: [{'name': 'Windows WASAPI'}],
            query_devices=lambda: [{'max_input_channels': 1, 'hostapi': 0, 'name': 'Mic'}],
            InputStream=Mock(return_value=stream), WasapiSettings=Mock())
        with patch.dict('sys.modules', {'sounddevice': sd}):
            recorder = MicrophoneRecorder(SimpleNamespace(name='Mic', channels=1))
            with self.assertRaises(RuntimeError):
                recorder.__enter__()
        stream.close.assert_called_once()


class EngineCaptureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.engine = TranscriptionEngine(Settings(), queue.Queue(), pending_dir=self.temp.name)
        self.engine.recognizer = FakeRecognizer()

    def test_capture_initialization_failure_is_reported_and_retried(self):
        engine = self.engine
        attempts = []
        class Capture:
            def __init__(self, *args): self.reports = 0
            def __enter__(self):
                attempts.append(1)
                if len(attempts) == 1:
                    raise OSError('COM initialization failed')
                return self
            def __exit__(self, *args): pass
            def receive(self, timeout):
                self.reports += 1
                if self.reports == 1:
                    return {'type': 'capture_ready', 'message': 'Reconnected'}
                engine.stop()
                return {'type': 'audio', 'data': np.zeros((FRAMES, 1), np.float32),
                        'captured': time.monotonic(), 'discontinuities': 0}
        with patch('transcription_engine.CaptureProcess', Capture), \
                patch('transcription_engine.StreamingVAD', return_value=Mock()):
            engine.start()
            self.assertTrue(engine.finished.wait(4))
            engine.thread.join(2)
        events = list(engine.events.queue)
        self.assertEqual(2, len(attempts))
        self.assertTrue(any(e['type'] == 'capture_error' for e in events))
        self.assertTrue(any(e['type'] == 'capture_ready' for e in events))

    def test_clear_resets_segmenter_and_pause_preserves_new_phrase(self):
        engine = self.engine
        counter = 0
        class Capture:
            def __init__(self, *args): pass
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def receive(self, timeout):
                nonlocal counter
                counter += 1
                if counter == 5:
                    engine.clear(1)
                if counter == 10:
                    engine.stop()
                return {'type': 'audio', 'data': np.ones((FRAMES, 2), np.float32) * .1,
                        'captured': time.monotonic(), 'discontinuities': 0}
        vad = SimpleNamespace(probability=lambda audio: .9)
        with patch('transcription_engine.CaptureProcess', Capture), \
                patch('transcription_engine.StreamingVAD', return_value=vad):
            engine.start()
            self.assertTrue(engine.finished.wait(4))
            engine.thread.join(2)
        entries = [e for e in engine.events.queue if e['type'] == 'transcript' and e['epoch'] == 1]
        self.assertEqual(1, len(entries))
        self.assertTrue(entries[0]['entry'].final)

    def test_pause_bounds_a_stalled_capture_and_flushes_buffered_speech(self):
        engine = self.engine
        entered = threading.Event()
        class Capture:
            def __init__(self, *args): self.blocks = 0
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def receive(self, timeout):
                if self.blocks < 4:
                    self.blocks += 1
                    return {'type': 'audio', 'data': np.ones((FRAMES, 1), np.float32) * .1,
                            'captured': time.monotonic(), 'discontinuities': 0}
                entered.set()
                time.sleep(timeout)
                return None
        with patch('transcription_engine.CaptureProcess', Capture), \
                patch('transcription_engine.StreamingVAD', return_value=SimpleNamespace(probability=lambda audio: .9)):
            engine.start()
            self.assertTrue(entered.wait(2))
            start = time.monotonic()
            engine.stop()
            self.assertTrue(engine.finished.wait(2))
            engine.thread.join(2)
        self.assertLess(time.monotonic() - start, 1.5)
        self.assertTrue(any(e['type'] == 'transcript' and e['entry'].final for e in engine.events.queue))


if __name__ == '__main__':
    unittest.main()
