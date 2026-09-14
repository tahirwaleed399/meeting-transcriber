"""Opt-in local hardware smoke check; only synthetic speech is printed.

Run from the repository root:
    python -m tests.smoke_runtime --wave path-to-synthetic-speech.wav
This briefly reads both default audio sources in memory. It does not save them.
"""
import argparse
import os
import queue
import tempfile
import time
from pathlib import Path

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'

from transcript_core import Settings
from transcription_engine import AudioJob, Recognizer, StreamingVAD, TranscriptionEngine, FRAMES, RATE


def until(predicate, seconds=45):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.05)
    raise AssertionError('Hardware check timed out')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wave', required=True, type=Path)
    args = parser.parse_args()
    from faster_whisper.audio import decode_audio
    audio = decode_audio(str(args.wave), sampling_rate=RATE)
    vad = StreamingVAD()
    peak = max(vad.probability(audio[i:i + FRAMES]) for i in range(0, len(audio), FRAMES))
    assert peak > .5, 'Synthetic speech did not trigger VAD'
    print('PASS: real VAD detects synthetic speech', flush=True)
    recognizer = Recognizer(Settings(), lambda kind, **data: print(kind + ': ' + data.get('message', ''), flush=True))
    recognizer.load()
    for final in (False, True):
        start = time.monotonic()
        text = recognizer.transcribe(AudioJob('synthetic', 0, 'Computer', 0., audio, final, 1, start))
        assert 'meeting' in text.lower() and 'project' in text.lower(), text
        print(f'PASS: {recognizer.device} {"final" if final else "draft"} ({time.monotonic() - start:.2f}s): {text}', flush=True)
    with tempfile.TemporaryDirectory(prefix='livescribe-hardware-') as folder:
        for run in range(2):
            events = queue.Queue()
            engine = TranscriptionEngine(Settings(source='Computer + microphone'), events,
                                         pending_dir=Path(folder) / str(run))
            engine.recognizer = recognizer
            start = time.monotonic()
            engine.start()
            try:
                def captured():
                    stats = engine.snapshot_stats()
                    return all(stats.get(source, {}).get('blocks_n', 0) >= 10 for source in ('Computer', 'You'))
                try:
                    until(captured)
                except AssertionError:
                    print('Capture diagnostics:', engine.snapshot_stats(), flush=True)
                    raise
                engine.disarm()
                until(engine.is_drained)
                assert not engine.mailbox.failed_count()
                print(f'PASS: both real sources capture and dictation drains; run {run + 1}, {time.monotonic() - start:.2f}s', flush=True)
            finally:
                stop_start = time.monotonic()
                engine.stop()
                assert engine.finished.wait(15), 'Engine did not stop'
                engine.thread.join(2)
                print(f'PASS: hardware shutdown completed in {time.monotonic() - stop_start:.2f}s', flush=True)
    cpu = Recognizer(Settings(), lambda kind, **data: None)
    cpu.load(force_cpu=True)
    start = time.monotonic()
    text = cpu.transcribe(AudioJob('cpu-synthetic', 0, 'Computer', 0., audio, True, 1, start))
    assert 'meeting' in text.lower() and 'project' in text.lower(), text
    print(f'PASS: real CPU final decode ({time.monotonic() - start:.2f}s): {text}', flush=True)
    import ctypes
    from ctypes import wintypes
    import tkinter as tk
    from global_hotkey import GlobalHotkey, WM_HOTKEY
    root = tk.Tk()
    root.withdraw()
    hotkey = GlobalHotkey()
    try:
        hotkey.attach(root)
        post = ctypes.windll.user32.PostMessageW
        post.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
        assert post(hotkey.hwnd, WM_HOTKEY, hotkey.ID, 0)
        root.update()
        assert hotkey.drain() == 1
        print('PASS: native Windows hotkey message reaches the Tk callback', flush=True)
    finally:
        hotkey.unregister()
        root.destroy()


if __name__ == '__main__':
    main()
