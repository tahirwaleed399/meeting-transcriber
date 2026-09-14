"""Compare final decoding against a local Git revision using identical cached models."""
import argparse
import os
import statistics
import subprocess
import sys
import time
import types

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'

from transcript_core import Settings
from transcription_engine import AudioJob, Recognizer, RATE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wave', required=True)
    parser.add_argument('--baseline', required=True)
    args = parser.parse_args()
    code = subprocess.check_output(['git', 'show', args.baseline + ':transcription_engine.py'], encoding='utf-8')
    module = types.ModuleType('baseline_engine')
    sys.modules[module.__name__] = module
    exec(compile(code, '<baseline transcription_engine.py>', 'exec'), module.__dict__)
    from faster_whisper.audio import decode_audio
    audio = decode_audio(args.wave, sampling_rate=RATE)
    current = Recognizer(Settings(), lambda *a, **k: None)
    current.load()
    baseline = module.Recognizer(Settings(), lambda *a, **k: None)
    baseline.models, baseline.device, baseline.compute = current.models, current.device, current.compute
    baseline.load()
    timings = {'baseline': [], 'current': []}
    texts = []
    for iteration in range(3):
        pairs = [('baseline', baseline), ('current', current)]
        if iteration % 2:
            pairs.reverse()
        for name, recognizer in pairs:
            start = time.monotonic()
            texts.append(recognizer.transcribe(AudioJob('compare', 0, 'Computer', 0., audio, True, 1, start)))
            elapsed = time.monotonic() - start
            timings[name].append(elapsed)
            print(f'{name} round {iteration + 1}: {elapsed:.3f}s', flush=True)
    assert len(set(texts)) == 1, 'Baseline and current recognition differ'
    print('PASS: all six final transcripts are identical', flush=True)
    print('Median seconds:', {name: round(statistics.median(values), 3) for name, values in timings.items()}, flush=True)


if __name__ == '__main__':
    main()
