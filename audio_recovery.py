"""Recover locally buffered phrases into a separate transcript without recording."""
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import uuid

from audio_jobs import read_job
from transcript_core import Entry, atomic_write


def recover_files(paths, directory, emit=lambda message: None, recognizer_factory=None):
    if recognizer_factory is None:
        from transcription_engine import Recognizer
        recognizer_factory = Recognizer
    jobs, failures = [], []
    for path in paths:
        try:
            job, settings = read_job(path)
            jobs.append((job, settings, Path(path)))
        except Exception as exc:
            failures.append(f'{Path(path).name}: {exc}')
    jobs.sort(key=lambda item: (item[0].start, item[0].key))
    recognizer, current_settings = None, None
    entries = []
    for index, (job, settings, path) in enumerate(jobs, 1):
        emit(f'Recovering phrase {index}/{len(jobs)}…')
        try:
            signature = (settings.profile, settings.language, settings.vocabulary)
            if signature != current_settings:
                recognizer = recognizer_factory(settings, lambda kind, **data: emit(data.get('message', kind)))
                recognizer.load(force_cpu=True)
                current_settings = signature
            text = recognizer.transcribe(job)
            if text.strip():
                entries.append(Entry(job.key, job.source, job.start,
                    job.start + len(job.audio) / 16000, text, True, job.revision))
        except Exception as exc:
            failures.append(f'{path.name}: {exc}')
            current_settings = None
    path = Path(directory) / ('Recovered_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S_') + uuid.uuid4().hex[:6] + '.txt')
    text = '\n\n'.join(entry.text for entry in entries)
    # Preserve original audio even on success, so a failed export/recovery cannot destroy it.
    atomic_write(path, text)
    atomic_write(path.with_suffix('.json'), json.dumps({'entries': [asdict(e) for e in entries],
        'text': text, 'failures': failures}, ensure_ascii=False, indent=2))
    return path, len(entries), failures
