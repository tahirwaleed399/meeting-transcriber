"""Coalesced previews, bounded audio RAM, and recoverable overflow/failed phrases."""
from collections import OrderedDict
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import threading
import uuid

import numpy as np

from transcript_core import ROOT, Settings


@dataclass
class AudioJob:
    key: str
    epoch: int
    source: str
    start: float
    audio: np.ndarray
    final: bool
    revision: int
    created: float
    previous_key: str | None = None
    truncated: bool = False


@dataclass
class StoredJob:
    key: str
    epoch: int
    final: bool
    revision: int
    path: Path

    def load(self):
        return read_job(self.path)[0]


def read_job(path):
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive['metadata'].item()))
        audio = np.asarray(archive['audio'], dtype=np.float32)
    if audio.ndim != 1 or not len(audio) or not np.isfinite(audio).all():
        raise ValueError('Invalid saved audio.')
    return AudioJob(audio=audio, **metadata['job']), Settings(**metadata['settings'])


class Mailbox:
    """Finals overflow to local disk; previews never displace final audio."""
    def __init__(self, max_items=64, directory=None, settings=None):
        self.condition = threading.Condition()
        self.items = OrderedDict()
        self.failed = OrderedDict()
        self.leased = {}
        self.max_items = max_items
        self.directory = Path(directory) if directory else ROOT / 'sessions' / 'pending-audio' / uuid.uuid4().hex
        self.settings = settings or Settings()

    def _store(self, job):
        self.directory.mkdir(parents=True, exist_ok=True)
        # Keys loaded from files are data, never filesystem paths.
        path = self.directory / (uuid.uuid4().hex + '.npz')
        temporary = path.with_suffix('.tmp')
        metadata = {name: getattr(job, name) for name in AudioJob.__dataclass_fields__ if name != 'audio'}
        try:
            with temporary.open('wb') as stream:
                np.savez(stream, audio=job.audio,
                         metadata=json.dumps({'job': metadata, 'settings': asdict(self.settings)}))
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        return StoredJob(job.key, job.epoch, job.final, job.revision, path)

    def put(self, job):
        with self.condition:
            old = self.items.get(job.key)
            if old and (old.final or old.revision > job.revision):
                return False
            memory_items = sum(isinstance(item, AudioJob) for item in self.items.values())
            if old is None and memory_items >= self.max_items:
                if not job.final:
                    return False
                try:
                    job = self._store(job)
                except OSError:
                    self.failed[job.key] = job
                    raise
            self.items[job.key] = job
            self.condition.notify()
            return True

    def get(self, timeout=.1):
        with self.condition:
            if not self.items:
                self.condition.wait(timeout)
            if not self.items:
                return None
            key = next((key for key, job in self.items.items() if job.final), next(iter(self.items)))
            stored = self.items[key]
            try:
                job = stored.load() if isinstance(stored, StoredJob) else stored
            except Exception:
                self.failed[key] = self.items.pop(key)
                raise
            self.items.pop(key)
            if isinstance(stored, StoredJob):
                self.leased[(job.epoch, job.key)] = stored
            return job

    def acknowledge(self, job):
        with self.condition:
            stored = self.leased.pop((job.epoch, job.key), None)
            if stored:
                stored.path.unlink(missing_ok=True)

    def fail(self, job):
        with self.condition:
            stored = self.leased.pop((job.epoch, job.key), None)
            if stored is None:
                try:
                    stored = self._store(job)
                except OSError:
                    # Retain the audio in memory even if the disk is full.
                    self.failed[job.key] = job
                    raise
            self.failed[job.key] = stored

    def retry_failed(self):
        with self.condition:
            self.items.update(self.failed)
            self.failed.clear()
            self.condition.notify()

    def failed_count(self):
        with self.condition:
            return len(self.failed)

    def unsaved_count(self):
        with self.condition:
            return sum(isinstance(item, AudioJob) and item.final
                       for item in (*self.items.values(), *self.failed.values()))

    def preserve_pending(self):
        """Keep pending finals recoverable if model initialization/shutdown fails."""
        with self.condition:
            for collection in (self.items, self.failed):
                for key, item in list(collection.items()):
                    if isinstance(item, AudioJob) and item.final:
                        collection[key] = self._store(item)

    def clear(self):
        with self.condition:
            stored = [item for item in (*self.items.values(), *self.failed.values(), *self.leased.values())
                      if isinstance(item, StoredJob)]
            self.items.clear()
            self.failed.clear()
            self.leased.clear()
            for item in stored:
                try:
                    item.path.unlink(missing_ok=True)
                except OSError:
                    # Clear must still invalidate the epoch if cleanup is refused.
                    import logging
                    logging.exception('Could not delete cleared pending audio: %s', item.path)

    def __len__(self):
        with self.condition:
            return len(self.items)
