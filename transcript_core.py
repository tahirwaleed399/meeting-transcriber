"""Transcript state and local persistence. No GUI, microphone or model side effects."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from datetime import datetime
import json
from pathlib import Path
import queue
import threading
import uuid

ROOT = Path(__file__).resolve().parent
PROFILES = {
    'Balanced': ('tiny.en', 'small.en', .65, 3),
    'Fastest': ('tiny.en', 'tiny.en', .45, 1),
    'Best accuracy': ('base.en', 'small.en', .85, 5),
}
LANGUAGES = {'English': 'en', 'Auto detect': None, 'Urdu': 'ur', 'Hindi': 'hi',
             'Spanish': 'es', 'French': 'fr', 'German': 'de', 'Arabic': 'ar'}


@dataclass
class Settings:
    profile: str = 'Balanced'
    source: str = 'Computer audio'
    output_id: str = ''
    microphone_id: str = ''
    language: str = 'English'
    vocabulary: str = ''
    auto_copy: bool = False
    always_on_top: bool = False
    font_size: int = 16
    dictation: bool = False
    hotkey: str = 'Ctrl+Shift+Space'

    @classmethod
    def load(cls, path: Path = ROOT / 'settings.json'):
        try:
            raw = json.loads(path.read_text(encoding='utf-8'))
            if not isinstance(raw, dict):
                return cls()
            allowed = {field.name for field in fields(cls)}
            obj = cls(**{k: v for k, v in raw.items() if k in allowed})
            if obj.profile not in PROFILES or obj.language not in LANGUAGES:
                return cls()
            if obj.source not in ('Computer audio', 'Microphone', 'Computer + microphone'):
                obj.source = 'Computer audio'
            obj.font_size = max(12, min(26, int(obj.font_size)))
            obj.dictation = bool(obj.dictation)
            if not isinstance(obj.hotkey, str) or not obj.hotkey.strip():
                obj.hotkey = 'Ctrl+Shift+Space'
            return obj
        except (OSError, ValueError, TypeError):
            return cls()

    def save(self, path: Path = ROOT / 'settings.json'):
        atomic_write(path, json.dumps(asdict(self), indent=2))


def atomic_write(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8', newline='\n') as stream:
        stream.write(text)
        stream.flush()
        import os
        os.fsync(stream.fileno())
    temporary.replace(path)


@dataclass
class Entry:
    key: str
    source: str
    start: float
    end: float
    text: str
    final: bool
    revision: int = 0


class Transcript:
    """Replace a single utterance hypothesis; never append overlapping guesses."""
    def __init__(self):
        self.epoch = 0
        self.entries: dict[str, Entry] = {}
        self.undo_entries: list[Entry] = []

    def apply(self, epoch: int, entry: Entry) -> bool:
        if epoch != self.epoch:
            return False
        old = self.entries.get(entry.key)
        if old and (old.final or old.revision > entry.revision):
            return False
        if old == entry:
            return False
        if not entry.text.strip():
            if entry.final:
                return self.entries.pop(entry.key, None) is not None
            return False
        self.entries[entry.key] = entry
        return True

    def ordered(self):
        return sorted(self.entries.values(), key=lambda entry: (entry.start, entry.key))

    def text(self, labels=False):
        entries = self.ordered()
        if labels:
            return '\n\n'.join(f'{entry.source}  {timestamp(entry.start)}\n{entry.text}' for entry in entries)
        return '\n\n'.join(entry.text for entry in entries)

    def clear(self):
        if self.entries:
            self.undo_entries = self.ordered()
        self.entries = {}
        self.epoch += 1
        return self.epoch

    def undo_clear(self):
        if not self.undo_entries:
            return False
        # Restored snapshots are immutable; in-flight old hypotheses remain rejected.
        for old in self.undo_entries:
            restored = Entry(**{**asdict(old), 'key': 'restored-' + uuid.uuid4().hex, 'final': True})
            self.entries[restored.key] = restored
        self.undo_entries = []
        return True

    def snapshot(self):
        return {'version': 1, 'updated': datetime.now().isoformat(timespec='seconds'),
                'entries': [asdict(entry) for entry in self.ordered()], 'text': self.text()}

    def restore(self, snapshot):
        restored = []
        for data in snapshot.get('entries', []):
            restored.append(Entry(**{**data, 'key': 'restored-' + uuid.uuid4().hex, 'final': True}))
        self.entries.update((entry.key, entry) for entry in restored)


def timestamp(seconds: float):
    seconds = max(0, int(seconds))
    return f'{seconds // 3600:02}:{seconds // 60 % 60:02}:{seconds % 60:02}'


def srt_text(entries):
    def stamp(seconds):
        ms = round(max(0, seconds) * 1000)
        return f'{ms // 3600000:02}:{ms // 60000 % 60:02}:{ms // 1000 % 60:02},{ms % 1000:03}'
    return '\n\n'.join(f'{i}\n{stamp(e.start)} --> {stamp(max(e.end, e.start + .1))}\n{e.text}'
                       for i, e in enumerate(entries, 1)) + '\n'


class Autosaver:
    """One writer, coalesced snapshots, atomic recovery file, explicit final flush."""
    def __init__(self, events, directory=ROOT / 'sessions'):
        self.directory = Path(directory)
        self.path = self.directory / (datetime.now().strftime('%Y-%m-%d_%H-%M-%S_') + uuid.uuid4().hex[:6] + '.txt')
        self.events = events
        self.pending = queue.Queue(maxsize=1)
        self.stop_event = threading.Event()
        self.last_error = None
        self.thread = threading.Thread(target=self._run, name='transcript-save', daemon=True)
        self.thread.start()

    def submit(self, snapshot):
        try:
            self.pending.put_nowait(snapshot)
        except queue.Full:
            try:
                self.pending.get_nowait()
            except queue.Empty:
                pass
            self.pending.put_nowait(snapshot)

    def _run(self):
        while not self.stop_event.is_set() or not self.pending.empty():
            try:
                snapshot = self.pending.get(timeout=.15)
            except queue.Empty:
                continue
            try:
                atomic_write(self.path, snapshot['text'])
                atomic_write(self.directory / 'latest.json', json.dumps(snapshot, ensure_ascii=False, indent=2))
                self.last_error = None
                self.events.put({'type': 'saved', 'path': str(self.path)})
            except OSError as exc:
                self.last_error = str(exc)
                self.events.put({'type': 'save_error', 'message': str(exc)})

    def close(self):
        self.stop_event.set()
        self.thread.join()
