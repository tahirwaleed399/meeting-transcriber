"""Local streaming speech recognition with bounded drafts and distinct audio sources."""
from __future__ import annotations

from collections import deque
import logging
import os
from pathlib import Path
import queue
import site
import threading
import time
import uuid

import numpy as np

from transcript_core import Entry, LANGUAGES, PROFILES, ROOT, Settings
from audio_capture import CaptureProcess
from audio_jobs import AudioJob, Mailbox

RATE = 16000
FRAMES = 1536  # 96 ms; exactly three Silero frames.
_dll_handles = []


def configure_runtime():
    os.environ.setdefault('HF_HOME', str(ROOT / 'hf_cache'))
    if os.name != 'nt' or _dll_handles:
        return
    folders = []
    for folder in site.getsitepackages():
        folders.extend(Path(folder).glob('nvidia/*/bin'))
    if folders:
        os.environ['PATH'] = os.pathsep.join(map(str, folders)) + os.pathsep + os.environ.get('PATH', '')
        for folder in folders:
            try:
                _dll_handles.append(os.add_dll_directory(str(folder)))
            except OSError:
                logging.warning('Could not add optional GPU library directory: %s', folder)


class StreamingVAD:
    """Stateful adapter for the bundled Silero v6 ONNX model (faster-whisper 1.2.1)."""
    def __init__(self):
        from faster_whisper.vad import get_vad_model
        self.session = get_vad_model().session
        self.h = np.zeros((1, 1, 128), np.float32)
        self.c = self.h.copy()
        self.context = np.zeros(64, np.float32)

    def probability(self, audio):
        rms = float(np.sqrt(np.mean(audio ** 2)))
        # Modest gain helps quiet microphones; the neural detector still rejects noise.
        audio = audio * min(6., max(1., .02 / max(rms, 1e-6)))
        audio = np.pad(audio, (0, (-len(audio)) % 512))
        frames = audio.reshape(-1, 512)
        contexts = np.vstack((self.context, frames[:-1, -64:]))
        batch = np.concatenate((contexts, frames), axis=1).astype(np.float32)
        out, self.h, self.c = self.session.run(None, {'input': batch, 'h': self.h, 'c': self.c})
        self.context = frames[-1, -64:].copy()
        return float(np.max(out))


class SpeechSegmenter:
    """Preserve pre-roll, revise one utterance, finalize once at a speech boundary."""
    def __init__(self, source, epoch, publish, interval=.65, endpoint=.576):
        self.source, self.epoch, self.publish = source, epoch, publish
        self.interval, self.endpoint = interval, endpoint
        self.pre_roll = deque(maxlen=3)
        self.chunks = []
        self.key = None
        self.start = 0.
        self.samples = 0
        self.silence = 0.
        self.last_preview = 0.
        self.revision = 0
        self.previous_key = None
        self.carry = None
        self.carry_start = 0.

    def reset(self, epoch):
        self.__init__(self.source, epoch, self.publish, self.interval, self.endpoint)

    def feed(self, audio, start, probability):
        speech = probability >= (.32 if self.key else .5)
        if not self.key:
            if self.carry is not None:
                self.start = self.carry_start
                self.chunks = [self.carry, audio.copy()]
                self.carry = None
            else:
                self.pre_roll.append((start, audio.copy()))
                if not speech:
                    return
                self.start = self.pre_roll[0][0]
                self.chunks = [item[1] for item in self.pre_roll]
            self.key = uuid.uuid4().hex
            self.samples = sum(len(chunk) for chunk in self.chunks)
            self.pre_roll.clear()
        else:
            self.chunks.append(audio.copy())
            self.samples += len(audio)
        self.silence = 0. if speech else self.silence + len(audio) / RATE
        duration = self.samples / RATE
        if self.silence >= self.endpoint or (duration >= 18. and self.silence >= .192):
            self.flush()
        elif duration >= 24.:
            self.flush(truncated=True)
        elif duration >= .65 and duration - self.last_preview >= self.interval:
            self._publish(False)
            self.last_preview = duration

    def _publish(self, final, truncated=False):
        self.revision += 1
        audio = np.concatenate(self.chunks).astype(np.float32)
        self.publish(AudioJob(self.key, self.epoch, self.source, self.start, audio,
                              final, self.revision, time.monotonic(), self.previous_key, truncated))

    def flush(self, truncated=False):
        if self.key is None and self.carry is not None:
            self.key = uuid.uuid4().hex
            self.chunks = [self.carry]
            self.samples = len(self.carry)
            self.start = self.carry_start
            self.carry = None
        if self.key and self.samples >= RATE * .15:
            self._publish(True, truncated)
        if truncated:
            self.carry = np.concatenate(self.chunks)[-RATE:].copy()
            self.carry_start = self.start + (self.samples - len(self.carry)) / RATE
            self.previous_key = self.key
        else:
            self.previous_key = None
        self.chunks, self.key, self.samples = [], None, 0
        self.last_preview = self.silence = 0.
        self.pre_roll.clear()


class Recognizer:
    def __init__(self, settings, emit):
        self.settings, self.emit = settings, emit
        self.models = {}
        self.device = 'cpu'
        self.compute = 'int8'
        self.boundaries = {}

    def load(self, force_cpu=False):
        configure_runtime()
        import ctranslate2
        from faster_whisper import WhisperModel
        preview, final, _, _ = PROFILES[self.settings.profile]
        if self.settings.language != 'English':
            preview, final = preview.removesuffix('.en'), final.removesuffix('.en')
        self.preview_name, self.final_name = preview, final
        if not force_cpu and preview in self.models and final in self.models:
            return f'{self.device.upper()} · {preview} drafts / {final} final'
        self.device, self.compute = 'cpu', 'int8'
        if not force_cpu:
            try:
                if ctranslate2.get_cuda_device_count():
                    supported = ctranslate2.get_supported_compute_types('cuda')
                    self.device = 'cuda'
                    self.compute = 'int8_float16' if 'int8_float16' in supported else 'float16'
            except Exception as exc:
                self.emit('warning', message=f'GPU detection failed; using CPU. {exc}')
                self.device, self.compute = 'cpu', 'int8'
        self.models.clear()
        try:
            for name in dict.fromkeys((preview, final)):
                self.emit('status', message=f'Loading {name} on {self.device.upper()}… First use may download the model.')
                kwargs = dict(device=self.device, compute_type=self.compute,
                              download_root=str(ROOT / 'models'), cpu_threads=min(6, os.cpu_count() or 4), num_workers=1)
                try:
                    model = WhisperModel(name, local_files_only=True, **kwargs)
                except Exception as exc:
                    # Only missing model files justify a network download; runtime errors do not.
                    from huggingface_hub.errors import LocalEntryNotFoundError
                    if not isinstance(exc, LocalEntryNotFoundError):
                        raise
                    model = WhisperModel(name, **kwargs)
                # Construction is not proof of working CUDA. Consume a real decode.
                warmup = (.015 * np.sin(np.arange(RATE, dtype=np.float32) * (2 * np.pi * 220 / RATE))).astype(np.float32)
                segments, _ = model.transcribe(warmup, language='en', beam_size=1,
                                               vad_filter=False, condition_on_previous_text=False)
                list(segments)
                self.models[name] = model
        except Exception as exc:
            if self.device == 'cuda':
                self.emit('warning', message=f'GPU unavailable; switching to CPU. {exc}')
                self.models.clear()
                return self.load(force_cpu=True)
            raise
        return f'{self.device.upper()} · {preview} drafts / {final} final'

    def transcribe(self, job):
        model = self.models[self.final_name if job.final else self.preview_name]
        audio = job.audio
        rms = float(np.sqrt(np.mean(audio ** 2)))
        if 1e-6 < rms < .025:
            audio = audio * min(6., .05 / rms)
        beam = PROFILES[self.settings.profile][3] if job.final else 1
        segments, _ = model.transcribe(
            audio, language=LANGUAGES[self.settings.language], beam_size=beam,
            best_of=1, temperature=0., condition_on_previous_text=False,
            vad_filter=False, word_timestamps=bool(job.previous_key or job.truncated),
            initial_prompt=self.settings.vocabulary.strip() or None,
            no_speech_threshold=.6, log_prob_threshold=-1., compression_ratio_threshold=2.4,
        )
        segments = list(segments)
        if job.previous_key or job.truncated:
            lower = self.boundaries.get(job.previous_key, job.start)
            upper = job.start + len(audio) / RATE - (.5 if job.truncated else 0.)
            words = [word for segment in segments for word in (segment.words or [])
                     if job.start + (word.start + word.end) / 2 > lower - .02
                     and job.start + word.end <= upper + .02]
            if job.truncated:
                self.boundaries[job.key] = job.start + words[-1].end if words else upper
            if job.final and job.previous_key:
                self.boundaries.pop(job.previous_key, None)
            return ''.join(word.word for word in words).strip()
        return ' '.join(segment.text.strip() for segment in segments if segment.text.strip()).strip()


class TranscriptionEngine:
    def __init__(self, settings: Settings, events: queue.Queue, epoch=0, time_offset=0., *, pending_dir=None):
        self.settings, self.events, self.epoch = settings, events, epoch
        self.time_offset = time_offset
        self.mailbox = Mailbox(directory=pending_dir, settings=settings)
        self.state_lock = threading.RLock()
        self.decoding = False
        self.flush_pending = set()
        self.disarmed_at = None
        self.reset_at = time.monotonic()
        self.stop_at = None
        self.backlog_warned = False
        self.stop_event = threading.Event()
        self.captures_done = threading.Event()
        self.finished = threading.Event()
        self.origin = time.monotonic()
        self.capture_threads = []
        self.thread = None
        self.recognizer = Recognizer(settings, self.emit)
        self.ready = threading.Event()
        # Always-on pipeline counters, surfaced by the debug panel.
        self.stats = {}
        self.stats_lock = threading.Lock()
        # Dictation mode keeps capture threads alive but discards audio while disarmed,
        # so a burst starts instantly instead of reopening devices and reloading models.
        self.armed = threading.Event()
        self.armed.set()

    def sources(self):
        if self.settings.source == 'Computer + microphone':
            return ['Computer', 'You']
        return ['Computer'] if self.settings.source == 'Computer audio' else ['You']

    def can_record(self):
        stats = self.snapshot_stats()
        now = time.monotonic()
        return self.ready.is_set() and all(
            stats.get(source, {}).get('state') in ('capturing', 'idle')
            and now - stats[source].get('last_block', 0.) < 8.
            for source in self.sources())

    def arm(self):
        with self.state_lock:
            self.reset_at = time.monotonic()
            self.disarmed_at = None
            self.flush_pending.clear()
            self.armed.set()

    def disarm(self):
        with self.state_lock:
            if self.armed.is_set():
                self.disarmed_at = time.monotonic()
                self.flush_pending = set(self.sources()) if self.capture_threads else set()
                self.armed.clear()

    def is_drained(self):
        with self.state_lock:
            return not self.armed.is_set() and not self.flush_pending and not self.decoding and not len(self.mailbox)

    def retry_failed(self):
        with self.state_lock:
            self.mailbox.retry_failed()

    def note(self, source, **fields):
        """Record the latest per-source pipeline state for diagnostics."""
        with self.stats_lock:
            entry = self.stats.setdefault(source, {})
            for key, value in fields.items():
                if key.endswith('_n'):
                    entry[key] = entry.get(key, 0) + value
                elif key.endswith('_max'):
                    entry[key] = max(entry.get(key, 0.), value)
                else:
                    entry[key] = value

    def snapshot_stats(self):
        with self.stats_lock:
            return {source: dict(values) for source, values in self.stats.items()}

    def emit(self, kind, **data):
        self.events.put({'type': kind, 'engine': id(self), **data})

    def start(self, capture=True):
        self.thread = threading.Thread(target=self._run, args=(capture,), name='speech-decoder', daemon=True)
        self.thread.start()

    def clear(self, epoch):
        with self.state_lock:
            self.epoch = epoch
            self.reset_at = time.monotonic()
            self.mailbox.clear()
            if not self.armed.is_set():
                self.flush_pending.clear()

    def stop(self):
        with self.state_lock:
            if self.stop_event.is_set():
                return
            self.stop_at = time.monotonic()
            self.disarm()
            self.stop_event.set()

    def submit(self, job):
        self.note(job.source, jobs_n=1, last_job=('final' if job.final else 'draft'),
                  job_seconds=len(job.audio) / RATE)
        with self.state_lock:
            if job.epoch == self.epoch:
                self.mailbox.put(job)
                if len(self.mailbox) >= self.mailbox.max_items and not self.backlog_warned:
                    self.backlog_warned = True
                    self.emit('warning', message='Transcription is behind. Extra phrases are buffered locally; capture continues. Fastest mode reduces the delay.')
            else:
                self.note(job.source, dropped_n=1)

    def _decode(self, job):
        for attempt in range(3):
            if job.epoch != self.epoch:
                return None
            try:
                if (attempt == 1 and self.recognizer.device == 'cuda') or attempt == 2:
                    self.recognizer.load(force_cpu=True)
                return self.recognizer.transcribe(job)
            except Exception as exc:
                self.emit('warning', message=f'Decode attempt {attempt + 1}/3 failed; '
                          + ('retrying. ' if attempt < 2 else 'keeping the phrase for recovery. ') + str(exc))
                if attempt < 2:
                    time.sleep(.1 * (attempt + 1))
        if job.final:
            with self.state_lock:
                if job.epoch != self.epoch:
                    return None
                try:
                    self.mailbox.fail(job)
                    message = 'A phrase could not be decoded. Audio is saved locally; use Retry failed phrases or Recover saved audio.'
                except OSError as exc:
                    message = f'Cannot save failed audio: {exc}. Audio is retained in memory. Free disk space, then Resume to retry.'
                    self.stop()
                self.emit('decode_error', message=message, pending=self.mailbox.failed_count())
        return None

    def _run(self, capture):
        try:
            description = self.recognizer.load()
            if self.stop_event.is_set():
                return
            self.origin = time.monotonic()
            if capture:
                for source in self.sources():
                    thread = threading.Thread(target=self._capture, args=(source,),
                                              name=f'capture-{source}', daemon=True)
                    self.capture_threads.append(thread)
                    thread.start()
            self.ready.set()
            self.emit('ready', message=description)
            boundary_epoch = self.epoch
            while True:
                if self.stop_event.is_set() and not any(t.is_alive() for t in self.capture_threads) and not len(self.mailbox):
                    break
                with self.state_lock:
                    try:
                        job = self.mailbox.get(timeout=0)
                    except Exception as exc:
                        self.emit('decode_error', message=f'Cannot read a saved phrase: {exc}. Other phrases will continue.',
                                  pending=self.mailbox.failed_count())
                        job = None
                    self.decoding = job is not None
                if job is None or job.epoch != self.epoch:
                    self.decoding = False
                    time.sleep(.02)
                    continue
                started = time.monotonic()
                try:
                    if boundary_epoch != self.epoch:
                        self.recognizer.boundaries.clear()
                        boundary_epoch = self.epoch
                    text = self._decode(job)
                    if text is None:
                        continue
                    self.note(job.source, decodes_n=1, inference=time.monotonic() - started,
                              text_chars=len(text.strip()))
                    if job.epoch == self.epoch:
                        entry = Entry(job.key, job.source, job.start, job.start + len(job.audio) / RATE,
                                      text, job.final, job.revision)
                        self.emit('transcript', epoch=job.epoch, entry=entry,
                                  inference=time.monotonic() - started, backlog=len(self.mailbox),
                                  latency=time.monotonic() - job.created)
                    try:
                        self.mailbox.acknowledge(job)
                    except OSError as exc:
                        self.emit('warning', message=f'Phrase decoded, but its temporary audio could not be removed: {exc}')
                finally:
                    with self.state_lock:
                        self.decoding = False
        except Exception as exc:
            logging.exception('Transcription engine failed')
            self.emit('error', message=str(exc))
            self.stop()
        finally:
            self.stop()
            for thread in self.capture_threads:
                thread.join()
            try:
                self.mailbox.preserve_pending()
            except OSError as exc:
                self.emit('error', message=f'Some pending audio could not be saved: {exc}. Free disk space and Resume before closing.')
            self.finished.set()
            self.emit('stopped')

    def _capture(self, source):
        """Keep segmentation in the parent; restart native capture after any failure."""
        selection = self.settings.output_id if source == 'Computer' else self.settings.microphone_id
        while not self.stop_event.is_set():
            segmenter = SpeechSegmenter(source, self.epoch, self.submit, PROFILES[self.settings.profile][2])
            try:
                vad = StreamingVAD()
                with CaptureProcess(source, selection, RATE, FRAMES) as capture:
                    self._capture_loop(source, vad, segmenter, capture)
            except Exception as exc:
                self.note(source, state='error', error=f'{type(exc).__name__}: {exc}', errors_n=1)
                self.emit('capture_error', source=source, message=f'{source}: {exc} Retrying?')
            finally:
                try:
                    with self.state_lock:
                        segmenter.flush()
                except Exception as exc:
                    self.emit('error', message=f'Cannot buffer pending audio: {exc}. Free disk space, then Resume.')
                    self.stop()
                with self.state_lock:
                    self.flush_pending.discard(source)
            if not self.stop_event.is_set():
                self.stop_event.wait(1.)
        self.note(source, state='stopped')

    def _capture_loop(self, source, vad, segmenter, capture):
        silence_seconds = 0.
        device_name = source
        audio_ready = False
        while True:
            with self.state_lock:
                if self.stop_event.is_set() and source not in self.flush_pending:
                    return
                if self.stop_event.is_set() and time.monotonic() - self.stop_at >= .5:
                    self.emit('warning', message=f'{source} did not finish its audio read. Keeping buffered speech and closing the device.')
                    return
            message = capture.receive(timeout=.1)
            if message is None:
                continue
            kind = message.pop('type')
            if kind in ('capture_error', 'device_changed'):
                raise RuntimeError(message['message'])
            if kind != 'audio':
                if kind == 'capture_ready':
                    device_name = message['message']
                    self.note(source, state='opening', device=device_name, error='',
                              digital_silence=0., last_block=0.)
                    continue
                self.emit(kind, source=source, **message)
                continue
            if message['discontinuities']:
                self.emit('discontinuity', source=source, count=message['discontinuities'])
            captured = message['captured']
            data = np.asarray(message['data'], dtype=np.float32)
            if not len(data):
                continue
            audio = data.mean(axis=1) if data.ndim == 2 else data
            if audio.ndim != 1 or not np.isfinite(audio).all():
                raise RuntimeError('The audio device returned invalid samples.')
            if not audio_ready:
                audio_ready = True
                self.note(source, state='capturing', last_block=captured, error='')
                self.emit('capture_ready', source=source, message=device_name)
            with self.state_lock:
                if segmenter.epoch != self.epoch:
                    segmenter.reset(self.epoch)
                    vad = StreamingVAD()
                if captured < self.reset_at:
                    continue
                if not self.armed.is_set() and source not in self.flush_pending:
                    self.note(source, state='idle', digital_silence=0., last_block=captured)
                    silence_seconds = 0.
                    continue
                start = self.time_offset + captured - self.origin - len(audio) / RATE
                rms = float(np.sqrt(np.mean(audio ** 2)))
                self.emit('level', source=source, rms=rms)
                probability = vad.probability(audio)
                segmenter.feed(audio, max(0., start), probability)
                block_peak = float(np.max(np.abs(audio)))
                silence_seconds = silence_seconds + len(audio) / RATE if block_peak < 1e-4 else 0.
                self.note(source, state='capturing', blocks_n=1, rms=rms,
                          peak=block_peak, peak_max=block_peak, digital_silence=silence_seconds,
                          last_block=captured, speech=probability,
                          channels=data.shape[1] if data.ndim == 2 else 1,
                          utterance=bool(segmenter.key), buffered=segmenter.samples / RATE,
                          silence=segmenter.silence)
                if not self.armed.is_set() and captured >= self.disarmed_at:
                    segmenter.flush()
                    self.flush_pending.discard(source)
                    self.note(source, state='idle', digital_silence=0.)
