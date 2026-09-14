"""Microphone capture for WASAPI devices with non-extensible mix formats."""
import contextlib
import os
import multiprocessing
import time
import warnings


@contextlib.contextmanager
def com_apartment():
    """Join this thread to the COM multithreaded apartment for its lifetime.

    WASAPI is COM, and COM apartments are per-thread. soundcard only calls
    CoInitializeEx once, on whichever thread imports it, so every call from a
    capture worker otherwise fails with CO_E_NOTINITIALIZED (0x800401f0).
    """
    if os.name != 'nt':
        yield
        return
    # Import soundcard first. Its module-level _COMLibrary() calls CoInitializeEx
    # and rejects anything but S_OK, so it fails with S_FALSE (0x1) if this thread
    # is already in an apartment. Importing first makes that call the one that wins.
    import soundcard  # noqa: F401
    import ctypes
    COINIT_MULTITHREADED = 0x0
    RPC_E_CHANGED_MODE = -2147417850  # 0x80010106: already in a different apartment.
    ole32 = ctypes.windll.ole32
    hresult = ole32.CoInitializeEx(None, COINIT_MULTITHREADED)
    # A thread already in an apartment stays in it, and must not uninitialize.
    owned = hresult >= 0
    if hresult < 0 and hresult != RPC_E_CHANGED_MODE:
        raise OSError(f'Cannot initialize COM on this thread (0x{hresult & 0xffffffff:08x}).')
    try:
        yield
    finally:
        if owned:
            ole32.CoUninitialize()


class MicrophoneRecorder:
    """Adapt sounddevice's blocking input stream to our capture loop."""
    def __init__(self, microphone, samplerate=16000):
        import sounddevice as sd
        self.sd, self.microphone, self.samplerate = sd, microphone, samplerate
        self.stream = None

    def __enter__(self):
        sd = self.sd
        def candidates():
            apis = sd.query_hostapis()
            return [(index, device) for index, device in enumerate(sd.query_devices())
                    if device['max_input_channels'] > 0
                    and apis[device['hostapi']]['name'] == 'Windows WASAPI'
                    and device['name'] == self.microphone.name]
        matches = candidates()
        if not matches:
            # PortAudio caches endpoints. There is no sounddevice stream open
            # at this point, and the app has one microphone track per engine.
            sd._terminate()
            sd._initialize()
            matches = candidates()
        if len(matches) != 1:
            raise RuntimeError('Cannot uniquely match the selected microphone. Refresh devices or select another microphone.')
        index, info = matches[0]
        self.stream = sd.InputStream(device=index, samplerate=self.samplerate,
            channels=min(self.microphone.channels, info['max_input_channels']),
            dtype='float32', blocksize=0, latency='low',
            extra_settings=sd.WasapiSettings(auto_convert=True))
        try:
            self.stream.start()
        except Exception:
            self.stream.close()
            self.stream = None
            raise
        return self

    def record(self, numframes):
        data, overflowed = self.stream.read(numframes)
        if overflowed:
            warnings.warn('Microphone audio discontinuity', RuntimeWarning)
        return data

    def __exit__(self, exc_type, exc, tb):
        if self.stream:
            try:
                self.stream.stop()
            finally:
                self.stream.close()


def open_recorder(microphone, source, samplerate=16000):
    if source == 'You':
        return MicrophoneRecorder(microphone, samplerate)
    return microphone.recorder(samplerate=samplerate,
        channels=list(range(microphone.channels)), blocksize=samplerate // 4)


def capture_worker(source, selection, messages, stop, rate, frames):
    """Own all native device calls in a process that the parent can restart."""
    def send(kind, **data):
        if not stop.is_set():
            messages.send({'type': kind, **data})

    try:
        with com_apartment():
            import soundcard as sc

            def device():
                if source == 'Computer':
                    speaker = sc.get_speaker(selection) if selection else sc.default_speaker()
                    if speaker is None:
                        raise RuntimeError('The selected output is unavailable. Choose another output or Follow Windows default.')
                    return sc.get_microphone(speaker.id, include_loopback=True)
                return sc.get_microphone(selection) if selection else sc.default_microphone()

            mic = device()
            if mic is None:
                raise RuntimeError('No audio device is available.')
            send('device', message=mic.name)
            if source == 'Computer' and selection:
                default = sc.default_speaker()
                if default is not None and default.id != mic.id:
                    send('wrong_output', selected=mic.name, playing=default.name)
            with open_recorder(mic, source, rate) as recorder:
                send('capture_ready', message=mic.name)
                check_at = time.monotonic() + 1.
                while not stop.is_set():
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter('always')
                        data = recorder.record(numframes=frames)
                    send('audio', data=data, captured=time.monotonic(), discontinuities=len(caught))
                    if not selection and time.monotonic() >= check_at:
                        check_at = time.monotonic() + 1.
                        if device().id != mic.id:
                            send('device_changed', message=f'{source} device changed. Reconnecting…')
                            return
    except Exception as exc:
        send('capture_error', message=f'{source}: {type(exc).__name__}: {exc}')
    finally:
        # A single writer and no background queue feeder: EOF remains observable
        # if a native crash interrupts a message.
        messages.close()


class CaptureProcess:
    """Bounded reads and shutdown, including device opening and native driver hangs."""
    def __init__(self, source, selection, rate=16000, frames=1536, *, timeout=8., startup_timeout=30., target=capture_worker):
        context = multiprocessing.get_context('spawn')
        self.messages, self.writer = context.Pipe(duplex=False)
        self.stop_event = context.Event()
        self.process = context.Process(target=target,
            args=(source, selection, self.writer, self.stop_event, rate, frames),
            name=f'audio-{source}', daemon=True)
        self.timeout = timeout
        self.startup_timeout = startup_timeout
        self.received = False
        self.last_message = time.monotonic()
        self.started = False

    def __enter__(self):
        try:
            self.process.start()
            self.started = True
            self.writer.close()
            self.last_message = time.monotonic()
            return self
        except Exception:
            self.messages.close()
            self.writer.close()
            raise

    def receive(self, timeout=.1):
        if not self.messages.poll(timeout):
            if not self.process.is_alive():
                raise RuntimeError(f'Audio worker exited (code {self.process.exitcode}).')
            if time.monotonic() - self.last_message > (self.timeout if self.received else self.startup_timeout):
                raise TimeoutError('The audio device stopped responding. Reconnecting…')
            return None
        try:
            message = self.messages.recv()
        except (EOFError, OSError) as exc:
            raise RuntimeError('Audio worker disconnected. Reconnecting…') from exc
        self.last_message = time.monotonic()
        self.received = True
        return message

    def __exit__(self, *args):
        self.stop_event.set()
        if self.started:
            self.process.join(timeout=.3)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=1.)
            if not self.process.is_alive():
                self.process.close()
        self.messages.close()
