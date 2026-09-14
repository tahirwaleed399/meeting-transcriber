"""Microphone capture for WASAPI devices with non-extensible mix formats."""
import contextlib
import os
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
        self.stream.start()
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
