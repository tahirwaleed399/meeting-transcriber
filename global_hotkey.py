"""System-wide hotkey registration for Windows, polled from the Tk main loop."""
from __future__ import annotations

import ctypes
from ctypes import wintypes

MOD_ALT, MOD_CONTROL, MOD_SHIFT, MOD_WIN = 0x0001, 0x0002, 0x0004, 0x0008
MOD_NOREPEAT = 0x4000
WM_HOTKEY = 0x0312
PM_REMOVE = 0x0001

# Only the keys worth binding to a push-to-talk style action.
VK_CODES = {
    'F1': 0x70, 'F2': 0x71, 'F3': 0x72, 'F4': 0x73, 'F5': 0x74, 'F6': 0x75,
    'F7': 0x76, 'F8': 0x77, 'F9': 0x78, 'F10': 0x79, 'F11': 0x7A, 'F12': 0x7B,
    'Space': 0x20, 'Insert': 0x2D, 'Pause': 0x13, 'ScrollLock': 0x91,
}
for _c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789':
    VK_CODES[_c] = ord(_c)

MODIFIER_FLAGS = {'Ctrl': MOD_CONTROL, 'Shift': MOD_SHIFT, 'Alt': MOD_ALT, 'Win': MOD_WIN}


def parse_hotkey(text):
    """Turn 'Ctrl+Shift+Space' into (modifier flags, virtual key code)."""
    parts = [part.strip() for part in str(text).split('+') if part.strip()]
    if not parts:
        raise ValueError('Empty hotkey')
    modifiers = 0
    for part in parts[:-1]:
        name = part.capitalize() if part.lower() != 'win' else 'Win'
        if name not in MODIFIER_FLAGS:
            raise ValueError(f'Unknown modifier: {part}')
        modifiers |= MODIFIER_FLAGS[name]
    key = parts[-1]
    code = VK_CODES.get(key if len(key) > 1 else key.upper())
    if code is None:
        raise ValueError(f'Unsupported key: {key}')
    if not modifiers:
        raise ValueError('Choose at least one modifier so the key still works in other apps.')
    return modifiers, code


class GlobalHotkey:
    """Register one system-wide hotkey and deliver presses to a callback.

    RegisterHotKey(None, ...) posts WM_HOTKEY to the *thread* queue, which Tk's own
    event loop drains and discards before we can peek at it. Passing a real window
    handle instead routes the message through that window's WndProc, so we subclass
    a dedicated hidden Tk window and read WM_HOTKEY there.
    """
    ID = 0xA71E

    def __init__(self, on_press=None):
        self.user32 = ctypes.windll.user32 if hasattr(ctypes, 'windll') else None
        self.registered = None
        self.on_press = on_press
        self.hwnd = None
        self._old_proc = None
        self._proc = None
        self._pending = 0

    def attach(self, widget):
        """Subclass a Tk widget's window so it receives WM_HOTKEY."""
        if self.user32 is None or self.hwnd is not None:
            return
        widget.update_idletasks()
        self.hwnd = int(widget.winfo_id())
        GWLP_WNDPROC = -4
        proc_type = ctypes.WINFUNCTYPE(ctypes.c_longlong, wintypes.HWND, ctypes.c_uint,
                                       ctypes.c_ulonglong, ctypes.c_longlong)
        set_long = getattr(self.user32, 'SetWindowLongPtrW', self.user32.SetWindowLongW)
        call_old = self.user32.CallWindowProcW
        call_old.restype = ctypes.c_longlong
        call_old.argtypes = [ctypes.c_void_p, wintypes.HWND, ctypes.c_uint,
                             ctypes.c_ulonglong, ctypes.c_longlong]

        def wnd_proc(hwnd, message, wparam, lparam):
            if message == WM_HOTKEY and wparam == self.ID:
                self._pending += 1
                if self.on_press:
                    self.on_press()
                return 0
            return call_old(self._old_proc, hwnd, message, wparam, lparam)

        self._proc = proc_type(wnd_proc)
        set_long.restype = ctypes.c_void_p
        set_long.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_void_p]
        self._old_proc = set_long(self.hwnd, GWLP_WNDPROC,
                                  ctypes.cast(self._proc, ctypes.c_void_p))

    def register(self, hotkey):
        """Bind the hotkey, replacing any previous one. Returns None or an error string."""
        if self.user32 is None:
            return 'Global hotkeys need Windows.'
        try:
            modifiers, code = parse_hotkey(hotkey)
        except ValueError as exc:
            return str(exc)
        self.unregister()
        # MOD_NOREPEAT stops held keys from firing a stream of toggles.
        if not self.user32.RegisterHotKey(self.hwnd, self.ID, modifiers | MOD_NOREPEAT, code):
            error = ctypes.get_last_error() if ctypes.get_last_error() else 0
            if error == 1409 or not error:
                return f'{hotkey} is already taken by another application. Choose a different one.'
            return f'Could not register {hotkey} (Windows error {error}).'
        self.registered = hotkey
        return None

    def unregister(self):
        if self.user32 is not None and self.registered:
            self.user32.UnregisterHotKey(self.hwnd, self.ID)
        self.registered = None

    def drain(self):
        """Return how many presses arrived since the last call."""
        presses, self._pending = self._pending, 0
        return presses
