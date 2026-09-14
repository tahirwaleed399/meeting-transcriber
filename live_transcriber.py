"""LiveScribe desktop application. Importing this module has no startup side effects."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import queue
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from audio_capture import com_apartment
from global_hotkey import GlobalHotkey
from transcript_core import Autosaver, LANGUAGES, PROFILES, ROOT, Settings, Transcript, atomic_write, srt_text, timestamp
from transcription_engine import TranscriptionEngine

BG, PANEL, FIELD = '#0b1220', '#121e30', '#19283d'
TEXT, MUTED, ACCENT, LINE, RED = '#ecf2fa', '#9cafc8', '#68e0c4', '#293950', '#f9a5ae'


class TranscriberApp:
    def __init__(self, root=None, *, autostart=False, session_dir=None):
        if root is None and os.name == 'nt':
            try:
                import ctypes
                ctypes.windll.shcore.SetProcessDpiAwareness(1)
            except (AttributeError, OSError):
                pass
        self.root = root or tk.Tk()
        self.root.title('LiveScribe · Meeting transcript')
        width = min(1400, self.root.winfo_screenwidth() - 80)
        height = min(980, self.root.winfo_screenheight() - 100)
        self.root.geometry(f'{width}x{height}')
        self.root.minsize(min(1040, width), min(680, height))
        self.root.configure(bg=BG)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=1)
        self.settings = Settings.load()
        self.events = queue.Queue()
        self.transcript = Transcript()
        self.session_dir = Path(session_dir) if session_dir else ROOT / 'sessions'
        self.previous = None
        try:
            self.previous = json.loads((self.session_dir / 'latest.json').read_text(encoding='utf-8'))
            if not isinstance(self.previous, dict):
                self.previous = None
        except (OSError, ValueError):
            pass
        self.saver = Autosaver(self.events, self.session_dir)
        self.engine = None
        self.closing = False
        self.dirty = False
        self.engine_error = ''
        self.capture_errors = {}
        self.save_finished = threading.Event()
        self.session_started = time.monotonic()
        self.rendered = ''
        self.last_autocopy = ''
        self.copy_after = self.toast_after = None
        self.device_maps = {'output': {'Follow Windows default': ''}, 'microphone': {'Follow Windows default': ''}}
        self.last_levels = {'Computer': 0., 'You': 0.}
        self.discontinuities = 0
        self.settings_widgets = []
        self.debug_window = self.debug_text = None
        self.debug_history, self.debug_rows = [], {}
        self.debug_summary = None
        self.silence_warned = False
        self.hotkey = GlobalHotkey()
        self.dictating = False
        self.burst_started = 0.
        self.burst_settle_ms = 1400  # Let the endpoint pause and final decode land.
        self._style()
        self._build()
        self._shortcuts()
        self.hotkey.attach(self.root)
        # Restore a saved dictation binding; a taken hotkey disables the mode with a notice.
        if self.settings.dictation:
            error = self.hotkey.register(self.settings.hotkey)
            if error:
                self.dictation_var.set(False)
                self.settings.dictation = False
                self.root.after(600, lambda: self.toast(error, True))
        self.update_dictation_hint()
        self.root.protocol('WM_DELETE_WINDOW', self.close)
        self.root.attributes('-topmost', self.settings.always_on_top)
        self.root.after(50, self.poll)
        self.refresh_devices()
        if autostart:
            self.root.after(300, self.toggle_recording)

    def _style(self):
        style = ttk.Style(self.root)
        style.theme_use('clam')
        style.configure('TCombobox', fieldbackground=FIELD, background=FIELD, foreground=TEXT,
                        arrowcolor=MUTED, bordercolor=LINE, lightcolor=FIELD, darkcolor=FIELD, padding=7)
        style.map('TCombobox', fieldbackground=[('readonly', FIELD), ('disabled', PANEL)],
                  foreground=[('readonly', TEXT), ('disabled', MUTED)], selectbackground=[('readonly', FIELD)],
                  selectforeground=[('readonly', TEXT)])
        style.configure('Vertical.TScrollbar', background=FIELD, troughcolor=PANEL,
                        bordercolor=PANEL, arrowcolor=MUTED, width=11)
        self.root.option_add('*TCombobox*Listbox.background', FIELD)
        self.root.option_add('*TCombobox*Listbox.foreground', TEXT)
        self.root.option_add('*TCombobox*Listbox.selectBackground', LINE)

    def label(self, parent, text='', *, size=10, color=MUTED, bold=False, **kwargs):
        return tk.Label(parent, text=text, bg=parent.cget('bg'), fg=color,
                        font=('Segoe UI', size, 'bold' if bold else 'normal'), **kwargs)

    def button(self, parent, text, command, *, primary=False, danger=False, compact=False):
        return tk.Button(parent, text=text, command=command,
                         bg=ACCENT if primary else FIELD, fg=BG if primary else RED if danger else TEXT,
                         activebackground='#93efda' if primary else LINE, activeforeground=BG if primary else TEXT,
                         disabledforeground='#7386a1', relief='flat', bd=0, cursor='hand2',
                         padx=12 if compact else 17, pady=7 if compact else 11,
                         font=('Segoe UI', 10, 'bold'), highlightthickness=1,
                         highlightbackground=ACCENT if primary else LINE, takefocus=True)

    def _build(self):
        header = tk.Frame(self.root, bg=BG, padx=26, pady=20)
        header.grid(row=0, column=0, sticky='ew')
        brand = tk.Frame(header, bg=BG)
        brand.pack(side='left')
        self.label(brand, 'LiveScribe', size=24, color=TEXT, bold=True).pack(anchor='w')
        self.label(brand, 'A clear record of the conversation.', size=10).pack(anchor='w', pady=(2, 0))
        badge = tk.Frame(header, bg=PANEL, padx=14, pady=8, highlightthickness=1, highlightbackground=LINE)
        badge.pack(side='right')
        self.label(badge, 'LOCAL TRANSCRIPTION', size=9, color=ACCENT, bold=True).pack()
        self.label(badge, 'Audio stays on this computer', size=9).pack(pady=(3, 0))
        body = tk.Frame(self.root, bg=BG, padx=26)
        body.grid(row=1, column=0, sticky='nsew')
        sidebar_holder = tk.Frame(body, bg=PANEL, width=292, highlightthickness=1, highlightbackground=LINE)
        sidebar_holder.pack(side='left', fill='y', padx=(0, 18))
        sidebar_holder.pack_propagate(False)
        sidebar_scroll = ttk.Scrollbar(sidebar_holder, orient='vertical')
        sidebar_scroll.pack(side='right', fill='y')
        sidebar_canvas = tk.Canvas(sidebar_holder, bg=PANEL, highlightthickness=0, yscrollcommand=sidebar_scroll.set)
        sidebar_canvas.pack(side='left', fill='both', expand=True)
        sidebar_scroll.config(command=sidebar_canvas.yview)
        sidebar = tk.Frame(sidebar_canvas, bg=PANEL, padx=15, pady=18)
        sidebar_window = sidebar_canvas.create_window(0, 0, window=sidebar, anchor='nw')
        sidebar.bind('<Configure>', lambda event: sidebar_canvas.configure(scrollregion=sidebar_canvas.bbox('all')))
        sidebar_canvas.bind('<Configure>', lambda event: sidebar_canvas.itemconfigure(sidebar_window, width=event.width))
        sidebar_canvas.bind('<MouseWheel>', lambda event: sidebar_canvas.yview_scroll(-int(event.delta / 120), 'units'))
        self.label(sidebar, 'SESSION SETUP', size=10, bold=True, color=TEXT).pack(anchor='w')
        self.label(sidebar, 'Pause to change your audio settings.', size=9).pack(anchor='w', pady=(4, 13))
        self.source_var = tk.StringVar(value=self.settings.source)
        self.combo(sidebar, 'Capture', self.source_var, ['Computer audio', 'Microphone', 'Computer + microphone'])
        self.output_var = tk.StringVar(value='Follow Windows default')
        self.output_combo = self.combo(sidebar, 'Computer output', self.output_var, ['Follow Windows default'])
        self.mic_var = tk.StringVar(value='Follow Windows default')
        self.mic_combo = self.combo(sidebar, 'Microphone', self.mic_var, ['Follow Windows default'])
        self.refresh_button = self.button(sidebar, 'Refresh audio devices', self.refresh_devices, compact=True)
        self.refresh_button.pack(fill='x', pady=(0, 14))
        self.profile_var = tk.StringVar(value=self.settings.profile)
        self.combo(sidebar, 'Transcription mode', self.profile_var, list(PROFILES))
        self.profile_hint = self.label(sidebar, '', size=9, wraplength=230, justify='left')
        self.profile_hint.pack(anchor='w', pady=(0, 10))
        self.profile_var.trace_add('write', lambda *_: self.update_profile_hint())
        self.update_profile_hint()
        self.language_var = tk.StringVar(value=self.settings.language)
        self.combo(sidebar, 'Language', self.language_var, list(LANGUAGES))
        self.label(sidebar, 'Names & specialist vocabulary', size=9).pack(anchor='w', pady=(2, 5))
        self.vocabulary = tk.Entry(sidebar, bg=FIELD, fg=TEXT, insertbackground=TEXT, relief='flat',
                                   font=('Segoe UI', 10), highlightthickness=1, highlightbackground=LINE)
        self.vocabulary.insert(0, self.settings.vocabulary)
        self.vocabulary.pack(fill='x', ipady=7, pady=(0, 13))
        self.settings_widgets.append(self.vocabulary)
        self.label(sidebar, 'DICTATION MODE', size=9, bold=True, color=TEXT).pack(anchor='w', pady=(4, 5))
        self.dictation_var = tk.BooleanVar(value=self.settings.dictation)
        self.dictation_check = tk.Checkbutton(
            sidebar, text='Hotkey bursts', variable=self.dictation_var, command=self.toggle_dictation_mode,
            bg=PANEL, fg=TEXT, selectcolor=FIELD, activebackground=PANEL, activeforeground=TEXT,
            font=('Segoe UI', 10), relief='flat', highlightthickness=0, cursor='hand2', anchor='w')
        self.dictation_check.pack(anchor='w')
        self.hotkey_var = tk.StringVar(value=self.settings.hotkey)
        hotkey_row = tk.Frame(sidebar, bg=PANEL)
        hotkey_row.pack(fill='x', pady=(4, 0))
        self.label(hotkey_row, 'Hotkey', size=9, width=7, anchor='w').pack(side='left')
        self.hotkey_combo = ttk.Combobox(hotkey_row, textvariable=self.hotkey_var, state='readonly',
                                         values=['Ctrl+Shift+Space', 'Ctrl+Shift+D', 'Ctrl+Alt+Space',
                                                 'Ctrl+Alt+D', 'Shift+F9', 'Ctrl+Shift+F9', 'Alt+F9'],
                                         font=('Segoe UI', 9))
        self.hotkey_combo.pack(side='left', fill='x', expand=True)
        self.hotkey_combo.bind('<<ComboboxSelected>>', self.change_hotkey)
        self.dictation_hint = self.label(sidebar, '', size=9, wraplength=225, justify='left')
        self.dictation_hint.pack(anchor='w', pady=(5, 12))
        self.label(sidebar, 'INPUT ACTIVITY', size=9, bold=True, color=TEXT).pack(anchor='w', pady=(4, 7))
        self.meters = {}
        for source in ('Computer', 'You'):
            row = tk.Frame(sidebar, bg=PANEL)
            row.pack(fill='x', pady=3)
            self.label(row, source, size=9, width=9, anchor='w').pack(side='left')
            meter = tk.Canvas(row, bg=FIELD, height=7, width=138, highlightthickness=0)
            meter.pack(side='left')
            meter.create_rectangle(0, 0, 0, 8, fill=ACCENT, outline='', tags='level')
            self.meters[source] = meter
        self.source_hint = self.label(sidebar, 'Use headphones when capturing both sources.', size=9,
                                      wraplength=225, justify='left')
        self.source_hint.pack(anchor='w', pady=(8, 10))
        self.button(sidebar, 'Open saved sessions', self.open_sessions, compact=True).pack(side='bottom', fill='x')
        self.button(sidebar, 'Diagnostics (F12)', self.toggle_debug, compact=True).pack(side='bottom', fill='x', pady=(0, 6))
        def scroll_sidebar(event):
            sidebar_canvas.yview_scroll(-int(event.delta / 120), 'units')
            return 'break'
        def bind_scroll(widget):
            widget.bind('<MouseWheel>', scroll_sidebar)
            for child in widget.winfo_children():
                bind_scroll(child)
        bind_scroll(sidebar)
        main = tk.Frame(body, bg=BG)
        main.pack(side='left', fill='both', expand=True)
        main.grid_columnconfigure(0, weight=1)
        main.grid_rowconfigure(1, weight=1)
        toolbar = tk.Frame(main, bg=BG)
        toolbar.grid(row=0, column=0, sticky='ew', pady=(0, 12))
        self.record_button = self.button(toolbar, 'Start listening', self.toggle_recording, primary=True)
        self.record_button.pack(side='left', padx=(0, 10))
        self.copy_button = self.button(toolbar, 'Copy all', self.copy_all)
        self.copy_button.pack(side='left', padx=(0, 7))
        self.copy_clear_button = self.button(toolbar, 'Copy & clear', self.copy_and_clear)
        self.copy_clear_button.pack(side='left', padx=(0, 7))
        self.clear_button = self.button(toolbar, 'Clear', self.clear, danger=True)
        self.clear_button.pack(side='left', padx=(0, 7))
        self.save_button = self.button(toolbar, 'Save as…', self.save_as)
        self.save_button.pack(side='left')
        card = tk.Frame(main, bg=PANEL, highlightthickness=1, highlightbackground=LINE)
        card.grid(row=1, column=0, sticky='nsew')
        card.grid_columnconfigure(0, weight=1)
        card.grid_rowconfigure(2, weight=1)
        card_header = tk.Frame(card, bg=PANEL, padx=20, pady=15)
        card_header.grid(row=0, column=0, sticky='ew')
        self.label(card_header, 'Live transcript', size=14, bold=True, color=TEXT).pack(side='left')
        self.count_label = self.label(card_header, '0 words', size=10)
        self.count_label.pack(side='right')
        tk.Frame(card, bg=LINE, height=1).grid(row=1, column=0, sticky='ew')
        content = tk.Frame(card, bg=PANEL)
        content.grid(row=2, column=0, sticky='nsew')
        scrollbar = ttk.Scrollbar(content, orient='vertical')
        scrollbar.pack(side='right', fill='y')
        self.text_box = tk.Text(content, wrap='word', bg=PANEL, fg=TEXT, relief='flat', bd=0,
                                font=('Segoe UI', self.settings.font_size), padx=22, pady=20,
                                selectbackground='#31566b', selectforeground='white',
                                spacing1=2, spacing3=9, insertbackground=ACCENT,
                                yscrollcommand=scrollbar.set, state='disabled', cursor='xterm')
        self.text_box.pack(side='left', fill='both', expand=True)
        scrollbar.config(command=self.text_box.yview)
        self.text_box.tag_configure('draft', foreground='#87cbbf')
        self.empty_frame = tk.Frame(content, bg=PANEL)
        self.empty_frame.place(relx=.5, rely=.42, anchor='center')
        self.label(self.empty_frame, 'Ready when you are.', size=23, bold=True, color=TEXT).pack()
        self.label(self.empty_frame, 'Choose your audio source and press Start listening.\nWords appear here as the conversation unfolds.',
                   size=11, justify='center').pack(pady=(12, 0))
        self.label(self.empty_frame, 'Copy everything at any time. Clear starts a fresh transcript.', size=9).pack(pady=(24, 0))
        bottom = tk.Frame(card, bg=PANEL, padx=16, pady=10)
        bottom.grid(row=3, column=0, sticky='ew')
        self.selection_button = self.button(bottom, 'Copy selection', self.copy_selection, compact=True)
        self.selection_button.pack(side='left')
        self.undo_button = self.button(bottom, 'Undo clear', self.undo_clear, compact=True)
        self.undo_button.pack(side='left', padx=7)
        self.undo_button.config(state='disabled')
        self.restore_button = self.button(bottom, 'Restore previous', self.restore_previous, compact=True)
        if self.previous and self.previous.get('entries'):
            self.restore_button.pack(side='left')
        self.button(bottom, 'A−', lambda: self.zoom(-1), compact=True).pack(side='right')
        self.button(bottom, 'A+', lambda: self.zoom(1), compact=True).pack(side='right', padx=6)
        options = tk.Frame(main, bg=BG, pady=11)
        options.grid(row=2, column=0, sticky='ew')
        self.follow_var = tk.BooleanVar(value=True)
        self.autocopy_var = tk.BooleanVar(value=self.settings.auto_copy)
        self.top_var = tk.BooleanVar(value=self.settings.always_on_top)
        for text, variable, command in [('Follow live', self.follow_var, self.follow_live),
                                        ('Auto-copy', self.autocopy_var, self.save_preferences),
                                        ('Keep on top', self.top_var, self.top_changed)]:
            tk.Checkbutton(options, text=text, variable=variable, command=command,
                           bg=BG, fg=MUTED, activebackground=BG, activeforeground=TEXT,
                           selectcolor=FIELD, font=('Segoe UI', 10), bd=0).pack(side='left', padx=(0, 15))
        self.timer_label = self.label(options, '00:00:00', size=10)
        self.timer_label.pack(side='right')
        status_frame = tk.Frame(self.root, bg=BG, padx=26, pady=12)
        status_frame.grid(row=2, column=0, sticky='ew')
        self.status_label = self.label(status_frame, '●  Ready · Select a source to begin', size=10, color=ACCENT, anchor='w')
        self.status_label.pack(fill='x')
        detail_row = tk.Frame(status_frame, bg=BG)
        detail_row.pack(fill='x', pady=(5, 0))
        self.detail_label = self.label(detail_row, 'Ctrl+Shift+C copy all  ·  Ctrl+Shift+X copy & clear  ·  Ctrl+L clear  ·  Ctrl+S save  ·  Ctrl+Space pause  ·  F12 diagnostics', size=9, anchor='w')
        self.detail_label.pack(side='left')
        self.saved_label = self.label(detail_row, 'Autosave ready', size=9)
        self.saved_label.pack(side='right')
        self.toast_label = self.label(self.root, '', size=10, color=ACCENT, anchor='w', padx=26)
        self.toast_label.grid(row=3, column=0, sticky='ew', pady=(0, 8))
        self.menu = tk.Menu(self.root, tearoff=False, bg=FIELD, fg=TEXT)
        for text, action in [('Copy selection', self.copy_selection), ('Copy all', self.copy_all),
                             ('Copy all & clear', self.copy_and_clear),
                             ('Copy latest phrase', self.copy_latest),
                             ('Select all', self.select_all), ('Clear transcript', self.clear)]:
            self.menu.add_command(label=text, command=action)
        self.text_box.bind('<Button-3>', lambda event: self.menu.tk_popup(event.x_root, event.y_root))
        self.text_box.bind('<MouseWheel>', self.on_scroll, add='+')
        self.text_box.bind('<Control-a>', lambda event: self.select_all())
        self.source_var.trace_add('write', lambda *_: self.update_source_controls())
        self.update_source_controls()

    def update_source_controls(self):
        if self.engine and not self.engine.finished.is_set():
            return
        source = self.source_var.get()
        self.output_combo.config(state='disabled' if source == 'Microphone' else 'readonly')
        self.mic_combo.config(state='disabled' if source == 'Computer audio' else 'readonly')
        hints = {'Computer audio': 'Captures what you hear. Choose Computer + microphone to include your voice.',
                 'Microphone': 'Captures your voice from the selected microphone.',
                 'Computer + microphone': 'Captures both sides separately. Use headphones to avoid speaker echo.'}
        self.source_hint.config(text=hints[source])

    def combo(self, parent, label, variable, values):
        self.label(parent, label, size=9).pack(anchor='w', pady=(0, 5))
        combo = ttk.Combobox(parent, textvariable=variable, values=values, state='readonly',
                             font=('Segoe UI', 10), width=23)
        combo.pack(fill='x', pady=(0, 12))
        self.settings_widgets.append(combo)
        return combo

    def update_profile_hint(self):
        hints = {'Balanced': 'Quick live drafts, refined when you pause.',
                 'Fastest': 'Lowest delay. Less accurate on difficult speech.',
                 'Best accuracy': 'Larger model; slower. First use downloads about 500 MB.'}
        self.profile_hint.config(text=hints[self.profile_var.get()])

    def _shortcuts(self):
        bindings = {'<Control-Shift-C>': self.copy_all, '<Control-Shift-c>': self.copy_all,
                    '<Control-l>': self.clear, '<Control-s>': self.save_as,
                    '<Control-space>': self.toggle_recording, '<Control-Shift-Z>': self.undo_clear,
                    '<Control-d>': self.toggle_debug, '<F12>': self.toggle_debug,
                    '<Control-Shift-X>': self.copy_and_clear, '<Control-Shift-x>': self.copy_and_clear}
        for key, action in bindings.items():
            self.root.bind(key, lambda event, action=action: (action(), 'break')[-1])

    def toast(self, text, error=False):
        if self.toast_after:
            self.root.after_cancel(self.toast_after)
        self.toast_label.config(text=text, fg=RED if error else ACCENT)
        self.toast_after = self.root.after(6500, lambda: self.toast_label.config(text=''))

    def refresh_devices(self):
        self.refresh_button.config(state='disabled')
        def worker():
            try:
                import soundcard as sc
                output, microphone = {'Follow Windows default': ''}, {'Follow Windows default': ''}
                # Enumerating WASAPI endpoints is a COM call, and apartments are per-thread.
                with com_apartment():
                    for i, device in enumerate(sc.all_speakers(), 1):
                        output[f'{i}. {device.name}'] = device.id
                    for i, device in enumerate(sc.all_microphones(), 1):
                        microphone[f'{i}. {device.name}'] = device.id
                self.events.put({'type': 'devices', 'output': output, 'microphone': microphone})
            except Exception as exc:
                self.events.put({'type': 'device_list_error', 'message': str(exc)})
        threading.Thread(target=worker, name='device-list', daemon=True).start()

    def save_preferences(self):
        self.settings.auto_copy = self.autocopy_var.get()
        self.settings.always_on_top = self.top_var.get()
        try:
            self.settings.save()
        except OSError as exc:
            self.toast(f'Could not save settings: {exc}', True)

    def read_settings(self):
        return Settings(profile=self.profile_var.get(), source=self.source_var.get(),
                        output_id=self.device_maps['output'].get(self.output_var.get(), ''),
                        microphone_id=self.device_maps['microphone'].get(self.mic_var.get(), ''),
                        language=self.language_var.get(), vocabulary=self.vocabulary.get(),
                        auto_copy=self.autocopy_var.get(), always_on_top=self.top_var.get(),
                        font_size=self.settings.font_size)

    def toggle_recording(self):
        if self.closing:
            return
        if self.engine and not self.engine.finished.is_set():
            self.engine.stop()
            self.record_button.config(text='Finishing…', state='disabled')
            self.status_label.config(text='●  Finishing the last phrase…', fg=ACCENT)
            return
        # A keyboard shortcut can arrive before the next GUI poll after shutdown.
        # Accept the previous engine's final events before switching engine IDs.
        while not self.events.empty():
            self.handle_event(self.events.get_nowait())
        old_engine = self.engine
        self.settings = self.read_settings()
        self.engine_error = ''
        self.capture_errors = {}
        self.save_preferences()
        offset = time.monotonic() - self.session_started
        self.engine = TranscriptionEngine(self.settings, self.events, self.transcript.epoch, offset)
        if old_engine and (old_engine.settings.profile, old_engine.settings.language) == (self.settings.profile, self.settings.language):
            self.engine.recognizer.models = old_engine.recognizer.models
            self.engine.recognizer.device = old_engine.recognizer.device
            self.engine.recognizer.compute = old_engine.recognizer.compute
        self.record_button.config(text='Cancel loading', state='normal')
        self.status_label.config(text='●  Preparing transcription…', fg=ACCENT)
        self.set_settings_enabled(False)
        if self.dictation_var.get():
            self.engine.armed.clear()  # Warm up the models without recording anything.
        self.engine.start()

    def set_settings_enabled(self, enabled):
        for widget in self.settings_widgets:
            widget.config(state=('readonly' if isinstance(widget, ttk.Combobox) else 'normal') if enabled else 'disabled')
        self.refresh_button.config(state='normal' if enabled else 'disabled')
        if enabled:
            self.update_source_controls()

    def copy_text(self, text):
        if not text.strip():
            self.toast('There is no text to copy yet.')
            return False
        try:
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            return True
        except tk.TclError as exc:
            self.toast(f'Clipboard unavailable. Try Copy again. {exc}', True)
            return False

    def copy_all(self):
        text = self.transcript.text()
        if self.copy_text(text):
            self.toast(f'Copied all {len(text.split())} words, including the current live draft.')

    def copy_and_clear(self):
        """Copy the whole transcript, then clear it only if the copy succeeded."""
        text = self.transcript.text()
        if not self.copy_text(text):
            return  # Never discard text that did not reach the clipboard.
        words = len(text.split())
        self.clear()
        self.toast(f'Copied {words} words and cleared. Undo clear restores the text.')

    def copy_selection(self):
        try:
            text = self.text_box.get('sel.first', 'sel.last')
        except tk.TclError:
            self.toast('Select some text first, or use Copy all.')
            return
        if self.copy_text(text):
            self.toast('Selected text copied.')

    def copy_latest(self):
        entries = self.transcript.ordered()
        if self.copy_text(entries[-1].text if entries else ''):
            self.toast('Latest phrase copied.')

    def select_all(self):
        self.text_box.tag_add('sel', '1.0', 'end-1c')
        self.text_box.focus_set()
        return 'break'

    def clear(self):
        if self.closing:
            return
        epoch = self.transcript.clear()
        self.dirty = True
        if self.engine:
            self.engine.clear(epoch)
        if self.copy_after:
            self.root.after_cancel(self.copy_after)
            self.copy_after = None
        self.last_autocopy = ''
        self.render()
        self.saver.submit(self.transcript.snapshot())
        self.undo_button.config(state='normal' if self.transcript.undo_entries else 'disabled')
        self.restore_button.pack_forget()
        self.toast('Transcript cleared. New speech starts fresh; Undo clear restores the previous text.')

    def undo_clear(self):
        if self.transcript.undo_clear():
            self.dirty = True
            self.render()
            self.saver.submit(self.transcript.snapshot())
            self.undo_button.config(state='disabled')
            self.toast('Previous text restored.')

    def restore_previous(self):
        if self.previous:
            try:
                self.transcript.restore(self.previous)
                self.dirty = True
                self.session_started = time.monotonic() - max((entry.end for entry in self.transcript.ordered()), default=0.)
                self.previous = None
                self.restore_button.pack_forget()
                self.render()
                self.saver.submit(self.transcript.snapshot())
                self.toast('Previous session restored into this transcript.')
            except (TypeError, ValueError, KeyError) as exc:
                self.toast(f'Could not restore this session: {exc}', True)

    def save_as(self, path=None):
        if not self.transcript.entries:
            self.toast('There is no transcript to save yet.')
            return
        if path is None:
            path = filedialog.asksaveasfilename(parent=self.root, title='Save transcript',
                initialfile=datetime.now().strftime('Transcript_%Y-%m-%d_%H-%M.txt'), defaultextension='.txt',
                filetypes=[('Plain text', '*.txt'), ('Markdown with sources', '*.md'),
                           ('Subtitles', '*.srt'), ('Session JSON', '*.json')])
        if not path:
            return
        path = Path(path)
        if path.suffix.lower() == '.json':
            text = json.dumps(self.transcript.snapshot(), ensure_ascii=False, indent=2)
        elif path.suffix.lower() == '.srt':
            text = srt_text(self.transcript.ordered())
        elif path.suffix.lower() == '.md':
            text = '# Meeting transcript\n\n' + self.transcript.text(labels=True) + '\n'
        else:
            text = self.transcript.text() + '\n'
        try:
            atomic_write(path, text)
            self.toast(f'Saved {path.name}')
        except OSError as exc:
            self.toast(f'Could not save: {exc}', True)

    def open_sessions(self):
        try:
            self.session_dir.mkdir(parents=True, exist_ok=True)
            os.startfile(str(self.session_dir))
        except OSError as exc:
            self.toast(f'Could not open sessions: {exc}', True)

    def top_changed(self):
        self.root.attributes('-topmost', self.top_var.get())
        self.save_preferences()

    def zoom(self, delta):
        self.settings.font_size = max(12, min(26, self.settings.font_size + delta))
        self.text_box.config(font=('Segoe UI', self.settings.font_size))
        self.save_preferences()

    def follow_live(self):
        if self.follow_var.get():
            self.text_box.see('end')

    def on_scroll(self, event):
        if event.delta > 0:
            self.follow_var.set(False)

    def render(self):
        entries = self.transcript.ordered()
        labels = self.settings.source == 'Computer + microphone'
        new_text = self.transcript.text(labels=labels)
        if new_text != self.rendered:
            old_scroll = self.text_box.yview()
            selection = tuple(str(index) for index in self.text_box.tag_ranges('sel'))
            common = 0
            for a, b in zip(self.rendered, new_text):
                if a != b:
                    break
                common += 1
            self.text_box.config(state='normal')
            self.text_box.delete(f'1.0+{common}c', 'end-1c')
            self.text_box.insert('end-1c', new_text[common:])
            self.text_box.tag_remove('draft', '1.0', 'end')
            offset = 0
            for entry in entries:
                section = f'{entry.source}  {timestamp(entry.start)}\n{entry.text}' if labels else entry.text
                if not entry.final:
                    self.text_box.tag_add('draft', f'1.0+{offset}c', f'1.0+{offset + len(section)}c')
                offset += len(section) + 2
            if selection:
                self.text_box.tag_add('sel', *selection)
            self.text_box.config(state='disabled')
            if self.follow_var.get() and not selection:
                self.text_box.see('end')
            elif old_scroll:
                self.text_box.yview_moveto(old_scroll[0])
            self.rendered = new_text
        if new_text:
            self.empty_frame.place_forget()
        else:
            self.empty_frame.place(relx=.5, rely=.42, anchor='center')
        drafts = sum(not entry.final for entry in entries)
        suffix = ' · refining live draft' if drafts else ''
        self.count_label.config(text=f'{len(self.transcript.text().split()):,} words{suffix}')
        # Finalization may keep the same characters while changing their status.
        if not drafts:
            self.text_box.tag_remove('draft', '1.0', 'end')

    def auto_copy(self):
        self.copy_after = None
        if not self.autocopy_var.get():
            return
        text = self.transcript.text()
        if text and text != self.last_autocopy and self.copy_text(text):
            self.last_autocopy = text

    def toggle_dictation_mode(self):
        """Turn hotkey-burst dictation on or off."""
        enabled = self.dictation_var.get()
        self.settings.dictation = enabled
        if enabled:
            error = self.hotkey.register(self.hotkey_var.get())
            if error:
                self.dictation_var.set(False)
                self.settings.dictation = False
                self.toast(error, True)
                self.update_dictation_hint()
                return
            self.toast('Dictation mode on. Press Start listening to warm up, then use '
                       + self.hotkey_var.get() + ' anywhere.')
        else:
            self.hotkey.unregister()
            # Leaving the mode must not strand the engine in a disarmed state.
            if self.engine:
                self.engine.armed.set()
            self.dictating = False
        self.save_preferences()
        self.update_dictation_hint()

    def change_hotkey(self, event=None):
        """Rebind the global hotkey, reverting if Windows refuses the new one."""
        previous = self.settings.hotkey
        chosen = self.hotkey_var.get()
        if self.dictation_var.get():
            error = self.hotkey.register(chosen)
            if error:
                self.hotkey_var.set(previous)
                self.hotkey.register(previous)
                self.toast(error, True)
                return
        self.settings.hotkey = chosen
        self.save_preferences()
        self.update_dictation_hint()
        self.toast('Hotkey set to ' + chosen + '.')

    def update_dictation_hint(self):
        """Keep the sidebar hint describing the actual current state."""
        if not self.dictation_var.get():
            text = 'Off. Transcription runs continuously while listening.'
            colour = MUTED
        elif not self.engine or self.engine.finished.is_set():
            text = 'Press Start listening once to load the models, then ' + self.hotkey_var.get() + ' starts a burst.'
            colour = MUTED
        elif self.dictating:
            text = 'Recording. Press ' + self.hotkey_var.get() + ' to stop, copy and reset.'
            colour = ACCENT
        else:
            text = 'Ready. Press ' + self.hotkey_var.get() + ' anywhere to start a burst.'
            colour = ACCENT
        self.dictation_hint.config(text=text, fg=colour)
        self.hotkey_combo.config(state='readonly' if self.dictation_var.get() else 'disabled')

    def poll_hotkey(self):
        """Drain global hotkey presses from the Tk loop."""
        if not self.dictation_var.get():
            return
        for _ in range(self.hotkey.drain()):
            self.toggle_burst()

    def toggle_burst(self):
        """Start a dictation burst, or end one and put its text on the clipboard."""
        if not self.engine or self.engine.finished.is_set() or not self.engine.ready.is_set():
            self.toast('Press Start listening first so the models are loaded.', True)
            return
        if self.dictating:
            self.end_burst()
        else:
            self.begin_burst()

    def begin_burst(self):
        self.dictating = True
        self.engine.armed.set()
        self.burst_started = time.monotonic()
        self.status_label.config(text='●  Recording · press ' + self.hotkey_var.get() + ' to stop and copy',
                                 fg=ACCENT)
        self.update_dictation_hint()
        self.log_debug('dictation: burst started')

    def end_burst(self):
        """Stop capture, wait for the last phrase, then copy and reset."""
        self.dictating = False
        self.engine.armed.clear()
        self.status_label.config(text='●  Finishing the last phrase…', fg=ACCENT)
        self.update_dictation_hint()
        self.log_debug('dictation: burst ended, waiting for final phrases')
        # The tail of speech is still decoding; collect it before copying.
        self.root.after(self.burst_settle_ms, self.finish_burst)

    def finish_burst(self):
        text = self.transcript.text()
        if not text.strip():
            self.status_label.config(text='●  Nothing was captured. Press ' + self.hotkey_var.get()
                                     + ' to try again.', fg=MUTED)
            self.update_dictation_hint()
            return
        words = len(text.split())
        if self.copy_text(text):
            self.clear()
            self.status_label.config(text='●  Copied ' + str(words) + ' words. Ready for the next burst.',
                                     fg=ACCENT)
            self.toast('Copied ' + str(words) + ' words to the clipboard. Paste anywhere.')
        else:
            # Copy failed, so keep the text rather than discarding it.
            self.status_label.config(text='●  Clipboard unavailable; text kept. Use Copy all.', fg=RED)
        self.update_dictation_hint()
        self.log_debug('dictation: burst finished, %d words' % words)

    def toggle_debug(self, event=None):
        """Open or close the diagnostics window."""
        if self.debug_window and self.debug_window.winfo_exists():
            self.debug_window.destroy()
            self.debug_window = None
            return 'break'
        self._build_debug_window()
        return 'break'

    def _build_debug_window(self):
        window = tk.Toplevel(self.root, bg=BG)
        window.title('LiveScribe · Diagnostics')
        window.geometry('780x580')
        self.debug_window = window
        window.protocol('WM_DELETE_WINDOW', self.toggle_debug)

        head = tk.Frame(window, bg=BG, padx=18, pady=14)
        head.pack(fill='x')
        self.label(head, 'Pipeline diagnostics', size=14, bold=True, color=TEXT).pack(anchor='w')
        self.label(head, 'Audio flows left to right. The first column that stops moving is where it breaks.',
                   size=9, wraplength=720, justify='left').pack(anchor='w', pady=(2, 0))

        body = tk.Frame(window, bg=BG, padx=18)
        body.pack(fill='x')
        self.debug_rows = {}
        columns = ('Source', 'State', 'Blocks', 'RMS', 'Peak', 'Loudest', 'Speech', 'Buffered', 'Jobs', 'Decoded', 'Chars')
        grid = tk.Frame(body, bg=PANEL, padx=12, pady=10)
        grid.pack(fill='x', pady=(6, 10))
        for column, name in enumerate(columns):
            self.label(grid, name, size=9, bold=True, color=TEXT).grid(row=0, column=column, sticky='w', padx=6)
        for row, source in enumerate(('Computer', 'You'), start=1):
            cells = []
            for column in range(len(columns)):
                cell = self.label(grid, source if column == 0 else '—', size=9,
                                  color=TEXT if column == 0 else MUTED)
                cell.grid(row=row, column=column, sticky='w', padx=6, pady=2)
                cells.append(cell)
            self.debug_rows[source] = cells

        self.debug_summary = self.label(body, 'Waiting for the engine…', size=9,
                                        wraplength=720, justify='left')
        self.debug_summary.pack(anchor='w', pady=(0, 8))

        controls = tk.Frame(body, bg=BG)
        controls.pack(fill='x', pady=(0, 8))
        self.button(controls, 'Copy report', self.copy_debug_report, compact=True).pack(side='left')
        self.button(controls, 'Probe devices', self.probe_devices, compact=True).pack(side='left', padx=(8, 0))
        self.button(controls, 'Clear log', self.clear_debug_log, compact=True).pack(side='left', padx=(8, 0))

        log_frame = tk.Frame(window, bg=BG, padx=18)
        log_frame.pack(fill='both', expand=True, pady=(0, 16))
        self.label(log_frame, 'EVENT LOG', size=9, bold=True, color=TEXT).pack(anchor='w', pady=(0, 6))
        scroll = tk.Scrollbar(log_frame)
        scroll.pack(side='right', fill='y')
        self.debug_text = tk.Text(log_frame, bg=FIELD, fg=TEXT, relief='flat', wrap='word',
                                  font=('Consolas', 9), insertbackground=TEXT,
                                  yscrollcommand=scroll.set, highlightthickness=0)
        self.debug_text.pack(fill='both', expand=True)
        scroll.config(command=self.debug_text.yview)
        self.debug_text.insert('end', '\n'.join(self.debug_history))
        self.debug_text.see('end')
        self.refresh_debug()

    def clear_debug_log(self):
        self.debug_history.clear()
        if self.debug_text:
            self.debug_text.delete('1.0', 'end')

    def log_debug(self, message):
        """Append one line to the rolling diagnostics log."""
        line = datetime.now().strftime('%H:%M:%S.%f')[:-3] + '  ' + message
        self.debug_history.append(line)
        if len(self.debug_history) > 400:
            del self.debug_history[:-400]
        if self.debug_window and self.debug_text:
            try:
                at_end = self.debug_text.yview()[1] > .999
                self.debug_text.insert('end', line + '\n')
                if int(self.debug_text.index('end-1c').split('.')[0]) > 400:
                    self.debug_text.delete('1.0', '2.0')
                if at_end:
                    self.debug_text.see('end')
            except tk.TclError:
                pass

    def refresh_debug(self):
        """Repaint the diagnostics table from engine counters."""
        if not (self.debug_window and self.debug_window.winfo_exists()):
            return
        stats = self.engine.snapshot_stats() if self.engine else {}
        for source, cells in self.debug_rows.items():
            values = stats.get(source)
            if not values:
                for cell in cells[1:]:
                    cell.config(text='—', fg=MUTED)
                continue
            state = values.get('state', '—')
            blocks = values.get('blocks_n', 0)
            readings = [state, str(blocks),
                        '%.4f' % values.get('rms', 0.),
                        '%.3f' % values.get('peak', 0.),
                        '%.3f' % values.get('peak_max', 0.),
                        '%.2f' % values.get('speech', 0.),
                        '%.1fs' % values.get('buffered', 0.),
                        str(values.get('jobs_n', 0)),
                        str(values.get('decodes_n', 0)),
                        str(values.get('text_chars', 0))]
            for cell, text in zip(cells[1:], readings):
                cell.config(text=text, fg=MUTED)
            cells[1].config(fg=RED if state == 'error' else ACCENT if state == 'capturing' else MUTED)
            # Digital silence is the most common failure; make it obvious.
            if state == 'capturing' and blocks > 10 and values.get('peak_max', 0.) < 1e-4:
                cells[5].config(fg=RED)
        self.debug_summary.config(text=self.debug_diagnosis(stats))
        self.root.after(400, self.refresh_debug)

    def debug_diagnosis(self, stats):
        """Translate the counters into the most likely explanation."""
        if not self.engine:
            return 'Engine not running. Press Start listening.'
        if not stats:
            return 'Engine starting; no capture thread has reported yet.'
        notes = []
        for source, values in sorted(stats.items()):
            state, blocks = values.get('state', '?'), values.get('blocks_n', 0)
            peak, jobs = values.get('peak_max', 0.), values.get('jobs_n', 0)
            if state == 'error':
                notes.append(source + ': capture failed — ' + str(values.get('error', 'unknown')))
            elif blocks == 0:
                notes.append(source + ': device open but no audio blocks returned yet.')
            elif peak < 1e-4 and not values.get('decodes_n', 0):
                notes.append(source + ': every block so far is digital silence (peak 0). Nothing is playing '
                             'to this endpoint, or the selected output is not the one actually playing.')
            elif jobs == 0:
                notes.append(source + ': audio is arriving (loudest %.3f) but the detector has not called it '
                             'speech yet (last %.2f).' % (peak, values.get('speech', 0.)))
            elif values.get('decodes_n', 0) == 0:
                notes.append(source + ': %d phrase(s) queued but none decoded — the model may still be loading.'
                             % jobs)
            elif values.get('text_chars', 0) == 0:
                notes.append(source + ': decoding runs but returns empty text.')
            else:
                notes.append(source + ': healthy — %d decode(s), last %.2fs.'
                             % (values.get('decodes_n', 0), values.get('inference', 0.)))
        return '\n'.join(notes)

    def debug_report(self):
        """Plain-text snapshot for sharing."""
        lines = ['LiveScribe diagnostics',
                 'Time: ' + datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                 'Source setting: ' + str(self.settings.source),
                 'Output id: ' + (self.settings.output_id or '(Windows default)'),
                 'Microphone id: ' + (self.settings.microphone_id or '(Windows default)'),
                 'Profile: %s  Language: %s' % (self.settings.profile, self.settings.language),
                 'Engine running: %s' % bool(self.engine and not self.engine.finished.is_set()),
                 'Discontinuities: %d' % self.discontinuities, '']
        stats = self.engine.snapshot_stats() if self.engine else {}
        for source, values in sorted(stats.items()):
            lines.append('[' + source + ']')
            for key in sorted(values):
                lines.append('  %s = %s' % (key, values[key]))
        lines.extend(['', self.debug_diagnosis(stats), '', '--- recent events ---'])
        lines.extend(self.debug_history[-120:])
        return '\n'.join(lines)

    def copy_debug_report(self):
        self.root.clipboard_clear()
        self.root.clipboard_append(self.debug_report())
        self.toast('Diagnostics report copied to the clipboard.')

    def probe_devices(self):
        """Independently verify WASAPI enumeration and a one-second loopback read."""
        self.log_debug('probe: starting…')
        settings = self.read_settings()
        def report(message):
            self.events.put({'type': 'debug', 'message': message})
        def worker():
            try:
                import numpy as np
                import soundcard as sc
                from audio_capture import com_apartment, open_recorder
                with com_apartment():
                    speaker = (sc.get_speaker(settings.output_id) if settings.output_id
                               else sc.default_speaker())
                    report('probe: output endpoint = ' + speaker.name)
                    mic = sc.get_microphone(speaker.id, include_loopback=True)
                    report('probe: loopback device = ' + mic.name)
                    with open_recorder(mic, 'Computer', 16000) as recorder:
                        data = np.asarray(recorder.record(numframes=16000), dtype=np.float32)
                peak = float(np.max(np.abs(data)))
                rms = float(np.sqrt(np.mean(data ** 2)))
                verdict = ('SILENT - play audio through this exact output, then probe again'
                           if peak < 1e-4 else 'audio present')
                report('probe: 1s read shape=%s peak=%.5f rms=%.5f -> %s'
                       % (data.shape, peak, rms, verdict))
            except Exception as exc:
                report('probe FAILED: %s: %s' % (type(exc).__name__, exc))
        threading.Thread(target=worker, name='debug-probe', daemon=True).start()

    def handle_event(self, event):
        kind = event['type']
        if kind == 'debug':
            self.log_debug(event['message'])
            return
        if 'engine' in event and (not self.engine or event['engine'] != id(self.engine)):
            self.log_debug('(stale engine) ' + kind)
            return
        if kind != 'level':  # Level events fire ~10x/second per source.
            detail = event.get('message') or event.get('source') or ''
            self.log_debug(kind + (': ' + str(detail) if detail else ''))
        if kind == 'transcript':
            if self.transcript.apply(event['epoch'], event['entry']):
                self.dirty = True
                self.render()
                self.saver.submit(self.transcript.snapshot())
                if self.autocopy_var.get() and self.copy_after is None:
                    self.copy_after = self.root.after(450, self.auto_copy)
            pending = f" · {event['backlog']} phrases waiting" if event['backlog'] else ''
            self.detail_label.config(text=f"Updated in {event['inference']:.2f}s{pending}  ·  Green text is being refined")
        elif kind == 'ready':
            if not self.engine.stop_event.is_set():
                self.record_button.config(text='Pause', state='normal')
                if self.dictation_var.get():
                    self.status_label.config(text='●  Ready · press ' + self.hotkey_var.get()
                                             + ' anywhere to dictate', fg=ACCENT)
                    self.update_dictation_hint()
                elif not self.capture_errors:
                    self.status_label.config(text='●  Listening · ' + event['message'], fg=ACCENT)
        elif kind == 'status':
            self.status_label.config(text='●  ' + event['message'], fg=ACCENT)
        elif kind in ('error', 'capture_error'):
            if kind == 'error':
                self.engine_error = event['message']
            else:
                self.capture_errors[event['source']] = event['message']
            self.status_label.config(text='●  ' + event['message'], fg=RED)
            logging.error(event['message'])
        elif kind == 'warning':
            self.toast(event['message'], True)
            logging.warning(event['message'])
        elif kind == 'stopped':
            if not self.closing:
                self.record_button.config(text='Resume' if self.transcript.entries else 'Start listening', state='normal')
                message = self.engine_error or 'Paused · Copy, save, or resume at any time'
                self.status_label.config(text='●  ' + message, fg=RED if self.engine_error else MUTED)
                self.set_settings_enabled(True)
        elif kind == 'level':
            self.last_levels[event['source']] = min(1., max(0., (event['rms'] / .12) ** .5))
        elif kind == 'device':
            self.toast(f"{event['source']}: {event['message']}")
        elif kind == 'capture_ready':
            self.silence_warned = False
            self.capture_errors.pop(event['source'], None)
            if (not self.capture_errors and self.engine and self.engine.ready.is_set()
                    and not self.engine.stop_event.is_set() and not self.dictation_var.get()):
                self.status_label.config(text='●  Listening · ' + self.settings.source, fg=ACCENT)
        elif kind == 'wrong_output':
            message = ('Capturing "%s" but Windows is playing to "%s". You will record silence. '
                       'Change Computer output, or choose Follow Windows default.'
                       % (event['selected'], event['playing']))
            self.status_label.config(text='●  ' + message, fg=RED)
            self.toast(message, True)
            logging.warning(message)
        elif kind == 'discontinuity':
            self.discontinuities += event['count']
            if self.discontinuities == 1 or self.discontinuities % 20 == 0:
                self.toast(f'Audio interruptions detected ({self.discontinuities}). Check the device if words are missing.', True)
        elif kind == 'devices':
            for source, combo, variable, selected_id in (
                ('output', self.output_combo, self.output_var, self.settings.output_id),
                ('microphone', self.mic_combo, self.mic_var, self.settings.microphone_id)):
                self.device_maps[source] = event[source]
                combo.config(values=list(event[source]))
                variable.set(next((name for name, value in event[source].items() if value == selected_id), 'Follow Windows default'))
            if not self.engine or self.engine.finished.is_set():
                self.refresh_button.config(state='normal')
        elif kind == 'device_list_error':
            self.refresh_button.config(state='normal')
            self.toast('Could not list audio devices: ' + event['message'], True)
        elif kind == 'saved':
            self.saved_label.config(text='Saved locally · ' + datetime.now().strftime('%H:%M:%S'), fg=MUTED)
        elif kind == 'save_error':
            self.saved_label.config(text='Autosave failed · use Save as', fg=RED)
            self.toast(event['message'], True)

    def poll(self):
        for _ in range(250):
            try:
                event = self.events.get_nowait()
            except queue.Empty:
                break
            self.handle_event(event)
        for source, meter in self.meters.items():
            level = self.last_levels[source]
            meter.coords('level', 0, 0, int(level * 138), 8)
            self.last_levels[source] *= .8
        self.poll_hotkey()
        self.check_silence()
        self.timer_label.config(text=timestamp(time.monotonic() - self.session_started))
        if self.closing and (not self.engine or self.engine.finished.is_set()):
            while not self.events.empty():
                self.handle_event(self.events.get_nowait())
            if self.dirty:
                self.saver.submit(self.transcript.snapshot())
            threading.Thread(target=self.finish_save, name='close-save', daemon=True).start()
            self.root.after(50, self.check_closed)
            return
        self.root.after(50, self.poll)

    def check_silence(self):
        """Warn once when capture is healthy but every block has been pure silence."""
        if self.silence_warned or not self.engine or not self.engine.ready.is_set():
            return
        if self.engine.stop_event.is_set():
            return
        for source, values in self.engine.snapshot_stats().items():
            # ~30s of blocks at 96 ms, all digitally silent.
            if (values.get('state') == 'capturing' and values.get('blocks_n', 0) > 310
                    and values.get('peak_max', 0.) < 1e-4):
                self.silence_warned = True
                message = ('%s has received only silence for 30s. Check that audio is playing to the '
                           'selected device, then press F12 for diagnostics.' % source)
                self.status_label.config(text='●  ' + message, fg=RED)
                self.toast(message, True)
                logging.warning(message)
                return

    def finish_save(self):
        self.saver.close()
        self.save_finished.set()

    def check_closed(self):
        if self.save_finished.is_set():
            if self.saver.last_error and not messagebox.askyesno('Transcript could not be saved',
                'Automatic saving failed. Close anyway?\n\nChoose No to return and use Copy all or Save as.\n\n' + self.saver.last_error,
                parent=self.root):
                self.closing = False
                self.save_finished.clear()
                self.saver = Autosaver(self.events, self.session_dir)
                self.record_button.config(text='Start listening', state='normal')
                self.clear_button.config(state='normal')
                self.set_settings_enabled(True)
                self.root.after(50, self.poll)
                return
            self.hotkey.unregister()
            self.root.destroy()
        else:
            self.root.after(50, self.check_closed)

    def close(self):
        if self.closing:
            return
        self.closing = True
        self.record_button.config(text='Finishing…', state='disabled')
        self.clear_button.config(state='disabled')
        self.status_label.config(text='●  Finishing pending speech and saving your transcript…', fg=ACCENT)
        if self.engine:
            self.engine.stop()


def main():
    parser = argparse.ArgumentParser(description='LiveScribe local meeting transcription')
    parser.add_argument('--start', action='store_true', help='Start listening automatically')
    args = parser.parse_args()
    ROOT.joinpath('sessions').mkdir(exist_ok=True)
    handler = RotatingFileHandler(ROOT / 'sessions' / 'app.log', maxBytes=1_000_000, backupCount=2, encoding='utf-8')
    logging.basicConfig(level=logging.INFO, handlers=[handler], format='%(asctime)s %(levelname)s %(message)s')
    if os.name == 'nt':
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except (AttributeError, OSError):
            pass
    app = TranscriberApp(autostart=args.start)
    app.root.mainloop()


if __name__ == '__main__':
    main()
