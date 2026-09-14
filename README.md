# LiveScribe

A local Windows meeting transcriber with fast live drafts, phrase refinement, Copy, Clear, Undo clear, and automatic saving.

Double-click **Start LiveScribe.vbs**, choose an audio source, and click **Start listening**. Alternatively, run:

```powershell
.\venv\Scripts\python.exe live_transcriber.py
```

Add `--start` to begin listening when the app opens. Model loading happens in the background, so the window opens immediately. The launcher does not open a console window.

`transcriber.py` also opens the new screen.

| Control | Behavior |
| --- | --- |
| Copy all | Copies every word currently shown, including the live draft, without source/timestamp labels. |
| Copy selection | Copies only the highlighted text. Standard Ctrl+C also works. |
| Dictation mode | Hotkey bursts. Warm up once, then a global hotkey starts and stops recording from any app. Stopping copies that burst to the clipboard and resets for the next one. |
| Copy & clear | Copies everything, then clears in one step. Nothing is cleared if the copy fails. Undo clear restores it. |
| Clear | Clears the current transcript and invalidates pending audio/results. New speech starts fresh. |
| Undo clear | Restores the previous text while keeping speech received after Clear. |
| Pause / Resume | Stops capture, finishes pending phrases, and resumes with the same loaded models when settings are unchanged. |
| Save as | Exports TXT, Markdown with source labels, SRT, or session JSON. |
| Restore previous | Restores the last saved session when one is available at startup. |
| Auto-copy | Optional; off by default. Copies the current transcript after updates. Clear does not erase the Windows clipboard. |
| Follow live | Scrolls to new text. Scrolling upward turns it off so you can read earlier text. |
| Keep on top | Keeps the window above other apps. |
| A+ / A− | Changes transcript font size. |

**Dictation mode** suits short dictation rather than a whole meeting. Tick **Hotkey bursts** in the sidebar, press **Start listening** once to load the models, and then the hotkey (default **Ctrl+Shift+Space**) toggles recording from whatever app you are in. Press it, speak, press it again: the burst lands on your clipboard and the transcript resets. Models stay loaded between bursts, so activation is immediate. Choose a different hotkey from the dropdown if another application already owns it.

Right-click the transcript for Copy latest phrase and Select all. Keyboard shortcuts: **Ctrl+Shift+C** copy all, **Ctrl+Shift+X** copy & clear, **Ctrl+L** clear, **Ctrl+S** save, **Ctrl+Space** pause/resume, **Ctrl+Shift+Z** undo clear, **F12** diagnostics.

Press **F12** (or the sidebar Diagnostics button) to open a live pipeline view: per-source capture state, block counts, audio peak, speech probability, queued phrases and decode counts, with a plain-language reading of where audio stops flowing. Copy report puts the whole snapshot on the clipboard; Probe devices runs an independent one-second loopback read.

Green text is a provisional draft and can change. Final text is white. The transcript is read-only so live updates cannot overwrite manual edits; export it to edit elsewhere. Copy and Save include the draft visible at the time you click.

| Mode | Preview / final model | Use |
| --- | --- | --- |
| Balanced — default | tiny / small, final beam 3 | Fast previews followed by a more accurate final pass. |
| Fastest | tiny / tiny, beam 1 | Minimum processing delay, with lower recognition accuracy. |
| Best accuracy | base / small, final beam 5 | More accurate drafts and wider final decoding; uses more processing time. |

English uses the `.en` models. Other language choices use multilingual models and may require a first-use download. Auto detect is intended for a consistent language within each phrase. Models are cached under `models/`; no transcription service or API key is required. This machine already has the English tiny, base, and small models cached.

**Computer audio** captures the selected playback device. **Microphone** captures your voice. **Computer + microphone** captures both as separate `Computer` and `You` tracks; these are source labels, not identification of individual remote speakers. Use headphones with both sources to prevent your microphone from hearing speaker playback. Acoustic echo cancellation and remote-speaker diarization are not implemented.

Follow Windows default reconnects when the default endpoint ID changes. Select a specific device when the meeting app uses a different output. Pause before changing capture settings. Input meters, device errors, retries, and transcription errors appear in the window.

Speaker capture uses SoundCard. Microphone capture uses sounddevice/PortAudio with Windows format conversion; this handles the USB microphone mix format that SoundCard could not open on this machine. Duplicate microphone names that cannot be uniquely matched produce a visible error instead of selecting an arbitrary device.

Transcripts are automatically saved as local text files in `sessions/`, with an atomic `latest.json` recovery snapshot. Live drafts are saved too. Clear resets the current session's saved transcript; it does not delete previously saved sessions, exported files, or clipboard contents. Closing the app waits for pending speech and saving. An autosave failure is shown and requires an explicit choice before closing without a successful save.

GPU setup adds installed NVIDIA DLL folders to the current process's search path and runs a real warm-up decode. Supported GPUs use `int8_float16`, measured faster on this Quadro T2000. GPU initialization or inference failure falls back to CPU/int8. No system-wide PATH changes are needed. CUDA availability alone is not treated as readiness.

**Setup on another Windows machine**

The current environment was tested with Python 3.14.4. Install Python with Tcl/Tk support, then:

```powershell
py -3.14 -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

For NVIDIA acceleration, use a compatible installed NVIDIA driver and install `requirements-gpu.txt` instead. The versions are pinned to the environment tested here. The GPU libraries are substantial downloads. CPU operation uses the core requirements only. The app handles model downloads on first use; an initial download needs internet access and can delay cancellation until the download call returns.

**Verification**

Verify by running the app against a real call. The development scaffolding used during the rebuild (benchmarks, synthetic-speech replays, and the pre-rebuild sources) has been removed from the repository. Current design decisions and upstream sources are in [RESEARCH.md](RESEARCH.md).
