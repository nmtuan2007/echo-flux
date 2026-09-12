# EchoFlux Architecture

Two processes that share nothing but a localhost WebSocket.

```
Desktop App (Tauri + React)            Engine (Python)
  engineStore.ts  ── JSON ──────────>  websocket_server.py
                  <───── JSON ───────  main.py  ──> audio / ASR / NMT / LLM
```

The engine never knows a UI exists — it broadcasts JSON. The UI never touches the
filesystem, Python, or models — it sends commands and renders replies. The CLI
(`apps/cli/main.py`) drives the same engine headlessly. Breaking that separation is the
single most damaging change you can make here; see `.agent/rules/03-architecture.md`.

The message contract itself lives in `.agent/rules/03-architecture.md` and is pinned by
`tests/test_ws_contract.py`. It is not repeated here.

## Thread model

This is where the engine is easy to break. `engine/main.py` mixes `asyncio` (WebSocket
I/O) with `threading` (blocking audio and ML work). Nothing blocking may touch the event
loop.

```
Capture-{stream}  thread   audio device ──> _audio_queues[stream]  (Queue, maxsize=500)
Process-{stream}  thread   audio queue  ──> VAD ──> ASR ──> _result_queue
TranslationThread thread   _translation_queue (maxsize=100) ──> _result_queue
_broadcast_loop   asyncio  _result_queue ──get_nowait()──> ws_server.broadcast()
```

- One `Capture-*` and one `Process-*` thread **per audio stream**. With
  `audio.source = "both"` there are two of each, keyed `mic` and `system`, each with its
  own queue and its own VAD instance. This is why every ASR method takes a `stream_id`
  and why backends must hold per-stream state.
- All threads are daemons. Cross-thread handoff is always `queue.Queue`, never shared
  mutable state.
- `_broadcast_loop` is the only bridge back into asyncio. It drains at most 10 messages
  per pass with `get_nowait()`, then sleeps 20 ms when the queue is empty. Code running
  on a thread that needs the loop uses `asyncio.run_coroutine_threadsafe`.
- Model download and Hub search each spawn their own short-lived daemon thread for the
  same reason: they block.

## Transcription flow

1. `_capture_loop` reads `int16` PCM `bytes` and puts them on the stream's queue.
2. `_process_loop` runs VAD, then `transcribe_stream()`, and on a speech boundary
   `finalize_current()`.
3. `_enqueue_asr_result` emits `partial` while speech is in progress and `final` at a
   boundary. Every `final` gets a stable `entry_id` (`e-<timestamp>`).
4. Translation is **always asynchronous**. `final` ships with `translation: null`; the
   text is queued, translated on `TranslationThread`, and returned later as a
   `translation_update` carrying the same `entry_id`. The UI matches them on that id.

The UI must therefore tolerate a final transcript that has no translation yet — and
`partial` messages arriving every few hundred milliseconds.

## Modules

| Path | Role |
| --- | --- |
| `engine/main.py` | `EchoFluxEngine` — pipeline construction, threads, broadcast loop |
| `engine/server/websocket_server.py` | Socket lifecycle, command dispatch, device enumeration |
| `engine/core/config.py` | `Config`, `TranscriptionConfig`, env map, data dirs |
| `engine/core/model_manager.py` | Download / delete / list / Hub search, progress callback |
| `engine/audio/` | `MicrophoneInput`, `SystemAudioInput` (WASAPI loopback), `VAD`, mixer |
| `engine/asr/` | `ASRBackend` ABC, `AutoModelAdapter`, faster-whisper + transformers adapters |
| `engine/translation/` | `TranslationBackend` ABC, Marian / online / fallback backends |
| `engine/llm/assistant.py` | Optional suggestions and streamed summaries |
| `apps/desktop/src/store/engineStore.ts` | All UI business logic, socket handling, reconnect |
| `apps/cli/main.py` | Headless entry point |

Backend selection is indirect: `AutoModelAdapter.load()` reads the downloaded model's
`manifest.json` and picks the `ctranslate2` or `transformers` runtime from it. Translation
always goes through `FallbackTranslationBackend`, which wraps the configured backend and
degrades rather than raising.

## Configuration

Resolved later-wins: built-in defaults → `config.json` in the data dir → `.env` (project
root, then data dir) → environment variables. `_load_dotenv` will not overwrite a variable
that is already set, which is what makes real environment variables outrank `.env`.

The UI sends **dotted keys** in its `start` payload (`"asr.model_size"`), which
`_initialize_pipeline` reads directly from the settings dict, falling back to
`ECHOFLUX_*` for CLI use. Adding a setting means touching `_DEFAULT_CONFIG`, `_ENV_MAP`,
`_initialize_pipeline`, and the store — see `.agent/workflows/new-feature.md`.

Pinned by `tests/test_config.py`.

## Models and failure behaviour

Models are never bundled. They download on demand to the platform data directory
(`%USERPROFILE%\.echoflux\models` on Windows, `~/Library/Application Support/EchoFlux/models`
on macOS, `~/.local/share/echoflux/models` on Linux), overridable with
`ECHOFLUX_MODELS_DIR`.

Degradation is a design rule, not a nicety: GPU load failure falls back to CPU, VAD
initialisation failure disables VAD instead of killing the pipeline, translation failure
falls through to the next backend, and device enumeration returns a placeholder while
capture is active rather than crashing PyAudio. Preserve this when adding backends.
