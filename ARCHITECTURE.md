# EchoFlux Architecture

EchoFlux is a local-first, privacy-preserving desktop application for real-time speech-to-text, live translation, and context-aware conversation assistance.

The application is structured into three strictly decoupled tiers:

```
┌────────────────────────────────────────────────────────┐
│               Frontend UI (React / TypeScript)         │
│  - Zustand Slices: session, transcript, settings, conv │
│  - Panels: Transcript, Assist, History, Profiles, etc. │
│  - Holds ZERO raw secrets; uses credential references  │
└──────────────────────────┬─────────────────────────────┘
                           │ Tauri IPC (invoke)
┌──────────────────────────▼─────────────────────────────┐
│             Desktop Bridge (Rust / Tauri)              │
│  - OS Credential Vault (Windows Credential Manager /   │
│    macOS Keychain) via secure_store.rs                 │
│  - Engine process lifecycle & sidecar supervisor       │
│  - Privileged control dispatcher & secret injector     │
└──────────────────────────┬─────────────────────────────┘
                           │ Authenticated WebSocket (localhost)
┌──────────────────────────▼─────────────────────────────┐
│               Backend Engine (Python)                  │
│  - Dual-stream audio capture & Silero VAD              │
│  - Provider Registry (STT, Translation, Assistant)     │
│  - Session Orchestration & Sticky Fallback to Local    │
│  - SQLite repository: conversations, messages, snapshots│
└────────────────────────────────────────────────────────┘
```

The engine never knows a UI exists — it communicates over authenticated localhost WebSockets. The UI never touches the filesystem, Python environment, or models — it dispatches commands and renders event streams. The CLI (`apps/cli/main.py`) drives the same engine headlessly. Breaking this separation is prohibited; see `.agent/rules/03-architecture.md`.

---

## 1. System Tiers and Ownership

| Layer | Owns | Prohibitions |
| --- | --- | --- |
| **Frontend UI (React)** | Presentation, interaction states, sanitized preferences, modular Zustand slices (`useEngineStore`). | NEVER reads or stores raw third-party API keys; NEVER interacts directly with local filesystem or model files. |
| **Desktop Bridge (Rust)** | System integration, OS Credential Vault (Windows Credential Manager / macOS Keychain), child process lifecycle (sidecar `echoflux-engine`), privileged control proxying, secret redaction. | NEVER implements session, transcription, translation, or prompt generation logic. |
| **Backend Engine (Python)** | Audio capture, Silero VAD, STT streaming, translation scheduling, assistant summaries, local SQLite persistence (`ConversationRepository`), context profile snapshotting. | NEVER assumes a GUI exists; NEVER persists raw secrets to disk, database, or logs. |
| **Provider Adapters** | Normalized audio/text streams and standardized errors for cloud/local backends. | NEVER decide fallback, manage UI state, or emit raw UI notifications. |

---

## 2. Core Architectural Invariants

### 2.1 Zero-Secret Persistence
- Third-party API keys (OpenAI, Gemini, Deepgram, AssemblyAI, Azure Speech) are stored exclusively in the OS vault:
  - Windows: **Windows Credential Manager**
  - macOS: **macOS Keychain**
- The webview only handles opaque credential references (`CredentialRef`) and configured status flags.
- Tauri's privileged bridge (`control_bridge.rs`) resolves credential references into short-lived in-memory strings strictly at command execution time, injecting them into the engine over the control channel.
- The Python engine consumes secrets ephemerally for the duration of the session and discards them upon termination. Secrets are never saved to SQLite, `config.json`, logs, or error responses.

### 2.2 Immutable Session Profile Snapshotting
- When a user starts a meeting session with an attached Context Profile, `ConversationRepository` immutably captures the profile definition into `session_profile_snapshots`.
- Subsequent edits to the user's Context Profiles do not mutate active or historical meeting contexts.

### 2.3 Sticky Fallback Orchestration
- If a cloud STT or Translation provider encounters a network error, rate limit, or authentication failure during a live meeting, `SessionOrchestrator` automatically degrades that capability to the **Local** provider for the remainder of the session without speech or transcript loss.
- Sticky fallback emits a structured status event to notify the UI without terminating the audio pipeline.
- Assistant errors fail visibly without fallback, as assistant queries are user-triggered rather than continuous streaming.

### 2.4 Dual-Stream Audio & Concurrency
- `engine/main.py` coordinates concurrent `threading` (blocking PortAudio / WASAPI capture, VAD, model inference) with `asyncio` (WebSocket event broadcasting and control requests).
- Supports independent `mic` and `system` (loopback) audio streams, each running isolated capture and process worker threads connected via thread-safe bounded `queue.Queue`.
- `_broadcast_loop` drains up to 10 messages per batch via non-blocking `get_nowait()` polling, preventing thread contention on the event loop.

---

## 3. Module Layout

| Path | Role |
| --- | --- |
| `engine/main.py` | `EchoFluxEngine` composition root, thread orchestration, pipeline runner |
| `engine/server/websocket_server.py` | Authenticated WebSocket server (control role & event broadcast plane) |
| `engine/server/control_dispatcher.py` | Control command dispatcher and error boundary |
| `engine/session/session_orchestrator.py` | Live STT/Translation session lifecycle, stream coordination, sticky fallback |
| `engine/conversation/repository.py` | Local SQLite database schema, conversations, messages, translations, snapshots |
| `engine/conversation/context_profiles.py` | Context profile management, vocabulary, instructions |
| `engine/assistant/service.py` | Assistant suggestion and clarification generation |
| `engine/assistant/summary_service.py` | Real-time meeting summary chunk streaming |
| `engine/providers/` | Capability provider registry (`stt/`, `translation/`, `assistant/`) |
| `engine/local_models/` | Local model pack catalog, hardware tier inspection, on-demand downloader |
| `engine/audio/` | Microphone, WASAPI system audio loopback, Silero VAD |
| `apps/desktop/src/store/` | Unified Zustand store composed from typed slices (`session`, `transcript`, `settings`, `conversation`) |
| `apps/desktop/src/components/` | Modular React panels (`TranscriptPanel`, `AssistPanel`, `HistoryPanel`, `ContextProfilesPanel`, `OnboardingDialog`, etc.) |
| `apps/desktop/src-tauri/src/` | Rust desktop bridge (`control_bridge`, `engine_process`, `secure_store`, `main`) |
| `packaging/echoflux-engine.spec` | PyInstaller standalone freeze specification for sidecar binary |
| `scripts/build_sidecar.ps1` | Sidecar cross-compilation & deployment automation script |

---

## 4. Transcription & Translation Flow

```
1. Audio Capture (PortAudio / WASAPI Loopback)
   └─> int16 PCM bytes ──> Queue ──> Silero VAD
2. Speech Processing
   └─> Transcribe chunk ──> partial event (streamed to UI)
   └─> Boundary detected ──> final event (persisted to SQLite with message_id)
3. Asynchronous Translation
   └─> final text enqueued to TranslationThread
   └─> NMT / LLM translation completed
   └─> translation_update event emitted (correlated by message_id & entry_id)
   └─> Updated translation persisted to SQLite
4. Real-time Assistance (Optional)
   └─> ID-only request_assist ──> context assembled from SQLite ──> suggestion_result
   └─> request_summary ──> streaming llm_chunk ──> llm_done
```

---

## 5. Reliability & Soak Target

EchoFlux is validated against a 4-hour continuous session soak test simulating continuous dual-stream meeting speech. The pipeline maintains strict queue caps (`maxsize=500` for audio chunks, `maxsize=100` for translation items) to guarantee memory stability, zero unbounded buffer growth, and consistent event ordering under sustained load.
