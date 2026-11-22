---
description: Overall Architecture & Decoupling rules
globs: "**/*.py, **/*.ts, **/*.tsx"
---

# Architecture Rules

## strict Decoupling

EchoFlux uses a separated engine + UI architecture. These rules hold in both the current
runtime and the approved target state.

- The **Python Engine** knows NOTHING about the UI. It only emits JSON events.
- The **React App** knows NOTHING about the local filesystem, Python environment, or models. It only sends commands and receives JSON.
- Do NOT introduce tight coupling (e.g., the UI directly reading local Python files or executing Python scripts).
- **Python** owns audio, provider adapters, provider/session orchestration, fallback, transcript reconciliation, context assembly, and local conversation persistence.
- **Tauri** is an explicit secure broker: OS credential vault, engine process lifecycle, ephemeral process tokens, and forwarding of sensitive control commands. It owns no session, provider, or transcript logic.
- **React** owns presentation and non-secret UI state only.
- **Provider adapters** return normalized results and errors. They NEVER choose fallback, call UI code, or emit UI-specific behavior; orchestration in the engine decides fallback and the events that report it.

## Implemented System Architecture

EchoFlux is built upon an explicit 3-tier decoupled architecture:

| Concern | Implementation |
| --- | --- |
| Engine socket | Authenticated event plane; clients present an ephemeral token issued at engine bootstrap |
| Sensitive control | Travels React → Tauri IPC → privileged Tauri-to-engine control role; never over the public event plane |
| Secrets | Stored in OS vault (Windows Credential Manager / macOS Keychain) via Tauri; the webview receives only credential references and configured flags |
| Engine lifecycle | Launched and supervised by Tauri desktop bridge (bundled sidecar binary in release) |
| Providers | Independent STT, Translation, and Assistant services, each selected from a provider registry |
| Fallback | Session orchestration falls cloud STT/Translation back to Local on failure; Assistant errors are reported directly without fallback |

## WebSocket API Contract

ALL communication between Engine and UI must follow this exact JSON structure.

Every message carries a `type` field. Adding a type means updating BOTH sides plus this
file — `tests/test_ws_contract.py` fails if the three ever disagree.

### UI to Engine (6 commands)

| Type | Payload |
| --- | --- |
| `start` | `{"config": {"asr.model_size": ..., "asr.language": ..., "asr.device": ..., "translation.enabled": ..., "translation.backend": ..., "translation.source_lang": ..., "translation.target_lang": ..., "vad.enabled": ..., "audio.source": ..., "audio.mic_device_id": ..., "audio.speaker_device_id": ..., "llm.*": ...}}` — dotted config keys |
| `stop` | none |
| `list_devices` | none |
| `request_suggestion` | `{"entry_id": "...", "target_text": "...", "context": [...]}` |
| `request_summary` | `{"entries": [{"source": "...", "text": "..."}, ...]}` |
| `test_connection` | `{"capability": "stt|translation|assistant", "service": {...}}` |

An unrecognized type is answered with an `error` message, not ignored.

### Engine to UI (11 messages)

**Transcription**

- `{"type": "partial", "text": "...", "translation": null, "is_final": false, "timestamp": 123.4, "source": "mic|system"?, "translation_backend": "..."?}`
- `{"type": "final", "entry_id": "e-<ts>", "text": "...", "translation": null, "is_final": true, "timestamp": 123.4, "source": "mic|system"?, "translation_backend": "..."?}`
- `{"type": "translation_update", "entry_id": "...", "source_text": "...", "translation": "...", "timestamp": 123.4, "is_final": true, "translation_backend": "..."?}`

Translation is always asynchronous: `final` ships with `translation: null` and the text
arrives later in a `translation_update` matched on `entry_id`.

**Lifecycle**

- `{"type": "status", "status": "started|stopped"}`
- `{"type": "error", "message": "..."}`
- `{"type": "audio_level", "source": "mic|system", "level": 0.45}`

**Devices**

- `{"type": "devices_list", "microphones": [{"id", "name"}], "speakers": [{"id", "name"}]}`

**Validation**

- `{"type": "test_connection_result", "id": "...", "result": {}}`

**LLM assistant**

- `{"type": "suggestion_result", "entry_id": "...", "options": [...]}` or `{"entry_id": "...", "error": "..."}`
- `{"type": "llm_chunk", "text": "..."}` — streamed, repeated
- `{"type": "llm_done"}` — terminates a `llm_chunk` stream

## Model Management & Resource Handling

- Models MUST NOT be bundled in the codebase.
- Models MUST be dynamically downloaded to the OS-specific data directory (`~/.echoflux/models`).
- ALWAYS implement safe fallbacks. If GPU inference fails (e.g., missing cuDNN DLLs), catch the exception and fallback to CPU gracefully.
