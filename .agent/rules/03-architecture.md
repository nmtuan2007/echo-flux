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

## Current Runtime vs Approved Target State

This is a staged migration. Do not describe target-state features as if they exist.

| Concern | Current runtime | Approved target state |
| --- | --- | --- |
| Engine socket | Unauthenticated localhost WebSocket carrying every command and event | Authenticated event plane; clients present an ephemeral token issued at engine bootstrap |
| Sensitive control | Sent over the WebSocket, including API keys in the `start` config | Travels React → Tauri IPC → privileged Tauri-to-engine control role; never over the event plane |
| Secrets | LLM API key held in Zustand and persisted to `localStorage` | Windows Credential Manager / macOS Keychain via Tauri; the webview gets references and configured state only |
| Engine lifecycle | Started separately from the desktop app | Launched and supervised by Tauri |
| Providers | ASR/translation backends chosen from downloaded model manifests | Independent STT, Translation, and Assistant services, each selected from a provider registry |
| Fallback | `FallbackTranslationBackend` wraps the configured backend | Session orchestration falls cloud STT/Translation back to Local; Assistant errors are reported, never fallen back |

Migration boundary: new secure modes and provider code are added alongside the legacy path
and stay dormant until their integration task switches the runtime over. Until then the
legacy path below must keep working.

## WebSocket API Contract

This section documents the **current runtime** contract. It stays unchanged until the code
changes; target-state messages are added here only in the same change that implements them.

ALL communication between Engine and UI must follow this exact JSON structure.

Every message carries a `type` field. Adding a type means updating BOTH sides plus this
file — `tests/test_ws_contract.py` fails if the three ever disagree.

### UI to Engine (9 commands)

| Type | Payload |
| --- | --- |
| `start` | `{"config": {"asr.model_size": ..., "asr.language": ..., "asr.device": ..., "translation.enabled": ..., "translation.backend": ..., "translation.source_lang": ..., "translation.target_lang": ..., "vad.enabled": ..., "audio.source": ..., "audio.mic_device_id": ..., "audio.speaker_device_id": ..., "llm.*": ...}}` — dotted config keys |
| `stop` | none |
| `list_devices` | none |
| `request_models_list` | none |
| `download_model` | `{"model_id": "...", "model_type": "asr\|translation", "hf_token": "..."}` |
| `delete_model` | `{"model_id": "...", "model_type": "asr\|translation"}` |
| `search_hub` | `{"query": "...", "task": "asr", "hf_token": "..."}` |
| `request_suggestion` | `{"entry_id": "...", "target_text": "...", "context": [...]}` |
| `request_summary` | `{"entries": [{"source": "...", "text": "..."}, ...]}` |

An unrecognized type is answered with an `error` message, not ignored.

### Engine to UI (13 messages)

**Transcription**

- `{"type": "partial", "text": "...", "translation": null, "is_final": false, "timestamp": 123.4, "source": "mic|system"?, "translation_backend": "..."?}`
- `{"type": "final", "entry_id": "e-<ts>", "text": "...", "translation": null, "is_final": true, "timestamp": 123.4, "source": "mic|system"?, "translation_backend": "..."?}`
- `{"type": "translation_update", "entry_id": "...", "source_text": "...", "translation": "...", "timestamp": 123.4, "is_final": true, "translation_backend": "..."?}`

Translation is always asynchronous: `final` ships with `translation: null` and the text
arrives later in a `translation_update` matched on `entry_id`.

**Lifecycle**

- `{"type": "status", "status": "started|stopped"}`
- `{"type": "error", "message": "..."}`

**Devices & models**

- `{"type": "devices_list", "microphones": [{"id", "name"}], "speakers": [{"id", "name"}]}`
- `{"type": "models_list", "asr": [...], "translation": [...]}`
- `{"type": "download_progress", "model": "...", "percent": 0-100}`
- `{"type": "model_action_result", "action": "download|delete", "model_id": "...", "success": bool, "error": "..."|null}`
- `{"type": "hub_search_results", "results": [...]}`

**LLM assistant**

- `{"type": "suggestion_result", "entry_id": "...", "options": [...]}` or `{"entry_id": "...", "error": "..."}`
- `{"type": "llm_chunk", "text": "..."}` — streamed, repeated
- `{"type": "llm_done"}` — terminates a `llm_chunk` stream

## Model Management & Resource Handling

- Models MUST NOT be bundled in the codebase.
- Models MUST be dynamically downloaded to the OS-specific data directory (`~/.echoflux/models`).
- ALWAYS implement safe fallbacks. If GPU inference fails (e.g., missing cuDNN DLLs), catch the exception and fallback to CPU gracefully.
