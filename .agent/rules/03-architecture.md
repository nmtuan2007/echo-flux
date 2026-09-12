---
description: Overall Architecture & Decoupling rules
globs: "**/*.py, **/*.ts, **/*.tsx"
---

# Architecture Rules

## strict Decoupling

EchoFlux uses a separated engine + UI architecture.

- The **Python Engine** knows NOTHING about the UI. It only broadcasts JSON over a WebSocket.
- The **React App** knows NOTHING about the local filesystem, Python environment, or models. It only sends commands and receives JSON over a WebSocket.
- Do NOT introduce tight coupling (e.g., the UI directly reading local Python files or executing Python scripts).

## WebSocket API Contract

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
