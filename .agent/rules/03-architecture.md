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

ALL communication between Engine and UI must follow this exact JSON structure:

**UI to Engine:**

- Start: `{"type": "start", "config": { ... }}`
- Stop: `{"type": "stop"}`

**Engine to UI:**

- Partial Transcript: `{"type": "partial", "text": "...", "translation": "...", "timestamp": 123.4}`
- Final Transcript: `{"type": "final", "entry_id": "...", "text": "...", "translation": "...", "timestamp": 123.4}`
- Status Update: `{"type": "status", "status": "started|stopped"}`
- Async Translation: `{"type": "translation_update", "source_text": "...", "translation": "...", "timestamp": 123.4}`
- Error: `{"type": "error", "message": "..."}`

## Model Management & Resource Handling

- Models MUST NOT be bundled in the codebase.
- Models MUST be dynamically downloaded to the OS-specific data directory (`~/.echoflux/models`).
- ALWAYS implement safe fallbacks. If GPU inference fails (e.g., missing cuDNN DLLs), catch the exception and fallback to CPU gracefully.
