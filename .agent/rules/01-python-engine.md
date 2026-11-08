---
description: Python Engine rules — Threading, Asyncio, State, Providers, and Audio/ML Pipelines
globs: "**/*.py"
---

# Python Engine Rules

## Concurrency & Threading (CRITICAL)

The EchoFlux engine mixes `asyncio` (for WebSockets and control channels) and `threading` (for blocking Audio/ML tasks).

- NEVER run blocking ML inference or audio capture directly on the `asyncio` event loop.
- ALWAYS use `threading.Thread` for `CaptureThread`, `ProcessThread`, and `TranslationThread`.
- ALWAYS communicate between threads using thread-safe `queue.Queue`.
- ALWAYS communicate from threads back to the asyncio loop via thread-safe queues or `asyncio.Queue` using `get_nowait()` polling in a dedicated broadcast loop.

## Provider Registry & Adapter Architecture

EchoFlux organizes AI capabilities into three independent services: `stt`, `translation`, and `assistant`.

- ALWAYS register new providers via `engine.providers.manifests.build_default_registry()`.
- Every provider MUST specify its capability, protocol family, field schema, and adapter factory.
- Provider adapters MUST implement the respective capability interface:
  - `engine.providers.stt.base.STTProvider`
  - `engine.providers.translation.base.TranslationProvider`
  - `engine.providers.assistant.base.AssistantProvider`
- Adapters MUST emit normalized results and errors. Adapters NEVER decide fallback, manage UI state, or emit raw UI events. Fallback is strictly managed by `SessionOrchestrator`.

## Ephemeral Secrets & Zero-Persistence Rules

- NEVER persist raw API keys or third-party secrets to disk, SQLite database, logs, or error frames.
- Secrets are passed ephemerally by the Tauri desktop bridge into the control channel or via stdin for validation probes.
- Once a session terminates or a probe completes, in-memory secrets MUST be dropped.
- Error reporting MUST be sanitized: emit standardized error codes (e.g., `command_failed`, `invalid_payload`) without echoing raw secrets, exception traces, or provider endpoint tokens.

## Audio Processing Pipeline

- Audio capture (`read_chunk()`) MUST return `bytes` of `int16` PCM data.
- NEVER block the capture thread. If the processing queue is full, drop frames or handle gracefully.
- VAD (Voice Activity Detection) MUST process audio in standard window sizes (e.g., 512 samples for Silero).
- Keep latency minimal. Avoid unnecessary memory copies of large `numpy` arrays; use `np.frombuffer` instead of `np.fromstring`.

## Backend Interfaces (Legacy & Local Adapters)

Any local ASR, Translation, or Audio component MUST implement the corresponding abstract base class:

- `engine.asr.base.ASRBackend`
- `engine.translation.base.TranslationBackend`
- `engine.audio.input_manager.AudioInput`

## Configuration Management

- NEVER hardcode configuration values (ports, thresholds, model names).
- ALWAYS read from `engine.core.config.Config`.
- Config fallback order: Defaults -> `config.json` -> `.env` -> Environment Variables.

## Logging

- NEVER use `print()`.
- ALWAYS use the central logger: `from engine.core.logging import get_logger`.
- Include meaningful context in logs. Do not log inside high-frequency loops (like per-audio-chunk) unless using `logger.debug`.
- NEVER log API keys, Bearer tokens, or decrypted credential payloads.
