---
description: Python Engine rules — Threading, Asyncio, State, and Audio/ML Pipelines
globs: "*_/_.py"
---

# Python Engine Rules

## Concurrency & Threading (CRITICAL)

The EchoFlux engine mixes `asyncio` (for WebSockets) and `threading` (for blocking Audio/ML tasks).

- NEVER run blocking ML inference or audio capture directly on the `asyncio` event loop.
- ALWAYS use `threading.Thread` for `CaptureThread`, `ProcessThread`, and `TranslationThread`.
- ALWAYS communicate between threads using thread-safe `queue.Queue`.
- ALWAYS communicate from threads back to the asyncio loop via thread-safe queues or `asyncio.Queue` using `get_nowait()` polling in a dedicated broadcast loop.

## Audio Processing Pipeline

- Audio capture (`read_chunk()`) MUST return `bytes` of `int16` PCM data.
- NEVER block the capture thread. If the processing queue is full, drop frames or handle gracefully.
- VAD (Voice Activity Detection) MUST process audio in standard window sizes (e.g., 512 samples for Silero).
- Keep latency minimal. Avoid unnecessary memory copies of large `numpy` arrays; use `np.frombuffer` instead of `np.fromstring`.

## Backend Interfaces

Any new ASR, Translation, or Audio component MUST implement the corresponding abstract base class:

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
