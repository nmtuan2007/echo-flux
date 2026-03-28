---
description: Systematic bug fix workflow for the Python Engine.
---

# Workflow: Bug Fix (Python Engine)

## Step 1: Isolate the Subsystem
Identify where the bug is happening:
- **Audio Capture (`engine.audio`)**: VAD misfiring, PyAudio crashing, WASAPI loopback not finding devices.
- **ASR/Translation (`engine.asr`, `engine.translation`)**: Out of memory, hallucination loops, CTranslate2 DLL errors.
- **Threading/Queues (`engine.main`)**: Deadlocks, audio queue overflowing, slow processing.
- **WebSocket (`engine.server`)**: Connection drops, malformed JSON.

## Step 2: Reproduce & Diagnose
1. Check the logs in `~/.echoflux/logs/`.
2. Determine if the error is environment-specific (e.g., Windows WASAPI vs macOS, or CUDA missing).
3. If it's a threading issue, check if a queue is blocking without a timeout.

## Step 3: Implement Fix
- Ensure exceptions are caught at the subsystem boundary and logged via `get_logger()`.
- If an ML model fails on GPU, ensure it recursively falls back to CPU (see `MarianBackend._load_translator_safe`).
- If VAD fails to initialize, gracefully set `self._enabled = False` instead of crashing the pipeline.

## Step 4: Verify
- Run `make test` (if applicable).
- Run `make cli ARGS="--model tiny"` to verify the engine runs headless.
- Connect the desktop app and ensure real-time flow is undisturbed.
