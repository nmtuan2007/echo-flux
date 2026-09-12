---
description: Diagnose and fix a bug in the EchoFlux Python engine
argument-hint: <symptom, error message, or log excerpt>
---

Diagnose and fix: $ARGUMENTS

Read `.agent/workflows/bug-fix-engine.md` and follow it. It is the canonical workflow,
shared with Gemini Antigravity.

Start with its step 1 triage — isolate the subsystem (audio capture, ASR/translation,
threading/queues, or WebSocket) before proposing a fix. Logs are in `~/.echoflux/logs/`.

Bias toward the failure modes this engine actually has: a queue blocking without a
timeout, an exception escaping a subsystem boundary, or a GPU path that should have
degraded to CPU. Keep the fix minimal and preserve the graceful-degradation behaviour
described in ARCHITECTURE.md.

Verify with `.venv/Scripts/ruff.exe check .` and `.venv/Scripts/pytest.exe -q`.
