---
description: Add a full-stack feature across the Python engine and the React desktop UI
argument-hint: <what the feature should do>
---

Implement this feature: $ARGUMENTS

Read `.agent/workflows/new-feature.md` and follow its five steps in order. It is the
canonical workflow, shared with Gemini Antigravity.

Step 1 is load-bearing: define the WebSocket payload **before** writing Python or React.
Both sides plus `.agent/rules/03-architecture.md` must be updated together, or
`tests/test_ws_contract.py` will fail.

Verify with `.venv/Scripts/ruff.exe check .` and `.venv/Scripts/pytest.exe -q`.
Ask before running the engine — it needs audio hardware and may download models.
