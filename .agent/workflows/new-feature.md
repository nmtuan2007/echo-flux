---
description: Workflow for adding a full-stack feature (Engine + UI).
---

# Workflow: New Full-Stack Feature

## Step 1: Define the WebSocket Payload
- Before touching Python or React, define exactly what JSON payload will be passed over the WebSocket.
- Update `engineStore.ts` (`handleMessage`) to accommodate the new payload type or config parameter.

## Step 2: Update Python Engine Configuration
- Add new config variables to `_DEFAULT_CONFIG` and `_ENV_MAP` in `engine/core/config.py`.
- Read the new config parameters inside `EchoFluxEngine._initialize_pipeline` in `main.py`.

## Step 3: Implement Python Logic
- If adding a new backend, use the `add-engine-backend` skill.
- If altering the audio pipeline, modify `engine/main.py`'s `_process_loop` or `_capture_loop`. Ensure thread safety.

## Step 4: Update React Desktop UI
- Update `engineStore.ts` to include the new UI state and default configuration values.
- Update `SettingsPanel.tsx` to include UI toggles for the new configuration.
- Implement UI rendering in the corresponding panel. Use existing CSS classes from `global.css`.

## Step 5: Test Cross-Platform
- Verify the feature works on CPU.
- Verify the feature works with VAD enabled and disabled.
- Ensure stopping and starting the pipeline re-initializes the feature cleanly without memory leaks.
