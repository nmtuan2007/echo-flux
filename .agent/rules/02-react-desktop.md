---
description: React/Tauri Frontend rules — Zustand, WebSockets, Styling
globs: "apps/desktop/src/**/*.ts, apps/desktop/src/**/*.tsx"
---

# React Desktop Rules

## State Management (Zustand)

- ALL business logic, WebSocket connection handling, and transcript accumulation MUST live in the store (`src/store/`), never in components.
- The store is organized into typed Zustand slices (`sessionSlice.ts`, `transcriptSlice.ts`, `settingsSlice.ts`, `conversationSlice.ts`) under `src/store/`.
- All slices MUST be composed into the single unified `useEngineStore` hook; do not create competing top-level stores.
- Components MUST be strictly presentational or dispatchers.
- Components MUST use the `useEngineStore` hook to access state.
- NEVER implement WebSocket `onmessage` logic inside a `.tsx` component.
- The store holds presentation and non-secret UI state only. Raw secret values (API keys) are NEVER held in Zustand or persisted to browser storage (`localStorage` / `sessionStorage`). The store maintains credential references (`cred_ref`) and configured/unconfigured flags. Raw secrets are saved directly to the OS vault via Tauri IPC.

## WebSockets & Real-Time Updates

- Handle rapid updates efficiently. The `engineStore` implements heuristics to prevent flickering on `partial` updates.
- Ensure the WebSocket automatically reconnects if the Python engine goes down (handled by `reconnectTimer`).
- Keep UI operations lightweight. Do not block the React main thread, as partial updates arrive every few hundred milliseconds.

## Styling (Plain CSS)

- NEVER use inline styles unless absolutely necessary for dynamic layout.
- ALWAYS use the CSS variables defined in `src/styles/global.css` (e.g., `var(--bg-primary)`, `var(--accent)`).
- The app is dark-mode by default. Maintain the `var(--bg-*)` and `var(--text-*)` variable system for consistency.
- Maintain the `-webkit-app-region: drag` rule in the header to allow Tauri window dragging.

## Tauri & Desktop Bridge

- The UI is running in a Tauri webview. Avoid heavy computation in the browser.
- The store calls Tauri IPC commands (`invoke`) for OS credential vault operations (`credential_save`, `credential_replace`, `credential_delete`), engine process lifecycle, and sensitive control commands (`control_forward`, `test_connection`).
- Tauri operates as an explicit secure broker: it stores secrets in Windows Credential Manager or macOS Keychain, holds short-lived engine process tokens, and forwards privileged control commands to the engine control port.
- Session, provider, and transcript logic stay in Python; presentation and UI interaction state stay in the React store.
- Components MUST NOT call `invoke` directly for engine or credential operations; route them through the store.
- File exports (TXT, SRT, JSON) use standard Web APIs (`Blob`, `URL.createObjectURL`) for cross-platform compatibility without native Tauri filesystem calls, keeping the web view decoupled.
