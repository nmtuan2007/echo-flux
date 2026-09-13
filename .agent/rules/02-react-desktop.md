---
description: React/Tauri Frontend rules — Zustand, WebSockets, Styling
globs: "apps/desktop/src/**/*.ts, apps/desktop/src/**/*.tsx"
---

# React Desktop Rules

## State Management (Zustand)

- ALL business logic, WebSocket connection handling, and transcript accumulation MUST live in the store (`src/store/`), never in components.
- **Current runtime:** the store is one module, `src/store/engineStore.ts`.
- **Approved target state:** the store MAY be split into typed Zustand slices (for example connection, session, transcript, providers, assistant) under `src/store/`. Slices MUST be composed into the single `useEngineStore` hook; do not create competing top-level stores.
- Components MUST be strictly presentational or dispatchers.
- Components MUST use the `useEngineStore` hook to access state.
- NEVER implement WebSocket `onmessage` logic inside a `.tsx` component.
- The store holds presentation and non-secret UI state only. In the target state, secret values are write-only from the webview: the store keeps credential references and configured/not-configured flags, never raw keys. (Today `config.llmApiKey` is still persisted to `localStorage`; that is legacy to migrate, not a pattern to copy.)

## WebSockets & Real-Time Updates

- Handle rapid updates efficiently. The `engineStore` implements heuristics to prevent flickering on `partial` updates.
- Ensure the WebSocket automatically reconnects if the Python engine goes down (handled by `reconnectTimer`).
- Keep UI operations lightweight. Do not block the React main thread, as partial updates arrive every few hundred milliseconds.

## Styling (Plain CSS)

- NEVER use inline styles unless absolutely necessary for dynamic layout.
- ALWAYS use the CSS variables defined in `src/styles/global.css` (e.g., `var(--bg-primary)`, `var(--accent)`).
- The app is dark-mode by default. Maintain the `var(--bg-*)` and `var(--text-*)` variable system for consistency.
- Maintain the `-webkit-app-region: drag` rule in the header to allow Tauri window dragging.

## Tauri & Desktop

- The UI is running in a Tauri webview. Avoid heavy computation in the browser.
- **Current runtime:** Tauri commands cover window management (overlay, stealth, cursor events) and `get_engine_url`. No secure bridge exists yet.
- **Approved target state:** the store MAY call Tauri IPC commands for credential storage, engine process lifecycle, and sensitive control (for example applying provider config or Test Connection). Tauri is a secure broker, not a business-logic owner: it stores secrets, holds process tokens, and forwards privileged commands to the engine. Session, provider, and transcript logic stay in Python; UI logic stays in the store.
- Components MUST NOT call `invoke` directly for engine or credential operations; route them through the store.
- File exports (TXT, SRT, JSON) use standard Web APIs (`Blob`, `URL.createObjectURL`) for cross-platform compatibility without native Tauri filesystem calls, keeping the web view decoupled.
