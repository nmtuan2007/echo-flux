---
description: React/Tauri Frontend rules — Zustand, WebSockets, Styling
globs: "apps/desktop/src/**/*.ts, apps/desktop/src/**/*.tsx"
---

# React Desktop Rules

## State Management (Zustand)

- ALL business logic, WebSocket connection handling, and transcript accumulation MUST live in `src/store/engineStore.ts`.
- Components MUST be strictly presentational or dispatchers.
- Components MUST use the `useEngineStore` hook to access state.
- NEVER implement WebSocket `onmessage` logic inside a `.tsx` component.

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
- File exports (TXT, SRT, JSON) use standard Web APIs (`Blob`, `URL.createObjectURL`) for cross-platform compatibility without native Tauri filesystem calls, keeping the web view decoupled.
