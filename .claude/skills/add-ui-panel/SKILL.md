---
name: add-ui-panel
description: Add a new view or panel to the EchoFlux Tauri/React desktop app. Use when creating a new screen, tab, or panel component under apps/desktop/src/components/ and wiring it into navigation and the Zustand store.
---

Read `.agent/skills/add-ui-panel/SKILL.md` and follow it exactly. It is the canonical
version of this skill, shared with Gemini Antigravity.

It covers the four files a new panel always touches: the `AppView` type in the store,
the component itself, the conditional render in `App.tsx`, and the navigation control in
`Header.tsx`.
