---
name: code-review
description: Perform automated code review against project rules.
---

# Code Review Skill

When reviewing code for EchoFlux, check for these categories:

## Category 1: Threading & Async (CRITICAL - Python)

- Are there blocking network/ML calls inside an `async def` function?
- Is thread-safe queueing (`queue.Queue`) used between audio/ml threads?
- Is `asyncio.Queue` or a polling mechanism used to bridge threads to WebSockets?

## Category 2: Decoupling (CRITICAL - General)

- Is the Python code trying to manipulate the UI directly? (Forbidden)
- Is the React code trying to load models or read OS paths? (Forbidden)
- Do all engine messages strictly adhere to the defined JSON structure in `engineStore.ts`?

## Category 3: Resource Leaks (HIGH)

- Python: Are models explicitly deleted in `unload_model()` and is garbage collection called if necessary?
- Python: Are Audio streams properly closed in `stop()` and `_cleanup()`?
- React: Does `useEffect` have a cleanup function (e.g., `disconnect()`)?

## Category 4: UX/UI (MEDIUM)

- Are CSS classes matching existing patterns (`btn`, `btn-icon`, `status-bar`)?
- Are colors hardcoded instead of using `var(--accent)`, `var(--bg-secondary)`, etc.?
- React: Does the UI flicker during rapid `partial` WebSocket messages?

## Output Format

For each issue found:

[CATEGORY] [SEVERITY] — [File:Line]
Description: [What is wrong]
Fix: [Exact code change needed]
