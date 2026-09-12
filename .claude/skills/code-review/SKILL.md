---
name: code-review
description: Review EchoFlux code against the project's threading, decoupling, resource-leak, and UI rules. Use when reviewing a diff, a new backend, or a UI change in this repo, or when asked to check work before committing.
---

Read `.agent/skills/code-review/SKILL.md` and follow it exactly, including its output
format. It is the canonical version of this skill, shared with Gemini Antigravity.

For a large or engine-heavy review, delegate instead of reading everything inline:

- `engine-reviewer` — threading and resource-leak sweep (categories 1 and 3)
- `ws-contract-checker` — engine↔UI message contract drift (category 2)
