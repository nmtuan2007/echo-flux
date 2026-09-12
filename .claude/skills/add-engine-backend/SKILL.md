---
name: add-engine-backend
description: Add a new ASR or translation backend to the EchoFlux Python engine. Use when adding, replacing, or wiring up a speech-recognition or machine-translation model implementation under engine/asr/ or engine/translation/.
---

Read `.agent/skills/add-engine-backend/SKILL.md` and follow it exactly. It is the
canonical version of this skill, shared with Gemini Antigravity.

Do not reimplement the steps from memory — the abstract method signatures it lists are
pinned by `tests/test_backend_contract.py`, and a class missing one cannot be
instantiated.
