# Design: Dual-Target AI Agent Harness

**Status:** Implemented
**Date:** 2026-09-12
**Scope:** Repository tooling only — no engine or UI behaviour changes

---

## 1. Understanding Summary

- **What:** A single-source agent harness serving two tools — Claude Code and Gemini Antigravity. `.agent/` stays canonical; a new `.claude/` layer plus `CLAUDE.md` point into it rather than copying it.
- **Why:** Claude Code currently reads nothing in this repo automatically. The rules in `.agent/` encode non-obvious constraints (asyncio/threading separation, the engine↔UI WebSocket contract, backend ABCs) where a plausible-looking change breaks the system at runtime, not at compile time.
- **Who:** Solo developer. All harness files are committed to git.
- **Constraints:** Zero content duplication between the two harnesses. `.agent/` file contents remain valid for Antigravity. Windows-first (PowerShell, `.venv\Scripts\`) but must not break on macOS/Linux. `CLAUDE.md` is loaded every session, so it delegates rather than restates. Hooks auto-format silently and lint advisorily — they never block.
- **Non-goals:** No Makefile rewrite. No cleanup of `devices.py` / `realtimesst.log`. No blocking rule-violation hooks. No test gating on Stop. No MCP config, no CI. No refactoring of engine or UI code.

---

## 2. Assumptions

1. Gemini Antigravity discovers `.agent/rules/`, `.agent/skills/`, `.agent/workflows/` from the repository root. Inferred from the existing layout; not verified against Antigravity's specification.
2. `.claude/` is not gitignored and should be committed. **Verified** — no entry in `.gitignore`.
3. Claude Code will follow a stub `SKILL.md` that delegates to the corresponding `.agent` file. One extra read hop, accepted as the cost of zero duplication.
4. Tests are smoke-level and dependency-light: no audio hardware, no model downloads, no GPU.
5. ~~Heavy ML imports may need `importorskip` guards.~~ **Disproven.** See Finding 3.
6. Harness correctness is validated by use; no meta-tests of the harness itself.
7. Hooks run local formatters only — no network I/O. `.env` reads stay denied.
8. `CLAUDE.md` under ~80 lines; hooks operate on the single edited file, never the tree.

---

## 3. Findings From Codebase Investigation

Evidence gathered before designing. Each finding changed the design.

### Finding 1 — The documented WebSocket contract is badly stale

`.agent/rules/03-architecture.md` documents 5 engine→UI message types and 2 UI→engine commands.

Actual implementation:

**Engine → UI (13):** `partial`, `final`, `translation_update`, `status`, `error`, `devices_list`, `download_progress`, `models_list`, `hub_search_results`, `model_action_result`, `suggestion_result`, `llm_chunk`, `llm_done`

**UI → Engine (9):** `start`, `stop`, `list_devices`, `request_suggestion`, `request_summary`, `request_models_list`, `download_model`, `delete_model`, `search_hub`

Note that the two *implementations* agree with each other exactly — the engine emits 13 and the UI handles all 13; the UI sends 9 and the server handles all 9. Only the documentation drifted. Sources: `engine/server/websocket_server.py`, `engine/main.py`, `apps/desktop/src/store/engineStore.ts`.

### Finding 2 — The `add-engine-backend` skill would produce broken code

`.agent/skills/add-engine-backend/SKILL.md` versus the real ABCs in `engine/asr/base.py` and `engine/translation/base.py`:

| Documented | Actual |
| --- | --- |
| `load_model(self, config: TranscriptionConfig)` | `load_model(self, config: dict)` |
| `transcribe_stream(self, audio_chunk: bytes) -> Optional[TranscriptResult]` | `transcribe_stream(self, audio_chunk: bytes, stream_id: str = "default") -> TranscriptResult` |
| 5 ASR methods listed | 6 abstract members — `finalize_current(stream_id)` is undocumented |

The omission is not cosmetic: a class written from the skill cannot be instantiated (`TypeError: Can't instantiate abstract class`).

### Finding 3 — Heavy ML imports are already deferred

`faster_whisper`, `transformers` and `torch` are imported inside `load_model()`, not at module level. Measured import cost: `engine.asr.base` 0.78s, `engine.translation.base` 0.15s, `engine.core.config` ~0s. Backend module top-levels need only stdlib and `numpy`.

**Consequence:** ABC-conformance tests can import and instantiate the real backend classes with no `importorskip` guards. Assumption 5 was wrong in our favour.

Caveat retained: the package `__init__.py` files are eager (`engine/asr/__init__.py` imports `FasterWhisperBackend`; `engine/translation/__init__.py` imports all three backends), so importing any submodule pulls its siblings. Harmless today because those modules are light; it would stop being harmless if a backend ever imports torch at module level.

### Finding 4 — Tooling reality

- `ruff` and `pytest` are present in `.venv`; `ruff check` on one file runs in ~0.5s.
- `pytest 9.0.2` + `pytest-asyncio 1.3.0` collect cleanly, so a suite runs as soon as `tests/` exists. (`make` itself turned out not to be installed — see D12.)
- **`prettier` is not installed.** `apps/desktop/package.json` has no prettier dependency, so `make format`'s `npx prettier` reaches the network.
- `tests/` does not exist, yet `pyproject.toml` sets `testpaths = ["tests"]` and both workflows end by telling the agent to run `make test`.
- **261 pre-existing ruff violations.** A naive format-on-edit hook would rewrite whole files and bury real changes.

### Finding 5 — Style rule contradicts linter

`.agent/rules/04-python-style.md` mandates `typing.Optional/List/Dict`. Ruff's enabled `UP` rules demand PEP-604 (`str | None`). The conflict accounts for 84 of the 261 violations (UP045 46, UP035 21, UP006 17). Left unresolved, the advisory hook would instruct the agent to violate rule 04 on every Python edit.

---

## 4. Decision Log

| # | Decision | Alternatives considered | Rationale |
| --- | --- | --- | --- |
| D1 | Support both Claude Code and Antigravity | Claude-native only; Antigravity only | User works across both tools. |
| D2 | Single source (`.agent/`) + pointer layer (`.claude/`) | Neutral `docs/agent/` core with both pointing in; full copies with a sync script; deliberately independent | Zero duplication and no new machinery. Moving content to a neutral core risked breaking Antigravity's glob-scoped rules; a sync script is machinery a solo repo does not need. |
| D3 | Include slash commands, subagents, hooks, and a permissions allowlist | Documentation-only harness | User selected all four. Commands make the existing workflows invocable; subagents keep large-file review out of the main context. |
| D4 | Auto-format silently, lint advisorily, never block | Blocking on rule violations; format only; full gate including tests | Pattern-matched rule enforcement produces false positives that stop work. Test gating is premature against a brand-new suite. |
| D5 | Close the verification gap and the docs gap; skip Makefile and stray-file cleanup | Harness only; harness + tests; everything including cleanup | Workflows and hooks depend on a working verify step, and both harnesses need one accurate architecture doc. Makefile portability is orthogonal. |
| D6 | Correct the stale docs **and** pin them with tests | Correct only; stop documenting the contract and generate it from source | Drift accumulated in roughly five commits and will recur. Code generation is a real pipeline neither tool benefits from. Pinning turns silent drift into a failing test. |
| D7 | Resolve the style conflict by ignoring `UP006/UP035/UP045` in ruff | Rewrite rule 04 to PEP-604 and `--fix` the code; rule change for new code only; leave conflicting and document it | Config-only change, zero code churn, keeps rule 04 true and silences 84 of 261 findings. |
| D8 | Hook formats only the edited file; prettier runs only if resolvable locally | Format the tree; use `npx prettier` | 261 pre-existing violations make tree-wide formatting destructive. `npx` means network I/O inside a hook, which assumption 7 forbids. |
| D9 | Subagents are read-only (`Read, Grep, Glob`) | Give them edit access | A reviewer that edits mid-review produces changes the user never reviewed. |
| D10 | Contract test parses source with scoped regexes | Import and introspect; AST parsing for TS | The TS side cannot be imported from Python, and AST-parsing TypeScript needs a Node toolchain. Regex is adequate if scoped narrowly and kept in one readable helper. |
| D11 | Auto-format only when the resulting diff is ≤ 20 lines | Always format the edited file; never auto-format | **Decided during implementation.** Measurement showed `ruff format` rewrites 226 of 827 lines in `engine/main.py`, so unconditional formatting would bury a one-line fix in a 227-line diff — the exact outcome D8 exists to prevent. The threshold makes formatting apply to files that are already clean (all new code) and stay out of the way on legacy files, and it self-heals as files are cleaned up. |
| D12 | Harness prescribes direct `.venv/Scripts/…` commands, not `make` | Keep `make` targets; install make | **Decided during implementation.** `make` is not installed on this machine (checked in both bash and PowerShell), so every documented verification command would have failed on first use. The Makefile remains as a statement of intent. |

---

## 5. Final Design

### 5.1 File plan

```
CLAUDE.md                              new — always-loaded entry point
.claude/
  settings.json                        permissions + hooks
  skills/
    add-engine-backend/SKILL.md        stub -> .agent/skills/add-engine-backend/
    add-ui-panel/SKILL.md              stub -> .agent/skills/add-ui-panel/
    code-review/SKILL.md               stub -> .agent/skills/code-review/
  commands/
    new-feature.md                     -> .agent/workflows/new-feature.md
    fix-engine-bug.md                  -> .agent/workflows/bug-fix-engine.md
  agents/
    engine-reviewer.md                 read-only
    ws-contract-checker.md             read-only
scripts/hooks/post_edit.py             format + advisory lint
tests/
  conftest.py                          env-isolated Config fixture
  test_config.py                       precedence chain
  test_backend_contract.py             pins ABCs against the skill doc
  test_ws_contract.py                  pins message types across 5 surfaces
ARCHITECTURE.md                        filled (currently 0 bytes)
```

Modified in place:

```
.agent/rules/03-architecture.md            corrected message lists (Finding 1)
.agent/skills/add-engine-backend/SKILL.md  corrected signatures (Finding 2)
pyproject.toml                             ruff ignore += UP006, UP035, UP045 (D7)
```

### 5.2 CLAUDE.md

Under ~80 lines. Contains only what the `.agent` rules do not:

- `@`-imports of the four rule files and `ARCHITECTURE.md`
- Environment facts: Windows-first, `.venv\Scripts\` vs `.venv/bin/`, the `make` targets that work
- An explicit instruction that `.agent/` is the source of truth — rule edits go there, never into `.claude/`
- A note that ~261 pre-existing ruff findings exist and unrelated ones are not to be "fixed" opportunistically

Accepted cost: five `@`-imports are always resident, roughly 130 lines of context per session.

### 5.3 Skill stubs

Each stub keeps its own `name` and `description` frontmatter — that is what Claude matches on during skill selection, so the wording is tuned for Claude. The body is one instruction: read the corresponding `.agent/skills/<name>/SKILL.md` and follow it verbatim. The `.agent` descriptions are left untouched for Antigravity.

### 5.4 Slash commands

- `/new-feature <description>` — executes the five steps of `.agent/workflows/new-feature.md`. Step 1 (define the WebSocket payload before writing either side) is the load-bearing step.
- `/fix-engine-bug <symptom>` — executes `.agent/workflows/bug-fix-engine.md`, starting from its subsystem-isolation triage.

Both carry `description` and `argument-hint` frontmatter, and pin verification to real commands: `make lint`, `make test`, `make cli ARGS="--model tiny"`.

### 5.5 Subagents

Both read-only (`Read, Grep, Glob`).

**`engine-reviewer`** — targets the two failure modes invisible in a diff but fatal at runtime:

- blocking calls introduced inside `async def`
- cross-thread handoff not using `queue.Queue`
- `unload_model()` / `_cleanup()` not releasing what `load_model()` / `start()` acquired
- `useEffect` without teardown

This is categories 1 and 3 of the `code-review` skill, given a dedicated context window.

**`ws-contract-checker`** — reads `engine/server/websocket_server.py`, `engine/main.py`, `apps/desktop/src/store/engineStore.ts` and the documented list, then reports any type present on one side but missing from another. It is the interactive twin of `test_ws_contract.py`: the test reports *that* surfaces diverged, the subagent explains *how*.

### 5.6 Hooks

**PostToolUse** matching `Edit|Write`, invoking `python scripts/hooks/post_edit.py` with the tool payload on stdin.

- `.py` → `ruff format` the edited file, then `ruff check` that file; findings returned to the agent as advisory text; exit 0 always.
- `.ts` / `.tsx` / `.css` → run `prettier` only if `apps/desktop/node_modules/.bin/prettier` resolves. Otherwise skip silently. Never `npx`.
- The script locates the venv itself (`Scripts/` on Windows, `bin/` elsewhere) and exits 0 on anything missing, so it degrades to a no-op rather than breaking a session or a non-Windows clone.

**Stop** — if `.py` or `.tsx` files changed during the session, remind to run `make lint` / `make test`. Advisory only.

### 5.7 Permissions

`.claude/settings.json`:

- **Allow:** `make lint|test|format`, `ruff check|format`, `pytest`, `git status|diff|log`, `npm run build`, `npx tsc --noEmit`
- **Deny:** `Read(./.env)`, `Read(**/.env)` — the file holds real values and is gitignored
- Everything else continues to prompt.

Machine-specific overrides belong in untracked `settings.local.json`.

### 5.8 Tests

**`conftest.py`** — the critical fixture. `Config.__init__` calls `_load_dotenv()` on both the repo `.env` and `~/.echoflux/.env`, mutating `os.environ`. Without isolation, tests read real settings and leak across runs. The fixture provides a `tmp_path` data dir, snapshots and restores `os.environ`, and clears `ECHOFLUX_*`.

**`test_config.py`** — the precedence chain the rules promise: defaults → `config.json` → `.env` → environment variables, each layer overriding the last. Plus dotted `get`/`set`, `_deep_merge` preserving sibling keys, and `models_dir`/`logs_dir` resolving under `data_dir`.

**`test_backend_contract.py`** — parses `def <name>` out of the Python fences in `.agent/skills/add-engine-backend/SKILL.md` and asserts that set equals `ASRBackend.__abstractmethods__`, and likewise for `TranslationBackend`. This compares documentation to code directly; it fails today on the missing `finalize_current`. It also instantiates `FasterWhisperBackend`, `MarianBackend`, `OnlineBackend` and `FallbackTranslationBackend` to prove ABC satisfaction — cheap, per Finding 3.

**`test_ws_contract.py`** — extracts message types from five surfaces (engine sends, engine handles, UI sends, UI handles, documented list) and asserts they agree.

Known fragility, accepted: this is regex over source. It must handle the ternary at `engine/main.py:560` (`"partial" if not asr_result.is_final else "final"`) and must not mistake TypeScript type annotations such as `type: "asr" | "translation"` at `engineStore.ts:149` for a message. The UI-send regex is scoped to `JSON.stringify({ type: ...`. All parsing lives in one readable helper so a false positive is obvious and fixable rather than mysterious.

### 5.9 Documentation corrections

- `.agent/rules/03-architecture.md` — all 13 engine→UI types and 9 UI→engine commands, with payload shapes.
- `.agent/skills/add-engine-backend/SKILL.md` — true signatures including `finalize_current`.
- `ARCHITECTURE.md` — filled from the code: thread topology and queue handoffs, the contract table, module map, config precedence, model storage paths per platform.

---

## 6. Risks

| Risk | Severity | Mitigation |
| --- | --- | --- |
| Antigravity's actual discovery rules differ from Assumption 1 | Medium | `.agent/` contents are unchanged apart from factual corrections, so nothing regresses even if the assumption is wrong. |
| Claude does not reliably follow stub skills to the `.agent` file | Medium | Stub bodies are a single imperative sentence with an explicit path. If this proves unreliable in use, D2 falls back to full copies plus a sync check. |
| Contract-test regexes produce false positives as the codebase grows | Medium | Parsing confined to one helper; failures name the exact surface and type. |
| Advisory lint noise from the remaining ~177 pre-existing violations | Low | Hook scopes to the edited file; `CLAUDE.md` states that unrelated findings are not to be fixed opportunistically. |
| Five `@`-imports consume context every session | Low | Rule files total ~130 lines; accepted deliberately. |
| Hook fails on a non-Windows clone | Low | Script exits 0 on any missing tool. |

---

## 7. Verification Plan

1. `.venv/Scripts/pytest.exe -q` — the new suite passes (after the doc corrections; `test_backend_contract.py` is expected to fail *before* them, which proves the pin works).
2. `.venv/Scripts/ruff.exe check .` — total violations drop by 84 via D7, with no code changes.
3. Edit a `.py` file in a Claude session — confirm the file is formatted, advisory findings appear, and nothing blocks.
4. Invoke `/new-feature` and `/fix-engine-bug` — confirm they resolve and read their `.agent` workflow.
5. Invoke each skill by name — confirm Claude reaches the `.agent` content.
6. Run `ws-contract-checker` — confirm it reports no divergence once docs are corrected.
7. Open the repo in Antigravity — confirm `.agent/` rules still apply.

---

## 8. Out of Scope

Makefile portability on Windows (`find`/`rm`/`cp` in recipes); `devices.py` and `realtimesst.log` at the repository root; adding prettier as a devDependency; the remaining ~177 ruff violations; MCP server configuration; CI workflows; any engine or UI behaviour change.
