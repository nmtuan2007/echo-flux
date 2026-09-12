# EchoFlux

Real-time local speech-to-text and translation. Python engine + Tauri/React desktop app,
decoupled over a localhost WebSocket.

## Rules

`.agent/` is the single source of truth for this project's rules, skills, and workflows.
It is shared with Gemini Antigravity. **Edit rules there, never in `.claude/`** — the
files under `.claude/` are pointers, and content added to them is invisible to the other
tool.

@.agent/rules/01-python-engine.md
@.agent/rules/02-react-desktop.md
@.agent/rules/03-architecture.md
@.agent/rules/04-python-style.md
@ARCHITECTURE.md

## Environment

Windows + PowerShell. The virtualenv is at `.venv` — binaries are in `.venv/Scripts/`
(`.venv/bin/` on macOS/Linux). Forward slashes work in both PowerShell and bash.

**`make` is not installed on this machine.** The Makefile documents intent; use these
directly.

| Task | Command |
| --- | --- |
| Lint | `.venv/Scripts/ruff.exe check .` |
| Format | `.venv/Scripts/ruff.exe format .` |
| Test | `.venv/Scripts/pytest.exe -q` |
| Run engine | `.venv/Scripts/python.exe -m engine.main` |
| Run CLI headless | `.venv/Scripts/python.exe -m apps.cli.main --model tiny` |
| Build UI | `cd apps/desktop; npm run build` |
| Typecheck UI | `cd apps/desktop; npx tsc --noEmit` |

Caveats worth knowing before you run something:

- PowerShell has no `&&`. Use `;`, or `cmd; if ($?) { next }` when the second step must
  only run on success.
- prettier is **not** a project dependency, so there is no local frontend formatter.
  Don't reach for `npx prettier` — it downloads on every call.
- Running the engine needs audio hardware and downloads models on first use. Prefer the
  lint and test commands for verification; ask before starting the engine.

## Verification

`.venv/Scripts/pytest.exe -q` runs a fast (<1s) suite that pins three things: config precedence, the
engine↔UI message contract across all five places it is expressed, and the backend ABCs
against what `add-engine-backend` documents.

If a contract test fails, the documentation and the code have diverged — fix the pair,
don't relax the test.

## Working in this codebase

- **Threading is the main hazard.** The engine mixes asyncio and threads; blocking ML or
  audio work must never reach the event loop. See the thread model in ARCHITECTURE.md.
- **The codebase has ~177 pre-existing ruff violations.** The post-edit hook reports
  findings for the file you touched. Fix what your change introduced; do not
  opportunistically clean unrelated lines — it buries the real diff.
- `typing.Optional/List/Dict` is the deliberate house style here (rule 04). Ruff's
  PEP-604 rules are disabled to match; don't "modernize" annotations.
- Adding a WebSocket message type means updating the engine, the store, and
  `.agent/rules/03-architecture.md` together.
