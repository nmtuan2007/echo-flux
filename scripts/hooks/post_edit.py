"""PostToolUse hook: format the edited file and report lint findings for it.

Deliberately conservative, for two reasons measured in this repo:

1. ``ruff format`` rewrites a whole file, and most of this codebase predates ruff --
   formatting engine/main.py changes 226 of its 827 lines. Reformatting on every edit
   would bury one-line fixes in hundred-line diffs. So the file is only rewritten when
   the resulting diff is small enough to be attributable to the edit that just happened;
   otherwise we report and leave it alone. As files get cleaned up, formatting starts
   applying to them on its own.
2. prettier is not a project dependency. We use the local binary if it exists and skip
   silently otherwise -- never ``npx``, which would mean network I/O inside a hook.

This hook is advisory. It never blocks, and it exits 0 on every path including its own
failures: a broken hook must not be able to stop a session.
"""

import contextlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Above this many changed lines, a reformat is assumed to be pre-existing drift
# rather than a consequence of the current edit.
MAX_ATTRIBUTABLE_FORMAT_LINES = 20

TIMEOUT = 15
PY_EXTS = {".py"}
JS_EXTS = {".ts", ".tsx", ".js", ".jsx", ".css", ".json"}
SKIP_PARTS = {".venv", "node_modules", "dist", "target", "__pycache__", ".git"}

ROOT = Path(__file__).resolve().parent.parent.parent


def _run(cmd, **kwargs):
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=TIMEOUT,
        cwd=str(ROOT),
        **kwargs,
    )


def _find_ruff():
    for candidate in (
        ROOT / ".venv" / "Scripts" / "ruff.exe",
        ROOT / ".venv" / "bin" / "ruff",
    ):
        if candidate.exists():
            return str(candidate)
    return shutil.which("ruff")


def _find_prettier():
    base = ROOT / "apps" / "desktop" / "node_modules" / ".bin"
    for name in ("prettier.cmd", "prettier"):
        candidate = base / name
        if candidate.exists():
            return str(candidate)
    return None  # Never fall back to npx: that is a network call.


def _format_python(ruff, path):
    """Format only if the change is attributable to this edit. Returns a note or None."""
    try:
        diff = _run([ruff, "format", "--diff", path])
    except Exception:
        return None

    changed = sum(
        1
        for line in diff.stdout.splitlines()
        if line[:1] in "+-" and not line.startswith(("+++", "---"))
    )

    if changed == 0:
        return None

    if changed > MAX_ATTRIBUTABLE_FORMAT_LINES:
        return (
            f"Skipped auto-format: {Path(path).name} is not ruff-formatted "
            f"({changed} lines would change). Left as-is so your diff stays readable."
        )

    _run([ruff, "format", path])
    return None


def _lint_python(ruff, path):
    try:
        result = _run([ruff, "check", path])
    except Exception:
        return None
    if result.returncode == 0:
        return None
    output = result.stdout.strip()
    return output or None


def _format_frontend(path):
    prettier = _find_prettier()
    if not prettier:
        return None
    with contextlib.suppress(Exception):
        _run([prettier, "--write", path])
    return None


def main():
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return

    raw_path = (payload.get("tool_input") or {}).get("file_path")
    if not raw_path:
        return

    path = Path(raw_path)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists() or not path.is_file():
        return
    if SKIP_PARTS & set(path.parts):
        return

    try:
        path.relative_to(ROOT)
    except ValueError:
        return  # Outside the project; not ours to touch.

    target = str(path)
    notes = []

    if path.suffix in PY_EXTS:
        ruff = _find_ruff()
        if not ruff:
            return
        note = _format_python(ruff, target)
        if note:
            notes.append(note)
        findings = _lint_python(ruff, target)
        if findings:
            notes.append(
                "ruff findings for this file (advisory — this codebase has known "
                "pre-existing violations; fix what your change introduced, not "
                "unrelated lines):\n" + findings
            )
    elif path.suffix in JS_EXTS:
        _format_frontend(target)

    if notes:
        print(
            json.dumps(
                {
                    "hookSpecificOutput": {
                        "hookEventName": "PostToolUse",
                        "additionalContext": "\n\n".join(notes),
                    }
                }
            )
        )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # A hook must never be able to break a session.
        pass
    finally:
        os._exit(0)
