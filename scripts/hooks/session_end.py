"""Stop hook: remind the user to verify, when source files were touched.

Prints to stdout and exits 0. It deliberately does NOT return a blocking decision --
a Stop hook that blocks can put the session into a continuation loop, and this is only
a reminder.
"""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
WATCHED = (".py", ".ts", ".tsx")


def main():
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True,
        text=True,
        timeout=10,
        cwd=str(ROOT),
    )
    if result.returncode != 0:
        return

    touched = {
        Path(line[3:].strip().strip('"')).suffix
        for line in result.stdout.splitlines()
        if line.strip()
    }

    if touched & set(WATCHED):
        targets = []
        if ".py" in touched:
            targets.append("make lint && make test")
        if {".ts", ".tsx"} & touched:
            targets.append("cd apps/desktop && npx tsc --noEmit")
        print("Unverified changes present. Suggested: " + "; ".join(targets))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        pass
    finally:
        sys.stdout.flush()
        os._exit(0)
