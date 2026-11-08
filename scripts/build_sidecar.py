#!/usr/bin/env python3
"""Build and deploy EchoFlux Python engine sidecar binary for Tauri externalBin.

Resolves the host target triple, compiles engine/main.py via PyInstaller
using packaging/echoflux-engine.spec, and places the frozen binary into
apps/desktop/src-tauri/binaries/echoflux-engine-$triple.exe.
"""

import argparse
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def resolve_target_triple() -> str:
    """Resolve Rust target triple via rustc, with standard platform fallback."""
    try:
        result = subprocess.run(
            ["rustc", "-vV"],
            capture_output=True,
            text=True,
            check=True,
        )
        for line in result.stdout.splitlines():
            if line.startswith("host:"):
                return line.split(":", 1)[1].strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        pass

    sys_name = sys.platform
    machine = platform.machine().lower()

    if sys_name == "win32":
        return "x86_64-pc-windows-msvc" if "64" in machine else "i686-pc-windows-msvc"
    elif sys_name == "darwin":
        return "aarch64-apple-darwin" if machine in ("arm64", "aarch64") else "x86_64-apple-darwin"
    elif sys_name.startswith("linux"):
        return "aarch64-unknown-linux-gnu" if machine in ("arm64", "aarch64") else "x86_64-unknown-linux-gnu"

    return f"{machine}-unknown-{sys_name}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build EchoFlux Python engine sidecar binary."
    )
    parser.add_argument(
        "--spec",
        default="packaging/echoflux-engine.spec",
        help="Path to PyInstaller spec file (default: packaging/echoflux-engine.spec)",
    )
    parser.add_argument(
        "--distpath",
        default="dist",
        help="Directory for PyInstaller output (default: dist)",
    )
    parser.add_argument(
        "--target-dir",
        default="apps/desktop/src-tauri/binaries",
        help="Destination directory for Tauri externalBin (default: apps/desktop/src-tauri/binaries)",
    )
    parser.add_argument(
        "--triple",
        default=None,
        help="Override target triple (e.g. x86_64-pc-windows-msvc)",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        default=True,
        help="Clean PyInstaller cache before building (default: True)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned operations and resolved paths without executing PyInstaller",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    spec_path = repo_root / args.spec
    dist_dir = repo_root / args.distpath
    target_dir = repo_root / args.target_dir

    if not spec_path.exists():
        print(f"Error: Spec file not found: {spec_path}", file=sys.stderr)
        return 1

    triple = args.triple or resolve_target_triple()
    is_windows = sys.platform == "win32"
    exe_suffix = ".exe" if is_windows else ""

    src_binary_name = f"echoflux-engine{exe_suffix}"
    dst_binary_name = f"echoflux-engine-{triple}{exe_suffix}"

    src_binary_path = dist_dir / src_binary_name
    dst_binary_path = target_dir / dst_binary_name

    print("EchoFlux Sidecar Build")
    print(f"  Target Triple  : {triple}")
    print(f"  Spec File      : {spec_path}")
    print(f"  Dist Binary    : {src_binary_path}")
    print(f"  Tauri Target   : {dst_binary_path}")

    pyinstaller_cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
    ]
    if args.clean:
        pyinstaller_cmd.append("--clean")
    pyinstaller_cmd.extend([
        "--distpath",
        str(dist_dir),
        str(spec_path),
    ])

    if args.dry_run:
        print("\n[Dry Run] Would execute:")
        print(f"  {' '.join(pyinstaller_cmd)}")
        print(f"  mkdir -p {target_dir}")
        print(f"  cp {src_binary_path} -> {dst_binary_path}")
        return 0

    print("\nRunning PyInstaller compilation...")
    result = subprocess.run(pyinstaller_cmd, cwd=str(repo_root))
    if result.returncode != 0:
        print(f"Error: PyInstaller compilation failed with code {result.returncode}", file=sys.stderr)
        return result.returncode

    if not src_binary_path.exists():
        print(f"Error: Built binary not found at {src_binary_path}", file=sys.stderr)
        return 1

    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nCopying {src_binary_name} -> {dst_binary_path}...")
    shutil.copy2(src_binary_path, dst_binary_path)

    print(f"Sidecar binary successfully deployed to {dst_binary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
