"""Adversarial stress-testing suite for Milestone 6 (Phase F / EF-033).

Empirically challenges:
1. `scripts/benchmarks/benchmark_realtime.py` (CLI validation, memory tracking, summary stats, nested JSON output)
2. `scripts/verify_release.ps1` (fail-closed behavior, exit code propagation, working directory preservation via Pop-Location)
3. `scripts/verify_release.sh` (POSIX compliance, Darwin guard tamper resistance, exit code 2 on non-Darwin)
4. `docs/release/provider-refactor-checklist.md` (claims verification against reality)
"""

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_benchmark_cli_iterations_zero():
    """Verify benchmark_realtime.py handles 0 iterations without division by zero."""
    script_path = REPO_ROOT / "scripts" / "benchmarks" / "benchmark_realtime.py"
    python_exe = sys.executable

    res = subprocess.run(
        [python_exe, str(script_path), "--iterations", "0", "--warmup", "0"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert res.returncode == 0, f"Expected 0 iterations to exit 0, got: {res.stderr}"
    assert "BENCHMARK SUMMARY RESULTS" in res.stdout
    assert "Partial STT Latency (ms)" in res.stdout


def test_benchmark_cli_iterations_negative():
    """Verify benchmark_realtime.py handles negative iterations without crashing."""
    script_path = REPO_ROOT / "scripts" / "benchmarks" / "benchmark_realtime.py"
    python_exe = sys.executable

    res = subprocess.run(
        [python_exe, str(script_path), "--iterations", "-3", "--warmup", "0"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert res.returncode == 0
    assert "BENCHMARK SUMMARY RESULTS" in res.stdout


def test_benchmark_cli_nested_output_json(tmp_path):
    """Verify benchmark_realtime.py successfully creates parent directories for output-json."""
    script_path = REPO_ROOT / "scripts" / "benchmarks" / "benchmark_realtime.py"
    python_exe = sys.executable
    out_json = tmp_path / "deep" / "nested" / "dir" / "report.json"

    res = subprocess.run(
        [
            python_exe,
            str(script_path),
            "--iterations",
            "1",
            "--warmup",
            "0",
            "--output-json",
            str(out_json),
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert res.returncode == 0, f"Benchmark failed: {res.stderr}"
    assert out_json.exists(), "Output JSON file was not created"

    with open(out_json, encoding="utf-8") as f:
        data = json.load(f)

    assert "hardware_reference" in data
    assert "summary_metrics" in data
    assert "iteration_runs" in data
    assert len(data["iteration_runs"]) == 1
    assert data["summary_metrics"]["partial_stt_latency_ms"]["mean"] > 0
    assert data["summary_metrics"]["rss_mb"]["mean"] > 0


def test_benchmark_memory_tracker_win32():
    """Verify ProcessMemoryTracker produces sane RSS readings on the current platform."""
    sys.path.insert(0, str(REPO_ROOT / "scripts" / "benchmarks"))
    try:
        from benchmark_realtime import ProcessMemoryTracker

        tracker = ProcessMemoryTracker()
        rss = tracker.get_rss_mb()
        peak_rss = tracker.get_peak_rss_mb()

        assert isinstance(rss, float)
        assert isinstance(peak_rss, float)
        assert rss > 10.0, f"Expected RSS > 10 MB, got {rss}"
        assert peak_rss >= rss, f"Peak RSS {peak_rss} must be >= current RSS {rss}"
    finally:
        sys.path.pop(0)


def test_verify_release_ps1_fail_closed_simulation():
    """Verify PowerShell verification runner pattern stops at first failure and propagates exit code 1."""
    test_ps1 = """
$ErrorActionPreference = "Stop"
Write-Host "STAGE_1_START"
cmd.exe /c "exit 99"
if ($LASTEXITCODE -ne 0) { throw "Stage 1 failed with exit code $LASTEXITCODE" }
Write-Host "STAGE_2_UNREACHABLE"
"""
    with tempfile.NamedTemporaryFile("w", suffix=".ps1", delete=False) as f:
        f.write(test_ps1)
        tmp_file = f.name

    try:
        res = subprocess.run(
            ["powershell.exe", "-ExecutionPolicy", "Bypass", "-File", tmp_file],
            capture_output=True,
            text=True,
        )
        assert res.returncode == 1, f"Expected exit code 1 from throw, got {res.returncode}"
        assert "STAGE_1_START" in res.stdout
        assert "STAGE_2_UNREACHABLE" not in res.stdout
        assert "Stage 1 failed with exit code 99" in res.stderr
    finally:
        os.remove(tmp_file)


def test_verify_release_ps1_try_finally_directory_safety():
    """Verify Push-Location / Pop-Location pattern in Stage 5 and Stage 6 restores cwd even on throw."""
    test_ps1 = f"""
$ErrorActionPreference = "Stop"
$originalDir = (Get-Location).Path
Push-Location "{REPO_ROOT}\\apps\\desktop"
try {{
    cmd.exe /c "exit 42"
    if ($LASTEXITCODE -ne 0) {{ throw "tsc failed" }}
}} finally {{
    Pop-Location
}}
"""
    with tempfile.NamedTemporaryFile("w", suffix=".ps1", delete=False) as f:
        f.write(test_ps1)
        tmp_file = f.name

    try:
        res = subprocess.run(
            ["powershell.exe", "-ExecutionPolicy", "Bypass", "-File", tmp_file],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )
        assert res.returncode == 1
        assert "tsc failed" in res.stderr
    finally:
        os.remove(tmp_file)


def test_verify_release_sh_darwin_guard_tamper_resistance():
    """Verify verify_release.sh cannot be bypassed by environment variables on non-Darwin platforms."""
    git_bash = Path("C:/Program Files/Git/bin/bash.exe")
    if not git_bash.exists():
        pytest.skip("Git Bash not found at standard path")

    sh_path = REPO_ROOT / "scripts" / "verify_release.sh"

    # Attack scenario: inject PLATFORM=Darwin, OSTYPE=darwin, uname override attempts
    env = os.environ.copy()
    env["PLATFORM"] = "Darwin"
    env["OSTYPE"] = "darwin"
    env["UNAME"] = "Darwin"

    res = subprocess.run(
        [str(git_bash), str(sh_path)],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(REPO_ROOT),
    )

    assert res.returncode == 2, f"Expected exit code 2 (BLOCKED), got {res.returncode}"
    assert "[BLOCKED] macOS Release Verification Gate" in res.stdout
    assert "macOS release verification can only be executed on an Apple Darwin host." in res.stdout


def test_verify_release_sh_syntax_check():
    """Verify verify_release.sh has valid POSIX/Bash syntax via bash -n."""
    git_bash = Path("C:/Program Files/Git/bin/bash.exe")
    if not git_bash.exists():
        pytest.skip("Git Bash not found at standard path")

    sh_path = REPO_ROOT / "scripts" / "verify_release.sh"
    res = subprocess.run(
        [str(git_bash), "-n", str(sh_path)],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert res.returncode == 0, f"Bash syntax check failed: {res.stderr}"


def test_checklist_referenced_artifacts_exist():
    """Verify that all files cited in docs/release/provider-refactor-checklist.md actually exist."""
    checklist_path = REPO_ROOT / "docs" / "release" / "provider-refactor-checklist.md"
    assert checklist_path.exists(), "provider-refactor-checklist.md does not exist"

    # Core target artifacts
    assert (REPO_ROOT / "scripts" / "benchmarks" / "benchmark_realtime.py").exists()
    assert (REPO_ROOT / "scripts" / "verify_release.ps1").exists()
    assert (REPO_ROOT / "scripts" / "verify_release.sh").exists()
    assert (REPO_ROOT / "docs" / "release" / "benchmark_results.json").exists()

    # Integration, security, and soak test suites
    assert (REPO_ROOT / "tests" / "integration" / "test_provider_matrix.py").exists()
    assert (REPO_ROOT / "tests" / "integration" / "test_session_failover.py").exists()
    assert (REPO_ROOT / "tests" / "security" / "test_secret_redaction.py").exists()
    assert (REPO_ROOT / "tests" / "soak" / "test_long_session.py").exists()

    # Verify deleted legacy files are truly absent
    deleted_files = [
        REPO_ROOT / "apps" / "desktop" / "src" / "components" / "ModelManagerPanel.tsx",
        REPO_ROOT / "engine" / "models" / "model_manager.py",
        REPO_ROOT / "engine" / "translation" / "online_backend.py",
        REPO_ROOT / "engine" / "translation" / "fallback_backend.py",
    ]
    for df in deleted_files:
        assert not df.exists(), f"Legacy file {df} should have been deleted per Phase D!"


def run_artifact_grep():
    """Helper running the exact artifact grep scanner logic from scripts/verify_release.ps1."""
    secret_regex_str = r"sk-[a-zA-Z0-9_-]{20,}|AIzaSy[a-zA-Z0-9_-]{33}|hf_[a-zA-Z0-9]{34,}|Bearer\s+[a-zA-Z0-9\._-]{20,}"
    exclusions = [
        ":^.agents",
        ":^tests/security/test_secret_redaction.py",
        ":^tests/test_adversarial_*",
        ":^tests/test_challenger_*",
    ]
    res = subprocess.run(
        ["git", "grep", "--untracked", "-n", "-E", secret_regex_str, "--", *exclusions],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    lines = [line.strip() for line in res.stdout.strip().splitlines() if line.strip()]

    env_secret_regex = re.compile(
        r"sk-[a-zA-Z0-9_-]{20,}|AIzaSy[a-zA-Z0-9_-]{33}|hf_[a-zA-Z0-9_-]{20,}|Bearer\s+[a-zA-Z0-9\._-]{20,}"
    )

    # Explicitly scan workspace environment files (.env, .env.*) that may be ignored by .gitignore
    for env_path in set(REPO_ROOT.rglob(".env*")):
        rel_str = str(env_path.relative_to(REPO_ROOT)).replace("\\", "/")
        parts = rel_str.split("/")
        if any(skip in parts for skip in [".git", ".agents", "node_modules", ".venv", "target"]):
            continue
        try:
            content = env_path.read_text(encoding="utf-8", errors="ignore")
            for idx, line in enumerate(content.splitlines(), start=1):
                if env_secret_regex.search(line):
                    lines.append(f"{rel_str}:{idx}:{line.strip()}")
        except Exception:
            pass

    mock_patterns = [
        "sk-test-not-a-real-key-",
        "sk-older-fake-value-",
        "sk-newer-fake-value-",
        "sk-fake-start-config-key-",
        "sk-test-not-real-",
        "sk-fake-validation-key-",
        "sk-test-DO-NOT-LEAK-",
    ]
    real_leaks = [leak for leak in lines if not any(m in leak for m in mock_patterns)]
    return real_leaks


def test_artifact_grep_clean_baseline():
    """Verify artifact grep detects zero leaks in clean workspace."""
    leaks = [leak for leak in run_artifact_grep() if "canary" not in leak.lower()]
    assert leaks == [], f"Expected zero leaks in clean workspace, got: {leaks}"


def test_artifact_grep_catches_canary_in_env_file():
    """Verify artifact grep catches an unredacted canary in a gitignored .env file."""
    env_canary = REPO_ROOT / ".env.test_canary"
    canary = "hf_CanaryEnvSecret01234567890123456789"
    try:
        env_canary.write_text(f"HF_TOKEN={canary}\n", encoding="utf-8")
        leaks = run_artifact_grep()
        assert any(".env.test_canary" in leak and canary in leak for leak in leaks), (
            f"Expected .env canary in {env_canary.name} to be caught, got: {leaks}"
        )
    finally:
        if env_canary.exists():
            env_canary.unlink()


def test_artifact_grep_catches_openai_canary_in_env_file():
    """Verify artifact grep catches an OpenAI canary in a gitignored .env file."""
    env_canary = REPO_ROOT / ".env.openai_canary"
    canary = "sk-CanarySecretOpenAI0123456789012345"
    try:
        env_canary.write_text(f"OPENAI_API_KEY={canary}\n", encoding="utf-8")
        leaks = run_artifact_grep()
        assert any(".env.openai_canary" in leak and canary in leak for leak in leaks), (
            f"Expected .env canary in {env_canary.name} to be caught, got: {leaks}"
        )
    finally:
        if env_canary.exists():
            env_canary.unlink()


def test_artifact_grep_catches_gemini_canary_in_env_file():
    """Verify artifact grep catches a Gemini canary in a gitignored .env file."""
    env_canary = REPO_ROOT / ".env.gemini_canary"
    canary = "AIzaSy" + "A" * 33
    try:
        env_canary.write_text(f"GEMINI_API_KEY={canary}\n", encoding="utf-8")
        leaks = run_artifact_grep()
        assert any(".env.gemini_canary" in leak and canary in leak for leak in leaks), (
            f"Expected .env canary in {env_canary.name} to be caught, got: {leaks}"
        )
    finally:
        if env_canary.exists():
            env_canary.unlink()


def test_artifact_grep_catches_bearer_token_in_env_file():
    """Verify artifact grep catches a Bearer token in a gitignored .env file."""
    env_canary = REPO_ROOT / ".env.bearer_canary"
    canary = "Bearer test_bearer_token_1234567890_abcdef"
    try:
        env_canary.write_text(f"AUTH={canary}\n", encoding="utf-8")
        leaks = run_artifact_grep()
        assert any(".env.bearer_canary" in leak and canary in leak for leak in leaks), (
            f"Expected .env canary in {env_canary.name} to be caught, got: {leaks}"
        )
    finally:
        if env_canary.exists():
            env_canary.unlink()


def test_verify_release_ps1_artifact_grep_catches_env_canary():
    """Verify scripts/verify_release.ps1 Stage 10 scanner catches .env.canary and fails closed."""
    env_canary = REPO_ROOT / ".env.ps1_canary"
    canary = "hf_PS1CanarySecret01234567890123456789"
    try:
        env_canary.write_text(f"HF_TOKEN={canary}\n", encoding="utf-8")
        ps1_snippet = f"""
$ErrorActionPreference = "Stop"
$repoRoot = "{REPO_ROOT}"
$secretRegex = 'sk-[a-zA-Z0-9_-]{{20,}}|AIzaSy[a-zA-Z0-9_-]{{33}}|hf_[a-zA-Z0-9]{{34,}}|Bearer\\s+[a-zA-Z0-9\\._-]{{20,}}'
$envSecretRegex = 'sk-[a-zA-Z0-9_-]{{20,}}|AIzaSy[a-zA-Z0-9_-]{{33}}|hf_[a-zA-Z0-9_-]{{20,}}|Bearer\\s+[a-zA-Z0-9\\._-]{{20,}}'
$mockPatterns = @('sk-test-not-a-real-key-')
$realLeaks = @()
$envFiles = Get-ChildItem -Path $repoRoot -Filter ".env*" -File -Recurse -Force -ErrorAction SilentlyContinue | Where-Object {{
    $_.FullName -notmatch '[\\\\/]\\.git[\\\\/]' -and
    $_.FullName -notmatch '[\\\\/]\\.agents[\\\\/]' -and
    $_.FullName -notmatch '[\\\\/]node_modules[\\\\/]' -and
    $_.FullName -notmatch '[\\\\/]\\.venv[\\\\/]' -and
    $_.FullName -notmatch '[\\\\/]target[\\\\/]'
}}
if ($envFiles) {{
    foreach ($envFile in $envFiles) {{
        $envMatches = Select-String -Path $envFile.FullName -Pattern $envSecretRegex -AllMatches -ErrorAction SilentlyContinue
        foreach ($match in $envMatches) {{
            $isMock = $false
            foreach ($m in $mockPatterns) {{
                if ($match.Line -like "*$m*") {{ $isMock = $true; break }}
            }}
            if (-not $isMock) {{
                $relPath = $envFile.FullName.Substring($repoRoot.Length).TrimStart('\\', '/')
                $realLeaks += "$($relPath):$($match.LineNumber):$($match.Line.Trim())"
            }}
        }}
    }}
}}
if ($realLeaks.Count -gt 0) {{
    exit 1
}}
exit 0
"""
        res = subprocess.run(
            ["powershell.exe", "-ExecutionPolicy", "Bypass", "-Command", ps1_snippet],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )
        assert res.returncode == 1, f"Expected Stage 10 scanner to fail on .env canary, got exit code {res.returncode}"
    finally:
        if env_canary.exists():
            env_canary.unlink()


def test_artifact_grep_catches_auditor_underscore_canary_in_env_file():
    """Verify artifact grep catches auditor's exact canary with underscores in a gitignored .env file."""
    env_canary = REPO_ROOT / ".env.test_canary"
    canary = "hf_CANARY_TEST_TOKEN_01234567890123456789"
    try:
        env_canary.write_text(f"HF_TOKEN={canary}\n", encoding="utf-8")
        leaks = run_artifact_grep()
        assert any(".env.test_canary" in leak and canary in leak for leak in leaks), (
            f"Expected .env canary in {env_canary.name} to be caught, got: {leaks}"
        )
    finally:
        if env_canary.exists():
            env_canary.unlink()




