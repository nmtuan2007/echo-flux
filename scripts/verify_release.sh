#!/usr/bin/env bash
set -euo pipefail

# EchoFlux Release Gate Verification Script for macOS (EF-033)

PLATFORM="$(uname -s)"

if [ "$PLATFORM" != "Darwin" ]; then
    echo "========================================================================"
    echo "  [BLOCKED] macOS Release Verification Gate"
    echo "========================================================================"
    echo "Current operating system: $PLATFORM"
    echo "macOS release verification can only be executed on an Apple Darwin host."
    echo "Per EF-033 Acceptance Criteria: Reporting BLOCKED honestly without falsifying PASS."
    echo "========================================================================"
    exit 2
fi

echo "========================================================================"
echo "  EchoFlux Release Verification Started on macOS Host: $(hostname)"
echo "========================================================================"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="./.venv/bin/python"
PYTEST_BIN="./.venv/bin/pytest"
RUFF_BIN="./.venv/bin/ruff"

if [ ! -f "$PYTHON_BIN" ]; then
    PYTHON_BIN="python3"
    PYTEST_BIN="pytest"
    RUFF_BIN="ruff"
fi

# Stage 1: Ruff check on touched files
echo "--> [Stage 1/10] Python Ruff Check on Touched Files"
"$RUFF_BIN" check \
    engine/asr/__init__.py \
    engine/conversation/repository.py \
    engine/main.py \
    engine/providers/stt/assemblyai.py \
    engine/providers/stt/azure_speech.py \
    engine/providers/stt/deepgram.py \
    engine/server/websocket_server.py \
    engine/server/control_dispatcher.py \
    engine/session/session_orchestrator.py \
    engine/translation/__init__.py \
    scripts/build_sidecar.py \
    scripts/benchmarks/benchmark_realtime.py \
    tests/integration/ \
    tests/security/ \
    tests/soak/ \
    tests/test_assistant_handlers.py \
    tests/test_backend_contract.py \
    tests/test_stt_provider_matrix.py \
    tests/test_ws_auth.py \
    tests/test_ws_contract.py \
    tests/test_adversarial_m1.py \
    tests/test_adversarial_m2.py \
    tests/test_adversarial_m2_c1.py \
    tests/test_challenger_m2_it2_c1.py \
    tests/test_challenger_m2_it2_c2.py \
    tests/test_control_apis.py

# Stage 2: Full Python pytest suite
echo "--> [Stage 2/10] Full Python Pytest Suite"
"$PYTEST_BIN" -q

# Stage 3: E2E Integration, Security, and Soak suites
echo "--> [Stage 3/10] E2E Integration, Security, and Soak Suites"
"$PYTEST_BIN" -q tests/integration/ tests/security/ tests/soak/

# Stage 4: Frontend Vitest tests
echo "--> [Stage 4/10] Frontend Vitest Tests"
npm --prefix apps/desktop test -- --run

# Stage 5: Frontend TypeScript verification
echo "--> [Stage 5/10] Frontend TypeScript Check (tsc --noEmit)"
(cd apps/desktop && npx tsc --noEmit)

# Stage 6: Frontend production build
echo "--> [Stage 6/10] Frontend Production Build (npm run build)"
(cd apps/desktop && npm run build)

# Stage 7: Rust Cargo check
echo "--> [Stage 7/10] Rust Cargo Check"
cargo check --manifest-path apps/desktop/src-tauri/Cargo.toml

# Stage 8: Rust Cargo test
echo "--> [Stage 8/10] Rust Cargo Tests"
cargo test --manifest-path apps/desktop/src-tauri/Cargo.toml

# Stage 9: Git whitespace & diff hygiene
echo "--> [Stage 9/10] Git Whitespace & Diff Hygiene Check"
git diff --check HEAD

# Stage 10: Secret leak scanner, Repository Artifact Grep & Realtime Latency Benchmark
echo "--> [Stage 10/10] Secret Redaction Tests, Artifact Grep & Realtime Latency Benchmark"
"$PYTEST_BIN" -q tests/security/test_secret_redaction.py

# Artifact Grep: Scan repository artifacts, fixtures, and source files for sensitive secret patterns
echo "--> Scanning repository artifacts for sensitive secret patterns..."
SECRET_REGEX='sk-[a-zA-Z0-9_-]{20,}|AIzaSy[a-zA-Z0-9_-]{33}|hf_[a-zA-Z0-9]{34,}|Bearer[[:space:]]+[a-zA-Z0-9._-]{20,}'
MOCK_PATTERNS='sk-test-not-a-real-key-|sk-older-fake-value-|sk-newer-fake-value-|sk-fake-start-config-key-|sk-test-not-real-|sk-fake-validation-key-|sk-test-DO-NOT-LEAK-'

RAW_MATCHES=$(git grep --untracked -n -E "$SECRET_REGEX" -- \
    ':^.agents' \
    ':^tests/security/test_secret_redaction.py' \
    ':^tests/test_adversarial_*' \
    ':^tests/test_challenger_*' 2>/dev/null || true)

# Explicitly scan workspace environment files (.env, .env.*) that may be ignored by .gitignore
ENV_SECRET_REGEX='sk-[a-zA-Z0-9_-]{20,}|AIzaSy[a-zA-Z0-9_-]{33}|hf_[a-zA-Z0-9_-]{20,}|Bearer[[:space:]]+[a-zA-Z0-9._-]{20,}'
ENV_FILES=$(find . -maxdepth 3 -name ".env*" \
    -not -path "*/.git/*" \
    -not -path "*/.agents/*" \
    -not -path "*/node_modules/*" \
    -not -path "*/.venv/*" \
    -not -path "*/target/*" 2>/dev/null || true)

if [ -n "$ENV_FILES" ]; then
    ENV_MATCHES=$(grep -H -n -E "$ENV_SECRET_REGEX" $ENV_FILES 2>/dev/null || true)
    if [ -n "$ENV_MATCHES" ]; then
        if [ -n "$RAW_MATCHES" ]; then
            RAW_MATCHES=$(printf "%s\n%s" "$RAW_MATCHES" "$ENV_MATCHES")
        else
            RAW_MATCHES="$ENV_MATCHES"
        fi
    fi
fi

REAL_LEAKS=""
if [ -n "$RAW_MATCHES" ]; then
    REAL_LEAKS=$(echo "$RAW_MATCHES" | grep -v -E "$MOCK_PATTERNS" || true)
fi

if [ -n "$REAL_LEAKS" ]; then
    echo "ERROR: Artifact grep detected unredacted secret patterns in repository artifacts:" >&2
    echo "$REAL_LEAKS" >&2
    exit 1
fi
echo "Artifact grep PASS: Zero secret leaks detected across repository."

mkdir -p docs/release
"$PYTHON_BIN" scripts/benchmarks/benchmark_realtime.py --iterations 3 --output-json docs/release/benchmark_results.json

echo "========================================================================"
echo "  VERDICT: PASS (All 10 macOS Release Gates Verified)"
echo "========================================================================"
exit 0
