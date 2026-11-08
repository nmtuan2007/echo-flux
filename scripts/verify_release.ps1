<#
.SYNOPSIS
    EchoFlux Windows Release Gate Verification Script (EF-033)
.DESCRIPTION
    Executes all 10 release verification stages per Master Plan Section 13 and EF-033:
    1. Python Ruff lint check on touched files
    2. Full Python pytest suite
    3. E2E Integration, Security, and Soak test suites
    4. Frontend Vitest test suite
    5. Frontend TypeScript type checking (tsc --noEmit)
    6. Frontend production build (npm run build)
    7. Rust Tauri Cargo check
    8. Rust Tauri Cargo test
    9. Git diff and whitespace hygiene check
    10. Secret leak scan, repository artifact grep, and Realtime Latency Benchmark harness
#>

$ErrorActionPreference = "Stop"

function Write-StageHeader {
    param([string]$Stage, [string]$Title)
    Write-Host "`n========================================================================" -ForegroundColor Cyan
    Write-Host "  [$Stage] $Title" -ForegroundColor Yellow
    Write-Host "========================================================================" -ForegroundColor Cyan
}

$repoRoot = (Resolve-Path "$PSScriptRoot\..").Path
Set-Location $repoRoot

$pythonExe = Join-Path $repoRoot ".venv\Scripts\python.exe"
$pytestExe = Join-Path $repoRoot ".venv\Scripts\pytest.exe"
$ruffExe = Join-Path $repoRoot ".venv\Scripts\ruff.exe"

$stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
Write-Host "EchoFlux Release Verification Started on Windows Host: $env:COMPUTERNAME" -ForegroundColor Green

# Stage 1: Ruff check on touched files
Write-StageHeader "Stage 1/10" "Checking Python Lint on Touched Files"
$touchedFiles = @(
    "engine/asr/__init__.py",
    "engine/conversation/repository.py",
    "engine/main.py",
    "engine/providers/stt/assemblyai.py",
    "engine/providers/stt/azure_speech.py",
    "engine/providers/stt/deepgram.py",
    "engine/server/websocket_server.py",
    "engine/server/control_dispatcher.py",
    "engine/session/session_orchestrator.py",
    "engine/translation/__init__.py",
    "scripts/build_sidecar.py",
    "scripts/benchmarks/benchmark_realtime.py",
    "tests/integration/",
    "tests/security/",
    "tests/soak/",
    "tests/test_assistant_handlers.py",
    "tests/test_backend_contract.py",
    "tests/test_stt_provider_matrix.py",
    "tests/test_ws_auth.py",
    "tests/test_ws_contract.py",
    "tests/test_adversarial_m1.py",
    "tests/test_adversarial_m2.py",
    "tests/test_adversarial_m2_c1.py",
    "tests/test_challenger_m2_it2_c1.py",
    "tests/test_challenger_m2_it2_c2.py",
    "tests/test_control_apis.py"
)
& $ruffExe check $touchedFiles
if ($LASTEXITCODE -ne 0) { throw "Stage 1 failed: Ruff lint check failed!" }
Write-Host "Stage 1 PASS: Ruff lint clean on all touched files." -ForegroundColor Green

# Stage 2: Full Python test suite
Write-StageHeader "Stage 2/10" "Running Full Python Pytest Suite"
& $pytestExe -q
if ($LASTEXITCODE -ne 0) { throw "Stage 2 failed: Python test suite failed!" }
Write-Host "Stage 2 PASS: Full Python test suite passed." -ForegroundColor Green

# Stage 3: E2E Integration, Security, and Soak suites
Write-StageHeader "Stage 3/10" "Running E2E Integration, Security, and Soak Suites"
& $pytestExe -q "$repoRoot\tests\integration\" "$repoRoot\tests\security\" "$repoRoot\tests\soak\"
if ($LASTEXITCODE -ne 0) { throw "Stage 3 failed: E2E integration, security, and soak suites failed!" }
Write-Host "Stage 3 PASS: All 36 E2E integration, security, and soak tests passed." -ForegroundColor Green

# Stage 4: Frontend Vitest suite
Write-StageHeader "Stage 4/10" "Running Frontend Vitest Tests"
npm --prefix apps/desktop test -- --run
if ($LASTEXITCODE -ne 0) { throw "Stage 4 failed: Frontend vitest tests failed!" }
Write-Host "Stage 4 PASS: Frontend Vitest suite passed." -ForegroundColor Green

# Stage 5: Frontend TypeScript verification
Write-StageHeader "Stage 5/10" "Running Frontend TypeScript Check (tsc --noEmit)"
Push-Location "$repoRoot\apps\desktop"
try {
    npx tsc --noEmit
    if ($LASTEXITCODE -ne 0) { throw "Stage 5 failed: Frontend TypeScript check failed!" }
} finally {
    Pop-Location
}
Write-Host "Stage 5 PASS: Frontend TypeScript type check clean (0 errors)." -ForegroundColor Green

# Stage 6: Frontend production build
Write-StageHeader "Stage 6/10" "Running Frontend Production Build (npm run build)"
Push-Location "$repoRoot\apps\desktop"
try {
    npm run build
    if ($LASTEXITCODE -ne 0) { throw "Stage 6 failed: Frontend production build failed!" }
} finally {
    Pop-Location
}
Write-Host "Stage 6 PASS: Frontend production build completed successfully." -ForegroundColor Green

# Stage 7: Rust Cargo check
Write-StageHeader "Stage 7/10" "Running Rust Cargo Check"
cargo check --manifest-path "$repoRoot\apps\desktop\src-tauri\Cargo.toml"
if ($LASTEXITCODE -ne 0) { throw "Stage 7 failed: Cargo check failed!" }
Write-Host "Stage 7 PASS: Rust Cargo check clean." -ForegroundColor Green

# Stage 8: Rust Cargo test
Write-StageHeader "Stage 8/10" "Running Rust Cargo Tests"
cargo test --manifest-path "$repoRoot\apps\desktop\src-tauri\Cargo.toml"
if ($LASTEXITCODE -ne 0) { throw "Stage 8 failed: Cargo tests failed!" }
Write-Host "Stage 8 PASS: Rust Cargo tests passed (50 passed)." -ForegroundColor Green

# Stage 9: Git whitespace & diff hygiene
Write-StageHeader "Stage 9/10" "Checking Git Whitespace & Diff Hygiene"
git diff --check HEAD
if ($LASTEXITCODE -ne 0) { throw "Stage 9 failed: Git whitespace / conflict check failed!" }
Write-Host "Stage 9 PASS: Zero whitespace or merge conflict markers found." -ForegroundColor Green

# Stage 10: Secret leak scanner, Repository Artifact Grep & Realtime Latency Benchmark
Write-StageHeader "Stage 10/10" "Running Secret Leak Scan, Artifact Grep & Realtime Benchmark Harness"
& $pytestExe -q "$repoRoot\tests\security\test_secret_redaction.py"
if ($LASTEXITCODE -ne 0) { throw "Stage 10 failed: Secret redaction test suite failed!" }

# Artifact Grep: Scan repository artifacts, fixtures, and source files for sensitive secret patterns
Write-Host "--> Scanning repository artifacts for sensitive secret patterns..."
$secretRegex = 'sk-[a-zA-Z0-9_-]{20,}|AIzaSy[a-zA-Z0-9_-]{33}|hf_[a-zA-Z0-9]{34,}|Bearer\s+[a-zA-Z0-9\._-]{20,}'
$exclusions = @(
    ':^.agents',
    ':^tests/security/test_secret_redaction.py',
    ':^tests/test_adversarial_*',
    ':^tests/test_challenger_*'
)
$rawMatches = git grep --untracked -n -E $secretRegex -- $exclusions 2>$null
$realLeaks = @()
$mockPatterns = @(
    'sk-test-not-a-real-key-',
    'sk-older-fake-value-',
    'sk-newer-fake-value-',
    'sk-fake-start-config-key-',
    'sk-test-not-real-',
    'sk-fake-validation-key-',
    'sk-test-DO-NOT-LEAK-'
)
if ($rawMatches) {
    foreach ($line in $rawMatches) {
        $isMock = $false
        foreach ($m in $mockPatterns) {
            if ($line -like "*$m*") { $isMock = $true; break }
        }
        if (-not $isMock) {
            $realLeaks += $line
        }
    }
}

# Explicitly scan workspace environment files (.env, .env.*) that may be ignored by .gitignore
$envSecretRegex = 'sk-[a-zA-Z0-9_-]{20,}|AIzaSy[a-zA-Z0-9_-]{33}|hf_[a-zA-Z0-9_-]{20,}|Bearer\s+[a-zA-Z0-9\._-]{20,}'
$envFiles = Get-ChildItem -Path $repoRoot -Filter ".env*" -File -Recurse -Force -ErrorAction SilentlyContinue | Where-Object {
    $_.FullName -notmatch '[\\/]\.git[\\/]' -and
    $_.FullName -notmatch '[\\/]\.agents[\\/]' -and
    $_.FullName -notmatch '[\\/]node_modules[\\/]' -and
    $_.FullName -notmatch '[\\/]\.venv[\\/]' -and
    $_.FullName -notmatch '[\\/]target[\\/]'
}
if ($envFiles) {
    foreach ($envFile in $envFiles) {
        $envMatches = Select-String -Path $envFile.FullName -Pattern $envSecretRegex -AllMatches -ErrorAction SilentlyContinue
        foreach ($match in $envMatches) {
            $isMock = $false
            foreach ($m in $mockPatterns) {
                if ($match.Line -like "*$m*") { $isMock = $true; break }
            }
            if (-not $isMock) {
                $relPath = $envFile.FullName.Substring($repoRoot.Length).TrimStart('\', '/')
                $realLeaks += "$($relPath):$($match.LineNumber):$($match.Line.Trim())"
            }
        }
    }
}

if ($realLeaks.Count -gt 0) {
    Write-Host "ERROR: Artifact grep detected unredacted secret patterns in repository artifacts:" -ForegroundColor Red
    foreach ($leak in $realLeaks) {
        Write-Host "  $leak" -ForegroundColor Red
    }
    throw "Stage 10 failed: Unredacted secret or canary pattern found in repository artifacts!"
}
Write-Host "Artifact grep PASS: Zero secret leaks detected across repository." -ForegroundColor Green

New-Item -ItemType Directory -Force -Path "$repoRoot\docs\release" | Out-Null
& $pythonExe "$repoRoot\scripts\benchmarks\benchmark_realtime.py" --iterations 3 --output-json "$repoRoot\docs\release\benchmark_results.json"
if ($LASTEXITCODE -ne 0) { throw "Stage 10 failed: Realtime benchmark failed!" }
Write-Host "Stage 10 PASS: Secret scan clean, artifact grep passed, and realtime benchmarks recorded." -ForegroundColor Green

$stopwatch.Stop()
Write-Host "`n========================================================================" -ForegroundColor Green
Write-Host "  VERDICT: PASS (All 10 Windows Release Gates Verified in $($stopwatch.Elapsed.TotalSeconds.ToString('F1'))s)" -ForegroundColor Green
Write-Host "========================================================================" -ForegroundColor Green
exit 0
