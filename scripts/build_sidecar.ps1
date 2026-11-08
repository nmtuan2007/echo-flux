<#
.SYNOPSIS
    Builds and packages the EchoFlux Python engine sidecar for Tauri.
.DESCRIPTION
    Determines host target triple, compiles engine/main.py using PyInstaller
    and packaging/echoflux-engine.spec, and deploys the binary to
    apps/desktop/src-tauri/binaries/echoflux-engine-$triple.exe.
.PARAMETER DryRun
    Show commands without executing PyInstaller compilation.
.PARAMETER Clean
    Clean PyInstaller cache before building (default: true).
.PARAMETER DistPath
    Directory for PyInstaller output (default: dist).
.PARAMETER TargetDir
    Destination directory for Tauri externalBin (default: apps/desktop/src-tauri/binaries).
.PARAMETER Triple
    Override target triple (e.g. x86_64-pc-windows-msvc).
#>

[CmdletBinding()]
param(
    [switch]$DryRun,
    [switch]$Clean = $true,
    [string]$DistPath = "dist",
    [string]$TargetDir = "apps/desktop/src-tauri/binaries",
    [string]$Triple = ""
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $scriptDir

# Locate Python in .venv or PATH
$pythonExe = Join-Path $repoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $pythonExe)) {
    $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCmd) {
        $pythonExe = $pythonCmd.Source
    } else {
        Write-Error "Could not find python executable in .venv or PATH."
        exit 1
    }
}

$buildPy = Join-Path $scriptDir "build_sidecar.py"
$pyArgs = @($buildPy, "--distpath", $DistPath, "--target-dir", $TargetDir)

if ($DryRun) {
    $pyArgs += "--dry-run"
}
if ($Clean) {
    $pyArgs += "--clean"
}
if ($Triple) {
    $pyArgs += @("--triple", $Triple)
}

Write-Host "Invoking sidecar builder with $pythonExe..." -ForegroundColor Cyan
& $pythonExe $pyArgs

if ($LASTEXITCODE -ne 0) {
    Write-Error "Sidecar build failed with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
}
