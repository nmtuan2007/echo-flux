$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $ProjectRoot

$PythonMinVersion = [version]"3.10"
$VenvDir = ".venv"
$VenvPython = "$VenvDir\Scripts\python.exe"

# --- Find Python ---
function Find-Python {
    $candidates = @("python3", "python")
    foreach ($cmd in $candidates) {
        try {
            $version = & $cmd -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')" 2>$null
            if ($version) {
                $parsed = [version]$version
                if ($parsed -ge $PythonMinVersion) {
                    return $cmd
                }
            }
        } catch {}
    }
    return $null
}

$PythonCmd = Find-Python

if (-not $PythonCmd) {
    Write-Error "Python >= $PythonMinVersion not found. Please install Python 3.10+ and try again."
    exit 1
}

$PythonVersion = & $PythonCmd --version
Write-Host "Using: $PythonVersion ($PythonCmd)"

# --- Create venv ---
if (Test-Path $VenvDir) {
    $confirm = Read-Host "Virtual environment exists. Recreate? (y/N)"
    if ($confirm -match "^[Yy]$") {
        Remove-Item -Recurse -Force $VenvDir
        & $PythonCmd -m venv $VenvDir
        Write-Host "Virtual environment recreated."
    }
} else {
    & $PythonCmd -m venv $VenvDir
    Write-Host "Virtual environment created at $VenvDir"
}

# --- Ensure venv python exists ---
if (-not (Test-Path $VenvPython)) {
    Write-Error "Virtual environment python not found."
    exit 1
}

# --- Upgrade packaging tools safely ---
& $VenvPython -m pip install --upgrade pip setuptools wheel

# --- Install dependencies ---
$Mode = if ($args.Count -gt 0) { $args[0] } else { "dev" }

switch ($Mode) {
    "prod" {
        Write-Host "Installing production dependencies..."
        & $VenvPython -m pip install -r requirements.txt
    }
    "dev" {
        Write-Host "Installing development dependencies..."
        & $VenvPython -m pip install -r requirements-dev.txt
    }
    "gpu" {
        Write-Host "Installing GPU dependencies..."
        & $VenvPython -m pip install -r requirements.txt
        & $VenvPython -m pip install "torch>=2.0"
    }
    default {
        Write-Error "Unknown mode: $Mode (use: dev, prod, gpu)"
        exit 1
    }
}

# --- Install project in editable mode ---
& $VenvPython -m pip install -e .

# --- Create .env if missing ---
if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "Created .env from .env.example"
}

Write-Host ""
Write-Host "=========================================="
Write-Host "  Setup complete!"
Write-Host "  Activate: .venv\Scripts\Activate.ps1"
Write-Host "  Run engine: make engine"
Write-Host "  Run CLI:    make cli"
Write-Host "=========================================="
