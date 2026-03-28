$ErrorActionPreference = "Stop"

$scriptPath = $PSScriptRoot
$rootPath = Split-Path -Parent $scriptPath

Write-Host "EchoFlux Runner" -ForegroundColor Cyan
Write-Host "----------------" -ForegroundColor Cyan

# 1. Start Python Engine
$pythonPath = Join-Path $rootPath ".venv\Scripts\python.exe"
if (-not (Test-Path $pythonPath)) {
    Write-Warning "Virtual environment not found at $pythonPath. Trying global 'python'..."
    $pythonPath = "python"
}

Write-Host "Starting Python Engine..." -ForegroundColor Green
# Using Start-Process to open in a new window
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$rootPath'; & '$pythonPath' -m engine.main; if (`$LASTEXITCODE -ne 0) { Read-Host 'Engine crashed. Press Enter to exit...' }"

# 2. Start Frontend
$frontendPath = Join-Path $rootPath "apps\desktop"
Write-Host "Starting Frontend..." -ForegroundColor Green
# Using Start-Process to open in a new window
# Note: 'npm run tauri -- dev' runs the development server and the Tauri app
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$frontendPath'; npm run tauri -- dev"

Write-Host "Done! Engine and Frontend are launching in separate windows." -ForegroundColor Cyan
