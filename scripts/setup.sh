#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

PYTHON_MIN_VERSION="3.10"
VENV_DIR=".venv"

# --- Find Python ---
find_python() {
    for cmd in python3.12 python3.11 python3.10 python3 python; do
        if command -v "$cmd" &> /dev/null; then
            version=$("$cmd" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null)
            major=$(echo "$version" | cut -d. -f1)
            minor=$(echo "$version" | cut -d. -f2)
            if [ "$major" -ge 3 ] && [ "$minor" -ge 10 ]; then
                echo "$cmd"
                return
            fi
        fi
    done
    echo ""
}

PYTHON_CMD=$(find_python)

if [ -z "$PYTHON_CMD" ]; then
    echo "Error: Python >= $PYTHON_MIN_VERSION not found."
    echo "Please install Python 3.10+ and try again."
    exit 1
fi

PYTHON_VERSION=$("$PYTHON_CMD" --version)
echo "Using: $PYTHON_VERSION ($PYTHON_CMD)"

# --- Create venv ---
if [ -d "$VENV_DIR" ]; then
    echo "Virtual environment already exists at $VENV_DIR"
    read -p "Recreate? (y/N): " confirm
    if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
        rm -rf "$VENV_DIR"
        "$PYTHON_CMD" -m venv "$VENV_DIR"
        echo "Virtual environment recreated."
    fi
else
    "$PYTHON_CMD" -m venv "$VENV_DIR"
    echo "Virtual environment created at $VENV_DIR"
fi

# --- Activate and install ---
source "$VENV_DIR/bin/activate"

pip install --upgrade pip setuptools wheel

MODE="${1:-dev}"

if [ "$MODE" = "prod" ]; then
    echo "Installing production dependencies..."
    pip install -r requirements.txt
elif [ "$MODE" = "dev" ]; then
    echo "Installing development dependencies..."
    pip install -r requirements-dev.txt
elif [ "$MODE" = "gpu" ]; then
    echo "Installing GPU dependencies..."
    pip install -r requirements.txt
    pip install torch>=2.0
else
    echo "Unknown mode: $MODE (use: dev, prod, gpu)"
    exit 1
fi

# --- Install project in editable mode ---
pip install -e .

# --- Create .env if missing ---
if [ ! -f .env ]; then
    cp .env.example .env
    echo "Created .env from .env.example"
fi

echo ""
echo "=========================================="
echo "  Setup complete!"
echo "  Activate: source .venv/bin/activate"
echo "  Run engine: make engine"
echo "  Run CLI:    make cli"
echo "=========================================="
