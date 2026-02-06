# EchoFlux

Real-time local speech-to-text and translation desktop application.

EchoFlux runs entirely on your machine — no cloud services, no data collection, no telemetry.

## Features

- Real-time speech-to-text using Faster Whisper
- Optional translation using MarianMT
- Microphone and system audio capture
- GPU accelerated with CPU fallback
- Floating overlay window mode
- Transcript export (TXT, SRT, JSON)
- WebSocket-based architecture — engine and UI are fully decoupled
- Plugin-ready design

## Requirements

- Python 3.10+
- Node.js 18+ (for desktop app)
- Rust toolchain (for Tauri)
- CUDA toolkit (optional, for GPU acceleration)

## Quick Start

### 1. Clone

```bash
git clone https://github.com/your-org/echoflux.git
cd echoflux
```

### 2. Setup Python Environment

macOS / Linux:

```bash
chmod +x scripts/setup.sh
./scripts/setup.sh dev
source .venv/bin/activate
```

Windows (PowerShell):

```
powershell
.\scripts\setup.ps1 dev
.venv\Scripts\Activate.ps1
```

### 3. Configure

```bash
cp .env.example .env
```

Edit `.env` to match your setup. All settings have sensible defaults.

### 4. Run the Engine

```bash
make engine
```

Or directly:

```bash
python -m engine.main
```

### 5. Run the CLI

```bash
echoflux --model small --lang en
```

With translation:

```bash
echoflux --model small --lang en --translate vi
```

### 6. Run the Desktop App

```bash
cd apps/desktop
npm install
npm run tauri dev
```

## Project Structure

```
echoflux/
├── apps/
│   ├── desktop/              # Tauri + React frontend
│   │   ├── src/
│   │   │   ├── components/   # React components
│   │   │   ├── store/        # Zustand state management
│   │   │   ├── styles/       # CSS
│   │   │   ├── utils/        # Helpers (export, etc.)
│   │   │   ├── App.tsx
│   │   │   └── main.tsx
│   │   └── src-tauri/        # Tauri Rust backend
│   └── cli/                  # CLI interface
│
├── engine/
│   ├── core/                 # Config, logging, exceptions
│   ├── audio/                # Audio capture, VAD
│   ├── asr/                  # Speech recognition backends
│   ├── translation/          # Translation backends
│   ├── diarization/          # Speaker diarization (future)
│   ├── server/               # WebSocket server
│   └── main.py               # Engine entry point
│
├── plugins/                  # Plugin directory (future)
├── scripts/                  # Setup scripts
├── tests/                    # Test suite
└── docs/                     # Documentation
```

## Architecture

EchoFlux uses a separated engine + UI architecture.

The **engine** is a standalone Python service that captures audio, runs ASR and translation models, and emits results over a local WebSocket. The **desktop app** is a Tauri + React frontend that connects to the engine and displays results. The engine can run without the UI, and the CLI provides a headless interface.

```
┌─────────────────────┐
│   Desktop App       │  Tauri + React
│   (UI Layer)        │
└──────────┬──────────┘
           │ WebSocket (localhost)
┌──────────▼──────────┐
│   Engine Service    │  Python
│   Real-time Core    │
└──────────┬──────────┘
           │
  ┌────────▼────────┐
  │   AI Backends   │
  │  (ASR / NMT)    │
  └─────────────────┘
```

## Configuration

Configuration is resolved in this order (later overrides earlier):

1. Built-in defaults
2. Config file (`~/.echoflux/config.json` or platform equivalent)
3. `.env` file in project root
4. Environment variables

All environment variables are prefixed with `ECHOFLUX_`. See `.env.example` for the full list.

## Model Management

Models are not bundled. On first run, the engine will download required models to the platform-specific data directory:

| Platform | Path                                             |
| -------- | ------------------------------------------------ |
| Windows  | `%USERPROFILE%\.echoflux\models\`                |
| macOS    | `~/Library/Application Support/EchoFlux/models/` |
| Linux    | `~/.local/share/echoflux/models/`                |

Override with `ECHOFLUX_MODELS_DIR` environment variable.

## Development

### Linting and Formatting

Python files are linted and formatted with [Ruff](https://docs.astral.sh/ruff/). Frontend files use [Prettier](https://prettier.io/).

```bash
make lint      # Check Python
make format    # Format Python + Frontend
```

### Pre-commit Hooks

```bash
pip install pre-commit
pre-commit install
```

### Testing

```bash
make test
```

### Available Make Commands

| Command              | Description                       |
| -------------------- | --------------------------------- |
| `make setup`         | Install production dependencies   |
| `make setup-dev`     | Install dev dependencies          |
| `make setup-desktop` | Install desktop app dependencies  |
| `make engine`        | Start the engine server           |
| `make cli`           | Run the CLI                       |
| `make lint`          | Run ruff linter                   |
| `make format`        | Format all code                   |
| `make test`          | Run test suite                    |
| `make clean`         | Remove caches and build artifacts |
| `make env`           | Create .env from example          |

## Security

- All communication is localhost only
- No remote connections by default
- No telemetry or data collection
- No analytics

## License

MIT
