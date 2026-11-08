# EchoFlux

**Local-first, privacy-preserving desktop application for real-time speech-to-text, live translation, and context-aware conversation assistance.**

EchoFlux is engineered to keep your conversations private. By default, audio capture, transcription, translation, and meeting histories run 100% locally on your device — no cloud services required, no data collection, no telemetry. When you choose to connect optional cloud AI providers for higher accuracy or specialized languages, your API keys are protected in your operating system's native hardware-backed credential vault.

---

## Capabilities

EchoFlux organizes AI capabilities into three independent services:

1. **Real-time Speech-to-Text (STT)**
   - **Local Default**: Faster-Whisper (CTranslate2) with GPU acceleration (NVIDIA CUDA) and automatic CPU fallback.
   - **Cloud Providers**: OpenAI Realtime API, Google Gemini Live, Deepgram Nova-2, AssemblyAI Streaming, Azure Speech SDK, and custom WebSocket endpoints.
   - **Dual Audio Streams**: Simultaneous capture from microphone and system audio (WASAPI loopback on Windows, BlackHole / CoreAudio on macOS).
   - **Sticky Fallback**: Automatic, zero-loss failover to the Local STT model if a cloud provider disconnects during a meeting.

2. **Live Translation**
   - **Local Default**: Offline Neural Machine Translation models.
   - **Cloud Providers**: OpenAI-compatible chat endpoints (GPT-4o, Claude via proxy, Groq, Ollama), Google Gemini, and custom translation endpoints.
   - **Asynchronous Execution**: Live translation never blocks real-time audio capture or transcription streaming.

3. **Context-Aware AI Assistant**
   - **Suggested Responses & Clarifications**: Per-message assistance triggered on demand with context dynamically retrieved from the local database.
   - **Real-Time Meeting Summaries**: Live streaming summaries of conversations with cancellation controls and token truncation guards.
   - **Context Profiles**: Customize vocabulary, speaker labels, industry jargon, and meeting goals with immutable session snapshotting.

---

## Security & Privacy Invariants

- **Zero Raw Secrets in UI**: The desktop frontend (React/Zustand) and configuration files (`config.json`, `.env`) never store raw API keys.
- **OS Credential Vault**: All third-party credentials are encrypted and stored in native OS keyrings:
  - **Windows**: Windows Credential Manager
  - **macOS**: Apple macOS Keychain
- **Ephemeral Process Consumption**: The Rust desktop bridge injects secrets in memory via authenticated localhost channels exclusively when a session begins and discards them immediately upon session termination.
- **Local SQLite Persistence**: Conversations, transcripts, translations, and context profiles are saved exclusively to a local SQLite database on your machine.

---

## System Requirements

| Component | Windows | macOS |
| --- | --- | --- |
| **Operating System** | Windows 10 / 11 (64-bit) | macOS 12 Monterey or newer (Apple Silicon & Intel) |
| **Python** | Python 3.10+ (64-bit) | Python 3.10+ |
| **Node.js** | Node.js 18+ & npm | Node.js 18+ & npm |
| **Rust Toolchain** | `cargo` & `rustc` (MSVC target) | `cargo` & `rustc` (Darwin target) |
| **GPU Acceleration** | NVIDIA GPU with CUDA 12 (optional, CPU fallback supported) | Apple Silicon Metal acceleration |
| **Audio** | WASAPI Loopback (built-in) | BlackHole or virtual audio device (for system audio capture) |

---

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/your-org/echoflux.git
cd echo-flux
```

### 2. Set Up Python Environment

**Windows (PowerShell):**
```powershell
.\scripts\setup.ps1 dev
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
chmod +x scripts/setup.sh
./scripts/setup.sh dev
source .venv/bin/activate
```

### 3. Run Development Desktop App

```bash
cd apps/desktop
npm install
npm run tauri dev
```

On first launch, EchoFlux presents an **Onboarding & Consent Wizard** guiding you through local model readiness, audio device selection, and optional cloud service consent.

---

## Desktop UI Features

- **Floating Overlay Mode (PiP)**: Keep live subtitles floating unobtrusively on top of video conferencing applications (Zoom, Microsoft Teams, Google Meet).
- **Stealth / Screen-Share Protection**: Native Windows display affinity hides the overlay window during screen shares and recordings.
- **Context Profiles**: Create dedicated profiles for client meetings, technical standups, or multilingual discussions with domain vocabulary boosting.
- **Conversation History**: Search, review, rename, and export finalized transcripts to TXT, SRT, or JSON format.
- **Provider Connection Testing**: Validate credentials and latency directly from the Settings panel before starting a live session.

---

## Packaging the Sidecar Engine Binary

EchoFlux packages the Python engine as a standalone frozen binary using PyInstaller, bundled directly into the Tauri desktop installer via `externalBin`:

```powershell
# Windows
.\scripts\build_sidecar.ps1
```

```bash
# macOS / Linux
python scripts/build_sidecar.py
```

The script compiles `engine/main.py` using `packaging/echoflux-engine.spec` and deploys the resulting binary to `apps/desktop/src-tauri/binaries/echoflux-engine-$triple.exe`.

To build the final production desktop installer:
```bash
cd apps/desktop
npm run tauri build
```

---

## Troubleshooting

### Windows
- **Missing CUDA DLLs**: If CUDA inference fails on startup, EchoFlux automatically degrades to CPU inference. Ensure you have installed the [NVIDIA CUDA Toolkit 12](https://developer.nvidia.com/cuda-toolkit) and latest NVIDIA drivers.
- **System Audio Capture**: On Windows, system audio capture uses WASAPI loopback. Verify that your default output device is configured properly in Windows Sound Settings.
- **PyAudioWPatch Installation**: If installing on Windows without pre-built wheels, ensure the Microsoft C++ Build Tools are installed.

### macOS
- **System Audio Capture**: macOS does not allow direct loopback of system speaker audio without a virtual audio loopback driver. Install [BlackHole](https://github.com/ExistentialAudio/BlackHole) (e.g. `brew install blackhole-2ch`) and set it as a multi-output device in Audio MIDI Setup.
- **Microphone Permissions**: If audio capture fails, ensure Terminal or EchoFlux has Microphone permissions in *System Settings > Privacy & Security > Microphone*.

---

## Testing & Verification

EchoFlux includes an automated verification suite covering contract synchronization, unit tests, and security scans:

```powershell
# Run backend tests
.\.venv\Scripts\pytest.exe -q

# Run frontend tests & TypeScript validation
npm test --prefix apps/desktop -- --run
.\apps\desktop\node_modules\.bin\tsc.cmd --noEmit -p apps/desktop/tsconfig.json

# Run Rust desktop bridge tests & Clippy
cargo test --manifest-path apps/desktop/src-tauri/Cargo.toml
cargo clippy --manifest-path apps/desktop/src-tauri/Cargo.toml
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
