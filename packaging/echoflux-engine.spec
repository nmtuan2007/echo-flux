# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for EchoFlux Python engine sidecar binary.

Compiles engine/main.py into a standalone executable (echoflux-engine)
packaged for Tauri externalBin integration.
"""

import os
import sys
import certifi

# Determine repository root relative to this spec file
spec_dir = os.path.dirname(os.path.abspath(SPEC)) if "SPEC" in locals() and SPEC else os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()
repo_root = os.path.abspath(os.path.join(spec_dir, ".."))
if not os.path.exists(os.path.join(repo_root, "engine", "main.py")):
    repo_root = os.path.abspath(".")

if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

entry_point = os.path.join(repo_root, "engine", "main.py")
block_cipher = None

# Bundle CA certificates for secure HTTPS/WSS communication with cloud AI providers
datas = [
    (certifi.where(), "certifi"),
]

hiddenimports = [
    # STT Dynamic Providers
    "engine.providers.stt.local",
    "engine.providers.stt.openai_realtime",
    "engine.providers.stt.gemini_live",
    "engine.providers.stt.deepgram",
    "engine.providers.stt.assemblyai",
    "engine.providers.stt.azure_speech",
    "engine.providers.stt.custom",
    # Translation Dynamic Providers
    "engine.providers.translation.local",
    "engine.providers.translation.openai_compatible",
    "engine.providers.translation.gemini",
    # Assistant Dynamic Providers
    "engine.providers.assistant.openai_compatible",
    "engine.providers.assistant.gemini",
    # Assistant Services
    "engine.assistant.service",
    "engine.assistant.summary_service",
    # Provider Validation (Test Connection)
    "engine.providers.validation",
    "engine.providers.text.openai_compatible",
    "engine.providers.text.gemini",
    "engine.providers.translation.openai_compatible",
    "engine.providers.translation.gemini",
    "engine.providers.assistant.openai_compatible",
    "engine.providers.assistant.gemini",
    # Session Orchestration
    "engine.session.session_orchestrator",
    "engine.session.translation_orchestrator",
    # Conversation & Context Profiles Persistence
    "engine.conversation.repository",
    "engine.conversation.context_profiles",
    # Local Model Catalog & Installer
    "engine.local_models.catalog",
    "engine.local_models.installer",
    "engine.local_models.selector",
    # Server & Protocol Dispatcher
    "engine.server.websocket_server",
    "engine.server.protocol",
    "engine.server.control_dispatcher",
    "engine.server.control_helpers",
    # Dynamic Websocket & Audio Dependencies
    "websockets.legacy",
    "websockets.legacy.server",
    "soundcard",
    "pyaudio",
    "pyaudiowpatch",
    "onnxruntime",
    "ctranslate2",
    "faster_whisper",
    "sqlite3",
]

# Intentionally exclude optional / non-bundled cloud SDKs & unused heavy GUI / test libs
excludes = [
    "azure.cognitiveservices.speech",
    "tkinter",
    "matplotlib",
    "PIL",
    "pytest",
    "_pytest",
    "py",
    "torch.testing",
    "torch.utils.tensorboard",
    "IPython",
    "jupyter",
]

a = Analysis(
    [entry_point],
    pathex=[repo_root],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="echoflux-engine",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
