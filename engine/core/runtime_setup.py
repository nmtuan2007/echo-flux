"""Runtime environment setup helpers for EchoFlux Engine startup and CLI overrides."""

import contextlib
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


def setup_cuda_paths() -> None:
    """Expose pip-installed NVIDIA runtime DLL directories on Windows before CTranslate2 loads."""
    if sys.platform != "win32":
        return

    venv_base = Path(sys.prefix)
    nvidia_path = venv_base / "Lib" / "site-packages" / "nvidia"
    targets = [
        nvidia_path / "cublas" / "bin",
        nvidia_path / "cudnn" / "bin",
        nvidia_path / "cuda_runtime" / "bin",
    ]

    for target in targets:
        if target.exists():
            abs_path = str(target.resolve())
            with contextlib.suppress(Exception):
                os.add_dll_directory(abs_path)
            os.environ["PATH"] = abs_path + os.pathsep + os.environ["PATH"]

    zlib_found = any(
        p and (Path(p) / "zlibwapi.dll").exists()
        for p in os.environ["PATH"].split(os.pathsep)
    ) or (Path.cwd() / "zlibwapi.dll").exists()
    if not zlib_found:
        logging.getLogger("echoflux.engine").warning(
            "zlibwapi.dll not found. CUDA might fail. Check System32."
        )


def load_dotenv_file() -> None:
    env_path = Path.cwd() / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        load_dotenv()


def setup_engine_logging() -> None:
    log_dir = Path.home() / ".echoflux" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"session_{ts}.log"

    level_str = os.getenv("ECHOFLUX_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_str, logging.INFO)

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    logging.getLogger("echoflux.engine").info("Logging initialized - file: %s", log_file)


def apply_cli_config_env(cli_config: Any) -> None:
    """Propagate Config/CLI overrides into ECHOFLUX_* env defaults before engine startup."""
    if cli_config is None:
        return
    mapping = {
        "engine.host": "ECHOFLUX_HOST",
        "engine.port": "ECHOFLUX_PORT",
        "asr.model_size": "ECHOFLUX_MODEL_SIZE",
        "asr.language": "ECHOFLUX_LANGUAGE",
        "asr.device": "ECHOFLUX_DEVICE",
        "translation.enabled": "ECHOFLUX_TRANSLATION_ENABLED",
        "translation.source_lang": "ECHOFLUX_SOURCE_LANG",
        "translation.target_lang": "ECHOFLUX_TARGET_LANG",
        "vad.enabled": "ECHOFLUX_VAD_ENABLED",
        "logging.level": "ECHOFLUX_LOG_LEVEL",
    }
    for cfg_key, env_key in mapping.items():
        val = cli_config.get(cfg_key)
        if val is not None:
            os.environ[env_key] = str(val)
