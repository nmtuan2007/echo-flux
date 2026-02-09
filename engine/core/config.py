import os
import sys
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


_DEFAULT_CONFIG = {
    "engine": {
        "host": "127.0.0.1",
        "port": 8765,
    },
    "audio": {
        "source": "microphone",
        "sample_rate": 16000,
        "channels": 1,
        "chunk_ms": 20,
        "format": "int16",
        "device_id": None,
    },
    "asr": {
        "backend": "faster_whisper",
        "model_size": "small",
        "language": "en",
        "compute_type": "float16",
        "device": "auto",
        "model_path": None,
    },
    "translation": {
        "enabled": False,
        "backend": "marian",
        "source_lang": "en",
        "target_lang": "vi",
        "model_path": None,
    },
    "vad": {
        "enabled": True,
        "threshold": 0.5,
    },
    "logging": {
        "level": "INFO",
        "max_bytes": 10_485_760,
        "backup_count": 5,
    },
}

_ENV_MAP = {
    "ECHOFLUX_HOST":                  ("engine.host",              str),
    "ECHOFLUX_PORT":                  ("engine.port",              int),
    "ECHOFLUX_AUDIO_SOURCE":          ("audio.source",             str),
    "ECHOFLUX_SAMPLE_RATE":           ("audio.sample_rate",        int),
    "ECHOFLUX_CHUNK_MS":              ("audio.chunk_ms",           int),
    "ECHOFLUX_AUDIO_DEVICE_ID":       ("audio.device_id",          str),
    "ECHOFLUX_ASR_BACKEND":           ("asr.backend",              str),
    "ECHOFLUX_MODEL_SIZE":            ("asr.model_size",           str),
    "ECHOFLUX_LANGUAGE":              ("asr.language",             str),
    "ECHOFLUX_COMPUTE_TYPE":          ("asr.compute_type",         str),
    "ECHOFLUX_DEVICE":                ("asr.device",               str),
    "ECHOFLUX_ASR_MODEL_PATH":        ("asr.model_path",           str),
    "ECHOFLUX_TRANSLATION_ENABLED":   ("translation.enabled",      lambda v: str(v).lower() in ("true", "1", "yes")),
    "ECHOFLUX_TRANSLATION_BACKEND":   ("translation.backend",      str),
    "ECHOFLUX_SOURCE_LANG":           ("translation.source_lang",  str),
    "ECHOFLUX_TARGET_LANG":           ("translation.target_lang",  str),
    "ECHOFLUX_TRANSLATION_MODEL_PATH":("translation.model_path",   str),
    "ECHOFLUX_VAD_ENABLED":           ("vad.enabled",              lambda v: str(v).lower() in ("true", "1", "yes")),
    "ECHOFLUX_VAD_THRESHOLD":         ("vad.threshold",            float),
    "ECHOFLUX_LOG_LEVEL":             ("logging.level",            str),
}


@dataclass
class TranscriptionConfig:
    """Configuration object for ASR backends."""
    model_size: str
    language: str
    device: str
    compute_type: str
    model_path: Optional[str] = None


def _get_data_dir() -> Path:
    override = os.environ.get("ECHOFLUX_DATA_DIR")
    if override:
        return Path(override)

    if sys.platform == "win32":
        base = os.environ.get("USERPROFILE", Path.home())
        return Path(base) / ".echoflux"
    elif sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "EchoFlux"
    else:
        xdg = os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")
        return Path(xdg) / "echoflux"


def _load_dotenv(env_path: Path):
    if not env_path.exists():
        return

    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            if value and value[0] in ('"', "'") and value[-1] == value[0]:
                value = value[1:-1]
            if key not in os.environ:
                os.environ[key] = value


class Config:
    def __init__(self, config_path: Optional[str] = None, env_path: Optional[str] = None):
        self._data: dict = json.loads(json.dumps(_DEFAULT_CONFIG))
        self._data_dir = _get_data_dir()
        self._config_path = Path(config_path) if config_path else self._data_dir / "config.json"

        # Override model/data dirs from env
        models_override = os.environ.get("ECHOFLUX_MODELS_DIR")
        if models_override:
            self._models_dir_override = Path(models_override)
        else:
            self._models_dir_override = None

        self._data_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        # Load .env (project root first, then data dir)
        project_env = Path(env_path) if env_path else Path.cwd() / ".env"
        _load_dotenv(project_env)
        _load_dotenv(self._data_dir / ".env")

        if self._config_path.exists():
            self._load_file()

        self._apply_env_overrides()

    def _load_file(self):
        with open(self._config_path, "r", encoding="utf-8") as f:
            user_config = json.load(f)
        self._deep_merge(self._data, user_config)

    def _deep_merge(self, base: dict, override: dict):
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value

    def _apply_env_overrides(self):
        for env_key, (dotted_key, cast_fn) in _ENV_MAP.items():
            value = os.environ.get(env_key)
            if value is not None:
                try:
                    self.set(dotted_key, cast_fn(value))
                except (ValueError, TypeError):
                    pass

    def get(self, dotted_key: str, default: Any = None) -> Any:
        keys = dotted_key.split(".")
        node = self._data
        for k in keys:
            if isinstance(node, dict) and k in node:
                node = node[k]
            else:
                return default
        return node

    def set(self, dotted_key: str, value: Any):
        keys = dotted_key.split(".")
        node = self._data
        for k in keys[:-1]:
            if k not in node or not isinstance(node[k], dict):
                node[k] = {}
            node = node[k]
        node[keys[-1]] = value

    def save(self):
        with open(self._config_path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2)

    @property
    def data_dir(self) -> Path:
        return self._data_dir

    @property
    def models_dir(self) -> Path:
        if self._models_dir_override:
            return self._models_dir_override
        return self._data_dir / "models"

    @property
    def logs_dir(self) -> Path:
        return self._data_dir / "logs"

    @property
    def raw(self) -> dict:
        return self._data
