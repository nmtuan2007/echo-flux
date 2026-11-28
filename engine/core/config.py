import contextlib
import copy
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlsplit, urlunsplit

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
        "mic_device_id": None,
        "speaker_device_id": None,
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
    "llm": {
        "enabled": False,
        "provider_url": "https://openrouter.ai/api/v1",
        "api_key": "",
        "model": "openai/gpt-4o-mini",
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
    "ECHOFLUX_MIC_DEVICE_ID":         ("audio.mic_device_id",      str),
    "ECHOFLUX_SPEAKER_DEVICE_ID":     ("audio.speaker_device_id",  str),
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
    "ECHOFLUX_LLM_ENABLED":          ("llm.enabled",              lambda v: str(v).lower() in ("true", "1", "yes")),
    "ECHOFLUX_LLM_PROVIDER_URL":     ("llm.provider_url",         str),
    "ECHOFLUX_LLM_API_KEY":          ("llm.api_key",              str),
    "ECHOFLUX_LLM_MODEL":            ("llm.model",                str),
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
    initial_prompt: Optional[str] = None


# Schema version 2: independent, provider-backed service sections under "services".
# Legacy flat sections (asr, translation, llm, ...) stay alongside them during the
# migration so the current runtime keeps reading the keys it already uses.

SCHEMA_VERSION = 2
LEGACY_SCHEMA_VERSION = 1
SERVICE_NAMES = ("stt", "translation", "assistant")

# D-003: Local STT and Local Translation are enabled by default; Assistant stays off until
# the user configures it.
_DEFAULT_SERVICES: Dict[str, Dict[str, Any]] = {
    "stt": {"enabled": True, "providerId": "local"},
    "translation": {"enabled": True, "providerId": "local"},
    "assistant": {"enabled": False, "providerId": "openai"},
}

# Legacy keys holding secret values. Readable in memory for the legacy runtime, never written.
LEGACY_SECRET_KEYS = ("llm.api_key",)
# Legacy URL keys whose credential-bearing parts are stripped before writing.
LEGACY_URL_KEYS = ("llm.provider_url",)

VALIDATION_UNVALIDATED = "unvalidated"
VALIDATION_VALID = "valid"
VALIDATION_INVALID = "invalid"
VALIDATION_STALE = "stale"

_SERVICE_FIELDS = frozenset(
    {"enabled", "providerId", "endpoint", "modelOverride", "options", "credentialRefs", "validation"}
)
_CREDENTIAL_REF_FIELDS = frozenset({"configId", "slot", "version"})
_VALIDATION_FIELDS = frozenset({"status", "fingerprint", "checkedAt", "errorCode"})
_STORED_VALIDATION_STATUSES = (VALIDATION_UNVALIDATED, VALIDATION_VALID, VALIDATION_INVALID)

_SECRET_KEY_NAMES = frozenset(
    {
        "apikey",
        "secret",
        "secretkey",
        "clientsecret",
        "token",
        "accesstoken",
        "authtoken",
        "password",
        "authorization",
        "bearer",
        "value",
    }
)

# Mirrors the vault id rules in apps/desktop/src-tauri/src/secure_store.rs so a pasted key
# can never become a provider id or credential reference.
_ID_RE = re.compile(r"[a-z0-9][a-z0-9_-]{0,47}")
_KEY_LIKE_PREFIXES = ("sk-", "sk_", "aiza", "gsk_", "xai-", "ghp_", "bearer")
_MAX_UNSEPARATED_RUN = 24
_ERROR_CODE_RE = re.compile(r"[a-z][a-z0-9_]{0,63}")
_FINGERPRINT_RE = re.compile(r"[0-9a-f]{64}")


class ConfigError(ValueError):
    """Invalid public config. Carries a location and a fixed code, never the offending value."""

    def __init__(self, path: str, code: str) -> None:
        super().__init__(f"{path}: {code}")
        self.path = path
        self.code = code


def is_secret_key_name(key: str) -> bool:
    """True for field names that would hold a raw credential rather than a reference."""
    return key.lower().replace("_", "").replace("-", "") in _SECRET_KEY_NAMES


def is_valid_id_segment(value: Any) -> bool:
    if not isinstance(value, str) or not _ID_RE.fullmatch(value):
        return False
    if value.startswith(_KEY_LIKE_PREFIXES):
        return False
    return max(len(run) for run in re.split(r"[-_]", value)) < _MAX_UNSEPARATED_RUN


def endpoint_has_embedded_credentials(endpoint: str) -> bool:
    """Detects user:pass@host and key-bearing query parameters such as ``?key=``."""
    try:
        parts = urlsplit(endpoint)
    except ValueError:
        return False
    if parts.username or parts.password:
        return True
    return any(_is_secret_query_pair(pair) for pair in parts.query.split("&"))


def _is_secret_query_pair(pair: str) -> bool:
    name = pair.split("=", 1)[0]
    return bool(name) and (name.lower() == "key" or is_secret_key_name(name))


def strip_endpoint_credentials(endpoint: str) -> str:
    """``endpoint`` without userinfo or key-bearing query parameters; otherwise unchanged."""
    if not endpoint_has_embedded_credentials(endpoint):
        return endpoint
    parts = urlsplit(endpoint)
    netloc = parts.netloc.rpartition("@")[2]
    query = "&".join(p for p in parts.query.split("&") if p and not _is_secret_query_pair(p))
    return urlunsplit((parts.scheme, netloc, parts.path, query, parts.fragment))


def read_schema_version(data: Any) -> int:
    """Schema version of a raw config document. Files written before v2 carry no marker."""
    if not isinstance(data, dict):
        raise ConfigError("config", "invalid_document")
    version = data.get("schema_version", LEGACY_SCHEMA_VERSION)
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise ConfigError("schema_version", "invalid_schema_version")
    return version


def public_config(data: Dict[str, Any]) -> Dict[str, Any]:
    """Deep copy of ``data`` with every legacy secret value removed, safe to serialize."""
    cleaned = copy.deepcopy(data)
    for dotted in LEGACY_SECRET_KEYS:
        cleaned.pop(dotted, None)
        section, _, leaf = dotted.rpartition(".")
        node = cleaned.get(section)
        if isinstance(node, dict):
            node.pop(leaf, None)
    for dotted in LEGACY_URL_KEYS:
        section, _, leaf = dotted.rpartition(".")
        node = cleaned.get(section)
        for container, key in ((cleaned, dotted), (node, leaf)):
            if isinstance(container, dict) and isinstance(container.get(key), str):
                container[key] = strip_endpoint_credentials(container[key])
    return cleaned


def _is_json_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, bool, int, float))


@dataclass(frozen=True)
class CredentialRef:
    """Pointer to one OS-vault entry: identity plus a version counter, never the secret.

    ``version`` is bumped whenever the stored secret is replaced, which is how a credential
    change reaches the validation fingerprint without the value ever touching config.
    """

    config_id: str
    slot: str
    version: int = 1

    def __post_init__(self) -> None:
        if not is_valid_id_segment(self.config_id):
            raise ConfigError("credentialRef.configId", "invalid_credential_ref")
        if not is_valid_id_segment(self.slot):
            raise ConfigError("credentialRef.slot", "invalid_credential_ref")
        if isinstance(self.version, bool) or not isinstance(self.version, int):
            raise ConfigError("credentialRef.version", "invalid_credential_version")
        if self.version < 0:
            raise ConfigError("credentialRef.version", "invalid_credential_version")

    def to_dict(self) -> Dict[str, Any]:
        return {"configId": self.config_id, "slot": self.slot, "version": self.version}

    @classmethod
    def from_dict(cls, path: str, data: Any) -> "CredentialRef":
        if not isinstance(data, dict):
            raise ConfigError(path, "invalid_credential_ref")
        for key in data:
            if key not in _CREDENTIAL_REF_FIELDS:
                secret_like = isinstance(key, str) and is_secret_key_name(key)
                raise ConfigError(path, "secret_value_rejected" if secret_like else "unknown_field")
        try:
            return cls(
                config_id=data.get("configId"),
                slot=data.get("slot"),
                version=data.get("version", 1),
            )
        except ConfigError as exc:
            raise ConfigError(path, exc.code) from None


@dataclass(frozen=True)
class ValidationMetadata:
    """Outcome of the last Test Connection, bound to the fingerprint it was run against."""

    status: str = VALIDATION_UNVALIDATED
    fingerprint: Optional[str] = None
    checked_at: Optional[float] = None
    # A ProviderErrorCode value. Free-text messages are not stored: they can echo secrets.
    error_code: Optional[str] = None

    def __post_init__(self) -> None:
        if self.status not in _STORED_VALIDATION_STATUSES:
            raise ConfigError("validation.status", "invalid_validation")
        if self.fingerprint is not None and not (
            isinstance(self.fingerprint, str) and _FINGERPRINT_RE.fullmatch(self.fingerprint)
        ):
            raise ConfigError("validation.fingerprint", "invalid_validation")
        if self.checked_at is not None and (
            isinstance(self.checked_at, bool) or not isinstance(self.checked_at, (int, float))
        ):
            raise ConfigError("validation.checkedAt", "invalid_validation")
        if self.error_code is not None and not (
            isinstance(self.error_code, str) and _ERROR_CODE_RE.fullmatch(self.error_code)
        ):
            raise ConfigError("validation.errorCode", "invalid_validation")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "fingerprint": self.fingerprint,
            "checkedAt": self.checked_at,
            "errorCode": self.error_code,
        }

    @classmethod
    def from_dict(cls, path: str, data: Any) -> "ValidationMetadata":
        if data is None:
            return cls()
        if not isinstance(data, dict) or any(key not in _VALIDATION_FIELDS for key in data):
            raise ConfigError(path, "invalid_validation")
        try:
            return cls(
                status=data.get("status", VALIDATION_UNVALIDATED),
                fingerprint=data.get("fingerprint"),
                checked_at=data.get("checkedAt"),
                error_code=data.get("errorCode"),
            )
        except ConfigError as exc:
            raise ConfigError(path, exc.code) from None


# Fields whose change alters what a Test Connection proved. ``enabled`` is deliberately absent:
# toggling a service off and on does not change whether its provider config works.
_FINGERPRINTED_CHANGES = frozenset(
    {"provider_id", "endpoint", "model_override", "options", "credential_refs"}
)


@dataclass(frozen=True)
class ServiceConfig:
    """Typed view of one ``services.<name>`` section."""

    service: str
    provider_id: str
    enabled: bool = False
    endpoint: str = ""
    model_override: str = ""
    options: Dict[str, Any] = field(default_factory=dict)
    credential_refs: Dict[str, CredentialRef] = field(default_factory=dict)
    validation: ValidationMetadata = field(default_factory=ValidationMetadata)

    def __post_init__(self) -> None:
        if self.service not in SERVICE_NAMES:
            raise ConfigError("services", "unknown_service")
        path = f"services.{self.service}"
        if not isinstance(self.enabled, bool):
            raise ConfigError(f"{path}.enabled", "invalid_type")
        if not is_valid_id_segment(self.provider_id):
            raise ConfigError(f"{path}.providerId", "invalid_provider_id")
        for name, value in (("endpoint", self.endpoint), ("modelOverride", self.model_override)):
            if not isinstance(value, str) or any(ord(ch) < 32 for ch in value):
                raise ConfigError(f"{path}.{name}", "invalid_type")
        if endpoint_has_embedded_credentials(self.endpoint):
            raise ConfigError(f"{path}.endpoint", "secret_value_rejected")
        if not isinstance(self.options, dict):
            raise ConfigError(f"{path}.options", "invalid_type")
        for key, value in self.options.items():
            if not isinstance(key, str):
                raise ConfigError(f"{path}.options", "invalid_option_key")
            if is_secret_key_name(key):
                raise ConfigError(f"{path}.options", "secret_value_rejected")
            if not _is_json_scalar(value):
                raise ConfigError(f"{path}.options.{key}", "invalid_option_value")
        if not isinstance(self.credential_refs, dict):
            raise ConfigError(f"{path}.credentialRefs", "invalid_type")
        for slot, ref in self.credential_refs.items():
            if not isinstance(ref, CredentialRef) or ref.slot != slot:
                raise ConfigError(f"{path}.credentialRefs", "invalid_credential_ref")
        if not isinstance(self.validation, ValidationMetadata):
            raise ConfigError(f"{path}.validation", "invalid_validation")
        # Detach from caller-owned dicts so later mutation cannot bypass validation.
        object.__setattr__(self, "options", dict(self.options))
        object.__setattr__(self, "credential_refs", dict(self.credential_refs))

    @classmethod
    def from_dict(cls, service: str, data: Any) -> "ServiceConfig":
        path = f"services.{service}"
        if not isinstance(data, dict):
            raise ConfigError(path, "invalid_section")
        for key in data:
            if key not in _SERVICE_FIELDS:
                secret_like = isinstance(key, str) and is_secret_key_name(key)
                raise ConfigError(path, "secret_value_rejected" if secret_like else "unknown_field")
        refs_raw = data.get("credentialRefs") or {}
        if not isinstance(refs_raw, dict):
            raise ConfigError(f"{path}.credentialRefs", "invalid_type")
        refs = {
            slot: CredentialRef.from_dict(f"{path}.credentialRefs", raw)
            for slot, raw in refs_raw.items()
        }
        model_override = data.get("modelOverride")
        options = data.get("options")
        return cls(
            service=service,
            provider_id=data.get("providerId"),
            enabled=data.get("enabled", False),
            endpoint=data.get("endpoint", ""),
            model_override="" if model_override is None else model_override,
            options={} if options is None else options,
            credential_refs=refs,
            validation=ValidationMetadata.from_dict(f"{path}.validation", data.get("validation")),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "providerId": self.provider_id,
            "endpoint": self.endpoint,
            "modelOverride": self.model_override,
            "options": dict(self.options),
            "credentialRefs": {slot: ref.to_dict() for slot, ref in self.credential_refs.items()},
            "validation": self.validation.to_dict(),
        }

    def fingerprint(self) -> str:
        """Digest of the public config a validation result depends on.

        Covers provider, endpoint, model override, options, and each credential reference
        including its version. There is no secret value in this object to include.
        """
        payload = {
            "schemaVersion": SCHEMA_VERSION,
            "service": self.service,
            "providerId": self.provider_id,
            "endpoint": self.endpoint.strip(),
            "modelOverride": self.model_override.strip(),
            "options": self.options,
            "credentialRefs": {slot: ref.to_dict() for slot, ref in self.credential_refs.items()},
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def validation_state(self) -> str:
        """``valid``/``invalid`` only while the stored fingerprint still matches; else stale."""
        if self.validation.status == VALIDATION_UNVALIDATED or not self.validation.fingerprint:
            return VALIDATION_UNVALIDATED
        if self.validation.fingerprint != self.fingerprint():
            return VALIDATION_STALE
        return self.validation.status

    def with_changes(self, **changes: Any) -> "ServiceConfig":
        """Copy with ``changes`` applied; clears validation when a fingerprinted field changes."""
        unknown = set(changes) - _FINGERPRINTED_CHANGES - {"enabled"}
        if unknown:
            raise ConfigError(f"services.{self.service}", "unknown_field")
        updated = replace(self, **changes)
        if updated.fingerprint() != self.fingerprint():
            updated = replace(updated, validation=ValidationMetadata())
        return updated

    def with_validation_result(
        self,
        success: bool,
        error_code: Optional[str] = None,
        checked_at: Optional[float] = None,
    ) -> "ServiceConfig":
        metadata = ValidationMetadata(
            status=VALIDATION_VALID if success else VALIDATION_INVALID,
            fingerprint=self.fingerprint(),
            checked_at=checked_at,
            error_code=None if success else error_code,
        )
        return replace(self, validation=metadata)


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

    with open(env_path, encoding="utf-8") as f:
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

        # No file means a fresh install: nothing to migrate.
        self._loaded_schema_version = SCHEMA_VERSION
        self._issues: List[str] = []

        if self._config_path.exists():
            self._load_file()

        self._apply_env_overrides()
        self._normalize_services()

    def _load_file(self):
        with open(self._config_path, encoding="utf-8") as f:
            user_config = json.load(f)
        try:
            self._loaded_schema_version = read_schema_version(user_config)
        except ConfigError as exc:
            self._issues.append(f"{exc.path}:{exc.code}")
            self._loaded_schema_version = LEGACY_SCHEMA_VERSION
            if not isinstance(user_config, dict):
                return
        self._deep_merge(self._data, user_config)
        if self._loaded_schema_version < SCHEMA_VERSION:
            # Legacy files stay readable: flat sections load untouched and the v2 services
            # are derived in memory. Nothing is written until migration or save() runs.
            # Imported here because config_migration depends on this module.
            from engine.core.config_migration import derive_services

            self._data["services"] = derive_services(user_config)
        elif self._loaded_schema_version > SCHEMA_VERSION:
            self._issues.append("schema_version:unsupported_schema_version")

    def _normalize_services(self):
        """Replace ``services`` with canonical v2 sections; invalid ones fall back to defaults.

        Degrades instead of raising so a hand-edited file cannot stop the engine, and records
        only location and code so no rejected value reaches logs or the UI.
        """
        services = self._data.get("services")
        if not isinstance(services, dict):
            if services is not None:
                self._issues.append("services:invalid_section")
            services = {}
        if any(name not in SERVICE_NAMES for name in services):
            self._issues.append("services:unknown_service")
        normalized = {}
        for name in SERVICE_NAMES:
            section = services.get(name, _DEFAULT_SERVICES[name])
            try:
                service = ServiceConfig.from_dict(name, section)
            except ConfigError as exc:
                self._issues.append(f"{exc.path}:{exc.code}")
                service = ServiceConfig.from_dict(name, _DEFAULT_SERVICES[name])
            normalized[name] = service.to_dict()
        self._data["services"] = normalized
        self._data["schema_version"] = SCHEMA_VERSION

    def _deep_merge(self, base: dict, override: dict):
        for key, value in override.items():
            if "." in key:
                self.set(key, value)
            elif key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value

    def _apply_env_overrides(self):
        for env_key, (dotted_key, cast_fn) in _ENV_MAP.items():
            value = os.environ.get(env_key)
            if value is not None:
                with contextlib.suppress(ValueError, TypeError):
                    self.set(dotted_key, cast_fn(value))

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
        """Write schema v2 JSON without secret values, backing up a legacy file first."""
        if self._loaded_schema_version < SCHEMA_VERSION and self._config_path.exists():
            from engine.core.config_migration import write_backup

            write_backup(self._config_path, self._loaded_schema_version)
        with open(self._config_path, "w", encoding="utf-8") as f:
            json.dump(public_config(self._data), f, indent=2)
        self._loaded_schema_version = SCHEMA_VERSION

    @property
    def schema_version(self) -> int:
        return SCHEMA_VERSION

    @property
    def loaded_schema_version(self) -> int:
        """Version of the file on disk when loaded; below SCHEMA_VERSION means legacy."""
        return self._loaded_schema_version

    @property
    def issues(self) -> List[str]:
        """``location:code`` entries for config that was rejected and replaced by defaults."""
        return list(self._issues)

    def service(self, name: str) -> ServiceConfig:
        if name not in SERVICE_NAMES:
            raise ConfigError("services", "unknown_service")
        return ServiceConfig.from_dict(name, self._data["services"][name])

    def set_service(self, service: ServiceConfig):
        self._data["services"][service.service] = service.to_dict()

    def update_service(self, name: str, **changes: Any) -> ServiceConfig:
        """Apply typed changes to one service; relevant changes invalidate its validation."""
        updated = self.service(name).with_changes(**changes)
        self.set_service(updated)
        return updated

    def record_validation(
        self,
        name: str,
        success: bool,
        error_code: Optional[str] = None,
        checked_at: Optional[float] = None,
    ) -> ServiceConfig:
        updated = self.service(name).with_validation_result(success, error_code, checked_at)
        self.set_service(updated)
        return updated

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
