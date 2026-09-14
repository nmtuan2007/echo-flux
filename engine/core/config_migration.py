"""Idempotent migration of ``config.json`` from the legacy flat schema (v1) to schema v2.

v2 adds ``schema_version`` and independent ``services.stt|translation|assistant`` sections.
Every other key, including the legacy flat sections the transitional runtime still reads, is
carried over. Raw secrets are not: ``llm.api_key`` is removed from the migrated document and
blanked in the backup, and credential-bearing parts of ``llm.provider_url`` are stripped from
both. ``MigrationResult.legacy_secret_detected`` tells the credential-import step to prompt
the user to re-enter the key into the OS vault.

File migration always writes a versioned backup before replacing the config, never
overwrites an existing backup, and re-running it on a v2 file is a no-op.
"""

import copy
import json
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urlsplit

from engine.core.config import (
    LEGACY_SCHEMA_VERSION,
    SCHEMA_VERSION,
    ConfigError,
    ServiceConfig,
    endpoint_has_embedded_credentials,
    public_config,
    read_schema_version,
    strip_endpoint_credentials,
)

LOCAL_PROVIDER = "local"
OPENAI_PROVIDER = "openai"
CUSTOM_OPENAI_COMPATIBLE_PROVIDER = "custom-openai-compatible"

# The only legacy translation backend with a v2 equivalent is local Marian. "online" is the
# unauthenticated path slated for removal, so it and unknown values also map to local.
_LOCAL_TRANSLATION_BACKENDS = frozenset({"marian"})

_OPENAI_HOST = "api.openai.com"

_LEGACY_API_KEY = "llm.api_key"
_LEGACY_PROVIDER_URL = "llm.provider_url"

_MAX_BACKUPS = 1000

# typing.Union is house style (rule 04); UP007 is the one PEP-604 rule not disabled globally.
PathInput = Union[str, Path]  # noqa: UP007


class ConfigMigrationError(Exception):
    """Migration refused or failed. Carries a fixed code; never file content or values."""

    def __init__(self, code: str) -> None:
        super().__init__(f"config migration failed: {code}")
        self.code = code


@dataclass(frozen=True)
class MigrationResult:
    from_version: int
    to_version: int
    migrated: bool
    backup_path: Optional[Path] = None
    legacy_secret_detected: bool = False
    # Fixed ``key:code`` notes describing lossy mappings. They never contain values.
    notes: Tuple[str, ...] = ()


def _effective_legacy(data: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve nested and dotted keys the way ``Config._deep_merge`` does, later keys winning."""
    view: Dict[str, Any] = {}

    def merge(base: Dict[str, Any], override: Dict[str, Any]) -> None:
        for key, value in override.items():
            if "." in key:
                node = view
                parts = key.split(".")
                for part in parts[:-1]:
                    if not isinstance(node.get(part), dict):
                        node[part] = {}
                    node = node[part]
                node[parts[-1]] = copy.deepcopy(value)
            elif isinstance(base.get(key), dict) and isinstance(value, dict):
                merge(base[key], value)
            else:
                base[key] = copy.deepcopy(value)

    merge(view, data)
    return view


def _lookup(view: Dict[str, Any], dotted: str) -> Any:
    node: Any = view
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def _clean_str(value: Any) -> Optional[str]:
    if not isinstance(value, str) or any(ord(ch) < 32 for ch in value):
        return None
    stripped = value.strip()
    return stripped or None


def redact_legacy_secrets(data: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    """Copy of a legacy document with ``llm.api_key`` blanked and URL credentials stripped.

    Handles both nested and dotted spellings and leaves every other setting as it was.
    Returns the copy and whether any secret material was found.
    """
    redacted = copy.deepcopy(data)
    detected = False
    section, _, key_leaf = _LEGACY_API_KEY.rpartition(".")
    url_leaf = _LEGACY_PROVIDER_URL.rpartition(".")[2]

    containers = [(redacted, _LEGACY_API_KEY, _LEGACY_PROVIDER_URL)]
    if isinstance(redacted.get(section), dict):
        containers.append((redacted[section], key_leaf, url_leaf))

    for container, key_name, url_name in containers:
        value = container.get(key_name)
        if isinstance(value, str) and value:
            container[key_name] = ""
            detected = True
        url = container.get(url_name)
        if isinstance(url, str) and url and endpoint_has_embedded_credentials(url):
            container[url_name] = strip_endpoint_credentials(url)
            detected = True
    return redacted, detected


def _derive(data: Dict[str, Any]) -> Tuple[Dict[str, Dict[str, Any]], List[str], bool]:
    view = _effective_legacy(data)
    notes: List[str] = []
    legacy_key = _lookup(view, _LEGACY_API_KEY)
    secret_detected = isinstance(legacy_key, str) and bool(legacy_key.strip())

    def options_from(mapping: Tuple[Tuple[str, str], ...]) -> Dict[str, Any]:
        options: Dict[str, Any] = {}
        for legacy_key_name, option_key in mapping:
            value = _clean_str(_lookup(view, legacy_key_name))
            if value is not None:
                options[option_key] = value
        return options

    # Local providers use curated model packs (D-032): asr.model_size and translation.model
    # stay in their legacy flat keys for the transitional runtime and are not carried into
    # modelOverride.
    stt = {
        "enabled": True,
        "providerId": LOCAL_PROVIDER,
        "options": options_from(
            (
                ("asr.language", "language"),
                ("asr.device", "device"),
                ("asr.compute_type", "compute_type"),
            )
        ),
    }

    backend = _clean_str(_lookup(view, "translation.backend"))
    if backend is not None and backend not in _LOCAL_TRANSLATION_BACKENDS:
        notes.append("translation.backend:mapped_to_local")
    translation_enabled = _lookup(view, "translation.enabled")
    translation = {
        # v2 default is enabled (D-003); an explicit legacy boolean is preserved.
        "enabled": translation_enabled if isinstance(translation_enabled, bool) else True,
        "providerId": LOCAL_PROVIDER,
        "options": options_from(
            (
                ("translation.source_lang", "source_lang"),
                ("translation.target_lang", "target_lang"),
            )
        ),
    }

    endpoint = _clean_str(_lookup(view, _LEGACY_PROVIDER_URL)) or ""
    if endpoint and endpoint_has_embedded_credentials(endpoint):
        notes.append("llm.provider_url:credentials_removed")
        secret_detected = True
        endpoint = strip_endpoint_credentials(endpoint)
    host = ""
    if endpoint:
        try:
            host = (urlsplit(endpoint).hostname or "").lower()
        except ValueError:
            host = ""
        if not host:
            notes.append("llm.provider_url:unparseable")
            endpoint = ""

    if not endpoint:
        # No usable legacy URL: select the approved default provider, left off until the user
        # configures a credential for it.
        assistant: Dict[str, Any] = {"enabled": False, "providerId": OPENAI_PROVIDER}
    else:
        assistant_enabled = _lookup(view, "llm.enabled")
        assistant = {
            "enabled": assistant_enabled if isinstance(assistant_enabled, bool) else False,
            "providerId": (
                OPENAI_PROVIDER if host == _OPENAI_HOST else CUSTOM_OPENAI_COMPATIBLE_PROVIDER
            ),
            "endpoint": endpoint,
            "modelOverride": _clean_str(_lookup(view, "llm.model")) or "",
        }
    # Legacy keys were never in the vault, so there is nothing to reference yet.
    assistant["credentialRefs"] = {}
    if secret_detected:
        notes.append("llm.api_key:requires_vault_import")

    services = {}
    for name, section in (("stt", stt), ("translation", translation), ("assistant", assistant)):
        services[name] = ServiceConfig.from_dict(name, section).to_dict()
    return services, notes, secret_detected


def derive_services(data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """v2 ``services`` sections derived from a legacy document. Never includes secrets."""
    services, _, _ = _derive(data)
    return services


def migrate_data(data: Any) -> Tuple[Dict[str, Any], MigrationResult]:
    """Pure v1-to-v2 migration of a parsed document. A v2 document is returned unchanged."""
    try:
        version = read_schema_version(data)
    except ConfigError as exc:
        raise ConfigMigrationError(exc.code) from None
    if version == SCHEMA_VERSION:
        return copy.deepcopy(data), MigrationResult(version, SCHEMA_VERSION, migrated=False)
    if version > SCHEMA_VERSION:
        raise ConfigMigrationError("unsupported_schema_version")

    services, notes, secret_detected = _derive(data)
    redacted, _ = redact_legacy_secrets(data)
    migrated = public_config(redacted)
    for key in [k for k in migrated if k == "services" or k.startswith("services.")]:
        del migrated[key]
    migrated["schema_version"] = SCHEMA_VERSION
    migrated["services"] = services
    result = MigrationResult(
        from_version=version,
        to_version=SCHEMA_VERSION,
        migrated=True,
        legacy_secret_detected=secret_detected,
        notes=tuple(notes),
    )
    return migrated, result


def _backup_candidates(config_path: Path, from_version: int):
    stem, suffix = config_path.stem, config_path.suffix
    yield config_path.with_name(f"{stem}.v{from_version}.backup{suffix}")
    for index in range(1, _MAX_BACKUPS):
        yield config_path.with_name(f"{stem}.v{from_version}.backup.{index}{suffix}")


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_bytes().decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        raise ConfigMigrationError("invalid_json") from None


def _has_content(path: Path, expected: Any) -> bool:
    try:
        return _read_json(path) == expected
    except (ConfigMigrationError, OSError):
        return False


def write_backup(config_path: PathInput, from_version: int = LEGACY_SCHEMA_VERSION) -> Path:
    """Write a secret-free copy of the config to ``<stem>.v<N>.backup.json``.

    A file without secrets is copied byte for byte; otherwise the redacted document is
    written. An existing backup with the same redacted content (left by an interrupted run)
    is reused; a different one is kept and the next free ``.backup.<i>.json`` slot is used.
    """
    path = Path(config_path)
    original = path.read_bytes()
    data = _read_json(path)
    redacted, detected = redact_legacy_secrets(data) if isinstance(data, dict) else (data, False)
    payload = json.dumps(redacted, indent=2).encode("utf-8") if detected else original

    for candidate in _backup_candidates(path, from_version):
        if candidate.exists():
            if _has_content(candidate, redacted):
                return candidate
            continue
        try:
            with open(candidate, "xb") as f:
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())
        except FileExistsError:
            continue
        return candidate
    raise ConfigMigrationError("backup_slots_exhausted")


def migrate_config_file(config_path: PathInput) -> MigrationResult:
    """Migrate a config file in place. Safe to re-run; a v2 or missing file is left alone."""
    path = Path(config_path)
    if not path.exists():
        return MigrationResult(SCHEMA_VERSION, SCHEMA_VERSION, migrated=False)
    migrated, result = migrate_data(_read_json(path))
    if not result.migrated:
        return result

    backup_path = write_backup(path, result.from_version)
    temp_path = path.with_name(f"{path.name}.migrating")
    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(migrated, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()
    return replace(result, backup_path=backup_path)
