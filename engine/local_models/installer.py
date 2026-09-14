"""Resumable, verified, atomically activated local pack installs.

Layout under ``Config.models_dir / "packs"`` (so ``ECHOFLUX_DATA_DIR``/``ECHOFLUX_MODELS_DIR``
and the Windows/macOS/Linux data directories all apply)::

    packs/<pack_id>/
        staging/<version>/      downloads in progress; ``*.part`` files resume from their size
        versions/<version>/     complete, verified, immutable installs (+ ``pack.json``)
        active.json             {"active": <version>, "previous": <last known-good version>}

Guarantees:

* A pack is checked for publication, signature, and hardware compatibility before any byte is
  requested, and its files are checked against the catalog sha256 before promotion.
* Staging is promoted to ``versions/`` with a directory rename and activation is a single
  ``os.replace`` of ``active.json``. A failure or cancellation at any point leaves the
  previously active pack active and its files untouched.
* The last known-good version is retained as ``previous`` and ``rollback`` restores it after
  re-verifying its files. Older versions are pruned only after a successful activation.

Everything here blocks on disk and network I/O. Call it from a worker thread (as model downloads
already are in ``engine/main.py``), never from the asyncio event loop.

Signatures: no public-key signature library is a dependency of this project yet, so the
default verifier fails closed and nothing can activate until a trusted verifier is supplied.
``HmacSha256Verifier`` is a real stdlib verifier, but a symmetric key shipped inside the app
proves integrity only, not publisher authenticity; the production publisher signature must be
an asymmetric scheme (for example Ed25519) plugged in through ``SignatureVerifier``.
"""

import hashlib
import hmac
import json
import os
import shutil
import threading
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

from engine.core.config import Config
from engine.core.logging import get_logger
from engine.local_models.catalog import (
    PACK_METADATA_FILE,
    ModelPack,
    PackArtifact,
    PackCatalog,
    PackSignature,
)
from engine.local_models.selector import (
    DEFAULT_POLICY,
    HardwareProbe,
    TierPolicy,
    compatibility_issues,
)

logger = get_logger("local_models.installer")

POINTER_SCHEMA_VERSION = 1
_POINTER_FILE = "active.json"
_STAGING_MARKER = ".staging.json"
_HASH_BLOCK_SIZE = 1024 * 1024

ProgressCallback = Callable[[str, int], None]


class PackInstallError(Exception):
    """Install/activation failure with a fixed code and an optional artifact path."""

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail


class InstallCancelledError(PackInstallError):
    def __init__(self) -> None:
        super().__init__("cancelled")


@dataclass(frozen=True)
class DownloadResponse:
    """``chunks`` start at byte ``offset``: the requested offset, or 0 if the range was ignored."""

    offset: int
    chunks: Iterable[bytes]


class DownloadTransport(ABC):
    @abstractmethod
    def open(self, url: str, offset: int) -> DownloadResponse:
        """Start streaming ``url`` from byte ``offset``."""


class SignatureVerifier(ABC):
    @abstractmethod
    def verify(self, signature: PackSignature, payload: bytes) -> bool:
        """True only if ``signature`` is valid for ``payload`` under a trusted key."""


class FailClosedVerifier(SignatureVerifier):
    """Default verifier: no trusted keys are configured, so no pack can activate."""

    def verify(self, signature: PackSignature, payload: bytes) -> bool:
        return False


class HmacSha256Verifier(SignatureVerifier):
    """HMAC-SHA256 over the signing payload with keys looked up by ``key_id``."""

    ALGORITHM = "hmac-sha256"

    def __init__(self, keys: Dict[str, bytes]) -> None:
        self._keys = dict(keys)

    @classmethod
    def sign(cls, key: bytes, payload: bytes) -> str:
        return hmac.new(key, payload, hashlib.sha256).hexdigest()

    def verify(self, signature: PackSignature, payload: bytes) -> bool:
        key = self._keys.get(signature.key_id)
        if signature.algorithm != self.ALGORITHM or key is None:
            return False
        expected = hmac.new(key, payload, hashlib.sha256).digest()
        return hmac.compare_digest(expected, signature.value_bytes())


@dataclass(frozen=True)
class InstalledPack:
    pack_id: str
    version: str
    path: Path


@dataclass(frozen=True)
class _Pointer:
    active: Optional[str]
    previous: Optional[str]


def packs_root(config: Config) -> Path:
    return config.models_dir / "packs"


class _ProgressReporter:
    def __init__(self, pack_id: str, total: int, callback: Optional[ProgressCallback]) -> None:
        self._pack_id = pack_id
        self._total = max(total, 1)
        self._callback = callback
        self._done = 0
        self._last = -1

    def advance(self, count: int) -> None:
        self._done += count
        percent = min(100, self._done * 100 // self._total)
        if self._callback is None or percent <= self._last:
            return
        self._last = percent
        try:
            self._callback(self._pack_id, percent)
        except Exception:
            logger.exception("Pack progress callback failed")


class LocalPackInstaller:
    """Installs and activates catalog packs only; there is no path for arbitrary model ids."""

    def __init__(
        self,
        root: Path,
        catalog: PackCatalog,
        hardware_probe: HardwareProbe,
        transport: Optional[DownloadTransport] = None,
        verifier: Optional[SignatureVerifier] = None,
        policy: TierPolicy = DEFAULT_POLICY,
    ) -> None:
        self._root = Path(root)
        self._catalog = catalog
        self._probe = hardware_probe
        self._transport = transport
        self._verifier = verifier or FailClosedVerifier()
        self._policy = policy
        self._locks: Dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()

    @classmethod
    def from_config(
        cls,
        config: Config,
        catalog: PackCatalog,
        hardware_probe: HardwareProbe,
        transport: Optional[DownloadTransport] = None,
        verifier: Optional[SignatureVerifier] = None,
        policy: TierPolicy = DEFAULT_POLICY,
    ) -> "LocalPackInstaller":
        return cls(packs_root(config), catalog, hardware_probe, transport, verifier, policy)

    @property
    def root(self) -> Path:
        return self._root

    # --- queries ---------------------------------------------------------------------------

    def active(self, pack_id: str) -> Optional[InstalledPack]:
        """The active install, or the retained previous one if the active files are gone."""
        pointer = self._read_pointer(pack_id)
        for version in (pointer.active, pointer.previous):
            if version and (self._version_dir(pack_id, version) / PACK_METADATA_FILE).is_file():
                if version != pointer.active:
                    logger.warning("Active %s pack missing; using previous %s", pack_id, version)
                return InstalledPack(pack_id, version, self._version_dir(pack_id, version))
        return None

    def previous_version(self, pack_id: str) -> Optional[str]:
        return self._read_pointer(pack_id).previous

    def installed_versions(self, pack_id: str) -> List[str]:
        versions_dir = self._pack_dir(pack_id) / "versions"
        if not versions_dir.is_dir():
            return []
        return sorted(p.name for p in versions_dir.iterdir() if p.is_dir())

    def check_activatable(self, pack: ModelPack) -> None:
        """Raise unless ``pack`` is published, validly signed, and fits this machine."""
        blockers = pack.activation_blockers()
        if blockers:
            raise PackInstallError(blockers[0], pack.pack_id)
        try:
            signed = pack.signature is not None and self._verifier.verify(
                pack.signature, pack.signing_payload()
            )
        except Exception:
            logger.warning("Signature verifier raised for %s; rejecting", pack.pack_id)
            signed = False
        if not signed:
            raise PackInstallError("signature_invalid", pack.pack_id)
        issues = compatibility_issues(pack, self._probe.probe(), self._policy)
        if issues:
            raise PackInstallError("incompatible_pack", ",".join(issues))

    def verify_installed(self, pack: ModelPack) -> None:
        """Re-hash an installed version against the catalog; raise on any difference."""
        version_dir = self._version_dir(pack.pack_id, pack.version)
        for artifact in pack.artifacts:
            _verify_file(_artifact_path(version_dir, artifact), artifact)

    # --- lifecycle -------------------------------------------------------------------------

    def install(
        self,
        pack_id: str,
        version: Optional[str] = None,
        progress: Optional[ProgressCallback] = None,
        cancel: Optional[threading.Event] = None,
    ) -> InstalledPack:
        """Download (resuming), verify, promote, and activate a catalog pack. Blocking."""
        pack = self._require_pack(pack_id, version)
        self.check_activatable(pack)
        with self._lock_for(pack.pack_id):
            if self._is_intact(pack):
                logger.info("Pack %s %s already installed", pack.pack_id, pack.version)
                return self._activate_locked(pack)
            logger.info("Installing pack %s %s", pack.pack_id, pack.version)
            staging = self._prepare_staging(pack)
            reporter = _ProgressReporter(pack.pack_id, pack.size_bytes, progress)
            for artifact in pack.artifacts:
                self._fetch_artifact(artifact, staging, reporter, cancel)
            self._promote(pack, staging)
            installed = self._activate_locked(pack)
            logger.info("Activated pack %s %s", pack.pack_id, pack.version)
            return installed

    def activate(self, pack_id: str, version: str) -> InstalledPack:
        pack = self._require_pack(pack_id, version)
        self.check_activatable(pack)
        with self._lock_for(pack.pack_id):
            self.verify_installed(pack)
            return self._activate_locked(pack)

    def rollback(self, pack_id: str) -> InstalledPack:
        """Re-activate the retained last known-good version after verifying it."""
        previous = self._read_pointer(pack_id).previous
        if not previous:
            raise PackInstallError("no_previous_version", pack_id)
        installed = self.activate(pack_id, previous)
        logger.info("Rolled back pack %s to %s", pack_id, previous)
        return installed

    # --- internals -------------------------------------------------------------------------

    def _require_pack(self, pack_id: str, version: Optional[str]) -> ModelPack:
        pack = self._catalog.get(pack_id, version)
        if pack is None:
            raise PackInstallError("unknown_pack", pack_id)
        return pack

    def _lock_for(self, pack_id: str) -> threading.Lock:
        with self._locks_guard:
            return self._locks.setdefault(pack_id, threading.Lock())

    def _pack_dir(self, pack_id: str) -> Path:
        return self._root / pack_id

    def _version_dir(self, pack_id: str, version: str) -> Path:
        return self._pack_dir(pack_id) / "versions" / version

    def _is_intact(self, pack: ModelPack) -> bool:
        if not (self._version_dir(pack.pack_id, pack.version) / PACK_METADATA_FILE).is_file():
            return False
        try:
            self.verify_installed(pack)
        except PackInstallError as exc:
            logger.warning("Installed %s %s is damaged (%s)", pack.pack_id, pack.version, exc.code)
            return False
        return True

    def _prepare_staging(self, pack: ModelPack) -> Path:
        staging = self._pack_dir(pack.pack_id) / "staging" / pack.version
        marker = staging / _STAGING_MARKER
        digest = pack.payload_digest()
        if staging.exists():
            try:
                recorded = json.loads(marker.read_text(encoding="utf-8")).get("payloadSha256")
            except (OSError, ValueError, AttributeError):
                recorded = None
            if recorded != digest:
                # Partial bytes belong to different file contents; resuming would corrupt them.
                logger.info("Discarding stale staging for %s %s", pack.pack_id, pack.version)
                shutil.rmtree(staging)
        staging.mkdir(parents=True, exist_ok=True)
        if not marker.exists():
            marker.write_text(json.dumps({"payloadSha256": digest}), encoding="utf-8")
        return staging

    def _fetch_artifact(
        self,
        artifact: PackArtifact,
        staging: Path,
        reporter: _ProgressReporter,
        cancel: Optional[threading.Event],
    ) -> None:
        final = _artifact_path(staging, artifact)
        part = final.with_name(final.name + ".part")
        if final.exists():
            try:
                _verify_file(final, artifact)
                reporter.advance(artifact.size_bytes)
                return
            except PackInstallError:
                final.unlink()
        final.parent.mkdir(parents=True, exist_ok=True)
        offset = part.stat().st_size if part.exists() else 0
        if offset > artifact.size_bytes:
            part.unlink()
            offset = 0
        reporter.advance(offset)
        if offset < artifact.size_bytes:
            self._download(artifact, part, offset, reporter, cancel)
        try:
            _verify_file(part, artifact)
        except PackInstallError:
            part.unlink(missing_ok=True)
            raise
        os.replace(part, final)

    def _download(
        self,
        artifact: PackArtifact,
        part: Path,
        offset: int,
        reporter: _ProgressReporter,
        cancel: Optional[threading.Event],
    ) -> None:
        if cancel is not None and cancel.is_set():
            raise InstallCancelledError()
        if self._transport is None:
            raise PackInstallError("transport_unavailable", artifact.path)
        try:
            response = self._transport.open(artifact.url, offset)
        except Exception as exc:
            raise PackInstallError("download_failed", artifact.path) from exc
        try:
            if response.offset not in (0, offset):
                raise PackInstallError("invalid_range_response", artifact.path)
            if response.offset != offset:
                # Server ignored the range request; the bytes already counted start over.
                reporter.advance(-offset)
            position = response.offset
            with open(part, "ab" if position else "wb") as handle:
                try:
                    for chunk in response.chunks:
                        if cancel is not None and cancel.is_set():
                            raise InstallCancelledError()
                        if not chunk:
                            continue
                        if position + len(chunk) > artifact.size_bytes:
                            raise PackInstallError("size_mismatch", artifact.path)
                        handle.write(chunk)
                        position += len(chunk)
                        reporter.advance(len(chunk))
                except PackInstallError:
                    raise
                except Exception as exc:
                    # Bytes already written stay on disk so the next attempt resumes.
                    raise PackInstallError("download_failed", artifact.path) from exc
                finally:
                    handle.flush()
                    os.fsync(handle.fileno())
        except PackInstallError as exc:
            if exc.code in ("size_mismatch", "invalid_range_response"):
                part.unlink(missing_ok=True)
            raise
        finally:
            close = getattr(response.chunks, "close", None)
            if callable(close):
                close()
        if position != artifact.size_bytes:
            raise PackInstallError("incomplete_download", artifact.path)

    def _promote(self, pack: ModelPack, staging: Path) -> None:
        (staging / _STAGING_MARKER).unlink(missing_ok=True)
        _write_json_atomic(staging / PACK_METADATA_FILE, pack.to_metadata())
        target = self._version_dir(pack.pack_id, pack.version)
        target.parent.mkdir(parents=True, exist_ok=True)
        trash: Optional[Path] = None
        if target.exists():
            # A damaged copy of this version; it is never the known-good pack we fall back to.
            trash = self._pack_dir(pack.pack_id) / "trash" / f"{pack.version}-{uuid.uuid4().hex}"
            trash.parent.mkdir(parents=True, exist_ok=True)
            os.replace(target, trash)
        try:
            os.replace(staging, target)
        except OSError as exc:
            if trash is not None:
                os.replace(trash, target)
            raise PackInstallError("promote_failed", pack.pack_id) from exc
        if trash is not None:
            shutil.rmtree(trash, ignore_errors=True)

    def _activate_locked(self, pack: ModelPack) -> InstalledPack:
        pointer = self._read_pointer(pack.pack_id)
        if pointer.active and pointer.active != pack.version:
            previous: Optional[str] = pointer.active
        else:
            previous = pointer.previous
        if previous == pack.version:
            previous = None
        pointer_path = self._pack_dir(pack.pack_id) / _POINTER_FILE
        document = {
            "schema": POINTER_SCHEMA_VERSION,
            "packId": pack.pack_id,
            "active": pack.version,
            "previous": previous,
        }
        try:
            _write_json_atomic(pointer_path, document)
        except OSError as exc:
            raise PackInstallError("activation_failed", pack.pack_id) from exc
        self._prune(pack.pack_id, {pack.version, previous})
        return InstalledPack(
            pack.pack_id, pack.version, self._version_dir(pack.pack_id, pack.version)
        )

    def _prune(self, pack_id: str, keep: set) -> None:
        pack_dir = self._pack_dir(pack_id)
        for version in self.installed_versions(pack_id):
            if version not in keep:
                shutil.rmtree(pack_dir / "versions" / version, ignore_errors=True)
        shutil.rmtree(pack_dir / "trash", ignore_errors=True)

    def _read_pointer(self, pack_id: str) -> _Pointer:
        path = self._pack_dir(pack_id) / _POINTER_FILE
        if not path.is_file():
            return _Pointer(None, None)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            active, previous = data.get("active"), data.get("previous")
        except (OSError, ValueError, AttributeError):
            logger.warning("Unreadable pack pointer for %s; treating as not installed", pack_id)
            return _Pointer(None, None)
        known = set(self.installed_versions(pack_id))
        return _Pointer(
            active if isinstance(active, str) and active in known else None,
            previous if isinstance(previous, str) and previous in known else None,
        )


def _artifact_path(root: Path, artifact: PackArtifact) -> Path:
    return root.joinpath(*artifact.path.split("/"))


def _verify_file(path: Path, artifact: PackArtifact) -> None:
    if path.is_symlink() or not path.is_file():
        raise PackInstallError("artifact_missing", artifact.path)
    if path.stat().st_size != artifact.size_bytes:
        raise PackInstallError("size_mismatch", artifact.path)
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(_HASH_BLOCK_SIZE), b""):
            digest.update(block)
    if not hmac.compare_digest(digest.hexdigest(), artifact.sha256):
        raise PackInstallError("checksum_mismatch", artifact.path)


def _write_json_atomic(path: Path, document: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(document, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
