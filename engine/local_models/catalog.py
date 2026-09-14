"""Curated local model pack catalog.

A pack is an immutable, versioned bundle of runtime files for one local service (STT or
Translation). Users never pick packs: the selector chooses one from this catalog from the
hardware profile, and the ``local`` provider manifest rejects model overrides (D-032). There is
deliberately no free-form model id or Hub search anywhere in this package.

Two gates are enforced here, and they are different on purpose:

* **Activation** needs a published, integrity-checkable pack: every artifact has an HTTPS URL,
  a size and a sha256, and the pack carries a signature over those values. Anything else can
  never be installed or activated.
* **Recommendation** (becoming a default, or being bundled) additionally needs a license that
  was reviewed and permits commercial distribution, plus benchmark evidence. Non-commercial and
  unknown-license packs therefore never become defaults, however well they benchmark.

Until a pack clears both gates, the models the legacy runtime already uses remain the
compatibility fallback. They are described here as ``LEGACY_RUNTIME`` packs, which the legacy
model manager still installs; this package never downloads them.
"""

import hashlib
import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Tuple

CATALOG_SCHEMA_VERSION = 1
# Written into every installed pack directory; no artifact may use this name.
PACK_METADATA_FILE = "pack.json"

_PACK_ID_RE = re.compile(r"[a-z0-9][a-z0-9-]{1,63}")
_VERSION_RE = re.compile(r"(0|[1-9]\d{0,5})\.(0|[1-9]\d{0,5})\.(0|[1-9]\d{0,5})")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_KEY_ID_RE = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}")
_SIGNATURE_RE = re.compile(r"(?:[0-9a-f]{2}){16,512}")
_ARTIFACT_SEGMENT_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9._-]{0,127}")


class CatalogError(ValueError):
    """Invalid catalog data. Carries a location and a fixed code."""

    def __init__(self, path: str, code: str) -> None:
        super().__init__(f"{path}: {code}")
        self.path = path
        self.code = code


class PackKind(str, Enum):
    STT = "stt"
    TRANSLATION = "translation"


class PackSource(str, Enum):
    # Downloaded, verified, and activated by engine.local_models.installer.
    CATALOG = "catalog"
    # Already provisioned by the legacy model manager; described for fallback selection only.
    LEGACY_RUNTIME = "legacy_runtime"


class HardwareTier(str, Enum):
    """Hardware class a pack is benchmarked for. See ``engine.local_models.selector``."""

    MINIMAL = "minimal"
    CPU_8GB = "cpu_8gb"
    CPU_16GB = "cpu_16gb"
    GPU = "gpu"


class CommercialUse(str, Enum):
    ALLOWED = "allowed"
    NON_COMMERCIAL = "non_commercial"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class LicenseInfo:
    """License metadata. ``reviewed`` is set only after an explicit distribution review."""

    spdx_id: str
    commercial_use: CommercialUse = CommercialUse.UNKNOWN
    reviewed: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.spdx_id, str) or not self.spdx_id.strip():
            raise CatalogError("license.spdx_id", "invalid_license")
        if not isinstance(self.commercial_use, CommercialUse):
            raise CatalogError("license.commercial_use", "invalid_license")

    @property
    def distributable(self) -> bool:
        """True only for a reviewed license that permits commercial distribution."""
        return self.commercial_use is CommercialUse.ALLOWED and self.reviewed

    def blockers(self) -> List[str]:
        if self.commercial_use is CommercialUse.NON_COMMERCIAL:
            return ["license_non_commercial"]
        if self.commercial_use is CommercialUse.UNKNOWN:
            return ["license_unknown"]
        if not self.reviewed:
            return ["license_not_reviewed"]
        return []


@dataclass(frozen=True)
class PackArtifact:
    """One file inside a pack. ``path`` is relative to the pack root, using ``/``."""

    path: str
    url: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        if not is_safe_relative_path(self.path):
            raise CatalogError("artifact.path", "unsafe_artifact_path")
        if not isinstance(self.url, str) or not self.url.startswith("https://"):
            raise CatalogError("artifact.url", "artifact_url_not_https")
        if isinstance(self.size_bytes, bool) or not isinstance(self.size_bytes, int):
            raise CatalogError("artifact.size_bytes", "invalid_artifact_size")
        if self.size_bytes <= 0:
            raise CatalogError("artifact.size_bytes", "invalid_artifact_size")
        if not isinstance(self.sha256, str) or not _SHA256_RE.fullmatch(self.sha256):
            raise CatalogError("artifact.sha256", "invalid_checksum")


@dataclass(frozen=True)
class PackSignature:
    """Detached signature over ``ModelPack.signing_payload()``, hex encoded."""

    key_id: str
    algorithm: str
    value: str

    def __post_init__(self) -> None:
        if not isinstance(self.key_id, str) or not _KEY_ID_RE.fullmatch(self.key_id):
            raise CatalogError("signature.key_id", "invalid_signature")
        if not isinstance(self.algorithm, str) or not _KEY_ID_RE.fullmatch(self.algorithm):
            raise CatalogError("signature.algorithm", "invalid_signature")
        if not isinstance(self.value, str) or not _SIGNATURE_RE.fullmatch(self.value):
            raise CatalogError("signature.value", "invalid_signature")

    def value_bytes(self) -> bytes:
        return bytes.fromhex(self.value)


@dataclass(frozen=True)
class GpuRequirement:
    backends: FrozenSet[str] = frozenset({"cuda"})
    min_vram_gb: float = 0.0


@dataclass(frozen=True)
class HardwareRequirements:
    """Minimum resources. Empty ``platforms``/``architectures`` mean any."""

    min_cpu_cores: int = 1
    min_ram_gb: float = 0.0
    gpu: Optional[GpuRequirement] = None
    platforms: FrozenSet[str] = frozenset()
    architectures: FrozenSet[str] = frozenset()


@dataclass(frozen=True)
class ModelPack:
    """Curated pack metadata. Identity is ``(pack_id, version)``."""

    pack_id: str
    version: str
    kind: PackKind
    display_name: str
    runtime: str
    license: LicenseInfo
    source: PackSource = PackSource.CATALOG
    requirements: HardwareRequirements = field(default_factory=HardwareRequirements)
    tiers: FrozenSet[HardwareTier] = frozenset()
    multilingual: bool = False
    languages: FrozenSet[str] = frozenset()
    language_pairs: FrozenSet[Tuple[str, str]] = frozenset()
    artifacts: Tuple[PackArtifact, ...] = ()
    signature: Optional[PackSignature] = None
    # Reference to benchmark results that justify recommending this pack; empty means none.
    benchmark_evidence: str = ""
    # Lower is preferred when several packs qualify for the same tier.
    rank: int = 100
    compatibility_fallback: bool = False
    bundled: bool = False
    # Model id the legacy runtime already understands; LEGACY_RUNTIME packs only.
    legacy_model_id: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.pack_id, str) or not _PACK_ID_RE.fullmatch(self.pack_id):
            raise CatalogError("pack.pack_id", "invalid_pack_id")
        path = f"packs.{self.pack_id}"
        if not isinstance(self.version, str) or not _VERSION_RE.fullmatch(self.version):
            raise CatalogError(f"{path}.version", "invalid_version")
        if not isinstance(self.kind, PackKind):
            raise CatalogError(f"{path}.kind", "invalid_kind")
        if not isinstance(self.source, PackSource):
            raise CatalogError(f"{path}.source", "invalid_source")
        if not isinstance(self.license, LicenseInfo):
            raise CatalogError(f"{path}.license", "invalid_license")
        seen = set()
        for artifact in self.artifacts:
            if not isinstance(artifact, PackArtifact):
                raise CatalogError(f"{path}.artifacts", "invalid_artifact")
            # Windows filesystems are case-insensitive, so compare folded paths.
            folded = artifact.path.lower()
            if folded in seen or folded == PACK_METADATA_FILE:
                raise CatalogError(f"{path}.artifacts", "duplicate_artifact_path")
            seen.add(folded)
        if self.source is PackSource.LEGACY_RUNTIME and (self.artifacts or self.signature):
            raise CatalogError(f"{path}.source", "legacy_pack_has_artifacts")
        if self.kind is PackKind.STT and self.language_pairs:
            raise CatalogError(f"{path}.language_pairs", "invalid_languages")
        if self.kind is PackKind.TRANSLATION and (self.languages or self.multilingual):
            raise CatalogError(f"{path}.languages", "invalid_languages")
        # Bundling is distribution: it is refused outright rather than silently ignored.
        if self.bundled and not self.license.distributable:
            raise CatalogError(f"{path}.bundled", "bundled_license_not_distributable")

    @property
    def key(self) -> Tuple[str, str]:
        return (self.pack_id, self.version)

    @property
    def version_tuple(self) -> Tuple[int, int, int]:
        major, minor, patch = (int(part) for part in self.version.split("."))
        return (major, minor, patch)

    @property
    def size_bytes(self) -> int:
        return sum(artifact.size_bytes for artifact in self.artifacts)

    def signing_payload(self) -> bytes:
        """Canonical bytes a pack signature covers: identity, runtime, and every checksum."""
        payload = {
            "schema": CATALOG_SCHEMA_VERSION,
            "packId": self.pack_id,
            "version": self.version,
            "kind": self.kind.value,
            "runtime": self.runtime,
            "artifacts": [
                {"path": a.path, "sizeBytes": a.size_bytes, "sha256": a.sha256}
                for a in sorted(self.artifacts, key=lambda a: a.path)
            ],
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")

    def payload_digest(self) -> str:
        return hashlib.sha256(self.signing_payload()).hexdigest()

    def activation_blockers(self) -> List[str]:
        """Codes that stop this pack from being installed or activated by the pack service."""
        if self.source is PackSource.LEGACY_RUNTIME:
            return ["legacy_managed"]
        blockers = []
        if not self.artifacts:
            blockers.append("artifacts_unpublished")
        if self.signature is None:
            blockers.append("signature_missing")
        return blockers

    def recommendation_blockers(self) -> List[str]:
        """Codes that stop this pack from becoming a recommended or bundled default."""
        blockers = self.activation_blockers() + self.license.blockers()
        if not self.benchmark_evidence:
            blockers.append("no_benchmark_evidence")
        return blockers

    def supports_language(
        self, source_language: Optional[str] = None, target_language: Optional[str] = None
    ) -> bool:
        source = None if source_language in (None, "", "auto") else source_language
        target = None if target_language in (None, "", "auto") else target_language
        if self.kind is PackKind.STT:
            return self.multilingual or (source is not None and source in self.languages)
        return any(
            (source is None or pair[0] == source) and (target is None or pair[1] == target)
            for pair in self.language_pairs
        )

    def to_metadata(self) -> Dict[str, Any]:
        """Record written next to installed files so a pack directory is self-describing."""
        return {
            "schema": CATALOG_SCHEMA_VERSION,
            "packId": self.pack_id,
            "version": self.version,
            "kind": self.kind.value,
            "runtime": self.runtime,
            "license": self.license.spdx_id,
            "payloadSha256": self.payload_digest(),
            "signatureKeyId": self.signature.key_id if self.signature else None,
            "artifacts": [
                {"path": a.path, "sizeBytes": a.size_bytes, "sha256": a.sha256}
                for a in self.artifacts
            ],
        }


def is_safe_relative_path(path: Any) -> bool:
    """Relative ``/``-separated path that cannot escape its root on Windows or POSIX."""
    if not isinstance(path, str) or not path or len(path) > 512:
        return False
    return all(
        _ARTIFACT_SEGMENT_RE.fullmatch(segment) and segment not in (".", "..")
        for segment in path.split("/")
    )


class PackCatalog:
    """Immutable, validated collection of packs with deterministic ordering."""

    def __init__(self, packs: Iterable[ModelPack], catalog_version: str = "1") -> None:
        packs = list(packs)
        if any(not isinstance(pack, ModelPack) for pack in packs):
            raise CatalogError("packs", "invalid_pack")
        ordered = sorted(packs, key=lambda p: (p.kind.value, p.pack_id, p.version_tuple))
        index: Dict[Tuple[str, str], ModelPack] = {}
        kinds: Dict[str, PackKind] = {}
        for pack in ordered:
            if pack.key in index:
                raise CatalogError(f"packs.{pack.pack_id}", "duplicate_pack_version")
            if kinds.setdefault(pack.pack_id, pack.kind) is not pack.kind:
                raise CatalogError(f"packs.{pack.pack_id}", "pack_kind_conflict")
            index[pack.key] = pack
        self._packs: Tuple[ModelPack, ...] = tuple(ordered)
        self._index = index
        self.catalog_version = catalog_version

    @property
    def packs(self) -> Tuple[ModelPack, ...]:
        return self._packs

    def get(self, pack_id: str, version: Optional[str] = None) -> Optional[ModelPack]:
        """A specific version, or the newest version of ``pack_id`` when none is given."""
        if version is not None:
            return self._index.get((pack_id, version))
        versions = [p for p in self._packs if p.pack_id == pack_id]
        return max(versions, key=lambda p: p.version_tuple) if versions else None

    def packs_for(self, kind: PackKind) -> Tuple[ModelPack, ...]:
        return tuple(p for p in self._packs if p.kind is kind)


# Pairs the legacy MarianBackend has presets for (engine/translation/marian_backend.py).
# Copied rather than imported: that module's package pulls in runtime dependencies.
_LEGACY_MARIAN_PAIRS = frozenset(
    {
        ("en", "vi"),
        ("en", "zh"),
        ("en", "ja"),
        ("en", "ko"),
        ("en", "de"),
        ("en", "fr"),
        ("en", "es"),
        ("vi", "en"),
    }
)

_ANY_TIER = frozenset(HardwareTier)

# Candidate entries exist so selection tiers are defined ahead of benchmarks. Their artifacts,
# checksums, and signatures are NOT published, so none of them can activate or be recommended;
# nothing below is a verified download location or checksum.
BUILTIN_PACKS: Tuple[ModelPack, ...] = (
    ModelPack(
        pack_id="legacy-faster-whisper-small",
        version="1.0.0",
        kind=PackKind.STT,
        display_name="Whisper small (current runtime)",
        runtime="ctranslate2",
        license=LicenseInfo("MIT", CommercialUse.ALLOWED),
        source=PackSource.LEGACY_RUNTIME,
        requirements=HardwareRequirements(min_cpu_cores=2, min_ram_gb=4.0),
        tiers=_ANY_TIER,
        multilingual=True,
        compatibility_fallback=True,
        legacy_model_id="small",
    ),
    ModelPack(
        pack_id="legacy-marian-opus-mt",
        version="1.0.0",
        kind=PackKind.TRANSLATION,
        display_name="OPUS-MT Marian (current runtime)",
        runtime="ctranslate2",
        # Per-pair upstream model cards; each preset pair permits commercial use.
        license=LicenseInfo("Apache-2.0 OR CC-BY-4.0", CommercialUse.ALLOWED),
        source=PackSource.LEGACY_RUNTIME,
        requirements=HardwareRequirements(min_cpu_cores=2, min_ram_gb=4.0),
        tiers=_ANY_TIER,
        language_pairs=_LEGACY_MARIAN_PAIRS,
        compatibility_fallback=True,
        legacy_model_id="Helsinki-NLP/opus-mt",
    ),
    ModelPack(
        pack_id="stt-cpu-standard",
        version="0.1.0",
        kind=PackKind.STT,
        display_name="Speech recognition (standard CPU)",
        runtime="ctranslate2",
        license=LicenseInfo("MIT", CommercialUse.ALLOWED),
        requirements=HardwareRequirements(min_cpu_cores=4, min_ram_gb=8.0),
        tiers=frozenset({HardwareTier.CPU_8GB}),
        multilingual=True,
        rank=10,
    ),
    ModelPack(
        pack_id="stt-cpu-16gb",
        version="0.1.0",
        kind=PackKind.STT,
        display_name="Speech recognition (16 GB CPU)",
        runtime="ctranslate2",
        license=LicenseInfo("MIT", CommercialUse.ALLOWED),
        requirements=HardwareRequirements(min_cpu_cores=4, min_ram_gb=16.0),
        tiers=frozenset({HardwareTier.CPU_16GB}),
        multilingual=True,
        rank=10,
    ),
    ModelPack(
        pack_id="stt-gpu",
        version="0.1.0",
        kind=PackKind.STT,
        display_name="Speech recognition (GPU accelerated)",
        runtime="ctranslate2",
        license=LicenseInfo("MIT", CommercialUse.ALLOWED),
        requirements=HardwareRequirements(
            min_cpu_cores=4, min_ram_gb=8.0, gpu=GpuRequirement(frozenset({"cuda"}), 4.0)
        ),
        tiers=frozenset({HardwareTier.GPU}),
        multilingual=True,
        rank=10,
    ),
)


def builtin_catalog() -> PackCatalog:
    return PackCatalog(BUILTIN_PACKS, catalog_version="2026.09.0")
