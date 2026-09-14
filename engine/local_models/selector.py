"""Deterministic, hardware-aware default pack selection.

The product targets CPU-only machines with 4-8 cores and 8-16 GB of RAM. Tiers:

* ``MINIMAL``   below the target (fewer than 4 cores or under 8 GB).
* ``CPU_8GB``   4+ cores and 8 GB up to 16 GB - the primary target.
* ``CPU_16GB``  4+ cores and 16 GB or more - the explicit 16 GB fallback tier.
* ``GPU``       a supported accelerator on top of a CPU tier. Acceleration is optional.

A machine is evaluated against a fallback chain, highest tier first: a GPU machine tries
``GPU -> CPU_16GB -> CPU_8GB`` (``CPU_16GB`` only when it has the RAM), so a GPU pack that is
unavailable or unrecommendable degrades to the CPU tiers instead of to nothing. If no
recommendable pack fits any tier in the chain, the compatibility fallback - the model the
current runtime already uses - is returned until benchmark evidence selects a new default.

The same catalog and hardware profile always produce the same result, and no function here
accepts a model or pack id from the caller: pack choice stays internal and hidden (D-032).
"""

import os
import platform
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple

from engine.core.logging import get_logger
from engine.local_models.catalog import (
    CommercialUse,
    HardwareTier,
    ModelPack,
    PackCatalog,
    PackKind,
)

logger = get_logger("local_models.selector")

REASON_RECOMMENDED = "recommended"
REASON_COMPATIBILITY_FALLBACK = "compatibility_fallback"
REASON_NO_COMPATIBLE_PACK = "no_compatible_pack"


@dataclass(frozen=True)
class GpuInfo:
    backend: str
    vram_gb: float


@dataclass(frozen=True)
class HardwareProfile:
    """What selection needs to know about the machine. Platform/arch use normalized names."""

    cpu_cores: int
    ram_gb: float
    platform: str
    architecture: str
    gpu: Optional[GpuInfo] = None


@dataclass(frozen=True)
class TierPolicy:
    """Tier boundaries. Product definitions, not user settings: pack choice is not exposed."""

    min_cpu_cores: int = 4
    cpu_8gb_ram_gb: float = 8.0
    cpu_16gb_ram_gb: float = 16.0
    gpu_min_vram_gb: float = 4.0
    gpu_backends: Tuple[str, ...] = ("cuda",)
    # Operating systems report slightly less than the installed RAM (a 16 GB laptop shows
    # ~15.7 GB), so nominal requirements are compared against ``ram_gb / ram_tolerance``.
    ram_tolerance: float = 0.9

    def effective_ram_gb(self, profile: HardwareProfile) -> float:
        return profile.ram_gb / self.ram_tolerance


DEFAULT_POLICY = TierPolicy()


@dataclass(frozen=True)
class PackSelection:
    kind: PackKind
    pack: Optional[ModelPack]
    reason: str
    tier: Optional[HardwareTier]
    tier_chain: Tuple[HardwareTier, ...]

    @property
    def is_compatibility_fallback(self) -> bool:
        return self.reason == REASON_COMPATIBILITY_FALLBACK


def classify_tier_chain(
    profile: HardwareProfile, policy: TierPolicy = DEFAULT_POLICY
) -> Tuple[HardwareTier, ...]:
    """Tiers this machine qualifies for, preferred first, always ending in ``MINIMAL``."""
    ram = policy.effective_ram_gb(profile)
    chain: List[HardwareTier] = []
    cpu_ok = profile.cpu_cores >= policy.min_cpu_cores
    if cpu_ok and ram >= policy.cpu_8gb_ram_gb:
        gpu = profile.gpu
        if (
            gpu is not None
            and gpu.backend in policy.gpu_backends
            and gpu.vram_gb >= policy.gpu_min_vram_gb
        ):
            chain.append(HardwareTier.GPU)
        if ram >= policy.cpu_16gb_ram_gb:
            chain.append(HardwareTier.CPU_16GB)
        chain.append(HardwareTier.CPU_8GB)
    chain.append(HardwareTier.MINIMAL)
    return tuple(chain)


def compatibility_issues(
    pack: ModelPack, profile: HardwareProfile, policy: TierPolicy = DEFAULT_POLICY
) -> List[str]:
    """Codes for every requirement of ``pack`` this machine does not meet."""
    req = pack.requirements
    issues = []
    if req.platforms and profile.platform not in req.platforms:
        issues.append("platform_unsupported")
    if req.architectures and profile.architecture not in req.architectures:
        issues.append("architecture_unsupported")
    if profile.cpu_cores < req.min_cpu_cores:
        issues.append("insufficient_cpu_cores")
    if policy.effective_ram_gb(profile) < req.min_ram_gb:
        issues.append("insufficient_ram")
    if req.gpu is not None:
        gpu = profile.gpu
        if gpu is None or gpu.backend not in req.gpu.backends:
            issues.append("gpu_required")
        elif gpu.vram_gb < req.gpu.min_vram_gb:
            issues.append("insufficient_vram")
    return issues


def _preference_key(pack: ModelPack) -> Tuple[int, str, Tuple[int, int, int]]:
    major, minor, patch = pack.version_tuple
    # Newest version wins within a pack; rank, then id, break ties across packs.
    return (pack.rank, pack.pack_id, (-major, -minor, -patch))


def select_default_pack(
    catalog: PackCatalog,
    kind: PackKind,
    profile: HardwareProfile,
    source_language: Optional[str] = None,
    target_language: Optional[str] = None,
    policy: TierPolicy = DEFAULT_POLICY,
) -> PackSelection:
    """Choose the default pack for ``kind`` on this machine. Pure and deterministic."""
    chain = classify_tier_chain(profile, policy)
    candidates = sorted(
        (
            pack
            for pack in catalog.packs_for(kind)
            if pack.supports_language(source_language, target_language)
            and not compatibility_issues(pack, profile, policy)
        ),
        key=_preference_key,
    )

    recommendable = [pack for pack in candidates if not pack.recommendation_blockers()]
    for tier in chain:
        for pack in recommendable:
            if tier in pack.tiers:
                return PackSelection(kind, pack, REASON_RECOMMENDED, tier, chain)

    for pack in candidates:
        # A fallback is what already runs today, but a license that forbids commercial use or
        # is unknown still disqualifies it.
        if pack.compatibility_fallback and pack.license.commercial_use is CommercialUse.ALLOWED:
            logger.debug("No recommendable %s pack for tiers %s; using fallback", kind.value, chain)
            return PackSelection(kind, pack, REASON_COMPATIBILITY_FALLBACK, None, chain)

    return PackSelection(kind, None, REASON_NO_COMPATIBLE_PACK, None, chain)


class HardwareProbe(ABC):
    """Source of the machine's hardware profile. Tests inject fakes."""

    @abstractmethod
    def probe(self) -> HardwareProfile: ...


def normalize_platform(value: str) -> str:
    if value.startswith("win"):
        return "windows"
    if value == "darwin":
        return "macos"
    if value.startswith("linux"):
        return "linux"
    return value


def normalize_architecture(value: str) -> str:
    lowered = value.lower()
    if lowered in ("amd64", "x86_64", "x64"):
        return "x86_64"
    if lowered in ("arm64", "aarch64"):
        return "arm64"
    return lowered


class SystemHardwareProbe(HardwareProbe):
    """Stdlib-only probe for CPU, RAM, platform, and architecture.

    It reports no GPU: detecting one reliably needs an ML runtime, which must not be imported
    here. The runtime integration supplies ``gpu`` after its own lazy device check.
    """

    def __init__(self, gpu: Optional[GpuInfo] = None) -> None:
        self._gpu = gpu

    def probe(self) -> HardwareProfile:
        return HardwareProfile(
            cpu_cores=os.cpu_count() or 1,
            ram_gb=_total_ram_bytes() / (1024**3),
            platform=normalize_platform(sys.platform),
            architecture=normalize_architecture(platform.machine()),
            gpu=self._gpu,
        )


def _total_ram_bytes() -> int:
    try:
        if sys.platform == "win32":
            import ctypes

            class _MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = _MemoryStatus()
            status.dwLength = ctypes.sizeof(_MemoryStatus)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.ullTotalPhys)
            return 0
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, OSError, ValueError):
        # Unknown RAM selects the MINIMAL tier, which only ever offers the fallback.
        logger.warning("Could not read total RAM; assuming minimal hardware")
        return 0
