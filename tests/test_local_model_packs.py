"""Curated local pack service: catalog gates, selection tiers, verified resumable installs.

Every download goes through an in-memory fake transport and every hardware profile is a fake
probe; nothing here touches the network or real hardware detection results.
"""

import hashlib
import inspect
import json
import os
import random
import subprocess
import sys
import threading
from dataclasses import replace
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import pytest

from engine.local_models import (
    CatalogError,
    CommercialUse,
    DownloadResponse,
    DownloadTransport,
    FailClosedVerifier,
    GpuInfo,
    GpuRequirement,
    HardwareProbe,
    HardwareProfile,
    HardwareRequirements,
    HardwareTier,
    HmacSha256Verifier,
    InstallCancelledError,
    LicenseInfo,
    LocalPackInstaller,
    ModelPack,
    PackArtifact,
    PackCatalog,
    PackInstallError,
    PackKind,
    PackSignature,
    PackSource,
    SystemHardwareProbe,
    builtin_catalog,
    classify_tier_chain,
    packs_root,
    select_default_pack,
)
from engine.local_models import installer as installer_module
from engine.local_models.selector import (
    REASON_COMPATIBILITY_FALLBACK,
    REASON_NO_COMPATIBLE_PACK,
    REASON_RECOMMENDED,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
KEY_ID = "test-key"
KEY = b"test-only-signing-key"
VERIFIER = HmacSha256Verifier({KEY_ID: KEY})
REVIEWED_MIT = LicenseInfo("MIT", CommercialUse.ALLOWED, reviewed=True)

CPU_8GB = HardwareProfile(cpu_cores=6, ram_gb=7.8, platform="windows", architecture="x86_64")
CPU_16GB = HardwareProfile(cpu_cores=8, ram_gb=15.7, platform="macos", architecture="arm64")
GPU_BOX = HardwareProfile(
    cpu_cores=8,
    ram_gb=15.9,
    platform="windows",
    architecture="x86_64",
    gpu=GpuInfo("cuda", 8.0),
)
LOW_END = HardwareProfile(cpu_cores=2, ram_gb=4.0, platform="linux", architecture="x86_64")


# --- helpers --------------------------------------------------------------------------------


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _url(pack_id: str, version: str, path: str) -> str:
    return f"https://packs.invalid/{pack_id}/{version}/{path}"


def make_pack(
    pack_id: str = "stt-test",
    version: str = "1.0.0",
    files: Optional[Dict[str, bytes]] = None,
    sign: bool = True,
    key: bytes = KEY,
    **overrides,
) -> ModelPack:
    files = files if files is not None else {"model.bin": b"weights-" + version.encode() * 50}
    artifacts = tuple(
        PackArtifact(
            path=path, url=_url(pack_id, version, path), size_bytes=len(data), sha256=_sha(data)
        )
        for path, data in files.items()
    )
    fields = {
        "kind": PackKind.STT,
        "display_name": pack_id,
        "runtime": "ctranslate2",
        "license": REVIEWED_MIT,
        "multilingual": True,
        "tiers": frozenset(HardwareTier),
        "benchmark_evidence": "bench/test",
    }
    fields.update(overrides)
    pack = ModelPack(pack_id=pack_id, version=version, artifacts=artifacts, **fields)
    if sign:
        value = HmacSha256Verifier.sign(key, pack.signing_payload())
        pack = replace(pack, signature=PackSignature(KEY_ID, HmacSha256Verifier.ALGORITHM, value))
    return pack


class FakeProbe(HardwareProbe):
    def __init__(self, profile: HardwareProfile = CPU_16GB) -> None:
        self.profile = profile

    def probe(self) -> HardwareProfile:
        return self.profile


class FakeTransport(DownloadTransport):
    """Serves bytes from memory in small chunks, with scripted failures."""

    def __init__(self, chunk_size: int = 16) -> None:
        self.blobs: Dict[str, bytes] = {}
        self.calls: List[Tuple[str, int]] = []
        self.chunk_size = chunk_size
        self.fail_after: Optional[int] = None
        self.ignore_range = False
        self.on_chunk = None

    def serve(self, pack: ModelPack, files: Dict[str, bytes]) -> None:
        for artifact in pack.artifacts:
            self.blobs[artifact.url] = files[artifact.path]

    def open(self, url: str, offset: int) -> DownloadResponse:
        self.calls.append((url, offset))
        data = self.blobs[url]
        start = 0 if self.ignore_range else offset
        return DownloadResponse(offset=start, chunks=self._chunks(data, start))

    def _chunks(self, data: bytes, start: int) -> Iterator[bytes]:
        sent = 0
        for index in range(start, len(data), self.chunk_size):
            if self.fail_after is not None and sent >= self.fail_after:
                self.fail_after = None
                raise ConnectionResetError("simulated drop")
            chunk = data[index : index + self.chunk_size]
            sent += len(chunk)
            if self.on_chunk is not None:
                self.on_chunk(sent)
            yield chunk


def make_installer(
    root: Path,
    packs: List[ModelPack],
    transport: Optional[FakeTransport] = None,
    profile: HardwareProfile = CPU_16GB,
    verifier=VERIFIER,
) -> LocalPackInstaller:
    return LocalPackInstaller(
        root,
        PackCatalog(packs),
        FakeProbe(profile),
        transport=transport,
        verifier=verifier,
    )


def _files(version: str) -> Dict[str, bytes]:
    return {
        "model.bin": (f"model-{version}-".encode()) * 40,
        "tokenizer/vocab.json": (f"vocab-{version}".encode()) * 10,
    }


def _pointer(root: Path, pack_id: str) -> dict:
    return json.loads((root / pack_id / "active.json").read_text(encoding="utf-8"))


# --- import hygiene -------------------------------------------------------------------------

_HEAVY_MODULES = [
    "torch",
    "ctranslate2",
    "faster_whisper",
    "transformers",
    "huggingface_hub",
    "tqdm",
    "numpy",
    "requests",
    "httpx",
    "aiohttp",
    "websockets",
    "urllib.request",
    "http.client",
    "engine.core.model_manager",
]

_IMPORT_PROBE = """
import json, sys
import engine.local_models
from engine.local_models import builtin_catalog, select_default_pack, PackKind
from engine.local_models import HardwareProfile
catalog = builtin_catalog()
profile = HardwareProfile(cpu_cores=8, ram_gb=16.0, platform="windows", architecture="x86_64")
select_default_pack(catalog, PackKind.STT, profile)
watched = json.loads(sys.argv[1])
print(json.dumps(sorted(name for name in sys.modules if name in watched)))
"""


def test_importing_pack_service_loads_no_ml_or_network_libraries():
    # A fresh interpreter keeps this independent of what other test modules imported.
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(REPO_ROOT), env.get("PYTHONPATH", "")) if part
    )
    completed = subprocess.run(
        [sys.executable, "-c", _IMPORT_PROBE, json.dumps(_HEAVY_MODULES)],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout.strip().splitlines()[-1]) == []


# --- catalog --------------------------------------------------------------------------------


def test_builtin_catalog_has_compatibility_fallbacks_for_both_services():
    catalog = builtin_catalog()
    for kind in PackKind:
        fallbacks = [p for p in catalog.packs_for(kind) if p.compatibility_fallback]
        assert len(fallbacks) == 1
        assert fallbacks[0].source is PackSource.LEGACY_RUNTIME
        assert fallbacks[0].license.commercial_use is CommercialUse.ALLOWED


def test_builtin_catalog_contains_nothing_activatable_or_recommendable_yet():
    # No checksum or signature has been published, so nothing may install or become a default.
    for pack in builtin_catalog().packs:
        assert pack.activation_blockers()
        assert pack.recommendation_blockers()
        assert not pack.bundled


def test_builtin_candidates_are_marked_unpublished():
    for pack in builtin_catalog().packs:
        if pack.source is PackSource.CATALOG:
            assert pack.activation_blockers() == ["artifacts_unpublished", "signature_missing"]


def test_pack_schema_carries_license_size_checksum_signature_and_requirements():
    pack = make_pack(
        requirements=HardwareRequirements(
            min_cpu_cores=4,
            min_ram_gb=8.0,
            gpu=GpuRequirement(frozenset({"cuda"}), 4.0),
            platforms=frozenset({"windows"}),
            architectures=frozenset({"x86_64"}),
        )
    )
    assert pack.size_bytes == sum(a.size_bytes for a in pack.artifacts)
    assert all(len(a.sha256) == 64 for a in pack.artifacts)
    assert pack.signature is not None and pack.signature.key_id == KEY_ID
    assert pack.license.spdx_id == "MIT"
    assert pack.requirements.gpu.min_vram_gb == 4.0
    assert pack.activation_blockers() == []
    assert pack.recommendation_blockers() == []


@pytest.mark.parametrize(
    ("license_info", "code"),
    [
        (
            LicenseInfo("CC-BY-NC-4.0", CommercialUse.NON_COMMERCIAL, reviewed=True),
            "license_non_commercial",
        ),
        (LicenseInfo("custom", CommercialUse.UNKNOWN), "license_unknown"),
        (LicenseInfo("Apache-2.0", CommercialUse.ALLOWED, reviewed=False), "license_not_reviewed"),
    ],
)
def test_undistributable_license_blocks_recommendation_and_bundling(license_info, code):
    pack = make_pack(license=license_info)
    assert code in pack.recommendation_blockers()
    with pytest.raises(CatalogError) as excinfo:
        make_pack(license=license_info, bundled=True)
    assert excinfo.value.code == "bundled_license_not_distributable"


def test_signature_is_bound_to_checksums():
    pack = make_pack()
    tampered_artifact = replace(pack.artifacts[0], sha256="0" * 64)
    tampered = replace(pack, artifacts=(tampered_artifact,))
    assert VERIFIER.verify(pack.signature, pack.signing_payload())
    assert not VERIFIER.verify(tampered.signature, tampered.signing_payload())


@pytest.mark.parametrize(
    ("kwargs", "code"),
    [
        ({"path": "../escape.bin"}, "unsafe_artifact_path"),
        ({"path": "C:/abs.bin"}, "unsafe_artifact_path"),
        ({"path": "dir\\file.bin"}, "unsafe_artifact_path"),
        ({"url": "http://insecure.invalid/x"}, "artifact_url_not_https"),
        ({"sha256": "abc"}, "invalid_checksum"),
        ({"size_bytes": 0}, "invalid_artifact_size"),
    ],
)
def test_artifact_validation_rejects_unsafe_values(kwargs, code):
    values = {
        "path": "model.bin",
        "url": "https://p.invalid/m",
        "size_bytes": 1,
        "sha256": "a" * 64,
    }
    values.update(kwargs)
    with pytest.raises(CatalogError) as excinfo:
        PackArtifact(**values)
    assert excinfo.value.code == code


def test_catalog_rejects_duplicate_versions():
    with pytest.raises(CatalogError) as excinfo:
        PackCatalog([make_pack(), make_pack()])
    assert excinfo.value.code == "duplicate_pack_version"


# --- selection ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("profile", "chain"),
    [
        (LOW_END, (HardwareTier.MINIMAL,)),
        (CPU_8GB, (HardwareTier.CPU_8GB, HardwareTier.MINIMAL)),
        (CPU_16GB, (HardwareTier.CPU_16GB, HardwareTier.CPU_8GB, HardwareTier.MINIMAL)),
        (
            GPU_BOX,
            (HardwareTier.GPU, HardwareTier.CPU_16GB, HardwareTier.CPU_8GB, HardwareTier.MINIMAL),
        ),
        (
            replace(CPU_8GB, gpu=GpuInfo("cuda", 2.0)),
            (HardwareTier.CPU_8GB, HardwareTier.MINIMAL),
        ),
        (replace(GPU_BOX, cpu_cores=2), (HardwareTier.MINIMAL,)),
    ],
)
def test_tier_chain(profile, chain):
    assert classify_tier_chain(profile) == chain


@pytest.mark.parametrize("profile", [LOW_END, CPU_8GB, CPU_16GB, GPU_BOX])
def test_builtin_catalog_selects_current_model_as_fallback(profile):
    stt = select_default_pack(builtin_catalog(), PackKind.STT, profile)
    assert stt.reason == REASON_COMPATIBILITY_FALLBACK
    assert stt.pack.pack_id == "legacy-faster-whisper-small"
    translation = select_default_pack(builtin_catalog(), PackKind.TRANSLATION, profile, "en", "vi")
    assert translation.pack.pack_id == "legacy-marian-opus-mt"
    assert translation.is_compatibility_fallback


def test_unsupported_language_pair_has_no_pack():
    selection = select_default_pack(builtin_catalog(), PackKind.TRANSLATION, CPU_8GB, "xx", "yy")
    assert selection.pack is None
    assert selection.reason == REASON_NO_COMPATIBLE_PACK


def _tiered_catalog(extra: Optional[List[ModelPack]] = None) -> List[ModelPack]:
    packs = [
        make_pack(
            "stt-cpu8",
            tiers=frozenset({HardwareTier.CPU_8GB}),
            requirements=HardwareRequirements(min_cpu_cores=4, min_ram_gb=8.0),
        ),
        make_pack(
            "stt-cpu16",
            tiers=frozenset({HardwareTier.CPU_16GB}),
            requirements=HardwareRequirements(min_cpu_cores=4, min_ram_gb=16.0),
        ),
        make_pack(
            "stt-gpu",
            tiers=frozenset({HardwareTier.GPU}),
            requirements=HardwareRequirements(
                min_cpu_cores=4, min_ram_gb=8.0, gpu=GpuRequirement(frozenset({"cuda"}), 4.0)
            ),
        ),
        *builtin_catalog().packs,
    ]
    return packs + (extra or [])


@pytest.mark.parametrize(
    ("profile", "expected", "tier"),
    [
        (CPU_8GB, "stt-cpu8", HardwareTier.CPU_8GB),
        (CPU_16GB, "stt-cpu16", HardwareTier.CPU_16GB),
        (GPU_BOX, "stt-gpu", HardwareTier.GPU),
        (LOW_END, "legacy-faster-whisper-small", None),
    ],
)
def test_selection_by_hardware_tier(profile, expected, tier):
    selection = select_default_pack(PackCatalog(_tiered_catalog()), PackKind.STT, profile)
    assert selection.pack.pack_id == expected
    assert selection.tier == tier


def test_gpu_machine_falls_back_to_16gb_tier_without_gpu_pack():
    packs = [p for p in _tiered_catalog() if p.pack_id != "stt-gpu"]
    selection = select_default_pack(PackCatalog(packs), PackKind.STT, GPU_BOX)
    assert selection.pack.pack_id == "stt-cpu16"
    assert selection.reason == REASON_RECOMMENDED


def test_selection_is_deterministic_regardless_of_catalog_order():
    packs = _tiered_catalog(
        [make_pack("stt-cpu8", version="1.2.0", tiers=frozenset({HardwareTier.CPU_8GB}))]
    )
    rng = random.Random(1234)
    results = set()
    for _ in range(20):
        rng.shuffle(packs)
        selection = select_default_pack(PackCatalog(packs), PackKind.STT, CPU_8GB)
        results.add(selection.pack.key)
    assert results == {("stt-cpu8", "1.2.0")}


@pytest.mark.parametrize(
    "license_info",
    [
        LicenseInfo("CC-BY-NC-4.0", CommercialUse.NON_COMMERCIAL, reviewed=True),
        LicenseInfo("unknown", CommercialUse.UNKNOWN, reviewed=True),
        LicenseInfo("Apache-2.0", CommercialUse.ALLOWED, reviewed=False),
    ],
)
def test_undistributable_pack_never_becomes_default_even_if_best(license_info):
    best = make_pack("aaa-best", license=license_info, rank=0)
    selection = select_default_pack(PackCatalog(_tiered_catalog([best])), PackKind.STT, CPU_16GB)
    assert selection.pack.pack_id == "stt-cpu16"


def test_unbenchmarked_or_unsigned_pack_never_becomes_default():
    unbenchmarked = make_pack("aaa-nobench", benchmark_evidence="", rank=0)
    unsigned = make_pack("aaa-unsigned", sign=False, rank=0)
    selection = select_default_pack(
        PackCatalog([unbenchmarked, unsigned, *builtin_catalog().packs]), PackKind.STT, GPU_BOX
    )
    assert selection.is_compatibility_fallback


def test_non_commercial_fallback_is_not_used():
    fallback = ModelPack(
        pack_id="nc-fallback",
        version="1.0.0",
        kind=PackKind.STT,
        display_name="nc",
        runtime="ctranslate2",
        license=LicenseInfo("CC-BY-NC-4.0", CommercialUse.NON_COMMERCIAL),
        source=PackSource.LEGACY_RUNTIME,
        multilingual=True,
        compatibility_fallback=True,
    )
    selection = select_default_pack(PackCatalog([fallback]), PackKind.STT, CPU_16GB)
    assert selection.pack is None


def test_pack_choice_is_not_exposed_as_a_parameter():
    params = set(inspect.signature(select_default_pack).parameters)
    assert params == {"catalog", "kind", "profile", "source_language", "target_language", "policy"}
    from engine.providers.manifests import BUILTIN_MANIFESTS

    local = next(m for m in BUILTIN_MANIFESTS if m.provider_id == "local")
    assert local.model_selectable is False


def test_system_probe_reports_sane_values():
    profile = SystemHardwareProbe().probe()
    assert profile.cpu_cores >= 1
    assert profile.ram_gb >= 0
    assert profile.platform in ("windows", "macos", "linux") or profile.platform
    assert profile.gpu is None


# --- data directories -----------------------------------------------------------------------


def test_packs_root_uses_config_data_dir(make_config, isolated_env):
    config = make_config()
    installer = LocalPackInstaller.from_config(config, PackCatalog([]), FakeProbe())
    assert installer.root == isolated_env / "models" / "packs"
    assert packs_root(config) == config.models_dir / "packs"


@pytest.mark.parametrize(
    ("platform_name", "expected_parts"),
    [
        ("win32", (".echoflux", "models", "packs")),
        ("darwin", ("Library", "Application Support", "EchoFlux", "models", "packs")),
    ],
)
def test_packs_root_follows_platform_data_dirs(
    make_config, monkeypatch, tmp_path, platform_name, expected_parts
):
    home = tmp_path / "home"
    monkeypatch.delenv("ECHOFLUX_DATA_DIR", raising=False)
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.setattr(sys, "platform", platform_name)
    root = packs_root(make_config())
    assert root == home.joinpath(*expected_parts)


# --- install / verification -----------------------------------------------------------------


def test_install_downloads_verifies_and_activates(tmp_path):
    files = _files("1.0.0")
    pack = make_pack(files=files)
    transport = FakeTransport()
    transport.serve(pack, files)
    installer = make_installer(tmp_path, [pack], transport)
    progress: List[int] = []

    installed = installer.install(pack.pack_id, progress=lambda _pid, pct: progress.append(pct))

    assert installed.version == "1.0.0"
    assert (installed.path / "tokenizer" / "vocab.json").read_bytes() == files[
        "tokenizer/vocab.json"
    ]
    metadata = json.loads((installed.path / "pack.json").read_text(encoding="utf-8"))
    assert metadata["payloadSha256"] == pack.payload_digest()
    assert installer.active(pack.pack_id) == installed
    assert progress[-1] == 100 and progress == sorted(progress)
    assert not (tmp_path / pack.pack_id / "staging" / "1.0.0").exists()


def test_reinstall_of_intact_pack_downloads_nothing(tmp_path):
    files = _files("1.0.0")
    pack = make_pack(files=files)
    transport = FakeTransport()
    transport.serve(pack, files)
    installer = make_installer(tmp_path, [pack], transport)
    installer.install(pack.pack_id)
    transport.calls.clear()
    installer.install(pack.pack_id)
    assert transport.calls == []


@pytest.mark.parametrize(
    ("pack_kwargs", "verifier", "code"),
    [
        ({"sign": False}, VERIFIER, "signature_missing"),
        ({"key": b"attacker-key"}, VERIFIER, "signature_invalid"),
        ({}, FailClosedVerifier(), "signature_invalid"),
    ],
)
def test_unsigned_or_untrusted_pack_cannot_install(tmp_path, pack_kwargs, verifier, code):
    files = _files("1.0.0")
    pack = make_pack(files=files, **pack_kwargs)
    transport = FakeTransport()
    transport.serve(pack, files)
    installer = make_installer(tmp_path, [pack], transport, verifier=verifier)
    with pytest.raises(PackInstallError) as excinfo:
        installer.install(pack.pack_id)
    assert excinfo.value.code == code
    assert transport.calls == []
    assert installer.active(pack.pack_id) is None


def test_incompatible_pack_cannot_install(tmp_path):
    files = _files("1.0.0")
    pack = make_pack(
        files=files,
        requirements=HardwareRequirements(gpu=GpuRequirement(frozenset({"cuda"}), 4.0)),
    )
    transport = FakeTransport()
    transport.serve(pack, files)
    installer = make_installer(tmp_path, [pack], transport, profile=CPU_16GB)
    with pytest.raises(PackInstallError) as excinfo:
        installer.install(pack.pack_id)
    assert excinfo.value.code == "incompatible_pack"
    assert "gpu_required" in excinfo.value.detail
    assert transport.calls == []


def test_legacy_and_unknown_packs_cannot_install(tmp_path):
    installer = make_installer(tmp_path, list(builtin_catalog().packs), FakeTransport())
    with pytest.raises(PackInstallError) as legacy:
        installer.install("legacy-faster-whisper-small")
    assert legacy.value.code == "legacy_managed"
    with pytest.raises(PackInstallError) as unpublished:
        installer.install("stt-cpu-standard")
    assert unpublished.value.code == "artifacts_unpublished"
    with pytest.raises(PackInstallError) as unknown:
        installer.install("Systran/faster-whisper-large-v3")
    assert unknown.value.code == "unknown_pack"


def _install_v1(tmp_path, extra: List[ModelPack], transport: FakeTransport):
    v1_files = _files("1.0.0")
    v1 = make_pack(files=v1_files)
    transport.serve(v1, v1_files)
    installer = make_installer(tmp_path, [v1, *extra], transport)
    installer.install(v1.pack_id, "1.0.0")
    return installer, v1


def test_corrupt_download_never_replaces_working_pack(tmp_path):
    transport = FakeTransport()
    v2_files = _files("2.0.0")
    v2 = make_pack(version="2.0.0", files=v2_files)
    installer, v1 = _install_v1(tmp_path, [v2], transport)
    corrupted = dict(v2_files)
    corrupted["model.bin"] = b"X" + v2_files["model.bin"][1:]
    transport.serve(v2, corrupted)

    with pytest.raises(PackInstallError) as excinfo:
        installer.install(v2.pack_id, "2.0.0")

    assert excinfo.value.code == "checksum_mismatch"
    assert installer.active(v1.pack_id).version == "1.0.0"
    installer.verify_installed(v1)
    assert not (tmp_path / v2.pack_id / "staging" / "2.0.0" / "model.bin.part").exists()
    assert "2.0.0" not in installer.installed_versions(v2.pack_id)


def test_oversized_stream_is_rejected(tmp_path):
    files = _files("1.0.0")
    pack = make_pack(files=files)
    transport = FakeTransport()
    transport.serve(pack, {**files, "model.bin": files["model.bin"] + b"extra"})
    installer = make_installer(tmp_path, [pack], transport)
    with pytest.raises(PackInstallError) as excinfo:
        installer.install(pack.pack_id)
    assert excinfo.value.code == "size_mismatch"
    assert installer.active(pack.pack_id) is None


def test_interrupted_download_resumes_from_partial_bytes(tmp_path):
    files = _files("1.0.0")
    pack = make_pack(files=files)
    transport = FakeTransport(chunk_size=32)
    transport.serve(pack, files)
    transport.fail_after = 96
    installer = make_installer(tmp_path, [pack], transport)

    with pytest.raises(PackInstallError) as excinfo:
        installer.install(pack.pack_id)
    assert excinfo.value.code == "download_failed"
    part = tmp_path / pack.pack_id / "staging" / "1.0.0" / "model.bin.part"
    assert part.stat().st_size == 96
    assert installer.active(pack.pack_id) is None

    transport.calls.clear()
    installed = installer.install(pack.pack_id)

    model_url = _url(pack.pack_id, "1.0.0", "model.bin")
    assert (model_url, 96) in transport.calls
    assert (installed.path / "model.bin").read_bytes() == files["model.bin"]


def test_resume_restarts_when_server_ignores_range(tmp_path):
    files = _files("1.0.0")
    pack = make_pack(files=files)
    transport = FakeTransport(chunk_size=32)
    transport.serve(pack, files)
    transport.fail_after = 64
    installer = make_installer(tmp_path, [pack], transport)
    with pytest.raises(PackInstallError):
        installer.install(pack.pack_id)

    transport.ignore_range = True
    installed = installer.install(pack.pack_id)
    assert (installed.path / "model.bin").read_bytes() == files["model.bin"]


def test_stale_partial_from_changed_catalog_entry_is_discarded(tmp_path):
    old_files = _files("1.0.0")
    old = make_pack(files=old_files)
    transport = FakeTransport(chunk_size=32)
    transport.serve(old, old_files)
    transport.fail_after = 64
    with pytest.raises(PackInstallError):
        make_installer(tmp_path, [old], transport).install(old.pack_id)

    new_files = {path: data.upper() for path, data in old_files.items()}
    republished = make_pack(files=new_files)
    transport.serve(republished, new_files)
    transport.calls.clear()
    installed = make_installer(tmp_path, [republished], transport).install(republished.pack_id)

    assert all(offset == 0 for _url_, offset in transport.calls)
    assert (installed.path / "model.bin").read_bytes() == new_files["model.bin"]


def test_cancelled_install_keeps_working_pack_and_can_resume(tmp_path):
    transport = FakeTransport(chunk_size=32)
    v2_files = _files("2.0.0")
    v2 = make_pack(version="2.0.0", files=v2_files)
    installer, v1 = _install_v1(tmp_path, [v2], transport)
    transport.serve(v2, v2_files)
    cancel = threading.Event()
    transport.on_chunk = lambda sent: cancel.set() if sent >= 64 else None

    with pytest.raises(InstallCancelledError):
        installer.install(v2.pack_id, "2.0.0", cancel=cancel)
    assert installer.active(v1.pack_id).version == "1.0.0"

    transport.on_chunk = None
    transport.calls.clear()
    installed = installer.install(v2.pack_id, "2.0.0")
    assert installed.version == "2.0.0"
    assert any(offset > 0 for _url_, offset in transport.calls)


def test_rollback_restores_last_known_good(tmp_path):
    transport = FakeTransport()
    v2_files = _files("2.0.0")
    v2 = make_pack(version="2.0.0", files=v2_files)
    installer, v1 = _install_v1(tmp_path, [v2], transport)
    transport.serve(v2, v2_files)
    installer.install(v2.pack_id, "2.0.0")
    assert _pointer(tmp_path, v2.pack_id) == {
        "schema": 1,
        "packId": v2.pack_id,
        "active": "2.0.0",
        "previous": "1.0.0",
    }

    restored = installer.rollback(v1.pack_id)

    assert restored.version == "1.0.0"
    assert installer.active(v1.pack_id).version == "1.0.0"
    assert installer.previous_version(v1.pack_id) == "2.0.0"


def test_rollback_without_previous_version_fails(tmp_path):
    installer, v1 = _install_v1(tmp_path, [], FakeTransport())
    with pytest.raises(PackInstallError) as excinfo:
        installer.rollback(v1.pack_id)
    assert excinfo.value.code == "no_previous_version"


def test_failed_pointer_swap_is_atomic(tmp_path, monkeypatch):
    transport = FakeTransport()
    v2_files = _files("2.0.0")
    v2 = make_pack(version="2.0.0", files=v2_files)
    installer, v1 = _install_v1(tmp_path, [v2], transport)
    transport.serve(v2, v2_files)
    real_replace = os.replace

    def failing_replace(src, dst):
        if Path(dst).name == "active.json":
            raise OSError("disk full")
        return real_replace(src, dst)

    monkeypatch.setattr(installer_module.os, "replace", failing_replace)
    with pytest.raises(PackInstallError) as excinfo:
        installer.install(v2.pack_id, "2.0.0")
    assert excinfo.value.code == "activation_failed"
    assert _pointer(tmp_path, v1.pack_id)["active"] == "1.0.0"
    assert installer.active(v1.pack_id).version == "1.0.0"
    assert not list((tmp_path / v1.pack_id).glob("*.tmp"))

    monkeypatch.setattr(installer_module.os, "replace", real_replace)
    assert installer.activate(v2.pack_id, "2.0.0").version == "2.0.0"


def test_corrupt_installed_pack_cannot_activate(tmp_path):
    transport = FakeTransport()
    v2_files = _files("2.0.0")
    v2 = make_pack(version="2.0.0", files=v2_files)
    installer, v1 = _install_v1(tmp_path, [v2], transport)
    transport.serve(v2, v2_files)
    installer.install(v2.pack_id, "2.0.0")
    tampered = tmp_path / v1.pack_id / "versions" / "1.0.0" / "model.bin"
    tampered.write_bytes(b"Z" * tampered.stat().st_size)

    with pytest.raises(PackInstallError) as excinfo:
        installer.rollback(v1.pack_id)

    assert excinfo.value.code == "checksum_mismatch"
    assert installer.active(v2.pack_id).version == "2.0.0"


def test_damaged_active_install_is_repaired_by_reinstall(tmp_path):
    files = _files("1.0.0")
    pack = make_pack(files=files)
    transport = FakeTransport()
    transport.serve(pack, files)
    installer = make_installer(tmp_path, [pack], transport)
    installed = installer.install(pack.pack_id)
    (installed.path / "model.bin").write_bytes(b"broken")

    transport.calls.clear()
    repaired = installer.install(pack.pack_id)

    assert transport.calls
    installer.verify_installed(pack)
    assert repaired.version == "1.0.0"


def test_only_active_and_previous_versions_are_retained(tmp_path):
    transport = FakeTransport()
    later = []
    for version in ("2.0.0", "3.0.0"):
        files = _files(version)
        pack = make_pack(version=version, files=files)
        later.append((pack, files))
    installer, v1 = _install_v1(tmp_path, [p for p, _ in later], transport)
    for pack, files in later:
        transport.serve(pack, files)
        installer.install(pack.pack_id, pack.version)
    assert installer.installed_versions(v1.pack_id) == ["2.0.0", "3.0.0"]
    assert installer.previous_version(v1.pack_id) == "2.0.0"


def test_active_falls_back_to_previous_when_active_files_vanish(tmp_path):
    transport = FakeTransport()
    v2_files = _files("2.0.0")
    v2 = make_pack(version="2.0.0", files=v2_files)
    installer, _v1 = _install_v1(tmp_path, [v2], transport)
    transport.serve(v2, v2_files)
    installer.install(v2.pack_id, "2.0.0")
    (tmp_path / v2.pack_id / "versions" / "2.0.0" / "pack.json").unlink()
    assert installer.active(v2.pack_id).version == "1.0.0"


def test_missing_transport_fails_without_touching_active(tmp_path):
    files = _files("1.0.0")
    pack = make_pack(files=files)
    installer = make_installer(tmp_path, [pack], transport=None)
    with pytest.raises(PackInstallError) as excinfo:
        installer.install(pack.pack_id)
    assert excinfo.value.code == "transport_unavailable"
    assert installer.active(pack.pack_id) is None
