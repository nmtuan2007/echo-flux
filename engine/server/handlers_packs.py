"""Control dispatcher handlers for Local Model Packs, Provider Manifests, and Audio Probe."""

import asyncio
import contextlib
import threading
from typing import Any, Dict

from engine.local_models import (
    HardwareProfile,
    InstallCancelledError,
    PackKind,
    PackSource,
    compatibility_issues,
    select_default_pack,
)
from engine.local_models.installer import PackInstallError
from engine.providers.registry import build_default_registry
from engine.providers.types import ProviderCapability
from engine.server.control_helpers import (
    DispatcherError,
    delete_local_model_files,
)
from engine.server.control_helpers import (
    is_local_model_ready as _is_local_model_ready,
)
from engine.server.control_helpers import (
    is_local_translation_pair_supported as _is_local_translation_pair_supported,
)
from engine.server.control_helpers import (
    probe_audio_stream as _probe_audio_stream,
)
from engine.server.control_helpers import (
    probe_hardware_with_gpu as _probe_hardware_with_gpu,
)
from engine.server.control_helpers import (
    serialize_manifest as _serialize_manifest,
)


async def handle_pack_status(dispatcher: Any, payload: Dict[str, Any]) -> Dict[str, Any]:
    pack_id = payload.get("pack_id")
    catalog = dispatcher._get_catalog()
    installer = dispatcher._get_installer()
    probe = dispatcher._get_probe()
    profile: HardwareProfile = await asyncio.to_thread(_probe_hardware_with_gpu, probe)

    hw_info = {
        "cpu_cores": profile.cpu_cores,
        "ram_gb": round(profile.ram_gb, 2),
        "platform": profile.platform,
        "architecture": profile.architecture,
        "has_gpu": profile.gpu is not None,
    }

    if pack_id:
        pack = catalog.get(pack_id)
        if pack is None:
            progress = dispatcher._pack_progress.get(pack_id)
            if progress is not None:
                return {
                    "pack_id": pack_id,
                    "hardware": hw_info,
                    "download_progress": progress,
                    "is_active": False,
                    "is_ready": False,
                    "installed": False,
                }
            raise DispatcherError("not_found", f"Pack '{pack_id}' not found in catalog")
        packs_to_check = [pack]
    else:
        packs_to_check = list(catalog.packs)

    models_dir = dispatcher._get_config().models_dir
    status_list = []
    for p in packs_to_check:
        active = await asyncio.to_thread(installer.active, p.pack_id)
        installed = await asyncio.to_thread(installer.installed_versions, p.pack_id)
        issues = compatibility_issues(p, profile)
        local_ready = _is_local_model_ready(p.kind.value, models_dir=models_dir)
        is_ready = (active is not None and len(issues) == 0) or local_ready
        is_installed = bool(installed) or (active is not None) or local_ready
        status_list.append(
            {
                "pack_id": p.pack_id,
                "version": p.version,
                "kind": p.kind.value,
                "display_name": p.display_name,
                "size_bytes": p.size_bytes,
                "tiers": [t.value for t in p.tiers],
                "active_version": active.version if active else ("1.0.0" if local_ready else None),
                "installed_versions": installed
                if installed
                else (["1.0.0"] if local_ready else []),
                "is_active": active is not None or local_ready,
                "is_ready": is_ready,
                "installed": is_installed,
                "compatibility_issues": [] if local_ready else issues,
                "is_compatible": len(issues) == 0 or local_ready,
            }
        )

    if pack_id and status_list:
        progress = dispatcher._pack_progress.get(pack_id)
        return {
            "pack": status_list[0],
            "hardware": hw_info,
            **({"download_progress": progress} if progress is not None else {}),
            **status_list[0],
        }
    active_downloads = {
        k: v for k, v in dispatcher._pack_progress.items() if v.get("active")
    }
    return {
        "packs": status_list,
        "hardware": hw_info,
        **({"active_downloads": active_downloads} if active_downloads else {}),
    }


async def handle_pack_install(dispatcher: Any, payload: Dict[str, Any]) -> Dict[str, Any]:
    pack_id = payload.get("pack_id")
    if not isinstance(pack_id, str) or not pack_id:
        raise DispatcherError("invalid_payload", "pack_id is required")
    version = payload.get("version")
    raw_src = payload.get("source_lang")
    src_lang = (
        raw_src if isinstance(raw_src, str) and raw_src.strip() and raw_src.lower() != "auto" else "en"
    )
    raw_tgt = payload.get("target_lang")
    tgt_lang = (
        raw_tgt if isinstance(raw_tgt, str) and raw_tgt.strip() and raw_tgt.lower() != "auto" else "vi"
    )

    lowered = pack_id.lower()
    models_dir = dispatcher._get_config().models_dir
    if dispatcher._pack_install_active is not None:
        raise DispatcherError(
            "install_in_progress",
            f"Another model installation ({dispatcher._pack_install_active}) is already running.",
        )
    dispatcher._pack_install_active = pack_id
    cancel_event = threading.Event()
    dispatcher._pack_cancels[pack_id] = cancel_event
    dispatcher._pack_progress[pack_id] = {
        "active": True,
        "percent": 0,
        "model": pack_id,
        "stage": "downloading",
    }

    from engine.core.progress import get_progress_stage, set_progress_callback

    def _on_progress(model_name: str, percent: int) -> None:
        if cancel_event.is_set():
            raise InstallCancelledError("cancelled")
        dispatcher._pack_progress[pack_id] = {
            "active": True,
            "percent": max(0, min(100, int(percent))),
            "model": model_name or pack_id,
            "stage": get_progress_stage(),
        }

    try:
        if "whisper" in lowered or lowered in ("base", "tiny", "small", "medium", "large"):
            size = "base"
            if "tiny" in lowered:
                size = "tiny"
            elif "small" in lowered:
                size = "small"
            elif "medium" in lowered:
                size = "medium"
            elif "large" in lowered:
                size = "large-v3"

            from faster_whisper.utils import download_model

            set_progress_callback(_on_progress)
            try:
                out_path = await asyncio.to_thread(download_model, size, cache_dir=str(models_dir))
                if cancel_event.is_set():
                    raise InstallCancelledError("cancelled")
                return {
                    "installed": True,
                    "pack_id": pack_id,
                    "version": "1.0.0",
                    "path": str(out_path),
                    "active": True,
                }
            except InstallCancelledError:
                await asyncio.to_thread(
                    delete_local_model_files, "stt", pack_id=pack_id, models_dir=models_dir
                )
                raise DispatcherError("cancelled", f"Install of pack '{pack_id}' cancelled") from None
            finally:
                set_progress_callback(None)
                dispatcher._pack_cancels.pop(pack_id, None)
                dispatcher._pack_progress.pop(pack_id, None)

        if "marian" in lowered or "opus" in lowered:
            from engine.translation.marian_backend import MarianBackend

            backend = MarianBackend()
            set_progress_callback(_on_progress)
            try:
                await asyncio.to_thread(
                    backend.load_model,
                    {"translation.source_lang": src_lang, "translation.target_lang": tgt_lang},
                )
                if cancel_event.is_set():
                    raise InstallCancelledError("cancelled")
                out_path = str(backend._ct2_model_path or models_dir)
                return {
                    "installed": True,
                    "pack_id": pack_id,
                    "version": "1.0.0",
                    "path": out_path,
                    "active": True,
                }
            except InstallCancelledError:
                await asyncio.to_thread(
                    delete_local_model_files,
                    "translation",
                    source_lang=src_lang,
                    target_lang=tgt_lang,
                    pack_id=pack_id,
                    models_dir=models_dir,
                )
                raise DispatcherError("cancelled", f"Install of pack '{pack_id}' cancelled") from None
            finally:
                set_progress_callback(None)
                with contextlib.suppress(Exception):
                    backend.unload_model()
                dispatcher._pack_cancels.pop(pack_id, None)
                dispatcher._pack_progress.pop(pack_id, None)

        installer = dispatcher._get_installer()
        try:
            installed = await asyncio.to_thread(
                installer.install,
                pack_id=pack_id,
                version=version,
                cancel=cancel_event,
            )
            return {
                "installed": True,
                "pack_id": installed.pack_id,
                "version": installed.version,
                "path": str(installed.path),
                "active": True,
            }
        except InstallCancelledError:
            raise DispatcherError("cancelled", f"Install of pack '{pack_id}' cancelled") from None
        except PackInstallError as err:
            raise DispatcherError(err.code, err.detail or err.code) from None
        finally:
            dispatcher._pack_cancels.pop(pack_id, None)
            dispatcher._pack_progress.pop(pack_id, None)
    finally:
        if dispatcher._pack_install_active == pack_id:
            dispatcher._pack_install_active = None


async def handle_pack_cancel(dispatcher: Any, payload: Dict[str, Any]) -> Dict[str, Any]:
    pack_id = payload.get("pack_id")
    if not isinstance(pack_id, str) or not pack_id:
        raise DispatcherError("invalid_payload", "pack_id is required")

    cancel_event = dispatcher._pack_cancels.get(pack_id)
    if cancel_event:
        cancel_event.set()
        return {"cancelled": True, "pack_id": pack_id}
    return {"cancelled": False, "pack_id": pack_id}


async def handle_pack_delete(dispatcher: Any, payload: Dict[str, Any]) -> Dict[str, Any]:
    if dispatcher._engine is not None and getattr(dispatcher._engine, "_running", False):
        raise DispatcherError("engine_busy", "Stop the active session before deleting a model")
    capability = payload.get("capability")
    pack_id = payload.get("pack_id")
    if not isinstance(capability, str) or capability not in ("stt", "translation"):
        raise DispatcherError("invalid_payload", "capability must be stt or translation")
    source_lang = payload.get("source_lang") if isinstance(payload.get("source_lang"), str) else None
    target_lang = payload.get("target_lang") if isinstance(payload.get("target_lang"), str) else None
    models_dir = dispatcher._get_config().models_dir
    removed = await asyncio.to_thread(
        delete_local_model_files,
        capability,
        source_lang=source_lang,
        target_lang=target_lang,
        pack_id=pack_id if isinstance(pack_id, str) else None,
        models_dir=models_dir,
    )
    return {"deleted": removed > 0, "removed_count": removed, "capability": capability}


async def handle_pack_check_readiness(dispatcher: Any, payload: Dict[str, Any]) -> Dict[str, Any]:
    capability = payload.get("capability", "stt")
    pack_id = payload.get("pack_id")
    quality_preset = payload.get("quality_preset")
    source_lang = payload.get("source_lang")
    target_lang = payload.get("target_lang")
    installer = dispatcher._get_installer()
    catalog = dispatcher._get_catalog()
    probe = dispatcher._get_probe()
    profile: HardwareProfile = await asyncio.to_thread(_probe_hardware_with_gpu, probe)

    kind = PackKind.STT if capability.lower() == "stt" else PackKind.TRANSLATION
    src_lang = source_lang if isinstance(source_lang, str) else None
    tgt_lang = target_lang if isinstance(target_lang, str) else None
    language_supported = (
        _is_local_translation_pair_supported(src_lang, tgt_lang)
        if kind is PackKind.TRANSLATION and tgt_lang
        else True
    )

    models_dir = dispatcher._get_config().models_dir
    if pack_id:
        pack = catalog.get(pack_id)
        if pack is None:
            raise DispatcherError("not_found", f"Pack '{pack_id}' not found")
        is_local = _is_local_model_ready(
            capability,
            source_lang=src_lang,
            target_lang=tgt_lang,
            pack_id=pack.pack_id,
            models_dir=models_dir,
        )
        issues = compatibility_issues(pack, profile)
        active = await asyncio.to_thread(installer.active, pack.pack_id)
        ready = (active is not None and len(issues) == 0) or is_local
        installed = (active is not None) or is_local
        tier_val = pack.tiers[0].value if pack.tiers else "minimal"
        size_bytes = pack.size_bytes or (500_000_000 if kind is PackKind.STT else 300_000_000)
        return {
            "capability": capability,
            "pack_id": pack.pack_id,
            "ready": ready,
            "issues": [] if ready else issues,
            "installed": installed,
            "tier": tier_val,
            "size_bytes": size_bytes,
            "language_supported": language_supported,
        }

    selection = select_default_pack(catalog, kind, profile)
    pack = selection.pack
    if pack is None:
        is_local = _is_local_model_ready(
            capability,
            source_lang=src_lang,
            target_lang=tgt_lang,
            models_dir=models_dir,
        )
        return {
            "capability": capability,
            "pack_id": None,
            "ready": is_local,
            "issues": [] if is_local else ["no_compatible_pack"],
            "installed": is_local,
            "tier": "minimal",
            "size_bytes": 500_000_000 if kind is PackKind.STT else 300_000_000,
            "language_supported": language_supported,
            "reason": selection.reason,
        }

    preset_packs = {
        "fast": ("legacy-faster-whisper-tiny", 75_000_000),
        "balanced": ("legacy-faster-whisper-small", 500_000_000),
        "accurate": ("legacy-faster-whisper-medium", 1_500_000_000),
    }
    effective_pack_id = pack.pack_id
    default_size = 500_000_000 if kind is PackKind.STT else 300_000_000
    effective_size = pack.size_bytes or default_size
    if (
        kind is PackKind.STT
        and pack.source is PackSource.LEGACY_RUNTIME
        and isinstance(quality_preset, str)
        and quality_preset in preset_packs
    ):
        effective_pack_id, effective_size = preset_packs[quality_preset]

    is_local = _is_local_model_ready(
        capability,
        source_lang=src_lang,
        target_lang=tgt_lang,
        pack_id=effective_pack_id,
        models_dir=models_dir,
    )
    active = await asyncio.to_thread(installer.active, pack.pack_id)
    ready = (active is not None) or is_local
    installed = (active is not None) or is_local
    tier_val = selection.tier.value if selection.tier else "minimal"
    return {
        "capability": capability,
        "pack_id": effective_pack_id,
        "ready": ready,
        "issues": [],
        "installed": installed,
        "tier": tier_val,
        "size_bytes": effective_size,
        "language_supported": language_supported,
        "reason": selection.reason,
    }


async def handle_get_provider_manifests(payload: Dict[str, Any]) -> Dict[str, Any]:
    capability = payload.get("capability")
    registry = build_default_registry()
    cap_enum = None
    if capability:
        try:
            cap_enum = ProviderCapability(capability.lower())
        except ValueError:
            raise DispatcherError("invalid_payload", f"Unknown capability '{capability}'") from None

    if cap_enum:
        manifests = registry.providers_for(cap_enum)
    else:
        with registry._lock:
            manifests = tuple(sorted(registry._manifests.values(), key=lambda m: m.provider_id))

    return {"manifests": [_serialize_manifest(m) for m in manifests]}


async def handle_audio_probe(engine: Any, payload: Dict[str, Any]) -> Dict[str, Any]:
    if engine is not None and getattr(engine, "_running", False):
        raise DispatcherError("engine_busy", "Stop the active session before testing audio")

    audio_source = payload.get("audio_source", "microphone")
    if audio_source not in ("microphone", "system", "both"):
        raise DispatcherError("invalid_payload", "audio_source must be microphone, system, or both")

    mic_id = payload.get("mic_device_id")
    speaker_id = payload.get("speaker_device_id")
    duration_ms = payload.get("duration_ms", 300)
    if not isinstance(duration_ms, int) or duration_ms < 100 or duration_ms > 1500:
        duration_ms = 300

    results: Dict[str, Any] = {}
    if audio_source in ("microphone", "both"):
        results["mic"] = await asyncio.to_thread(
            _probe_audio_stream, "mic", str(mic_id) if mic_id else None, duration_ms
        )
    if audio_source in ("system", "both"):
        results["system"] = await asyncio.to_thread(
            _probe_audio_stream, "system", str(speaker_id) if speaker_id else None, duration_ms
        )
    return {"streams": results}
