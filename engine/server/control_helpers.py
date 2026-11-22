"""Helpers extracted from control_dispatcher to keep request routing concise."""

import asyncio
import contextlib
import time
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Optional

from engine.core.config import Config
from engine.providers.manifests import ManifestField, ProviderManifest
from engine.server.protocol import suggestion_result_message


def stored_assist_payload(
    conversation_id: str,
    message_id: str,
    outcome: Any,
    provider: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Canonical suggestion_result payload shared by engine and control_dispatcher."""
    stored = outcome.stored
    resolved_provider = provider
    if resolved_provider is None and getattr(stored, "provider", None) is not None:
        resolved_provider = stored.provider.to_dict()
    return suggestion_result_message(
        conversation_id=conversation_id,
        message_id=message_id,
        assist_id=stored.assist_id,
        suggested_replies=[
            {"strategy": r.strategy, "text": r.text, "reasoning": r.reasoning}
            for r in stored.suggested_replies
        ],
        clarifying_questions=[
            {"question": q.question, "purpose": q.purpose}
            for q in stored.clarifying_questions
        ],
        provider=resolved_provider,
        context_window={
            "message_count": outcome.context_message_count,
            "omitted_message_count": outcome.omitted_message_count,
            "profile_count": outcome.profile_count,
            "omitted_profile_count": outcome.omitted_profile_count,
        },
    )


async def run_summary_stream(
    service: Any,
    conversation_id: str,
    on_chunk: Callable[[str], Awaitable[None]],
    on_failure: Callable[..., Awaitable[None]],
    on_success: Callable[[Any, Optional[Dict[str, Any]]], Awaitable[None]],
) -> None:
    """Shared summary streaming runner used by both EchoFluxEngine and ControlCommandDispatcher."""
    streamed = False

    async def _chunk_wrapper(text: str) -> None:
        nonlocal streamed
        streamed = True
        await on_chunk(text)

    try:
        outcome = await service.request_summary(conversation_id, on_chunk=_chunk_wrapper)
    except asyncio.CancelledError:
        await asyncio.shield(on_failure("cancelled", after_chunks=streamed))
        raise
    except Exception:
        await asyncio.shield(on_failure("internal_error", after_chunks=streamed))
        return

    w_payload = warning_dict(outcome.warning)
    if not outcome.ok:
        err = outcome.error
        await asyncio.shield(
            on_failure(
                err.code.value,
                after_chunks=streamed,
                message=err.message,
                retryable=err.retryable,
                provider_code=err.provider_code,
                warning=w_payload,
            )
        )
        return

    await asyncio.shield(on_success(outcome, w_payload))


def warning_dict(warning: Any) -> Optional[Dict[str, Any]]:
    if warning is None:
        return None
    return {
        "omitted_message_count": getattr(warning, "omitted_message_count", 0),
        "first_omitted_timestamp_ms": getattr(warning, "first_omitted_timestamp_ms", None),
        "last_omitted_timestamp_ms": getattr(warning, "last_omitted_timestamp_ms", None),
        "reason": getattr(warning, "reason", "context_limit"),
    }


def serialize_field(f: ManifestField) -> Dict[str, Any]:
    return {
        "key": f.key,
        "label": f.label,
        "field_type": f.field_type.value,
        "default": f.default,
        "required": f.required,
        "advanced": f.advanced,
        "secret": f.secret,
        "choices": list(f.choices),
        "minimum": f.minimum,
        "maximum": f.maximum,
        "description": f.description,
        "applies_to": [c.value for c in f.applies_to],
    }


def serialize_manifest(m: ProviderManifest) -> Dict[str, Any]:
    return {
        "provider_id": m.provider_id,
        "display_name": m.display_name,
        "capabilities": [c.value for c in m.capabilities],
        "protocol_family": m.protocol_family.value,
        "recommended_endpoint": m.recommended_endpoint,
        "recommended_model": m.recommended_model,
        "endpoint_editable": m.endpoint_editable,
        "requires_endpoint": m.requires_endpoint,
        "model_selectable": m.model_selectable,
        "fields": [serialize_field(f) for f in m.fields],
        "protocol_profiles": {k: v.value for k, v in m.protocol_profiles.items()},
        "profile_field": m.profile_field,
    }


def _expected_whisper_size(pack_id: Optional[str]) -> Optional[str]:
    if not pack_id:
        return None
    lowered = pack_id.lower()
    if "tiny" in lowered:
        return "tiny"
    if "base" in lowered:
        return "base"
    if "small" in lowered:
        return "small"
    if "medium" in lowered:
        return "medium"
    if "large" in lowered:
        return "large-v3"
    return None


def is_local_translation_pair_supported(
    source_lang: Optional[str], target_lang: Optional[str]
) -> bool:
    """True when offline Marian presets support translating into ``target_lang``."""
    from engine.local_models.catalog import _LEGACY_MARIAN_PAIRS

    src = (source_lang or "").strip().lower()
    tgt = (target_lang or "").strip().lower()
    if not tgt or tgt == "auto":
        return False
    if not src or src == "auto":
        return any(pair_tgt == tgt for _, pair_tgt in _LEGACY_MARIAN_PAIRS)
    return (src, tgt) in _LEGACY_MARIAN_PAIRS


def is_local_model_ready(
    capability: str,
    source_lang: Optional[str] = None,
    target_lang: Optional[str] = None,
    pack_id: Optional[str] = None,
    models_dir: Optional[Path] = None,
) -> bool:
    """Checks whether local STT or Translation models exist on disk in models_dir."""
    try:
        resolved_dir = models_dir if models_dir is not None else Config().models_dir
        if not resolved_dir or not resolved_dir.exists():
            return False
        cap = (capability or "").lower()
        if cap in ("stt", "speech", "whisper"):
            if (resolved_dir / "model.bin").is_file():
                return True
            expected_size = _expected_whisper_size(pack_id)
            if expected_size:
                stt_dirs = list(resolved_dir.glob(f"models--Systran--faster-whisper-{expected_size}*")) + list(
                    resolved_dir.glob(f"models--openai--whisper-{expected_size}*")
                )
            else:
                stt_dirs = list(resolved_dir.glob("models--Systran--faster-whisper-*")) + list(
                    resolved_dir.glob("models--openai--whisper-*")
                )
            return len(stt_dirs) > 0
        if cap in ("translation", "translate", "marian", "opus"):
            ct2_dir = resolved_dir / "ct2"
            if ct2_dir.exists():
                src = (source_lang or "").strip().lower()
                tgt = (target_lang or "").strip().lower()
                if src and src != "auto" and tgt and tgt != "auto":
                    trans_dirs = list(ct2_dir.glob(f"Helsinki-NLP_opus-mt-{src}-{tgt}*"))
                elif tgt and tgt != "auto":
                    trans_dirs = list(ct2_dir.glob(f"Helsinki-NLP_opus-mt-*-{tgt}*"))
                else:
                    trans_dirs = list(ct2_dir.glob("Helsinki-NLP_opus-mt-*"))
                return len(trans_dirs) > 0
        return False
    except Exception:
        return False


def probe_hardware_with_gpu(probe: Any) -> Any:
    """Run ``probe.probe()`` and lazily enrich GPU info via ctranslate2 when using SystemHardwareProbe."""
    from engine.local_models.selector import GpuInfo, HardwareProfile, SystemHardwareProbe

    profile: HardwareProfile = probe.probe()
    if profile.gpu is not None or type(probe) is not SystemHardwareProbe:
        return profile
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() <= 0:
            return profile
        vram_gb = 4.0
        with contextlib.suppress(Exception):
            import subprocess

            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                timeout=1.5,
                text=True,
            )
            first_mb = float(out.strip().splitlines()[0])
            if first_mb > 0:
                vram_gb = round(first_mb / 1024.0, 2)
        return HardwareProfile(
            cpu_cores=profile.cpu_cores,
            ram_gb=profile.ram_gb,
            platform=profile.platform,
            architecture=profile.architecture,
            gpu=GpuInfo(backend="cuda", vram_gb=vram_gb),
        )
    except Exception:
        return profile


def probe_audio_stream(kind: str, device_id: Optional[str], duration_ms: int) -> Dict[str, Any]:
    """Sample a single audio stream briefly to measure input RMS level."""
    import math

    import numpy as np

    cfg = {"sample_rate": 16000, "channels": 1, "chunk_ms": 20}
    inp = None
    try:
        if kind == "mic":
            from engine.audio.microphone import MicrophoneInput

            dev = int(device_id) if device_id else None
            inp = MicrophoneInput(cfg, device_id=dev)
        else:
            from engine.audio.system_audio import SystemAudioInput

            inp = SystemAudioInput(cfg, device_id=device_id or None)

        inp.start()
        deadline = time.monotonic() + (duration_ms / 1000.0)
        squares = 0.0
        samples = 0
        while time.monotonic() < deadline:
            chunk = inp.read_chunk()
            if chunk:
                arr = np.frombuffer(chunk, dtype=np.int16).astype(np.float64) / 32768.0
                if arr.size > 0:
                    squares += float(np.sum(arr * arr))
                    samples += int(arr.size)
        rms = math.sqrt(squares / samples) if samples > 0 else 0.0
        # Scale RMS so conversational speech (~0.05-0.2) reads cleanly on 0..1 meter.
        level = round(min(1.0, rms * 5.0), 3)
        return {"ok": True, "level": level, "rms": round(rms, 5), "samples": samples}
    except Exception as exc:
        return {"ok": False, "level": 0.0, "error": type(exc).__name__}
    finally:
        if inp is not None:
            with contextlib.suppress(Exception):
                inp.stop()
