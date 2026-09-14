"""Local Translation provider: the legacy Marian/CTranslate2 runtime behind ``TranslationProvider``.

Pair resolution. Callers never name a model. The effective pair is the request's languages,
falling back to the configured ``source_lang``/``target_lang``; ``"auto"`` is the default source.
``select_default_pack`` picks the translation pack for this machine and pair. For the
``LEGACY_RUNTIME`` pack the pair must be one of the Marian presets; an ``"auto"`` source resolves
only when exactly one supported source language translates into the target. The result is
cached per pair, so the same inputs always give the same pair or the same error.

Errors. An unsupported or ambiguous pair is ``UNSUPPORTED_LANGUAGE``; a runtime or model that
cannot load is ``MODEL_UNAVAILABLE`` (cached per pair until ``stop``); an inference failure is a
retryable ``INTERNAL_ERROR``. Each is raised for that request only. The provider stays usable and
nothing here touches the STT service, so a translation failure never stops transcription.
``start`` does not fail on a bad configured pair either; the error surfaces per request.

Threads. Model load, translation, and unload run through ``asyncio.to_thread`` and are serialized
by one lock, matching the engine's single ``TranslationThread``.

The adapter is dormant: nothing wires it into the engine yet, and it never chooses fallback.
Nothing here logs source or translated text.
"""

import asyncio
import importlib.util
import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

from engine.core.logging import get_logger
from engine.local_models.catalog import PackCatalog, PackKind, PackSource, builtin_catalog
from engine.local_models.selector import (
    HardwareProbe,
    HardwareProfile,
    SystemHardwareProbe,
    compatibility_issues,
    select_default_pack,
)
from engine.providers.contracts import TranslationProvider
from engine.providers.types import (
    ProviderCapability,
    ProviderError,
    ProviderErrorCode,
    ProviderHealthCheck,
    ProviderHealthStatus,
    ProviderRuntimeError,
    TranslationRequest,
    TranslationResult,
)

logger = get_logger("providers.translation.local")

AUTO_LANGUAGE = "auto"
DEVICES = ("auto", "cpu", "cuda")
# Each Marian pair holds its own model in memory; auto-detected sources can touch several.
DEFAULT_MAX_LOADED_PAIRS = 2

_LANGUAGE_RE = re.compile(r"[A-Za-z]{2,3}(?:[-_][A-Za-z0-9]{2,8})?")

Pair = Tuple[str, str]
PresetResolver = Callable[[str, str], Optional[str]]


def _error(
    code: ProviderErrorCode,
    message: str,
    retryable: bool = False,
    details: Optional[Dict[str, Any]] = None,
) -> ProviderError:
    return ProviderError(
        code=code,
        capability=ProviderCapability.TRANSLATION,
        message=message,
        retryable=retryable,
        details=details,
    )


def _invalid(field_name: str) -> ProviderRuntimeError:
    return ProviderRuntimeError(
        _error(
            ProviderErrorCode.INVALID_REQUEST,
            "Local translation rejected the request.",
            details={"field": field_name},
        )
    )


def _unsupported(reason: str, pair: Pair) -> ProviderRuntimeError:
    return ProviderRuntimeError(
        _error(
            ProviderErrorCode.UNSUPPORTED_LANGUAGE,
            "Local translation does not support this language pair.",
            details={"reason": reason, "source_lang": pair[0], "target_lang": pair[1]},
        )
    )


def _is_auto(language: Optional[str]) -> bool:
    return language is None or language.strip().lower() in ("", AUTO_LANGUAGE)


def _valid_language(language: Any, allow_auto: bool) -> bool:
    if not isinstance(language, str):
        return False
    if language.strip().lower() == AUTO_LANGUAGE:
        return allow_auto
    return _LANGUAGE_RE.fullmatch(language) is not None


# ── Settings ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class LocalTranslationSettings:
    """Local Translation options from the ``local`` manifest. There is no model field."""

    source_lang: str = AUTO_LANGUAGE
    target_lang: str = "vi"
    device: str = "auto"

    @classmethod
    def from_options(cls, options: Dict[str, Any]) -> "LocalTranslationSettings":
        """Build from registry-resolved options; keys for other capabilities are ignored."""
        defaults = cls()
        return cls(
            source_lang=options.get("source_lang", defaults.source_lang),
            target_lang=options.get("target_lang", defaults.target_lang),
            device=options.get("device", defaults.device),
        )


def validate_settings(settings: LocalTranslationSettings) -> None:
    if not _valid_language(settings.source_lang, allow_auto=True):
        raise _invalid("source_lang")
    if not _valid_language(settings.target_lang, allow_auto=False):
        raise _invalid("target_lang")
    if settings.device not in DEVICES:
        raise _invalid("device")


# ── Pair resolution ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ResolvedTranslationPair:
    pack_id: str
    version: str
    reason: str
    source_lang: str
    target_lang: str
    model_name: str


def _default_preset_resolver(source_lang: str, target_lang: str) -> Optional[str]:
    from engine.translation.marian_backend import MarianBackend

    return MarianBackend.resolve_preset_model(source_lang, target_lang)


def resolve_translation_pair(
    catalog: PackCatalog,
    profile: HardwareProfile,
    source_lang: str,
    target_lang: str,
    preset_resolver: PresetResolver = _default_preset_resolver,
) -> ResolvedTranslationPair:
    """Resolve a concrete, supported pair and the model that serves it. Deterministic."""
    requested = (source_lang, target_lang)
    source = None if _is_auto(source_lang) else source_lang
    selection = select_default_pack(catalog, PackKind.TRANSLATION, profile, source, target_lang)
    pack = selection.pack
    if pack is None:
        hardware_fits = any(
            not compatibility_issues(candidate, profile)
            for candidate in catalog.packs_for(PackKind.TRANSLATION)
        )
        if hardware_fits:
            raise _unsupported(selection.reason, requested)
        raise ProviderRuntimeError(
            _error(
                ProviderErrorCode.MODEL_UNAVAILABLE,
                "No local translation pack is compatible with this machine.",
                details={"reason": selection.reason},
            )
        )
    if pack.source is not PackSource.LEGACY_RUNTIME:
        # No catalog translation pack exists yet, so there is no installed-pack runtime to use.
        raise ProviderRuntimeError(
            _error(
                ProviderErrorCode.MODEL_UNAVAILABLE,
                "The selected local translation pack is not supported by this runtime.",
                details={"pack_id": pack.pack_id, "reason": "pack_runtime_unsupported"},
            )
        )
    if source is None:
        sources = sorted({s for s, t in pack.language_pairs if t == target_lang})
        if len(sources) != 1:
            raise _unsupported("source_language_required", requested)
        source = sources[0]
    model_name = preset_resolver(source, target_lang)
    if not model_name:
        raise _unsupported("no_preset_model", (source, target_lang))
    return ResolvedTranslationPair(
        pack.pack_id, pack.version, selection.reason, source, target_lang, model_name
    )


def _default_backend_factory() -> Any:
    from engine.translation.marian_backend import MarianBackend

    return MarianBackend()


def _runtime_available() -> bool:
    return all(
        importlib.util.find_spec(name) is not None for name in ("ctranslate2", "transformers")
    )


# ── Provider ─────────────────────────────────────────────────────────────────


class LocalTranslationProvider(TranslationProvider):
    """Marian translation with deterministic pair resolution and idempotent lifecycle."""

    def __init__(
        self,
        settings: Optional[LocalTranslationSettings] = None,
        *,
        provider_id: str = "local",
        catalog: Optional[PackCatalog] = None,
        hardware_probe: Optional[HardwareProbe] = None,
        backend_factory: Callable[[], Any] = _default_backend_factory,
        preset_resolver: PresetResolver = _default_preset_resolver,
        runtime_available: Callable[[], bool] = _runtime_available,
        max_loaded_pairs: int = DEFAULT_MAX_LOADED_PAIRS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._settings = settings or LocalTranslationSettings()
        validate_settings(self._settings)
        if max_loaded_pairs < 1:
            raise _invalid("max_loaded_pairs")
        self._provider_id = provider_id
        self._catalog = catalog or builtin_catalog()
        self._probe = hardware_probe or SystemHardwareProbe()
        self._backend_factory = backend_factory
        self._preset_resolver = preset_resolver
        self._runtime_available = runtime_available
        self._max_loaded_pairs = max_loaded_pairs
        self._clock = clock
        self._lifecycle = asyncio.Lock()
        # Guards everything below; held only on worker threads.
        self._lock = threading.Lock()
        self._profile: Optional[HardwareProfile] = None
        self._resolved: Dict[Pair, Any] = {}
        self._load_errors: Dict[Pair, ProviderError] = {}
        self._backends: OrderedDict[Pair, Any] = OrderedDict()
        self._started = False

    # ── BaseProvider ─────────────────────────────────────────────────────────

    @property
    def provider_id(self) -> str:
        return self._provider_id

    async def start(self) -> None:
        async with self._lifecycle:
            if self._started:
                return
            await asyncio.to_thread(self._prepare)
            self._started = True
            logger.info(
                "Local translation started: provider=%s target=%s",
                self._provider_id,
                self._settings.target_lang,
            )

    async def stop(self) -> None:
        async with self._lifecycle:
            if not self._started:
                return
            # Cleared first so new requests are refused while in-flight work drains.
            self._started = False
            await asyncio.to_thread(self._release)
            logger.info("Local translation stopped: provider=%s", self._provider_id)

    async def health_check(self) -> ProviderHealthCheck:
        """Report readiness without loading a model or translating text."""
        began = self._clock()
        try:
            details = await asyncio.to_thread(self._check_ready)
        except ProviderRuntimeError as err:
            return ProviderHealthCheck(
                status=ProviderHealthStatus.UNHEALTHY,
                message=err.error.message,
                details={"code": err.error.code.value, "retryable": err.error.retryable},
            )
        return ProviderHealthCheck(
            status=ProviderHealthStatus.HEALTHY,
            message="Local translation is available.",
            latency_ms=(self._clock() - began) * 1000.0,
            details=details,
        )

    # ── TranslationProvider ──────────────────────────────────────────────────

    async def translate(self, request: TranslationRequest) -> TranslationResult:
        if not self._started:
            raise _invalid("not_started")
        if not isinstance(request.text, str):
            raise _invalid("text")
        source = request.source_lang
        if _is_auto(source):
            source = self._settings.source_lang
        target = request.target_lang
        if _is_auto(target):
            target = self._settings.target_lang
        if not _valid_language(source, allow_auto=True):
            raise _invalid("source_lang")
        if not _valid_language(target, allow_auto=False):
            raise _invalid("target_lang")

        began = self._clock()
        if not request.text.strip():
            # Nothing to translate: no model is resolved or loaded.
            return self._result(request, "", source, target, began)
        if not _is_auto(source) and source == target:
            return self._result(request, request.text, source, target, began)

        translated, pair = await asyncio.to_thread(
            self._translate_blocking, request.text, source, target
        )
        detected = pair.source_lang if _is_auto(source) else None
        return self._result(request, translated, pair.source_lang, target, began, detected)

    def _result(
        self,
        request: TranslationRequest,
        translated: str,
        source: str,
        target: str,
        began: float,
        detected: Optional[str] = None,
    ) -> TranslationResult:
        return TranslationResult(
            translated_text=translated,
            source_text=request.text,
            source_lang=source,
            target_lang=target,
            stream_id=request.stream_id,
            sequence=request.sequence,
            revision=request.revision,
            is_final=request.is_final,
            detected_source_lang=detected,
            monotonic_timestamp=self._clock(),
            latency_ms=(self._clock() - began) * 1000.0,
        )

    # ── Blocking work (worker threads only) ──────────────────────────────────

    def _prepare(self) -> None:
        with self._lock:
            self._resolved.clear()
            self._load_errors.clear()
            self._profile = self._probe.probe()
            if _is_auto(self._settings.source_lang):
                return
            pair = (self._settings.source_lang, self._settings.target_lang)
            if pair[0] == pair[1]:
                return
            try:
                # Warm the configured pair so the first final is not slowed by a model load.
                self._backend_for(pair)
            except ProviderRuntimeError as err:
                logger.warning(
                    "Local translation pair not ready at start: code=%s", err.error.code.value
                )

    def _translate_blocking(
        self, text: str, source: str, target: str
    ) -> Tuple[str, ResolvedTranslationPair]:
        with self._lock:
            backend, pair = self._backend_for((source, target))
            try:
                result = backend.translate(text, pair.source_lang, pair.target_lang)
            except Exception as exc:
                logger.error("Local translation failed: error=%s", type(exc).__name__)
                raise ProviderRuntimeError(
                    _error(
                        ProviderErrorCode.INTERNAL_ERROR,
                        "Local translation failed.",
                        retryable=True,
                        details={"source_lang": pair.source_lang, "target_lang": pair.target_lang},
                    )
                ) from None
            return result.translated_text, pair

    def _resolve(self, requested: Pair) -> ResolvedTranslationPair:
        cached = self._resolved.get(requested)
        if isinstance(cached, ProviderError):
            raise ProviderRuntimeError(cached)
        if cached is not None:
            return cached
        profile = self._profile or self._probe.probe()
        try:
            pair = resolve_translation_pair(
                self._catalog, profile, requested[0], requested[1], self._preset_resolver
            )
        except ProviderRuntimeError as err:
            self._resolved[requested] = err.error
            raise
        self._resolved[requested] = pair
        return pair

    def _backend_for(self, requested: Pair) -> Tuple[Any, ResolvedTranslationPair]:
        pair = self._resolve(requested)
        key = (pair.source_lang, pair.target_lang)
        backend = self._backends.get(key)
        if backend is not None:
            self._backends.move_to_end(key)
            return backend, pair
        failure = self._load_errors.get(key)
        if failure is not None:
            raise ProviderRuntimeError(failure)
        while len(self._backends) >= self._max_loaded_pairs:
            _, evicted = self._backends.popitem(last=False)
            self._unload(evicted)
        backend = None
        try:
            backend = self._backend_factory()
            backend.load_model(
                {
                    "translation.source_lang": pair.source_lang,
                    "translation.target_lang": pair.target_lang,
                    # Always the resolved preset, so the legacy backend never guesses a Hub id.
                    "translation.model": pair.model_name,
                    "device": self._settings.device,
                }
            )
        except Exception as exc:
            logger.error(
                "Local translation model load failed: pack=%s pair=%s-%s error=%s",
                pair.pack_id,
                pair.source_lang,
                pair.target_lang,
                type(exc).__name__,
            )
            self._unload(backend)
            failure = _error(
                ProviderErrorCode.MODEL_UNAVAILABLE,
                "The local translation model could not be loaded.",
                details={"source_lang": pair.source_lang, "target_lang": pair.target_lang},
            )
            self._load_errors[key] = failure
            raise ProviderRuntimeError(failure) from None
        self._backends[key] = backend
        return backend, pair

    def _release(self) -> None:
        with self._lock:
            while self._backends:
                _, backend = self._backends.popitem(last=False)
                self._unload(backend)
            self._resolved.clear()
            self._load_errors.clear()

    def _check_ready(self) -> Dict[str, Any]:
        with self._lock:
            loaded = [f"{s}-{t}" for s, t in self._backends]
        if loaded:
            return {"loaded_pairs": loaded}
        if not self._runtime_available():
            raise ProviderRuntimeError(
                _error(
                    ProviderErrorCode.MODEL_UNAVAILABLE,
                    "The local translation runtime is not installed.",
                )
            )
        settings = self._settings
        if _is_auto(settings.source_lang) or settings.source_lang == settings.target_lang:
            return {"loaded_pairs": []}
        pair = resolve_translation_pair(
            self._catalog,
            self._probe.probe(),
            settings.source_lang,
            settings.target_lang,
            self._preset_resolver,
        )
        return {"loaded_pairs": [], "pack_id": pair.pack_id, "reason": pair.reason}

    @staticmethod
    def _unload(backend: Any) -> None:
        if backend is None:
            return
        try:
            backend.unload_model()
        except Exception as exc:
            logger.warning("Local translation unload failed: error=%s", type(exc).__name__)
