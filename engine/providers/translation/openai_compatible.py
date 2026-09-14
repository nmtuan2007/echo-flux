"""Canonical Translation adapter over an OpenAI-compatible Chat Completions API.

The adapter owns request/response mapping only. Messages come from an injected
``message_builder`` (prompt policy lives elsewhere), and endpoint, model, and credentials
arrive already resolved inside a ``TextEndpointStrategy``. It never falls back; failures
surface as ``ProviderRuntimeError`` with a stable code.
"""

import time
from typing import Callable, Optional, Sequence

from engine.core.logging import get_logger
from engine.providers.contracts import TranslationProvider
from engine.providers.text.openai_compatible import (
    OpenAICompatibleTextClient,
    TextConfigError,
    TextEndpointStrategy,
    TextGenerationOptions,
    TextMessage,
    TextTransportOptions,
    config_error_to_provider_error,
    make_text_error,
)
from engine.providers.types import (
    ProviderCapability,
    ProviderErrorCode,
    ProviderHealthCheck,
    ProviderHealthStatus,
    ProviderRuntimeError,
    TranslationRequest,
    TranslationResult,
)

logger = get_logger("providers.translation.openai_compatible")

_CAPABILITY = ProviderCapability.TRANSLATION

TranslationMessageBuilder = Callable[[TranslationRequest], Sequence[TextMessage]]


class OpenAICompatibleTranslationProvider(TranslationProvider):
    """Translation through OpenAI, Azure OpenAI, AssemblyAI, Local, or Custom endpoints."""

    def __init__(
        self,
        strategy: TextEndpointStrategy,
        model: str,
        *,
        message_builder: TranslationMessageBuilder,
        provider_id: str = "openai-compatible",
        generation: Optional[TextGenerationOptions] = None,
        transport_options: Optional[TextTransportOptions] = None,
        http_transport: Optional[object] = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not callable(message_builder):
            raise ProviderRuntimeError(
                make_text_error(
                    ProviderErrorCode.INVALID_REQUEST, _CAPABILITY, field_name="message_builder"
                )
            )
        try:
            self._client = OpenAICompatibleTextClient(
                strategy,
                capability=_CAPABILITY,
                options=transport_options,
                http_transport=http_transport,
                clock=clock,
            )
        except TextConfigError as exc:
            raise config_error_to_provider_error(exc, _CAPABILITY) from None
        self._model = model if isinstance(model, str) else ""
        self._builder = message_builder
        self._provider_id = provider_id
        self._generation = generation or TextGenerationOptions()

    def __repr__(self) -> str:
        return (
            f"OpenAICompatibleTranslationProvider(provider_id={self._provider_id!r}, "
            f"mode={self._client.mode!r})"
        )

    @property
    def provider_id(self) -> str:
        return self._provider_id

    async def start(self) -> None:
        self._client.open()
        logger.info(
            "Text translation provider started: provider=%s mode=%s endpoint=%s",
            self._provider_id,
            self._client.mode,
            self._client.redacted_endpoint,
        )

    async def stop(self) -> None:
        await self._client.aclose()
        logger.info("Text translation provider stopped: provider=%s", self._provider_id)

    async def health_check(self) -> ProviderHealthCheck:
        try:
            latency_ms = await self._client.probe()
        except ProviderRuntimeError as err:
            return ProviderHealthCheck(
                status=ProviderHealthStatus.UNHEALTHY,
                message=err.error.message,
                details={"code": err.error.code.value, "retryable": err.error.retryable},
            )
        if latency_ms is None:
            return ProviderHealthCheck(
                status=ProviderHealthStatus.DEGRADED,
                message="Configuration is valid; this mode has no connectivity probe.",
                details={"mode": self._client.mode},
            )
        return ProviderHealthCheck(
            status=ProviderHealthStatus.HEALTHY,
            message="Text provider reachable.",
            latency_ms=latency_ms,
            details={"mode": self._client.mode},
        )

    async def translate(self, request: TranslationRequest) -> TranslationResult:
        if not request.text.strip():
            return self._result(request, "", None)
        messages = self._builder(request)
        completion = await self._client.complete(self._model, messages, self._generation)
        translated = completion.text.strip()
        if not translated:
            raise ProviderRuntimeError(
                make_text_error(
                    ProviderErrorCode.INVALID_RESPONSE, _CAPABILITY, mode=self._client.mode
                )
            )
        return TranslationResult(
            translated_text=translated,
            source_text=request.text,
            source_lang=request.source_lang,
            target_lang=request.target_lang,
            stream_id=request.stream_id,
            sequence=request.sequence,
            revision=request.revision,
            is_final=request.is_final,
            provider_event_id=completion.response_id,
            input_tokens=completion.input_tokens,
            output_tokens=completion.output_tokens,
            latency_ms=completion.latency_ms,
        )

    @staticmethod
    def _result(
        request: TranslationRequest, text: str, latency_ms: Optional[float]
    ) -> TranslationResult:
        return TranslationResult(
            translated_text=text,
            source_text=request.text,
            source_lang=request.source_lang,
            target_lang=request.target_lang,
            stream_id=request.stream_id,
            sequence=request.sequence,
            revision=request.revision,
            is_final=request.is_final,
            latency_ms=latency_ms,
        )
