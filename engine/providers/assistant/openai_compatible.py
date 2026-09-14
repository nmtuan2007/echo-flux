"""Canonical Assistant adapter over an OpenAI-compatible Chat Completions API.

Transport and response mapping only. What to ask the model and how to turn its reply into
suggestions is an injected ``AssistantPromptPolicy``; the adapter never decides prompts,
context selection, UI schema, or fallback. Errors surface as ``ProviderRuntimeError``.
"""

import time
from contextlib import aclosing
from typing import AsyncIterator, Callable, Optional, Protocol, Sequence

from engine.core.logging import get_logger
from engine.providers.contracts import AssistantProvider
from engine.providers.text.openai_compatible import (
    OpenAICompatibleTextClient,
    TextCompletion,
    TextConfigError,
    TextEndpointStrategy,
    TextGenerationOptions,
    TextMessage,
    TextStreamChunk,
    TextTransportOptions,
    config_error_to_provider_error,
    make_text_error,
)
from engine.providers.types import (
    AssistRequest,
    AssistResult,
    ProviderCapability,
    ProviderErrorCode,
    ProviderHealthCheck,
    ProviderHealthStatus,
    ProviderRuntimeError,
    SummaryChunk,
    SummaryRequest,
    SummaryResult,
)

logger = get_logger("providers.assistant.openai_compatible")

_CAPABILITY = ProviderCapability.ASSISTANT


class AssistantPromptPolicy(Protocol):
    """Prompt construction and reply parsing, owned outside the adapter."""

    def build_assist_messages(self, request: AssistRequest) -> Sequence[TextMessage]: ...

    def parse_assist_result(
        self, request: AssistRequest, completion: TextCompletion
    ) -> AssistResult: ...

    def build_summary_messages(self, request: SummaryRequest) -> Sequence[TextMessage]: ...


class OpenAICompatibleAssistantProvider(AssistantProvider):
    """Assistant completions through OpenAI, Azure OpenAI, AssemblyAI, Local, or Custom."""

    def __init__(
        self,
        strategy: TextEndpointStrategy,
        model: str,
        *,
        prompt_policy: AssistantPromptPolicy,
        provider_id: str = "openai-compatible",
        generation: Optional[TextGenerationOptions] = None,
        transport_options: Optional[TextTransportOptions] = None,
        http_transport: Optional[object] = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        required = ("build_assist_messages", "parse_assist_result", "build_summary_messages")
        if prompt_policy is None or not all(
            callable(getattr(prompt_policy, name, None)) for name in required
        ):
            raise ProviderRuntimeError(
                make_text_error(
                    ProviderErrorCode.INVALID_REQUEST, _CAPABILITY, field_name="prompt_policy"
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
        self._policy = prompt_policy
        self._provider_id = provider_id
        self._generation = generation or TextGenerationOptions()

    def __repr__(self) -> str:
        return (
            f"OpenAICompatibleAssistantProvider(provider_id={self._provider_id!r}, "
            f"mode={self._client.mode!r})"
        )

    # ── BaseProvider ─────────────────────────────────────────────────────────

    @property
    def provider_id(self) -> str:
        return self._provider_id

    async def start(self) -> None:
        self._client.open()
        logger.info(
            "Text assistant provider started: provider=%s mode=%s endpoint=%s",
            self._provider_id,
            self._client.mode,
            self._client.redacted_endpoint,
        )

    async def stop(self) -> None:
        await self._client.aclose()
        logger.info("Text assistant provider stopped: provider=%s", self._provider_id)

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

    # ── Canonical completion primitives ──────────────────────────────────────

    async def complete_messages(
        self,
        messages: Sequence[TextMessage],
        generation: Optional[TextGenerationOptions] = None,
    ) -> TextCompletion:
        return await self._client.complete(self._model, messages, generation or self._generation)

    async def stream_messages(
        self,
        messages: Sequence[TextMessage],
        generation: Optional[TextGenerationOptions] = None,
    ) -> AsyncIterator[TextStreamChunk]:
        stream = self._client.stream(self._model, messages, generation or self._generation)
        async with aclosing(stream):
            async for chunk in stream:
                yield chunk

    # ── AssistantProvider ────────────────────────────────────────────────────

    async def assist(self, request: AssistRequest) -> AssistResult:
        completion = await self.complete_messages(self._policy.build_assist_messages(request))
        try:
            return self._policy.parse_assist_result(request, completion)
        except (ValueError, KeyError, TypeError) as exc:
            logger.warning(
                "Assistant reply could not be parsed: provider=%s error=%s",
                self._provider_id,
                type(exc).__name__,
            )
            raise ProviderRuntimeError(
                make_text_error(
                    ProviderErrorCode.INVALID_RESPONSE, _CAPABILITY, mode=self._client.mode
                )
            ) from None

    async def summarize(self, request: SummaryRequest) -> SummaryResult:
        completion = await self.complete_messages(
            self._policy.build_summary_messages(request), self._summary_generation(request)
        )
        return SummaryResult(
            summary_markdown=completion.text,
            provider_event_id=completion.response_id,
            input_tokens=completion.input_tokens,
            output_tokens=completion.output_tokens,
            latency_ms=completion.latency_ms,
        )

    async def stream_summary(self, request: SummaryRequest) -> AsyncIterator[SummaryChunk]:
        stream = self.stream_messages(
            self._policy.build_summary_messages(request), self._summary_generation(request)
        )
        async with aclosing(stream):
            async for chunk in stream:
                yield SummaryChunk(delta_text=chunk.delta_text, is_done=chunk.is_done)

    def _summary_generation(self, request: SummaryRequest) -> TextGenerationOptions:
        reserved = request.reserved_output_tokens
        if isinstance(reserved, bool) or not isinstance(reserved, int) or reserved < 1:
            return self._generation
        return TextGenerationOptions(
            temperature=self._generation.temperature, max_output_tokens=reserved
        )
