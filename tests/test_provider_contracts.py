"""Tests for EchoFlux provider domain contracts and canonical types.

Validates contract inheritance, capability distinction at registration, import cost,
reconciliation/revision metadata in canonical types, safe categorized errors, and
deterministic fake providers satisfying all interfaces.
"""

import asyncio
import inspect
import subprocess
import sys
import time
from typing import AsyncIterator, List, Optional

import pytest

from engine.providers import (
    # Contracts
    AssistantProvider,
    # Types
    AssistQuestion,
    AssistRequest,
    AssistResult,
    AssistSuggestion,
    AudioFrame,
    BaseProvider,
    ContextProfileSnapshot,
    ConversationMessage,
    # Enums
    ProviderCapability,
    ProviderError,
    ProviderErrorCode,
    ProviderException,
    ProviderHealthCheck,
    ProviderHealthStatus,
    StreamingSTTProvider,
    STTTranscriptResult,
    SummaryChunk,
    SummaryRequest,
    SummaryResult,
    SummaryWarning,
    TranslationProvider,
    TranslationRequest,
    TranslationResult,
    WordTiming,
)

# ── Deterministic Fake Providers ─────────────────────────────────────────────


class FakeSTTProvider(StreamingSTTProvider):
    """Deterministic in-memory STT provider for testing contracts."""

    def __init__(self, provider_id: str = "fake-stt") -> None:
        self._provider_id = provider_id
        self.started = False
        self.stopped = False
        self.received_frames: List[AudioFrame] = []
        self._event_queue: asyncio.Queue[STTTranscriptResult] = asyncio.Queue()

    @property
    def provider_id(self) -> str:
        return self._provider_id

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    async def health_check(self) -> ProviderHealthCheck:
        return ProviderHealthCheck(
            status=ProviderHealthStatus.HEALTHY,
            message="Fake STT is operational",
            latency_ms=1.5,
        )

    async def send_audio(self, frame: AudioFrame) -> None:
        self.received_frames.append(frame)
        # Deterministically emit a partial result then a final result
        partial = STTTranscriptResult(
            stream_id=frame.stream_id,
            sequence=frame.sequence,
            provider_event_id=f"evt-{frame.sequence}-1",
            text="hello",
            is_final=False,
            revision=1,
            start_time_ms=0.0,
            end_time_ms=frame.duration_ms,
            monotonic_timestamp=time.monotonic(),
            detected_language="en",
            confidence=0.85,
        )
        await self._event_queue.put(partial)

    async def finalize_stream(self, stream_id: str) -> None:
        final_evt = STTTranscriptResult(
            stream_id=stream_id,
            sequence=len(self.received_frames),
            provider_event_id=f"evt-{stream_id}-final",
            text="hello world",
            is_final=True,
            revision=2,
            start_time_ms=0.0,
            end_time_ms=1000.0,
            monotonic_timestamp=time.monotonic(),
            detected_language="en",
            confidence=0.98,
            is_endpoint=True,
            words=(
                WordTiming("hello", 0.0, 400.0, 0.99),
                WordTiming("world", 450.0, 900.0, 0.97),
            ),
        )
        await self._event_queue.put(final_evt)

    async def reset_stream(self, stream_id: str) -> None:
        self.received_frames = [f for f in self.received_frames if f.stream_id != stream_id]

    async def receive_events(self) -> AsyncIterator[STTTranscriptResult]:
        while not self._event_queue.empty():
            yield await self._event_queue.get()


class FakeTranslationProvider(TranslationProvider):
    """Deterministic translation provider for testing contracts."""

    def __init__(self, provider_id: str = "fake-trans") -> None:
        self._provider_id = provider_id
        self.started = False
        self.stopped = False

    @property
    def provider_id(self) -> str:
        return self._provider_id

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    async def health_check(self) -> ProviderHealthCheck:
        return ProviderHealthCheck(
            status=ProviderHealthStatus.HEALTHY,
            message="Fake translation is healthy",
            latency_ms=2.0,
        )

    async def translate(self, request: TranslationRequest) -> TranslationResult:
        if request.target_lang == "unsupported":
            raise ProviderException(
                ProviderError(
                    code=ProviderErrorCode.UNSUPPORTED_LANGUAGE,
                    capability=ProviderCapability.TRANSLATION,
                    message="Target language 'unsupported' is not available.",
                    retryable=False,
                )
            )

        return TranslationResult(
            translated_text=f"[{request.target_lang}] {request.text}",
            source_text=request.text,
            source_lang=request.source_lang,
            target_lang=request.target_lang,
            stream_id=request.stream_id,
            sequence=request.sequence,
            revision=request.revision,
            is_final=request.is_final,
            detected_source_lang="en" if request.source_lang == "auto" else request.source_lang,
            provider_event_id="trans-evt-1",
            monotonic_timestamp=time.monotonic(),
            input_tokens=len(request.text.split()),
            output_tokens=len(request.text.split()) + 1,
            latency_ms=10.0,
        )


class FakeAssistantProvider(AssistantProvider):
    """Deterministic LLM assistant provider for testing contracts."""

    def __init__(self, provider_id: str = "fake-assistant") -> None:
        self._provider_id = provider_id
        self.started = False
        self.stopped = False

    @property
    def provider_id(self) -> str:
        return self._provider_id

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    async def health_check(self) -> ProviderHealthCheck:
        return ProviderHealthCheck(
            status=ProviderHealthStatus.HEALTHY,
            message="Fake assistant ready",
            latency_ms=5.0,
        )

    async def assist(self, request: AssistRequest) -> AssistResult:
        return AssistResult(
            target_message_id=request.target_message.message_id,
            suggested_replies=(
                AssistSuggestion(strategy="Cooperative", text="I agree with this point."),
                AssistSuggestion(strategy="Assertive", text="We must review the proposal first."),
            ),
            clarifying_questions=(
                AssistQuestion(question="What is the targeted deadline?", purpose="Timeline check"),
            ),
            provider_event_id="assist-evt-1",
            monotonic_timestamp=time.monotonic(),
            input_tokens=150,
            output_tokens=45,
            latency_ms=120.0,
        )

    async def summarize(self, request: SummaryRequest) -> SummaryResult:
        warning: Optional[SummaryWarning] = None
        if len(request.messages) > 10:
            warning = SummaryWarning(
                omitted_message_count=len(request.messages) - 10,
                first_omitted_timestamp_ms=request.messages[0].timestamp_ms,
                last_omitted_timestamp_ms=request.messages[len(request.messages) - 11].timestamp_ms,
                reason="Transcript exceeded provider input token window.",
            )

        return SummaryResult(
            summary_markdown="## Meeting Summary\n\nKey discussion points.",
            bullet_points=("Point 1", "Point 2"),
            decisions=("Decision A approved",),
            action_items=("Implement EF-002",),
            risks=("Tight timeline",),
            warning=warning,
            provider_event_id="sum-evt-1",
            monotonic_timestamp=time.monotonic(),
            input_tokens=500,
            output_tokens=120,
            latency_ms=250.0,
        )

    async def stream_summary(self, request: SummaryRequest) -> AsyncIterator[SummaryChunk]:
        for token in ("Meeting ", "summary ", "streamed ", "text."):
            yield SummaryChunk(delta_text=token, is_done=False)
        yield SummaryChunk(delta_text="", is_done=True)


# ── Contract Distinction & Registration Tests ────────────────────────────────


def test_three_capability_contracts_are_distinct():
    """Verify that STT, Translation, and Assistant contracts are distinct classes."""
    classes = [StreamingSTTProvider, TranslationProvider, AssistantProvider]
    assert len(set(classes)) == 3

    for cls in classes:
        assert issubclass(cls, BaseProvider)

    # Cross-inheritance must not exist between capability interfaces
    assert not issubclass(StreamingSTTProvider, TranslationProvider)
    assert not issubclass(StreamingSTTProvider, AssistantProvider)
    assert not issubclass(TranslationProvider, StreamingSTTProvider)
    assert not issubclass(TranslationProvider, AssistantProvider)
    assert not issubclass(AssistantProvider, StreamingSTTProvider)
    assert not issubclass(AssistantProvider, TranslationProvider)


def test_capability_enums_are_distinct():
    """Ensure capability property values are distinct across capabilities."""
    assert StreamingSTTProvider.capability.fget(None) == ProviderCapability.STT  # type: ignore
    assert TranslationProvider.capability.fget(None) == ProviderCapability.TRANSLATION  # type: ignore
    assert AssistantProvider.capability.fget(None) == ProviderCapability.ASSISTANT  # type: ignore


def test_registration_prevents_capability_confusion():
    """Simulate a capability-checked registry to ensure providers cannot be confused."""
    registry = {
        ProviderCapability.STT: {},
        ProviderCapability.TRANSLATION: {},
        ProviderCapability.ASSISTANT: {},
    }

    def register(provider: BaseProvider) -> None:
        if not isinstance(provider, BaseProvider):
            raise TypeError("Must inherit BaseProvider")

        expected_type = {
            ProviderCapability.STT: StreamingSTTProvider,
            ProviderCapability.TRANSLATION: TranslationProvider,
            ProviderCapability.ASSISTANT: AssistantProvider,
        }[provider.capability]

        if not isinstance(provider, expected_type):
            raise TypeError(
                f"Provider with capability {provider.capability} must inherit {expected_type.__name__}"
            )
        registry[provider.capability][provider.provider_id] = provider

    stt = FakeSTTProvider("test-stt")
    trans = FakeTranslationProvider("test-trans")
    assistant = FakeAssistantProvider("test-ast")

    register(stt)
    register(trans)
    register(assistant)

    assert registry[ProviderCapability.STT]["test-stt"] is stt
    assert registry[ProviderCapability.TRANSLATION]["test-trans"] is trans
    assert registry[ProviderCapability.ASSISTANT]["test-ast"] is assistant

    # Intentional capability spoofing fails
    class ConfusedProvider(StreamingSTTProvider):
        @property
        def capability(self) -> ProviderCapability:
            return ProviderCapability.TRANSLATION  # Mismatch!

        @property
        def provider_id(self) -> str:
            return "confused"

        async def start(self) -> None: ...
        async def stop(self) -> None: ...
        async def health_check(self) -> ProviderHealthCheck:
            return ProviderHealthCheck(status=ProviderHealthStatus.HEALTHY)

        async def send_audio(self, frame: AudioFrame) -> None: ...
        async def receive_events(self) -> AsyncIterator[STTTranscriptResult]:
            return
            yield  # type: ignore

        async def finalize_stream(self, stream_id: str) -> None: ...
        async def reset_stream(self, stream_id: str) -> None: ...

    with pytest.raises(TypeError, match="must inherit TranslationProvider"):
        register(ConfusedProvider())


def test_unimplemented_abstract_methods_fail_instantiation():
    """Python must enforce ABC contract on incomplete implementations."""

    class IncompleteSTT(StreamingSTTProvider):
        @property
        def provider_id(self) -> str:
            return "inc"

        # Missing start, stop, health_check, send_audio, receive_events, etc.

    with pytest.raises(TypeError):
        IncompleteSTT()  # type: ignore


def test_streaming_methods_are_not_coroutine_functions_and_return_async_iterators():
    """Verify streaming methods on ABCs are not coroutines and fakes return __aiter__ objects."""
    assert not inspect.iscoroutinefunction(StreamingSTTProvider.receive_events)
    assert not inspect.iscoroutinefunction(AssistantProvider.stream_summary)

    fake_stt = FakeSTTProvider()
    stt_stream = fake_stt.receive_events()
    assert hasattr(stt_stream, "__aiter__")

    fake_assistant = FakeAssistantProvider()
    summary_stream = fake_assistant.stream_summary(SummaryRequest(messages=()))
    assert hasattr(summary_stream, "__aiter__")


# ── Canonical Types & Metadata Tests ─────────────────────────────────────────


def test_stt_result_contains_all_required_reconciliation_fields():
    """Verify STTTranscriptResult has stream ID, sequence, monotonic time, intervals, revision, etc."""
    result = STTTranscriptResult(
        stream_id="mic",
        sequence=42,
        provider_event_id="evt-999",
        text="testing audio reconciliation",
        is_final=False,
        revision=3,
        start_time_ms=1200.0,
        end_time_ms=2500.0,
        monotonic_timestamp=1000.5,
        detected_language="en",
        confidence=0.91,
        is_endpoint=False,
    )
    assert result.stream_id == "mic"
    assert result.sequence == 42
    assert result.provider_event_id == "evt-999"
    assert result.text == "testing audio reconciliation"
    assert result.is_final is False
    assert result.revision == 3
    assert result.start_time_ms == 1200.0
    assert result.end_time_ms == 2500.0
    assert result.monotonic_timestamp == 1000.5
    assert result.detected_language == "en"
    assert result.confidence == 0.91
    assert result.is_endpoint is False


def test_translation_types_contain_revision_and_stream_metadata():
    """Verify TranslationRequest and TranslationResult support revision tracking and finality."""
    req = TranslationRequest(
        text="Hello world",
        source_lang="en",
        target_lang="vi",
        stream_id="system",
        sequence=10,
        revision=2,
        is_final=True,
    )
    assert req.stream_id == "system"
    assert req.sequence == 10
    assert req.revision == 2
    assert req.is_final is True

    res = TranslationResult(
        translated_text="Xin chao the gioi",
        source_text="Hello world",
        source_lang="en",
        target_lang="vi",
        stream_id="system",
        sequence=10,
        revision=2,
        is_final=True,
        detected_source_lang="en",
        provider_event_id="trans-10",
        input_tokens=2,
        output_tokens=4,
    )
    assert res.revision == 2
    assert res.is_final is True
    assert res.detected_source_lang == "en"
    assert res.input_tokens == 2
    assert res.output_tokens == 4


def test_assist_result_requires_both_output_groups():
    """Verify AssistResult contains both suggested_replies and clarifying_questions."""
    res = AssistResult(
        target_message_id="msg-1",
        suggested_replies=(
            AssistSuggestion(strategy="Cooperative", text="Agreed.", reasoning="Builds consensus"),
        ),
        clarifying_questions=(
            AssistQuestion(question="What is the budget?", purpose="Budget alignment"),
        ),
    )
    assert len(res.suggested_replies) == 1
    assert res.suggested_replies[0].strategy == "Cooperative"
    assert len(res.clarifying_questions) == 1
    assert res.clarifying_questions[0].question == "What is the budget?"


def test_summary_warning_tracks_omitted_context():
    """Verify SummaryWarning captures omitted count and timestamps for context overflow."""
    warning = SummaryWarning(
        omitted_message_count=5,
        first_omitted_timestamp_ms=100.0,
        last_omitted_timestamp_ms=500.0,
        reason="Context overflow",
    )
    res = SummaryResult(
        summary_markdown="Summary text",
        bullet_points=("Bullet 1",),
        decisions=(),
        action_items=(),
        risks=(),
        warning=warning,
    )
    assert res.warning is not None
    assert res.warning.omitted_message_count == 5
    assert res.warning.first_omitted_timestamp_ms == 100.0
    assert res.warning.last_omitted_timestamp_ms == 500.0


# ── Categorized Error & Security Tests ───────────────────────────────────────


def test_categorized_provider_errors():
    """Verify provider errors have stable code, capability, retryability, and safe message."""
    error = ProviderError(
        code=ProviderErrorCode.RATE_LIMITED,
        capability=ProviderCapability.STT,
        message="STT provider rate limit reached. Backing off.",
        retryable=True,
        details={"retry_after_s": 5},
    )
    assert error.code == ProviderErrorCode.RATE_LIMITED
    assert error.capability == ProviderCapability.STT
    assert error.retryable is True
    assert str(error) == "[stt:rate_limited] STT provider rate limit reached. Backing off."

    exc = ProviderException(error)
    assert exc.error is error
    assert "[stt:rate_limited]" in str(exc)


def test_provider_error_does_not_contain_raw_config_or_request_fields():
    """Verify that ProviderError does not have raw request or config attributes."""
    fields = {f.name for f in ProviderError.__dataclass_fields__.values()}
    forbidden_terms = {
        "config",
        "raw_config",
        "request",
        "raw_request",
        "api_key",
        "secret",
        "headers",
    }
    assert fields.isdisjoint(forbidden_terms), (
        f"Found forbidden field in ProviderError: {fields & forbidden_terms}"
    )


# ── Deterministic Fake Provider Execution Tests ──────────────────────────────


@pytest.mark.asyncio
async def test_fake_stt_provider_execution():
    """Verify FakeSTTProvider streams partial and final results with monotonic ordering."""
    provider = FakeSTTProvider()
    assert provider.capability == ProviderCapability.STT

    await provider.start()
    assert provider.started is True

    health = await provider.health_check()
    assert health.status == ProviderHealthStatus.HEALTHY

    frame = AudioFrame(
        stream_id="mic",
        sequence=1,
        pcm_data=b"\x00" * 320,
        sample_rate=16000,
        channels=1,
        duration_ms=10.0,
    )
    await provider.send_audio(frame)
    await provider.finalize_stream("mic")

    events: List[STTTranscriptResult] = []
    async for evt in provider.receive_events():
        events.append(evt)

    assert len(events) == 2
    partial, final = events[0], events[1]

    # Partial event assertions
    assert partial.is_final is False
    assert partial.revision == 1
    assert partial.text == "hello"
    assert partial.stream_id == "mic"

    # Final event assertions
    assert final.is_final is True
    assert final.revision == 2
    assert final.text == "hello world"
    assert final.is_endpoint is True
    assert len(final.words or ()) == 2

    await provider.stop()
    assert provider.stopped is True


@pytest.mark.asyncio
async def test_fake_translation_provider_execution():
    """Verify FakeTranslationProvider handles translation requests and error propagation."""
    provider = FakeTranslationProvider()
    assert provider.capability == ProviderCapability.TRANSLATION

    await provider.start()
    req = TranslationRequest(
        text="EchoFlux",
        source_lang="auto",
        target_lang="vi",
        stream_id="mic",
        sequence=1,
        revision=1,
        is_final=True,
    )
    res = await provider.translate(req)
    assert res.translated_text == "[vi] EchoFlux"
    assert res.detected_source_lang == "en"
    assert res.is_final is True

    # Error handling test
    bad_req = TranslationRequest(
        text="EchoFlux",
        source_lang="en",
        target_lang="unsupported",
    )
    with pytest.raises(ProviderException) as exc_info:
        await provider.translate(bad_req)
    assert exc_info.value.error.code == ProviderErrorCode.UNSUPPORTED_LANGUAGE
    assert exc_info.value.error.retryable is False

    await provider.stop()
    assert provider.stopped is True


@pytest.mark.asyncio
async def test_fake_assistant_provider_execution():
    """Verify FakeAssistantProvider generates assist responses, summaries, and streams."""
    provider = FakeAssistantProvider()
    assert provider.capability == ProviderCapability.ASSISTANT

    await provider.start()

    # Assist
    msg = ConversationMessage(
        message_id="msg-101",
        speaker="Others",
        text="Could you clarify the budget for this phase?",
        timestamp_ms=1000.0,
    )
    req = AssistRequest(
        target_message=msg,
        context_messages=(msg,),
        context_profiles=(ContextProfileSnapshot("prof-1", "Budget Doc", "Cap is $10k"),),
    )
    assist_res = await provider.assist(req)
    assert assist_res.target_message_id == "msg-101"
    assert len(assist_res.suggested_replies) == 2
    assert len(assist_res.clarifying_questions) == 1

    # Summarize with overflow warning
    many_messages = tuple(
        ConversationMessage(
            message_id=f"msg-{i}",
            speaker="Me" if i % 2 == 0 else "Others",
            text=f"Line {i}",
            timestamp_ms=float(i * 1000),
        )
        for i in range(15)
    )
    sum_req = SummaryRequest(messages=many_messages)
    sum_res = await provider.summarize(sum_req)
    assert sum_res.warning is not None
    assert sum_res.warning.omitted_message_count == 5

    # Stream summary
    chunks: List[SummaryChunk] = []
    async for chunk in provider.stream_summary(sum_req):
        chunks.append(chunk)
    assert len(chunks) == 5
    assert "".join(c.delta_text for c in chunks) == "Meeting summary streamed text."
    assert chunks[-1].is_done is True

    await provider.stop()
    assert provider.stopped is True


# ── Import Cost & Zero-Heavy-SDK Isolation Tests ─────────────────────────────


def test_engine_providers_imports_cheaply_without_heavy_sdks():
    """Importing engine.providers must NOT import heavy ML or network packages."""
    code = inspect.cleandoc(
        """
        import sys
        import engine.providers

        forbidden = [
            "torch",
            "faster_whisper",
            "transformers",
            "ctranslate2",
            "openai",
            "websockets",
            "httpx",
            "aiohttp",
            "pyaudio",
            "soundcard",
            "numpy",
        ]
        loaded_forbidden = [pkg for pkg in forbidden if pkg in sys.modules]
        if loaded_forbidden:
            print("Loaded forbidden modules:", loaded_forbidden)
            sys.exit(1)
        sys.exit(0)
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"Importing engine.providers loaded forbidden modules:\n{result.stdout}\n{result.stderr}"
    )
