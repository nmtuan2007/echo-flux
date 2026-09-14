"""Fake-SDK lifecycle tests for the Azure Speech STT adapter.

No Azure account and no network: the SDK is an in-memory fake injected through
``sdk_loader``, and SDK events fire from real background threads so the callback bridge
is exercised the way the native SDK drives it.
"""

import asyncio
import functools
import json
import logging
import sys
import threading
from enum import Enum
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional

import pytest

from engine.providers.stt import azure_speech as azure
from engine.providers.stt.azure_speech import AzureSpeechSTTConfig, AzureSpeechSTTProvider
from engine.providers.types import (
    AudioFrame,
    ProviderError,
    ProviderErrorCode,
    ProviderHealthStatus,
    ProviderRuntimeError,
)

SECRET = "azure-TEST-secret-key-0123456789abcdef"
PCM_100MS = b"\x01\x00" * 1600  # 100 ms of 16 kHz PCM16 mono
TICKS_PER_MS = 10_000
RESOURCE_ENDPOINT = "https://myspeech.cognitiveservices.azure.com/"


# ── Fake SDK ─────────────────────────────────────────────────────────────────


class ResultReason(Enum):
    RecognizingSpeech = 1
    RecognizedSpeech = 2
    NoMatch = 3
    Canceled = 4


class CancellationReason(Enum):
    Error = 1
    EndOfStream = 2
    CancelledByUser = 3


class CancellationErrorCode(Enum):
    NoError = 0
    AuthenticationFailure = 1
    BadRequest = 2
    TooManyRequests = 3
    Forbidden = 4
    ConnectionFailure = 5
    ServiceTimeout = 6
    ServiceError = 7
    ServiceUnavailable = 8
    RuntimeError = 9
    ServiceRedirectTemporary = 10
    ServiceRedirectPermanent = 11
    EmbeddedModelError = 12


class PropertyId(Enum):
    SpeechServiceConnection_LanguageIdMode = 1


RECOGNIZER_SIGNALS = ("session_started", "recognizing", "recognized", "canceled", "session_stopped")
STOPPED = SimpleNamespace(session_id="session")


def run_in_thread(step: Callable[[], None]) -> None:
    worker = threading.Thread(target=step)
    worker.start()
    worker.join()


class FakeSignal:
    def __init__(self) -> None:
        self.callbacks: List[Callable[[Any], None]] = []

    def connect(self, callback: Callable[[Any], None]) -> None:
        self.callbacks.append(callback)

    def disconnect_all(self) -> None:
        self.callbacks.clear()

    def fire(self, evt: Any) -> None:
        for callback in list(self.callbacks):
            callback(evt)


class FakeFuture:
    def __init__(self, step: Callable[[], Any]) -> None:
        self._step = step

    def get(self) -> Any:
        return self._step()


class FakeSpeechConfig:
    def __init__(
        self,
        sdk: "FakeSpeechSDK",
        subscription: Optional[str] = None,
        region: Optional[str] = None,
        endpoint: Optional[str] = None,
    ) -> None:
        sdk.record_thread()
        self.subscription = subscription
        self.region = region
        self.endpoint = endpoint
        self.speech_recognition_language: Optional[str] = None
        self.properties: Dict[Any, str] = {}
        sdk.configs.append(self)

    def set_property(self, property_id: Any, value: str) -> None:
        self.properties[property_id] = value


class FakeStreamFormat:
    def __init__(self, samples_per_second: int, bits_per_sample: int, channels: int) -> None:
        self.samples_per_second = samples_per_second
        self.bits_per_sample = bits_per_sample
        self.channels = channels


class FakePushStream:
    def __init__(
        self, sdk: "FakeSpeechSDK", stream_format: Optional[FakeStreamFormat] = None
    ) -> None:
        self.sdk = sdk
        self.stream_format = stream_format
        self.chunks: List[bytes] = []
        self.closed = False
        self.recognizer: Optional[FakeRecognizer] = None
        sdk.push_streams.append(self)

    def write(self, buffer: bytes) -> None:
        if self.closed:
            raise RuntimeError("stream closed")
        self.chunks.append(bytes(buffer))

    def close(self) -> None:
        already_closed = self.closed
        self.closed = True
        if not already_closed and self.recognizer is not None:
            self.sdk.on_input_closed(self.recognizer)


class FakeAudioConfig:
    def __init__(self, stream: FakePushStream) -> None:
        self.stream = stream


class FakeAutoDetectConfig:
    def __init__(self, languages: List[str]) -> None:
        self.languages = list(languages)


class FakeRecognizer:
    def __init__(
        self,
        sdk: "FakeSpeechSDK",
        speech_config: FakeSpeechConfig,
        audio_config: FakeAudioConfig,
        auto_detect_source_language_config: Optional[FakeAutoDetectConfig] = None,
    ) -> None:
        sdk.record_thread()
        self.sdk = sdk
        self.speech_config = speech_config
        self.push = audio_config.stream
        self.push.recognizer = self
        self.auto_detect = auto_detect_source_language_config
        for name in RECOGNIZER_SIGNALS:
            setattr(self, name, FakeSignal())
        self.started = False
        self.stop_calls = 0
        self.pending_final: Optional[str] = None
        sdk.recognizers.append(self)

    def start_continuous_recognition_async(self) -> FakeFuture:
        def run() -> None:
            self.sdk.record_thread()
            if self.sdk.start_error is not None:
                raise self.sdk.start_error
            self.started = True
            self.session_started.fire(STOPPED)

        return FakeFuture(run)

    def stop_continuous_recognition_async(self) -> FakeFuture:
        def run() -> None:
            self.sdk.record_thread()
            self.stop_calls += 1

        return FakeFuture(run)

    def fire(self, *steps: Any) -> None:
        """Fire events from a separate thread, as the SDK does."""

        def run() -> None:
            for name, evt in steps:
                getattr(self, name).fire(evt)

        run_in_thread(run)

    def callback_count(self) -> int:
        return sum(len(getattr(self, name).callbacks) for name in RECOGNIZER_SIGNALS)


class FakeConnection:
    def __init__(self, sdk: "FakeSpeechSDK", recognizer: FakeRecognizer) -> None:
        self.sdk = sdk
        self.recognizer = recognizer
        self.connected = FakeSignal()
        self.disconnected = FakeSignal()
        self.opened = False
        self.closed = False
        sdk.connections.append(self)

    def open(self, for_continuous_recognition: bool) -> None:
        self.opened = for_continuous_recognition
        self.sdk.on_connection_open(self)

    def close(self) -> None:
        self.closed = True


def end_of_stream(recognizer: FakeRecognizer) -> None:
    steps: List[Any] = []
    if recognizer.pending_final is not None:
        steps.append(("recognized", recognized(recognizer.pending_final, duration_ms=500)))
    steps.append(("canceled", canceled(CancellationReason.EndOfStream)))
    steps.append(("session_stopped", STOPPED))
    recognizer.fire(*steps)


def connect_ok(connection: FakeConnection) -> None:
    run_in_thread(lambda: connection.connected.fire(SimpleNamespace()))


class FakeSpeechSDK:
    def __init__(self) -> None:
        self.configs: List[FakeSpeechConfig] = []
        self.recognizers: List[FakeRecognizer] = []
        self.push_streams: List[FakePushStream] = []
        self.connections: List[FakeConnection] = []
        self.sdk_threads: set = set()
        self.start_error: Optional[BaseException] = None
        self.on_input_closed: Callable[[FakeRecognizer], None] = end_of_stream
        self.on_connection_open: Callable[[FakeConnection], None] = connect_ok
        self.ResultReason = ResultReason
        self.CancellationReason = CancellationReason
        self.CancellationErrorCode = CancellationErrorCode
        self.PropertyId = PropertyId
        self.SpeechConfig = functools.partial(FakeSpeechConfig, self)
        self.SpeechRecognizer = functools.partial(FakeRecognizer, self)
        self.audio = SimpleNamespace(
            AudioStreamFormat=FakeStreamFormat,
            PushAudioInputStream=functools.partial(FakePushStream, self),
            AudioConfig=FakeAudioConfig,
        )
        self.languageconfig = SimpleNamespace(AutoDetectSourceLanguageConfig=FakeAutoDetectConfig)
        self.AutoDetectSourceLanguageResult = lambda result: SimpleNamespace(
            language=getattr(result, "language", None)
        )
        self.Connection = SimpleNamespace(from_recognizer=functools.partial(FakeConnection, self))

    def record_thread(self) -> None:
        self.sdk_threads.add(threading.get_ident())


# ── Helpers ──────────────────────────────────────────────────────────────────


def result_event(
    text: str,
    reason: ResultReason,
    offset_ms: float = 0.0,
    duration_ms: float = 0.0,
    language: Optional[str] = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        result=SimpleNamespace(
            text=text,
            reason=reason,
            offset=int(offset_ms * TICKS_PER_MS),
            duration=int(duration_ms * TICKS_PER_MS),
            language=language,
        )
    )


def recognizing(text: str, **kwargs: Any) -> SimpleNamespace:
    return result_event(text, ResultReason.RecognizingSpeech, **kwargs)


def recognized(text: str, **kwargs: Any) -> SimpleNamespace:
    return result_event(text, ResultReason.RecognizedSpeech, **kwargs)


def no_match() -> SimpleNamespace:
    return result_event("", ResultReason.NoMatch)


def canceled(
    reason: CancellationReason,
    code: Optional[CancellationErrorCode] = None,
    details: str = "",
) -> SimpleNamespace:
    return SimpleNamespace(
        cancellation_details=SimpleNamespace(reason=reason, code=code, error_details=details)
    )


def frame(
    sequence: int,
    pcm: bytes = PCM_100MS,
    stream_id: str = "mic",
    sample_rate: int = 16000,
    channels: int = 1,
) -> AudioFrame:
    return AudioFrame(
        stream_id=stream_id,
        sequence=sequence,
        pcm_data=pcm,
        sample_rate=sample_rate,
        channels=channels,
    )


def make_provider(sdk: Optional[FakeSpeechSDK], **overrides: Any) -> AzureSpeechSTTProvider:
    options = {
        "api_key": SECRET,
        "region": "westus",
        "recognition_language": "en-US",
        "start_timeout_s": 1.0,
        "stop_timeout_s": 1.0,
        "finalize_timeout_s": 1.0,
        "health_timeout_s": 1.0,
        **overrides,
    }
    return AzureSpeechSTTProvider(
        AzureSpeechSTTConfig.from_options(options), sdk_loader=lambda: sdk
    )


async def wait_until(predicate: Callable[[], bool], timeout: float = 2.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() > deadline:
            raise AssertionError("condition not reached")
        await asyncio.sleep(0.005)


def drain(provider: AzureSpeechSTTProvider) -> List[Any]:
    queue = provider._events
    assert queue is not None
    items = []
    while not queue.empty():
        items.append(queue.get_nowait())
    return items


async def collect(provider: AzureSpeechSTTProvider, count: int) -> List[Any]:
    queue = provider._events
    assert queue is not None
    await wait_until(lambda: queue.qsize() >= count)
    return drain(provider)


async def started_provider(sdk: FakeSpeechSDK, **overrides: Any) -> AzureSpeechSTTProvider:
    provider = make_provider(sdk, **overrides)
    await provider.start()
    await provider.send_audio(frame(1))
    return provider


# ── SDK availability ─────────────────────────────────────────────────────────


def test_module_import_does_not_load_sdk():
    assert azure.SDK_MODULE not in sys.modules


@pytest.mark.parametrize("failure", ["missing", "native"])
async def test_unavailable_sdk_reports_safe_provider_error(failure, monkeypatch, caplog):
    caplog.set_level(logging.DEBUG)
    config = AzureSpeechSTTConfig(api_key=SECRET, region="westus", recognition_language="en-US")
    if failure == "missing":
        monkeypatch.setitem(sys.modules, azure.SDK_MODULE, None)
        provider = AzureSpeechSTTProvider(config)
    else:

        def broken_loader() -> Any:
            raise OSError(f"native runtime failed to load {SECRET}")

        provider = AzureSpeechSTTProvider(config, sdk_loader=broken_loader)

    with pytest.raises(ProviderRuntimeError) as exc_info:
        await provider.start()
    error = exc_info.value.error
    assert error.code is ProviderErrorCode.MODEL_UNAVAILABLE
    assert error.retryable is False
    assert error.details == {"dependency": "azure-cognitiveservices-speech"}

    health = await provider.health_check()
    assert health.status is ProviderHealthStatus.UNHEALTHY
    assert health.details == {"code": "model_unavailable", "retryable": False}
    assert SECRET not in caplog.text and SECRET not in str(error)


# ── Configuration ────────────────────────────────────────────────────────────

ELEVEN_LOCALES = [
    "en-US",
    "de-DE",
    "fr-FR",
    "es-ES",
    "ja-JP",
    "zh-CN",
    "ko-KR",
    "it-IT",
    "pt-BR",
    "ru-RU",
    "nl-NL",
]
REGIONAL_V2 = "wss://westus.stt.speech.microsoft.com/speech/universal/v2"

CONFIG_ERRORS = [
    ({"api_key": ""}, ProviderErrorCode.AUTH_FAILED, "credential is missing"),
    ({"region": ""}, ProviderErrorCode.INVALID_REQUEST, "region or an endpoint"),
    ({"region": "West US"}, ProviderErrorCode.INVALID_REQUEST, "valid region name"),
    ({"region": "", "endpoint": REGIONAL_V2}, ProviderErrorCode.INVALID_REQUEST, "is required"),
    ({"region": "eastus", "endpoint": REGIONAL_V2}, ProviderErrorCode.INVALID_REQUEST, "match"),
    (
        {"endpoint": "http://westus.api.cognitive.microsoft.com/"},
        ProviderErrorCode.INVALID_REQUEST,
        "https or wss",
    ),
    (
        {"endpoint": "https://user:pw@myspeech.cognitiveservices.azure.com/"},
        ProviderErrorCode.INVALID_REQUEST,
        "credentials",
    ),
    (
        {"endpoint": RESOURCE_ENDPOINT + "?Subscription-Key=abc"},
        ProviderErrorCode.INVALID_REQUEST,
        "credentials",
    ),
    (
        {"endpoint": "https://speech.proxy.test/stt"},
        ProviderErrorCode.INVALID_REQUEST,
        "Custom Azure Speech endpoints",
    ),
    ({"recognition_language": "english"}, ProviderErrorCode.UNSUPPORTED_LANGUAGE, "locales"),
    (
        {"recognition_language": "", "candidate_languages": ELEVEN_LOCALES[:5]},
        ProviderErrorCode.INVALID_REQUEST,
        "at most 4",
    ),
    (
        {"recognition_language": "", "candidate_languages": ["en-US", "en-GB"]},
        ProviderErrorCode.INVALID_REQUEST,
        "one locale per base language",
    ),
    (
        {
            "recognition_language": "",
            "candidate_languages": ["en-US", "de-DE"],
            "language_id_mode": "Continuous",
        },
        ProviderErrorCode.INVALID_REQUEST,
        "requires an Azure Speech endpoint",
    ),
    (
        {
            "recognition_language": "",
            "candidate_languages": ELEVEN_LOCALES,
            "language_id_mode": "Continuous",
            "endpoint": RESOURCE_ENDPOINT,
        },
        ProviderErrorCode.INVALID_REQUEST,
        "at most 10",
    ),
    ({"candidate_languages": ["de-DE"]}, ProviderErrorCode.INVALID_REQUEST, "not both"),
    ({"recognition_language": ""}, ProviderErrorCode.INVALID_REQUEST, "candidate languages"),
    ({"language_id_mode": "Sometimes"}, ProviderErrorCode.INVALID_REQUEST, "AtStart"),
    ({"finalize_timeout_s": -1}, ProviderErrorCode.INVALID_REQUEST, "positive"),
]


@pytest.mark.parametrize(("overrides", "code", "fragment"), CONFIG_ERRORS)
async def test_config_errors_are_caught_before_streaming(overrides, code, fragment):
    sdk = FakeSpeechSDK()
    loads: List[int] = []

    def loader() -> FakeSpeechSDK:
        loads.append(1)
        return sdk

    options = {"api_key": SECRET, "region": "westus", "recognition_language": "en-US"}
    options.update(overrides)
    provider = AzureSpeechSTTProvider(AzureSpeechSTTConfig.from_options(options), sdk_loader=loader)

    with pytest.raises(ProviderRuntimeError) as exc_info:
        await provider.start()
    error = exc_info.value.error
    assert error.code is code
    assert fragment in error.message
    assert SECRET not in str(error) and SECRET not in json.dumps(error.details)
    assert loads == [] and sdk.recognizers == []

    with pytest.raises(ProviderRuntimeError):
        await provider.send_audio(frame(1))
    assert sdk.recognizers == []


def test_region_mismatch_reports_both_regions():
    config = AzureSpeechSTTConfig(
        api_key=SECRET, region="EastUS", endpoint=REGIONAL_V2, recognition_language="en-US"
    )
    with pytest.raises(ProviderRuntimeError) as exc_info:
        azure.validate_config(config)
    assert exc_info.value.error.details == {"region": "eastus", "endpoint_region": "westus"}


@pytest.mark.parametrize(
    ("overrides", "kind"),
    [
        ({}, azure.ENDPOINT_KIND_REGION),
        ({"region": "WestUS", "endpoint": REGIONAL_V2}, azure.ENDPOINT_KIND_REGIONAL),
        (
            {"region": "westus", "endpoint": "https://westus.api.cognitive.microsoft.com/"},
            azure.ENDPOINT_KIND_REGIONAL,
        ),
        ({"region": "", "endpoint": RESOURCE_ENDPOINT}, azure.ENDPOINT_KIND_RESOURCE),
        (
            {"endpoint": "wss://speech.proxy.test/stt", "allow_custom_endpoint": True},
            azure.ENDPOINT_KIND_CUSTOM,
        ),
        (
            {
                "recognition_language": "",
                "candidate_languages": ELEVEN_LOCALES[:10],
                "language_id_mode": "Continuous",
                "endpoint": RESOURCE_ENDPOINT,
            },
            azure.ENDPOINT_KIND_RESOURCE,
        ),
    ],
)
def test_valid_targets(overrides, kind):
    options = {"api_key": SECRET, "region": "westus", "recognition_language": "en-US"}
    options.update(overrides)
    target = azure.validate_config(AzureSpeechSTTConfig.from_options(options))
    assert target.endpoint_kind == kind


def test_target_description_and_config_repr_hide_secrets():
    config = AzureSpeechSTTConfig.from_options(
        {
            "api_key": SECRET,
            "region": "",
            "endpoint": RESOURCE_ENDPOINT + "stt?format=detailed",
            "language_hints": "en-US, vi-VN",
            "language_id_mode": "Continuous",
            "allow_custom_endpoint": True,
            "unknown": 1,
        }
    )
    assert config.candidate_languages == ("en-US", "vi-VN")
    assert config.allow_custom_endpoint is True
    assert SECRET not in repr(config) and "format=detailed" not in repr(config)
    description = azure.validate_config(config).describe()
    assert "myspeech.cognitiveservices.azure.com" in description
    assert "format=detailed" not in description and SECRET not in description


# ── Recognizer construction ──────────────────────────────────────────────────


async def test_region_config_builds_recognizer_lazily_off_the_loop():
    sdk = FakeSpeechSDK()
    provider = make_provider(sdk)
    await provider.start()
    assert sdk.recognizers == []

    await provider.send_audio(frame(1))
    await provider.send_audio(frame(2))

    [config] = sdk.configs
    assert (config.subscription, config.region, config.endpoint) == (SECRET, "westus", None)
    assert config.speech_recognition_language == "en-US"
    assert config.properties == {}
    [recognizer] = sdk.recognizers
    assert recognizer.started and recognizer.auto_detect is None
    stream_format = recognizer.push.stream_format
    assert (
        stream_format.samples_per_second,
        stream_format.bits_per_sample,
        stream_format.channels,
    ) == (16000, 16, 1)
    assert recognizer.push.chunks == [PCM_100MS, PCM_100MS]
    assert threading.get_ident() not in sdk.sdk_threads
    await provider.stop()


async def test_continuous_language_id_uses_endpoint_property_and_candidates():
    sdk = FakeSpeechSDK()
    provider = await started_provider(
        sdk,
        region="",
        endpoint=RESOURCE_ENDPOINT,
        recognition_language="",
        candidate_languages="en-US,vi-VN",
        language_id_mode="Continuous",
    )
    [config] = sdk.configs
    assert (config.region, config.endpoint) == (None, RESOURCE_ENDPOINT)
    assert config.speech_recognition_language is None
    assert config.properties == {PropertyId.SpeechServiceConnection_LanguageIdMode: "Continuous"}
    recognizer = sdk.recognizers[0]
    assert recognizer.auto_detect.languages == ["en-US", "vi-VN"]

    recognizer.fire(("recognized", recognized("Xin chào.", duration_ms=400, language="vi-VN")))
    [result] = await collect(provider, 1)
    assert result.detected_language == "vi-VN" and result.is_final
    await provider.stop()


async def test_start_failure_releases_resources_without_leaking(caplog):
    caplog.set_level(logging.DEBUG)
    sdk = FakeSpeechSDK()
    sdk.start_error = RuntimeError(f"Exception with error code 0x5 key={SECRET}")
    provider = make_provider(sdk)
    await provider.start()

    with pytest.raises(ProviderRuntimeError) as exc_info:
        await provider.send_audio(frame(1))
    error = exc_info.value.error
    assert error.code is ProviderErrorCode.INTERNAL_ERROR
    assert error.details == {"exception": "RuntimeError"}
    [recognizer] = sdk.recognizers
    assert recognizer.push.closed and recognizer.callback_count() == 0
    assert provider._sessions == {}
    assert SECRET not in caplog.text and SECRET not in str(error)

    sdk.start_error = None
    await provider.send_audio(frame(2))
    assert len(sdk.recognizers) == 2 and sdk.recognizers[1].started
    await provider.stop()


async def test_frame_validation_and_lifecycle_guards():
    sdk = FakeSpeechSDK()
    provider = make_provider(sdk)
    with pytest.raises(ProviderRuntimeError) as exc_info:
        await provider.send_audio(frame(1))
    assert exc_info.value.error.code is ProviderErrorCode.INVALID_REQUEST

    await provider.start()
    for bad in (frame(1, channels=2), frame(1, pcm=b"\x00"), frame(1, sample_rate=0)):
        with pytest.raises(ProviderRuntimeError) as exc_info:
            await provider.send_audio(bad)
        assert exc_info.value.error.code is ProviderErrorCode.INVALID_REQUEST
    await provider.send_audio(frame(1, pcm=b""))
    assert sdk.recognizers == []
    await provider.stop()


# ── Callback mapping ─────────────────────────────────────────────────────────


async def test_callback_order_and_finality_mapping_are_deterministic():
    sdk = FakeSpeechSDK()
    provider = make_provider(sdk)
    await provider.start()
    await provider.send_audio(frame(7))
    recognizer = sdk.recognizers[0]

    recognizer.fire(
        ("recognizing", recognizing("hel", offset_ms=100, duration_ms=200)),
        ("recognizing", recognizing("hel", offset_ms=100, duration_ms=200)),
        ("recognizing", recognizing("hello", offset_ms=100, duration_ms=400)),
        ("recognized", recognized("Hello world.", offset_ms=100, duration_ms=900)),
        ("recognizing", recognizing("next", offset_ms=1500, duration_ms=100)),
        ("recognized", no_match()),
        ("recognized", no_match()),
    )
    results = await collect(provider, 5)
    await asyncio.sleep(0.02)
    assert drain(provider) == []

    assert [(r.text, r.is_final, r.revision, r.provider_event_id) for r in results] == [
        ("hel", False, 1, "azure:mic:1:0"),
        ("hello", False, 2, "azure:mic:1:0"),
        ("Hello world.", True, 3, "azure:mic:1:0"),
        ("next", False, 4, "azure:mic:1:1"),
        ("", True, 5, "azure:mic:1:1"),
    ]
    assert [r.is_endpoint for r in results] == [False, False, True, False, True]
    assert (results[2].start_time_ms, results[2].end_time_ms) == (100.0, 1000.0)
    assert (results[4].start_time_ms, results[4].end_time_ms) == (1500.0, 1600.0)
    assert all(r.sequence == 7 and r.stream_id == "mic" for r in results)
    assert all(r.detected_language is None for r in results)
    await provider.stop()


async def test_burst_of_callbacks_keeps_arrival_order():
    sdk = FakeSpeechSDK()
    provider = await started_provider(sdk)
    recognizer = sdk.recognizers[0]
    steps = [("recognizing", recognizing("w" * (i + 1), duration_ms=i)) for i in range(50)]
    recognizer.fire(*steps)
    results = await collect(provider, 50)
    assert [len(r.text) for r in results] == list(range(1, 51))
    assert [r.revision for r in results] == list(range(1, 51))
    await provider.stop()


@pytest.mark.parametrize(
    ("name", "code", "retryable"),
    [
        ("AuthenticationFailure", ProviderErrorCode.AUTH_FAILED, False),
        ("Forbidden", ProviderErrorCode.AUTH_FAILED, False),
        ("TooManyRequests", ProviderErrorCode.RATE_LIMITED, True),
        ("BadRequest", ProviderErrorCode.INVALID_REQUEST, False),
        ("ConnectionFailure", ProviderErrorCode.NETWORK_UNAVAILABLE, True),
        ("ServiceTimeout", ProviderErrorCode.TIMEOUT, True),
        ("ServiceUnavailable", ProviderErrorCode.NETWORK_UNAVAILABLE, True),
        ("ServiceError", ProviderErrorCode.INTERNAL_ERROR, True),
        ("ServiceRedirectTemporary", ProviderErrorCode.CONNECTION_CLOSED, True),
        ("ServiceRedirectPermanent", ProviderErrorCode.CONNECTION_CLOSED, False),
        ("EmbeddedModelError", ProviderErrorCode.MODEL_UNAVAILABLE, False),
        ("RuntimeError", ProviderErrorCode.INTERNAL_ERROR, False),
        ("NoError", ProviderErrorCode.INTERNAL_ERROR, False),
        ("", ProviderErrorCode.INTERNAL_ERROR, False),
    ],
)
def test_cancellation_error_mapping(name, code, retryable):
    error = azure.map_cancellation_error(name)
    assert (error.code, error.retryable) == (code, retryable)
    assert error.details == {"provider_code": name or "unknown"}


def test_enum_name_only_passes_identifiers():
    assert azure._enum_name(CancellationErrorCode.ServiceTimeout) == "ServiceTimeout"
    assert azure._enum_name("CancellationErrorCode.Forbidden") == "Forbidden"
    assert azure._enum_name(f"key {SECRET}") == ""
    assert azure._enum_name(None) == ""


async def test_auth_cancellation_is_categorized_and_sticky(caplog):
    caplog.set_level(logging.DEBUG)
    sdk = FakeSpeechSDK()
    provider = await started_provider(sdk)
    recognizer = sdk.recognizers[0]
    events = provider.receive_events()

    recognizer.fire(("recognizing", recognizing("partial text", duration_ms=300)))
    partial = await asyncio.wait_for(events.__anext__(), 1.0)
    assert not partial.is_final

    details = f"WebSocket upgrade failed: key {SECRET} wss://westus.stt.speech.microsoft.com"
    recognizer.fire(
        (
            "canceled",
            canceled(
                CancellationReason.Error, CancellationErrorCode.AuthenticationFailure, details
            ),
        ),
        ("session_stopped", STOPPED),
    )
    promoted = await asyncio.wait_for(events.__anext__(), 1.0)
    assert (promoted.text, promoted.is_final, promoted.revision) == ("partial text", True, 2)
    with pytest.raises(ProviderRuntimeError) as exc_info:
        await asyncio.wait_for(events.__anext__(), 1.0)
    error = exc_info.value.error
    assert error.code is ProviderErrorCode.AUTH_FAILED and error.retryable is False
    assert error.details == {"provider_code": "AuthenticationFailure"}

    await wait_until(lambda: recognizer.stop_calls == 1 and recognizer.callback_count() == 0)
    assert recognizer.push.closed
    with pytest.raises(ProviderRuntimeError) as exc_info:
        await provider.send_audio(frame(2))
    assert exc_info.value.error.code is ProviderErrorCode.AUTH_FAILED
    assert len(sdk.recognizers) == 1
    assert SECRET not in caplog.text and SECRET not in json.dumps(error.details)
    await provider.stop()


async def test_retryable_cancellation_reopens_on_next_audio():
    sdk = FakeSpeechSDK()
    provider = await started_provider(sdk)
    first = sdk.recognizers[0]
    first.fire(
        (
            "canceled",
            canceled(CancellationReason.Error, CancellationErrorCode.ServiceTimeout),
        )
    )
    [error] = await collect(provider, 1)
    assert isinstance(error, ProviderError)
    assert error.code is ProviderErrorCode.TIMEOUT and error.retryable
    await wait_until(lambda: first.stop_calls == 1 and first.callback_count() == 0)

    await provider.send_audio(frame(2))
    assert len(sdk.recognizers) == 2 and sdk.recognizers[1].started
    await provider.stop()


async def test_unexpected_session_stop_promotes_partial_and_reopens():
    sdk = FakeSpeechSDK()
    provider = await started_provider(sdk)
    first = sdk.recognizers[0]
    first.fire(
        ("recognizing", recognizing("cut off", duration_ms=200)),
        ("session_stopped", STOPPED),
    )
    partial, final = await collect(provider, 2)
    assert (partial.is_final, final.text, final.is_final) == (False, "cut off", True)
    assert partial.provider_event_id == final.provider_event_id
    await wait_until(lambda: first.callback_count() == 0)

    await provider.send_audio(frame(2))
    assert len(sdk.recognizers) == 2
    await provider.stop()


# ── Finalize, reset, shutdown ────────────────────────────────────────────────


async def test_finalize_flushes_releases_and_keeps_timeline_monotonic():
    sdk = FakeSpeechSDK()
    provider = await started_provider(sdk)
    await provider.send_audio(frame(2))
    first = sdk.recognizers[0]
    first.fire(("recognizing", recognizing("good morn", duration_ms=150)))
    await collect(provider, 1)

    first.pending_final = "Good morning."
    await provider.finalize_stream("mic")
    [final] = drain(provider)
    assert (final.text, final.is_final, final.revision) == ("Good morning.", True, 2)
    assert first.push.closed and first.stop_calls == 1 and first.callback_count() == 0
    assert provider._sessions == {}

    await provider.send_audio(frame(3))
    second = sdk.recognizers[1]
    second.fire(("recognized", recognized("Again.", offset_ms=50, duration_ms=100)))
    [again] = await collect(provider, 1)
    assert again.revision == 3
    assert again.provider_event_id == "azure:mic:2:0"
    assert (again.start_time_ms, again.end_time_ms) == (250.0, 350.0)
    assert again.sequence == 3
    await provider.stop()


async def test_finalize_timeout_promotes_last_partial():
    sdk = FakeSpeechSDK()
    sdk.on_input_closed = lambda recognizer: None
    provider = await started_provider(sdk, finalize_timeout_s=0.05)
    recognizer = sdk.recognizers[0]
    recognizer.fire(("recognizing", recognizing("half a sent", duration_ms=300)))
    await collect(provider, 1)

    await provider.finalize_stream("mic")
    [final] = drain(provider)
    assert (final.text, final.is_final, final.revision, final.end_time_ms) == (
        "half a sent",
        True,
        2,
        300.0,
    )
    assert recognizer.push.closed and recognizer.stop_calls == 1
    assert recognizer.callback_count() == 0
    await provider.stop()


async def test_finalize_unknown_stream_is_noop():
    sdk = FakeSpeechSDK()
    provider = make_provider(sdk)
    await provider.start()
    await provider.finalize_stream("system")
    await provider.reset_stream("system")
    assert sdk.recognizers == []
    await provider.stop()


async def test_reset_discards_partial_and_ignores_late_events():
    sdk = FakeSpeechSDK()
    provider = await started_provider(sdk)
    first = sdk.recognizers[0]
    first.fire(("recognizing", recognizing("drop me", duration_ms=100)))
    await collect(provider, 1)

    await provider.reset_stream("mic")
    assert first.push.closed and first.stop_calls == 1 and first.callback_count() == 0
    assert provider._bridge is not None
    provider._bridge.post(azure._Signal(kind="recognized", generation=1, text="late"))
    await asyncio.sleep(0.02)
    assert drain(provider) == []

    await provider.send_audio(frame(2))
    sdk.recognizers[1].fire(("recognized", recognized("kept", duration_ms=100)))
    [kept] = await collect(provider, 1)
    assert (kept.text, kept.revision) == ("kept", 2)
    await provider.stop()


async def test_sample_rate_change_opens_new_recognizer():
    sdk = FakeSpeechSDK()
    provider = await started_provider(sdk)
    await provider.send_audio(frame(2, pcm=b"\x00\x00" * 800, sample_rate=8000))
    first, second = sdk.recognizers
    assert first.push.closed and first.callback_count() == 0
    assert second.push.stream_format.samples_per_second == 8000
    assert second.push.chunks == [b"\x00\x00" * 800]
    await provider.stop()


async def test_stop_releases_every_stream_and_ends_receive_events():
    sdk = FakeSpeechSDK()
    provider = make_provider(sdk)
    await provider.start()
    await provider.send_audio(frame(1, stream_id="mic"))
    await provider.send_audio(frame(1, stream_id="system"))
    assert len(sdk.recognizers) == 2
    events = provider.receive_events()

    await provider.stop()
    for recognizer in sdk.recognizers:
        assert recognizer.push.closed
        assert recognizer.stop_calls == 1
        assert recognizer.callback_count() == 0
    with pytest.raises(StopAsyncIteration):
        await asyncio.wait_for(events.__anext__(), 1.0)
    assert provider._tasks == set() and provider._sessions == {} and provider._bridge is None

    with pytest.raises(ProviderRuntimeError) as exc_info:
        await provider.send_audio(frame(2))
    assert exc_info.value.error.code is ProviderErrorCode.INVALID_REQUEST
    await provider.stop()


# ── Health check ─────────────────────────────────────────────────────────────


async def test_health_check_connects_and_releases():
    sdk = FakeSpeechSDK()
    provider = make_provider(sdk)
    health = await provider.health_check()
    assert health.status is ProviderHealthStatus.HEALTHY
    assert health.details == {"endpoint_kind": "region"}
    [connection] = sdk.connections
    assert connection.opened and connection.closed
    [recognizer] = sdk.recognizers
    assert recognizer.push.closed and recognizer.callback_count() == 0
    assert recognizer.stop_calls == 0
    assert connection.connected.callbacks == [] and provider._bridge is None


async def test_health_check_reports_auth_failure_without_secret(caplog):
    caplog.set_level(logging.DEBUG)
    sdk = FakeSpeechSDK()
    sdk.on_connection_open = lambda connection: connection.recognizer.fire(
        (
            "canceled",
            canceled(CancellationReason.Error, CancellationErrorCode.AuthenticationFailure, SECRET),
        )
    )
    provider = make_provider(sdk)
    health = await provider.health_check()
    assert health.status is ProviderHealthStatus.UNHEALTHY
    assert health.details == {"code": "auth_failed", "retryable": False}
    assert SECRET not in health.message and SECRET not in caplog.text
    assert sdk.connections[0].closed


async def test_health_check_timeout_and_config_error():
    sdk = FakeSpeechSDK()
    sdk.on_connection_open = lambda connection: None
    provider = make_provider(sdk, health_timeout_s=0.05)
    health = await provider.health_check()
    assert health.status is ProviderHealthStatus.UNHEALTHY
    assert health.details == {"code": "timeout", "retryable": True}
    assert sdk.connections[0].closed and sdk.recognizers[0].callback_count() == 0

    unconfigured = make_provider(FakeSpeechSDK(), region="")
    health = await unconfigured.health_check()
    assert health.details == {"code": "invalid_request", "retryable": False}
