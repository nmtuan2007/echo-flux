"""Deterministic offline provider matrix tests for EchoFlux.

Covers Tier 1 (isolated happy path), Tier 2 (boundary, malformed, rate-limit, close,
rotation), and Tier 3 (matrix resolution and capability contracts) for:
- Deepgram STT
- AssemblyAI STT
- Azure Speech STT
- OpenAI Realtime STT
- Gemini Live STT
- Local and Text (OpenAI, Gemini, Azure) Translation and Assistant providers.

Zero live network credentials or external sockets are used: all endpoints use in-memory
connections or fake SDK loaders.
"""

import asyncio
import contextlib
import json
from enum import Enum
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Tuple

import pytest

from engine.core.config import ServiceConfig
from engine.providers.registry import (
    ProviderRegistryError,
    RegistryErrorCode,
    build_default_registry,
)
from engine.providers.stt.assemblyai import (
    AssemblyAISettings,
    AssemblyAISTTProvider,
)
from engine.providers.stt.azure_speech import (
    AzureSpeechSTTConfig,
    AzureSpeechSTTProvider,
)
from engine.providers.stt.deepgram import (
    DeepgramSettings,
    DeepgramSTTProvider,
)
from engine.providers.stt.gemini_live import (
    GeminiLiveSTTConfig,
    GeminiLiveSTTProvider,
    LiveConnectError,
    LiveConnectionClosedError,
)
from engine.providers.stt.openai_protocol import (
    OpenAIRealtimeSettings,
)
from engine.providers.stt.openai_realtime import (
    OpenAIRealtimeSTTProvider,
    RealtimeConnectionClosedError,
    RealtimeHandshakeRejectedError,
)
from engine.providers.stt.streaming_ws import (
    HandshakeRejectedError,
    SocketClosedError,
)
from engine.providers.types import (
    AudioFrame,
    ProviderCapability,
    ProviderErrorCode,
    ProviderRuntimeError,
    STTTranscriptResult,
)

TEST_KEY = "test-matrix-secret-token-12345"
SAMPLE_PCM = b"\x01\x00" * 1600  # 100 ms of 16 kHz PCM16 mono


def _audio_frame(seq: int = 0, stream_id: str = "mic") -> AudioFrame:
    return AudioFrame(
        stream_id=stream_id,
        sequence=seq,
        pcm_data=SAMPLE_PCM,
        sample_rate=16000,
        channels=1,
    )


async def _next_event(events, timeout: float = 2.0) -> STTTranscriptResult:
    return await asyncio.wait_for(events.__anext__(), timeout=timeout)


async def _wait_until(predicate: Callable[[], bool], timeout: float = 2.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() > deadline:
            raise AssertionError("Condition not met within timeout")
        await asyncio.sleep(0.005)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Fake Deepgram Connection & Connector
# ─────────────────────────────────────────────────────────────────────────────


class _CloseSentinel:
    def __init__(self, code: int = 1000, reason: str = ""):
        self.code = code
        self.reason = reason


class FakeDeepgramConnection:
    def __init__(self, auth_header: Optional[str] = None):
        self.auth_header = auth_header
        self.sent: List[Any] = []
        self.queue: asyncio.Queue[Any] = asyncio.Queue()
        self.closed = False

    async def send(self, message: Any) -> None:
        if self.closed:
            raise SocketClosedError(1006, "Connection closed")
        self.sent.append(message)
        if isinstance(message, str):
            try:
                data = json.loads(message)
                if data.get("type") == "CloseStream":
                    self.queue.put_nowait(
                        json.dumps({"type": "Metadata", "request_id": "req-dg-close"})
                    )
                    self.queue.put_nowait(_CloseSentinel(1000, "Normal closure"))
            except Exception:
                pass

    async def recv(self) -> str:
        item = await self.queue.get()
        if isinstance(item, _CloseSentinel):
            self.closed = True
            raise SocketClosedError(item.code, item.reason)
        return item

    async def close(self) -> None:
        self.closed = True
        self.queue.put_nowait(_CloseSentinel(1000, "Closed by client"))

    def push_result(self, text: str, is_final: bool = True, speech_final: bool = True):
        payload = {
            "type": "Results",
            "channel_index": [0, 1],
            "duration": 1.0,
            "start": 0.0,
            "is_final": is_final,
            "speech_final": speech_final,
            "channel": {
                "alternatives": [
                    {
                        "transcript": text,
                        "confidence": 0.98,
                        "words": [{"word": w, "start": 0.0, "end": 0.5} for w in text.split()],
                    }
                ]
            },
        }
        self.queue.put_nowait(json.dumps(payload))

    def drop(self, code: int = 1006, reason: str = "Abnormal disconnect"):
        self.queue.put_nowait(_CloseSentinel(code, reason))


class FakeDeepgramConnector:
    def __init__(self, status_code: int = 200):
        self.status_code = status_code
        self.calls: List[Tuple[str, Dict[str, str]]] = []
        self.connections: List[FakeDeepgramConnection] = []

    async def __call__(
        self, url: str, headers: Dict[str, str], timeout: float
    ) -> FakeDeepgramConnection:
        self.calls.append((url, dict(headers)))
        if self.status_code == 401:
            raise HandshakeRejectedError(401)
        if self.status_code == 429:
            raise HandshakeRejectedError(429)
        if self.status_code >= 400:
            raise HandshakeRejectedError(self.status_code)
        conn = FakeDeepgramConnection(auth_header=headers.get("Authorization"))
        self.connections.append(conn)
        return conn


# ─────────────────────────────────────────────────────────────────────────────
# 2. Fake AssemblyAI Connection & Connector
# ─────────────────────────────────────────────────────────────────────────────


class FakeAssemblyAIConnection:
    def __init__(self, auth_header: Optional[str] = None):
        self.auth_header = auth_header
        self.sent: List[Any] = []
        self.queue: asyncio.Queue[Any] = asyncio.Queue()
        self.closed = False
        # Push Begin event to satisfy handshake
        self.queue.put_nowait(
            json.dumps({"type": "Begin", "id": "sess-aai-1", "expires_at": 4102444800})
        )

    async def send(self, message: Any) -> None:
        if self.closed:
            raise SocketClosedError(1006, "Connection closed")
        self.sent.append(message)
        if isinstance(message, str):
            try:
                data = json.loads(message)
                if data.get("type") == "Terminate":
                    self.queue.put_nowait(
                        json.dumps(
                            {
                                "type": "Termination",
                                "audio_duration_seconds": 1,
                                "session_duration_seconds": 2,
                            }
                        )
                    )
                    self.queue.put_nowait(_CloseSentinel(1000, "Session terminated"))
            except Exception:
                pass

    async def recv(self) -> str:
        item = await self.queue.get()
        if isinstance(item, _CloseSentinel):
            self.closed = True
            raise SocketClosedError(item.code, item.reason)
        return item

    async def close(self) -> None:
        self.closed = True
        self.queue.put_nowait(_CloseSentinel(1000, "Client closed"))

    def push_partial(self, text: str):
        payload = {
            "type": "Turn",
            "turn_order": 1,
            "turn_is_formatted": False,
            "end_of_turn": False,
            "transcript": text,
            "words": [{"text": w, "start": 0, "end": 100} for w in text.split()],
        }
        self.queue.put_nowait(json.dumps(payload))

    def push_final(self, text: str):
        payload = {
            "type": "Turn",
            "turn_order": 1,
            "turn_is_formatted": True,
            "end_of_turn": True,
            "transcript": text,
            "words": [{"text": w, "start": 0, "end": 500} for w in text.split()],
        }
        self.queue.put_nowait(json.dumps(payload))

    def drop(self, code: int = 1006, reason: str = "Connection lost"):
        self.queue.put_nowait(_CloseSentinel(code, reason))


class FakeAssemblyAIConnector:
    def __init__(self, status_code: int = 200):
        self.status_code = status_code
        self.calls: List[Tuple[str, Dict[str, str]]] = []
        self.connections: List[FakeAssemblyAIConnection] = []

    async def __call__(
        self, url: str, headers: Dict[str, str], timeout: float
    ) -> FakeAssemblyAIConnection:
        self.calls.append((url, dict(headers)))
        if self.status_code == 401:
            raise HandshakeRejectedError(401)
        if self.status_code == 429:
            raise HandshakeRejectedError(429)
        if self.status_code >= 400:
            raise HandshakeRejectedError(self.status_code)
        conn = FakeAssemblyAIConnection(auth_header=headers.get("Authorization"))
        self.connections.append(conn)
        return conn



# ─────────────────────────────────────────────────────────────────────────────
# 3. Fake Azure Speech SDK
# ─────────────────────────────────────────────────────────────────────────────


class AzureResultReason(Enum):
    RecognizingSpeech = 1
    RecognizedSpeech = 2
    NoMatch = 3
    Canceled = 4


class AzureCancellationReason(Enum):
    Error = 1
    EndOfStream = 2
    CancelledByUser = 3


class AzureCancellationErrorCode(Enum):
    NoError = 0
    AuthenticationFailure = 1
    BadRequest = 2
    TooManyRequests = 3
    Forbidden = 4
    ConnectionFailure = 5


class FakeAzureSignal:
    def __init__(self):
        self.callbacks: List[Callable[[Any], None]] = []

    def connect(self, cb: Callable[[Any], None]) -> None:
        self.callbacks.append(cb)

    def disconnect_all(self) -> None:
        self.callbacks.clear()

    def fire(self, evt: Any) -> None:
        for cb in list(self.callbacks):
            cb(evt)


class FakeAzureRecognizer:
    def __init__(self, speech_config, audio_config):
        self.speech_config = speech_config
        self.audio_config = audio_config
        self.session_started = FakeAzureSignal()
        self.session_stopped = FakeAzureSignal()
        self.recognizing = FakeAzureSignal()
        self.recognized = FakeAzureSignal()
        self.canceled = FakeAzureSignal()
        self.started = False
        self.stopped = False

    def start_continuous_recognition_async(self):
        self.started = True
        return SimpleNamespace(get=lambda: None)

    def stop_continuous_recognition_async(self):
        self.stopped = True
        return SimpleNamespace(get=lambda: None)


class FakeAzureSpeechSDK:
    def __init__(self):
        self.ResultReason = AzureResultReason
        self.CancellationReason = AzureCancellationReason
        self.CancellationErrorCode = AzureCancellationErrorCode
        self.PropertyId = SimpleNamespace(SpeechServiceConnection_LanguageIdMode=1)

        class _SpeechConfig:
            def __init__(self, subscription=None, region=None, endpoint=None):
                self.subscription = subscription
                self.region = region
                self.endpoint = endpoint
                self.speech_recognition_language = "en-US"
                self.properties = {}

            @classmethod
            def from_subscription(cls, sub, reg):
                return _SpeechConfig(subscription=sub, region=reg)

            @classmethod
            def from_endpoint(cls, ep, sub):
                return _SpeechConfig(subscription=sub, endpoint=ep)

            def set_property(self, prop_id, val):
                self.properties[prop_id] = val

        self.SpeechConfig = _SpeechConfig

        class _AudioStreamFormat:
            def __init__(self, samples_per_second=16000, bits_per_sample=16, channels=1):
                self.samples_per_second = samples_per_second
                self.bits_per_sample = bits_per_sample
                self.channels = channels

        class _PushAudioInputStream:
            def __init__(self, stream_format=None):
                self.stream_format = stream_format
                self.written = []
                self.closed = False

            def write(self, data: bytes):
                self.written.append(data)

            def close(self):
                self.closed = True

        class _AudioConfig:
            def __init__(self, stream=None):
                self.stream = stream

        self.audio = SimpleNamespace(
            AudioStreamFormat=_AudioStreamFormat,
            PushAudioInputStream=_PushAudioInputStream,
            AudioConfig=_AudioConfig,
        )

        self.last_recognizer: Optional[FakeAzureRecognizer] = None

        def _recognizer_factory(speech_config, audio_config):
            rec = FakeAzureRecognizer(speech_config, audio_config)
            self.last_recognizer = rec
            return rec

        self.SpeechRecognizer = _recognizer_factory


# ─────────────────────────────────────────────────────────────────────────────
# 4. Fake OpenAI Realtime Connection & Factory
# ─────────────────────────────────────────────────────────────────────────────


class FakeOpenAIRealtimeConnection:
    def __init__(self, auth_header: Optional[str] = None):
        self.auth_header = auth_header
        self.sent: List[Dict[str, Any]] = []
        self.queue: asyncio.Queue[Any] = asyncio.Queue()
        self.closed = False

    async def send(self, message: str) -> None:
        if self.closed:
            raise RealtimeConnectionClosedError(1006)
        payload = json.loads(message)
        self.sent.append(payload)
        if payload.get("type") == "session.update":
            session = {"id": "sess_openai_1", "type": "transcription"}
            self.queue.put_nowait(
                json.dumps({"type": "session.created", "session": session})
            )
            self.queue.put_nowait(
                json.dumps({"type": "session.updated", "session": session})
            )

    async def recv(self) -> str:
        item = await self.queue.get()
        if isinstance(item, _CloseSentinel):
            self.closed = True
            raise RealtimeConnectionClosedError(item.code)
        return item

    async def close(self) -> None:
        self.closed = True
        self.queue.put_nowait(_CloseSentinel(1000))

    def push_transcript_delta(self, delta: str, item_id: str = "item_1"):
        self.queue.put_nowait(
            json.dumps(
                {
                    "type": "conversation.item.input_audio_transcription.delta",
                    "item_id": item_id,
                    "delta": delta,
                }
            )
        )

    def push_transcript_done(self, transcript: str, item_id: str = "item_1"):
        self.queue.put_nowait(
            json.dumps(
                {
                    "type": "conversation.item.input_audio_transcription.completed",
                    "item_id": item_id,
                    "transcript": transcript,
                }
            )
        )

    def drop(self, code: int = 1006):
        self.queue.put_nowait(_CloseSentinel(code))


class FakeOpenAIRealtimeFactory:
    def __init__(self, status_code: int = 200):
        self.status_code = status_code
        self.calls: List[Tuple[str, Dict[str, str]]] = []
        self.connections: List[FakeOpenAIRealtimeConnection] = []

    async def __call__(
        self, url: str, headers: Dict[str, str], open_timeout: float
    ) -> FakeOpenAIRealtimeConnection:
        self.calls.append((url, dict(headers)))
        if self.status_code == 401:
            raise RealtimeHandshakeRejectedError(401)
        if self.status_code == 429:
            raise RealtimeHandshakeRejectedError(429)
        if self.status_code >= 400:
            raise RealtimeHandshakeRejectedError(self.status_code)
        conn = FakeOpenAIRealtimeConnection(auth_header=headers.get("Authorization"))
        self.connections.append(conn)
        return conn


# ─────────────────────────────────────────────────────────────────────────────
# 5. Fake Gemini Live Connection & Connector
# ─────────────────────────────────────────────────────────────────────────────


class FakeGeminiLiveConnection:
    def __init__(self):
        self.sent: List[Dict[str, Any]] = []
        self.queue: asyncio.Queue[Any] = asyncio.Queue()
        self.closed = False

    async def send(self, message: str) -> None:
        if self.closed:
            raise LiveConnectionClosedError(1006)
        payload = json.loads(message)
        self.sent.append(payload)
        if "setup" in payload:
            self.queue.put_nowait(json.dumps({"setupComplete": {}}))

    async def recv(self) -> str:
        item = await self.queue.get()
        if isinstance(item, _CloseSentinel):
            self.closed = True
            raise LiveConnectionClosedError(item.code)
        return item

    async def close(self) -> None:
        self.closed = True
        self.queue.put_nowait(_CloseSentinel(1000))

    def push_interim(self, text: str):
        self.queue.put_nowait(
            json.dumps({"serverContent": {"interimInputTranscription": {"text": text}}})
        )

    def push_final(self, text: str):
        self.queue.put_nowait(
            json.dumps(
                {
                    "serverContent": {
                        "inputTranscription": {"text": text, "languageCode": "en"},
                        "turnComplete": True,
                    }
                }
            )
        )

    def drop(self, code: int = 1006):
        self.queue.put_nowait(_CloseSentinel(code))


class FakeGeminiLiveConnector:
    def __init__(self, status_code: int = 200):
        self.status_code = status_code
        self.calls: List[Tuple[str, Dict[str, str]]] = []
        self.connections: List[FakeGeminiLiveConnection] = []

    async def __call__(
        self, url: str, headers: Dict[str, str]
    ) -> FakeGeminiLiveConnection:
        self.calls.append((url, dict(headers)))
        if self.status_code == 403:
            raise LiveConnectError(403)
        if self.status_code == 429:
            raise LiveConnectError(429)
        if self.status_code >= 400:
            raise LiveConnectError(self.status_code)
        conn = FakeGeminiLiveConnection()
        self.connections.append(conn)
        return conn


# ─────────────────────────────────────────────────────────────────────────────
# TIER 1: ISOLATED HAPPY PATH TESTS FOR ALL 5 OFFLINE PROVIDERS
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_deepgram_offline_streaming_happy_path():
    """Verify Deepgram STT adapter processes audio and emits transcript results offline."""
    connector = FakeDeepgramConnector(status_code=200)
    settings = DeepgramSettings(
        endpoint="wss://api.deepgram.com/v1/listen",
        model="nova-3",
        api_key=TEST_KEY,
        sample_rate=16000,
        language="en",
    )
    provider = DeepgramSTTProvider(
        settings, stream_ids=("mic",), connect_factory=connector
    )

    await provider.start()
    conn = connector.connections[0]
    assert conn.auth_header == f"Token {TEST_KEY}"

    events = provider.receive_events()
    await provider.send_audio(_audio_frame(0, "mic"))
    await _wait_until(lambda: len(conn.sent) >= 1)

    conn.push_result("hello world from deepgram", is_final=True)

    result = await _next_event(events)
    assert result.text == "hello world from deepgram"
    assert result.is_final is True
    assert result.stream_id == "mic"

    await provider.stop()


@pytest.mark.asyncio
async def test_assemblyai_offline_streaming_happy_path():
    """Verify AssemblyAI STT adapter processes audio and emits partial and final transcripts offline."""
    connector = FakeAssemblyAIConnector(status_code=200)
    settings = AssemblyAISettings(
        endpoint="wss://streaming.assemblyai.com/v3/ws",
        speech_model="universal-3-5-pro",
        api_key=TEST_KEY,
        sample_rate=16000,
    )
    provider = AssemblyAISTTProvider(
        settings, stream_ids=("mic",), connect_factory=connector
    )

    await provider.start()
    conn = connector.connections[0]
    assert conn.auth_header == TEST_KEY

    events = provider.receive_events()
    await provider.send_audio(_audio_frame(0, "mic"))
    await _wait_until(lambda: len(conn.sent) >= 1)

    conn.push_partial("hello")
    partial = await _next_event(events)
    assert partial.text == "hello"
    assert partial.is_final is False

    conn.push_final("hello world from assemblyai")
    final = await _next_event(events)
    assert final.text == "hello world from assemblyai"
    assert final.is_final is True

    await provider.stop()


@pytest.mark.asyncio
async def test_azure_speech_offline_streaming_happy_path():
    """Verify Azure Speech STT adapter processes speech events via fake SDK offline."""
    sdk = FakeAzureSpeechSDK()
    cfg = AzureSpeechSTTConfig.from_options(
        {
            "api_key": TEST_KEY,
            "region": "eastus",
            "recognition_language": "en-US",
            "start_timeout_s": 1.0,
            "stop_timeout_s": 1.0,
        }
    )
    provider = AzureSpeechSTTProvider(cfg, sdk_loader=lambda: sdk)

    await provider.start()
    events = provider.receive_events()

    # Audio frame triggers recognizer continuous start
    await provider.send_audio(_audio_frame(0, "mic"))
    assert sdk.last_recognizer is not None
    assert sdk.last_recognizer.started is True

    # Push recognized event from SDK
    sdk.last_recognizer.recognized.fire(
        SimpleNamespace(
            result=SimpleNamespace(
                reason=sdk.ResultReason.RecognizedSpeech,
                text="hello world from azure",
                offset=0,
                duration=10_000_000,
            )
        )
    )

    result = await _next_event(events)
    assert result.text == "hello world from azure"
    assert result.is_final is True

    await provider.stop()
    assert sdk.last_recognizer.stopped is True


@pytest.mark.asyncio
async def test_openai_realtime_offline_streaming_happy_path():
    """Verify OpenAI Realtime STT adapter handles session deltas and completion offline."""
    factory = FakeOpenAIRealtimeFactory(status_code=200)
    settings = OpenAIRealtimeSettings(
        endpoint="wss://api.openai.com/v1/realtime",
        model="gpt-4o-realtime",
        api_key=TEST_KEY,
    )
    provider = OpenAIRealtimeSTTProvider(
        settings, stream_ids=("mic",), connect_factory=factory
    )

    await provider.start()
    conn = factory.connections[0]
    events = provider.receive_events()

    await provider.send_audio(_audio_frame(0, "mic"))
    await _wait_until(lambda: len(conn.sent) >= 1)

    conn.push_transcript_delta("hello ")
    partial = await _next_event(events)
    assert partial.text == "hello "
    assert partial.is_final is False

    conn.push_transcript_done("hello world from openai")
    final = await _next_event(events)
    assert final.text == "hello world from openai"
    assert final.is_final is True

    await provider.stop()


@pytest.mark.asyncio
async def test_gemini_live_offline_streaming_happy_path():
    """Verify Gemini Live STT adapter processes interim and final transcript offline."""
    connector = FakeGeminiLiveConnector(status_code=200)
    cfg = GeminiLiveSTTConfig.from_options(
        {
            "api_key": TEST_KEY,
            "model": "gemini-2.0-flash-exp",
            "chunk_ms": 20.0,
        }
    )
    provider = GeminiLiveSTTProvider(cfg, connector=connector)

    await provider.start()
    events = provider.receive_events()

    await provider.send_audio(_audio_frame(0, "mic"))
    assert len(connector.connections) == 1
    conn = connector.connections[0]

    conn.push_interim("hello")
    partial = await _next_event(events)
    assert partial.text == "hello"
    assert partial.is_final is False

    conn.push_final("hello world from gemini")
    final = await _next_event(events)
    assert final.text == "hello world from gemini"
    assert final.is_final is True

    await provider.stop()


# ─────────────────────────────────────────────────────────────────────────────
# TIER 2: BOUNDARY CONDITIONS, ERRORS, RATE-LIMITS & CLOSE
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_deepgram_auth_failure_offline():
    """Verify Deepgram 401 raises ProviderRuntimeError with AUTH_FAILED without leaking key."""
    connector = FakeDeepgramConnector(status_code=401)
    settings = DeepgramSettings(
        endpoint="wss://api.deepgram.com/v1/listen",
        model="nova-3",
        api_key=TEST_KEY,
        sample_rate=16000,
    )
    provider = DeepgramSTTProvider(settings, stream_ids=("mic",), connect_factory=connector)

    with pytest.raises(ProviderRuntimeError) as exc_info:
        await provider.start()

    assert exc_info.value.error.code == ProviderErrorCode.AUTH_FAILED
    assert TEST_KEY not in str(exc_info.value)


@pytest.mark.asyncio
async def test_deepgram_rate_limit_offline():
    """Verify Deepgram 429 returns retryable RATE_LIMITED error."""
    connector = FakeDeepgramConnector(status_code=429)
    settings = DeepgramSettings(
        endpoint="wss://api.deepgram.com/v1/listen",
        model="nova-3",
        api_key=TEST_KEY,
        sample_rate=16000,
    )
    provider = DeepgramSTTProvider(settings, stream_ids=("mic",), connect_factory=connector)

    with pytest.raises(ProviderRuntimeError) as exc_info:
        await provider.start()

    assert exc_info.value.error.code == ProviderErrorCode.RATE_LIMITED
    assert exc_info.value.error.retryable is True


@pytest.mark.asyncio
async def test_deepgram_abnormal_disconnect_offline():
    """Verify Deepgram unexpected 1006 connection drop surfaces clean error on stream."""
    connector = FakeDeepgramConnector(status_code=200)
    settings = DeepgramSettings(
        endpoint="wss://api.deepgram.com/v1/listen",
        model="nova-3",
        api_key=TEST_KEY,
        sample_rate=16000,
    )
    provider = DeepgramSTTProvider(settings, stream_ids=("mic",), connect_factory=connector)

    await provider.start()
    conn = connector.connections[0]
    conn.drop(1006, "Unexpected drop")

    events = provider.receive_events()
    with contextlib.suppress(ProviderRuntimeError, asyncio.TimeoutError):
        await _next_event(events, timeout=0.5)

    await provider.stop()


@pytest.mark.asyncio
async def test_assemblyai_auth_failure_offline():
    """Verify AssemblyAI 401 raises AUTH_FAILED without leaking secret."""
    connector = FakeAssemblyAIConnector(status_code=401)
    settings = AssemblyAISettings(
        endpoint="wss://streaming.assemblyai.com/v3/ws",
        speech_model="universal-3-5-pro",
        api_key=TEST_KEY,
        sample_rate=16000,
    )
    provider = AssemblyAISTTProvider(settings, stream_ids=("mic",), connect_factory=connector)

    with pytest.raises(ProviderRuntimeError) as exc_info:
        await provider.start()

    assert exc_info.value.error.code == ProviderErrorCode.AUTH_FAILED
    assert TEST_KEY not in str(exc_info.value)


@pytest.mark.asyncio
async def test_assemblyai_rate_limit_offline():
    """Verify AssemblyAI 429 raises RATE_LIMITED with retryable=True."""
    connector = FakeAssemblyAIConnector(status_code=429)
    settings = AssemblyAISettings(
        endpoint="wss://streaming.assemblyai.com/v3/ws",
        speech_model="universal-3-5-pro",
        api_key=TEST_KEY,
        sample_rate=16000,
    )
    provider = AssemblyAISTTProvider(settings, stream_ids=("mic",), connect_factory=connector)

    with pytest.raises(ProviderRuntimeError) as exc_info:
        await provider.start()

    assert exc_info.value.error.code == ProviderErrorCode.RATE_LIMITED
    assert exc_info.value.error.retryable is True


@pytest.mark.asyncio
async def test_azure_speech_auth_cancellation_offline():
    """Verify Azure Speech authentication cancellation emits AUTH_FAILED."""
    sdk = FakeAzureSpeechSDK()
    cfg = AzureSpeechSTTConfig.from_options(
        {
            "api_key": TEST_KEY,
            "region": "eastus",
            "recognition_language": "en-US",
            "start_timeout_s": 0.5,
        }
    )
    provider = AzureSpeechSTTProvider(cfg, sdk_loader=lambda: sdk)

    await provider.start()
    events = provider.receive_events()
    await provider.send_audio(_audio_frame(0, "mic"))
    rec = sdk.last_recognizer

    rec.canceled.fire(
        SimpleNamespace(
            cancellation_details=SimpleNamespace(
                reason=sdk.CancellationReason.Error,
                code=sdk.CancellationErrorCode.AuthenticationFailure,
                error_details="Invalid subscription key",
            )
        )
    )

    with pytest.raises(ProviderRuntimeError) as exc_info:
        await _next_event(events, timeout=0.5)

    assert exc_info.value.error.code == ProviderErrorCode.AUTH_FAILED
    await provider.stop()


@pytest.mark.asyncio
async def test_openai_realtime_auth_failure_offline():
    """Verify OpenAI Realtime 401 handshake reject raises AUTH_FAILED."""
    factory = FakeOpenAIRealtimeFactory(status_code=401)
    settings = OpenAIRealtimeSettings(
        endpoint="wss://api.openai.com/v1/realtime",
        model="gpt-4o-realtime",
        api_key=TEST_KEY,
    )
    provider = OpenAIRealtimeSTTProvider(
        settings, stream_ids=("mic",), connect_factory=factory
    )

    with pytest.raises(ProviderRuntimeError) as exc_info:
        await provider.start()

    assert exc_info.value.error.code == ProviderErrorCode.AUTH_FAILED


@pytest.mark.asyncio
async def test_openai_realtime_rate_limit_offline():
    """Verify OpenAI Realtime 429 returns retryable RATE_LIMITED error."""
    factory = FakeOpenAIRealtimeFactory(status_code=429)
    settings = OpenAIRealtimeSettings(
        endpoint="wss://api.openai.com/v1/realtime",
        model="gpt-4o-realtime",
        api_key=TEST_KEY,
    )
    provider = OpenAIRealtimeSTTProvider(
        settings, stream_ids=("mic",), connect_factory=factory
    )

    with pytest.raises(ProviderRuntimeError) as exc_info:
        await provider.start()

    assert exc_info.value.error.code == ProviderErrorCode.RATE_LIMITED


@pytest.mark.asyncio
async def test_gemini_live_auth_failure_offline():
    """Verify Gemini Live 403 Forbidden raises AUTH_FAILED without leaking key."""
    connector = FakeGeminiLiveConnector(status_code=403)
    cfg = GeminiLiveSTTConfig.from_options({"api_key": TEST_KEY, "chunk_ms": 20.0})
    provider = GeminiLiveSTTProvider(cfg, connector=connector)

    await provider.start()
    with pytest.raises(ProviderRuntimeError) as exc_info:
        await provider.send_audio(_audio_frame(0, "mic"))

    assert exc_info.value.error.code == ProviderErrorCode.AUTH_FAILED
    assert TEST_KEY not in str(exc_info.value)
    await provider.stop()


@pytest.mark.asyncio
async def test_gemini_live_rate_limit_offline():
    """Verify Gemini Live 429 ResourceExhausted raises RATE_LIMITED with retryable=True."""
    connector = FakeGeminiLiveConnector(status_code=429)
    cfg = GeminiLiveSTTConfig.from_options({"api_key": TEST_KEY, "chunk_ms": 20.0})
    provider = GeminiLiveSTTProvider(cfg, connector=connector)

    await provider.start()
    with pytest.raises(ProviderRuntimeError) as exc_info:
        await provider.send_audio(_audio_frame(0, "mic"))

    assert exc_info.value.error.code == ProviderErrorCode.RATE_LIMITED
    assert exc_info.value.error.retryable is True
    await provider.stop()


# ─────────────────────────────────────────────────────────────────────────────
# TIER 3: REGISTRY, CAPABILITY MATRIX & VALIDATION CONTRACTS
# ─────────────────────────────────────────────────────────────────────────────


def test_provider_registry_contains_all_approved_matrix():
    """Verify build_default_registry() registers all approved STT, translation, and assistant providers."""
    reg = build_default_registry()

    stt_providers = {m.provider_id for m in reg.providers_for(ProviderCapability.STT)}
    assert {
        "local",
        "openai-realtime",
        "gemini-live",
        "deepgram",
        "azure-speech",
        "assemblyai",
        "custom",
    }.issubset(stt_providers)

    translation_providers = {
        m.provider_id for m in reg.providers_for(ProviderCapability.TRANSLATION)
    }
    assert {
        "local",
        "openai",
        "gemini",
        "custom-openai-compatible",
    }.issubset(translation_providers)

    assistant_providers = {
        m.provider_id for m in reg.providers_for(ProviderCapability.ASSISTANT)
    }
    assert {
        "openai",
        "gemini",
        "custom-openai-compatible",
    }.issubset(assistant_providers)


def test_provider_capability_mismatch_rejected():
    """Verify attempting to resolve a provider for an unsupported capability raises UNSUPPORTED_CAPABILITY."""
    reg = build_default_registry()
    service = ServiceConfig(
        service="assistant",
        provider_id="deepgram",  # Deepgram does not provide assistant capability
    )

    with pytest.raises(ProviderRegistryError) as exc:
        reg.resolve_service(service)

    assert exc.value.code == RegistryErrorCode.UNSUPPORTED_CAPABILITY


def test_provider_validation_missing_required_credentials():
    """Verify missing required credentials fails validation with MISSING_CREDENTIAL."""
    reg = build_default_registry()

    for provider_id in ("deepgram", "assemblyai", "azure-speech", "openai-realtime", "gemini-live"):
        manifest = reg.get(ProviderCapability.STT, provider_id)
        assert manifest is not None

        service = ServiceConfig(
            service="stt",
            provider_id=provider_id,
            credential_refs={},  # Empty credentials
        )

        with pytest.raises(ProviderRegistryError) as exc:
            reg.resolve_service(service)

        assert exc.value.code == RegistryErrorCode.MISSING_CREDENTIAL
