"""Provider validation strategies and the Test Connection process entrypoint.

Every adapter here is a fake and every registry is either a test registry whose factories
are in-memory or the built-in one with a provider that never touches the network. Nothing
contacts a real or paid provider.
"""

import asyncio
import io
import json
import logging
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Dict, List, Mapping, Optional

import pytest

from engine.core.config import CredentialRef, ServiceConfig
from engine.providers import validation
from engine.providers.contracts import (
    AssistantProvider,
    StreamingSTTProvider,
    TranslationProvider,
)
from engine.providers.manifests import (
    FieldType,
    ManifestField,
    ProtocolFamily,
    ProviderManifest,
)
from engine.providers.registry import ProviderRegistry, build_default_registry
from engine.providers.types import (
    AssistRequest,
    AssistResult,
    AudioFrame,
    ProviderCapability,
    ProviderError,
    ProviderErrorCode,
    ProviderHealthCheck,
    ProviderHealthStatus,
    ProviderRuntimeError,
    STTTranscriptResult,
    SummaryChunk,
    SummaryRequest,
    SummaryResult,
    TranslationRequest,
    TranslationResult,
)
from engine.providers.validation import (
    ERROR_NOT_SUPPORTED,
    PROBE_MAX_OUTPUT_TOKENS,
    PROBE_TEXT,
    AssistantProbeStrategy,
    HandshakeStrategy,
    TranslationProbeStrategy,
    ValidationResult,
    current_validation_status,
    select_strategy,
    validate_service,
)

ROOT = Path(__file__).resolve().parent.parent
SECRET = "sk-fake-validation-key-0000"
STT = ProviderCapability.STT
TRANSLATION = ProviderCapability.TRANSLATION
ASSISTANT = ProviderCapability.ASSISTANT
RESULT_KEYS = {
    "id",
    "capability",
    "providerId",
    "status",
    "latencyMs",
    "testedAt",
    "fingerprint",
    "credentialVersion",
    "errorCode",
}


# ── Fakes ────────────────────────────────────────────────────────────────────


class Tracker:
    """Shared record of what a fake adapter did, visible after validation returns."""

    def __init__(self) -> None:
        self.started = 0
        self.stopped = 0
        self.open_sockets = 0
        self.requests: List[Any] = []
        self.secrets_seen: Dict[str, str] = {}


class FakeSTT(StreamingSTTProvider):
    def __init__(self, tracker: Tracker, behaviour: Callable[[], Any]) -> None:
        self._tracker = tracker
        self._behaviour = behaviour

    @property
    def provider_id(self) -> str:
        return "fake-stt"

    async def start(self) -> None:
        self._tracker.started += 1

    async def stop(self) -> None:
        self._tracker.stopped += 1
        self._tracker.open_sockets = 0

    async def health_check(self) -> ProviderHealthCheck:
        self._tracker.open_sockets += 1
        outcome = self._behaviour()
        if asyncio.iscoroutine(outcome):
            outcome = await outcome
        self._tracker.open_sockets -= 1
        return outcome

    async def send_audio(self, frame: AudioFrame) -> None:
        return None

    async def receive_events(self) -> AsyncIterator[STTTranscriptResult]:
        if False:
            yield

    async def finalize_stream(self, stream_id: str) -> None:
        return None

    async def reset_stream(self, stream_id: str) -> None:
        return None


class FakeTranslation(TranslationProvider):
    def __init__(self, tracker: Tracker, behaviour: Callable[[TranslationRequest], Any]) -> None:
        self._tracker = tracker
        self._behaviour = behaviour

    @property
    def provider_id(self) -> str:
        return "fake-text"

    async def start(self) -> None:
        self._tracker.started += 1

    async def stop(self) -> None:
        self._tracker.stopped += 1

    async def health_check(self) -> ProviderHealthCheck:
        return ProviderHealthCheck(status=ProviderHealthStatus.HEALTHY)

    async def translate(self, request: TranslationRequest) -> TranslationResult:
        self._tracker.requests.append(request)
        outcome = self._behaviour(request)
        if asyncio.iscoroutine(outcome):
            outcome = await outcome
        return outcome


class FakeAssistant(AssistantProvider):
    def __init__(self, tracker: Tracker, behaviour: Callable[[AssistRequest], Any]) -> None:
        self._tracker = tracker
        self._behaviour = behaviour

    @property
    def provider_id(self) -> str:
        return "fake-text"

    async def start(self) -> None:
        self._tracker.started += 1

    async def stop(self) -> None:
        self._tracker.stopped += 1

    async def health_check(self) -> ProviderHealthCheck:
        return ProviderHealthCheck(status=ProviderHealthStatus.HEALTHY)

    async def assist(self, request: AssistRequest) -> AssistResult:
        self._tracker.requests.append(request)
        return self._behaviour(request)

    async def summarize(self, request: SummaryRequest) -> SummaryResult:
        raise AssertionError("validation must not summarize")

    async def stream_summary(self, request: SummaryRequest) -> AsyncIterator[SummaryChunk]:
        raise AssertionError("validation must not stream summaries")
        yield


def _api_key_field() -> ManifestField:
    return ManifestField(
        key="api_key", label="API key", field_type=FieldType.SECRET, required=True, secret=True
    )


TEST_MANIFESTS = (
    ProviderManifest(
        provider_id="fake-stt",
        display_name="Fake STT",
        capabilities=frozenset({STT}),
        protocol_family=ProtocolFamily.OPENAI_REALTIME,
        factories={STT: "fake.stt:Adapter"},
        recommended_endpoint="wss://stt.fake.invalid/v1",
        recommended_model="fake-model",
        fields=(_api_key_field(),),
    ),
    ProviderManifest(
        provider_id="fake-text",
        display_name="Fake text",
        capabilities=frozenset({TRANSLATION, ASSISTANT}),
        protocol_family=ProtocolFamily.OPENAI_COMPATIBLE_TEXT,
        factories={TRANSLATION: "fake.text:Translation", ASSISTANT: "fake.text:Assistant"},
        recommended_endpoint="https://text.fake.invalid/v1",
        recommended_model="fake-text-model",
        fields=(
            _api_key_field(),
            ManifestField(
                key="target_lang",
                label="Target",
                field_type=FieldType.STRING,
                default="vi",
                applies_to=frozenset({TRANSLATION}),
            ),
            ManifestField(
                key="max_output_tokens",
                label="Max output tokens",
                field_type=FieldType.INTEGER,
                default=1000,
                minimum=1,
            ),
        ),
    ),
    ProviderManifest(
        provider_id="fake-local",
        display_name="Fake local",
        capabilities=frozenset({STT}),
        protocol_family=ProtocolFamily.LOCAL,
        factories={STT: "fake.local:Adapter"},
        model_selectable=False,
    ),
    ProviderManifest(
        provider_id="fake-missing",
        display_name="Fake missing module",
        capabilities=frozenset({TRANSLATION}),
        protocol_family=ProtocolFamily.OPENAI_COMPATIBLE_TEXT,
        factories={TRANSLATION: "engine.providers.does_not_exist:Nothing"},
        recommended_endpoint="https://missing.fake.invalid/v1",
        recommended_model="m",
        fields=(_api_key_field(),),
    ),
)


class FakeRegistry(ProviderRegistry):
    """Real manifest resolution, in-memory factories. ``fake-missing`` uses the real import."""

    def __init__(self) -> None:
        super().__init__(TEST_MANIFESTS)
        self.factories: Dict[str, Any] = {"fake-stt": object(), "fake-text": object()}

    def load_factory(self, capability: Any, provider_id: str) -> Callable[..., Any]:
        if provider_id in self.factories:
            return self.factories[provider_id]
        return super().load_factory(capability, provider_id)


class Harness:
    def __init__(self) -> None:
        self.tracker = Tracker()
        self.registry = FakeRegistry()
        self.built: List[Any] = []
        self.stt_behaviour: Callable[[], Any] = lambda: ProviderHealthCheck(
            status=ProviderHealthStatus.HEALTHY
        )
        self.translate_behaviour: Callable[[TranslationRequest], Any] = _translated
        self.assist_behaviour: Callable[[AssistRequest], Any] = _assisted
        self.build_calls = 0

    def builders(self) -> Mapping[ProtocolFamily, Any]:
        def build_stt(factory, resolved, secrets):
            return self._record(resolved, secrets, FakeSTT(self.tracker, self.stt_behaviour))

        def build_text(factory, resolved, secrets):
            if resolved.capability is TRANSLATION:
                adapter = FakeTranslation(self.tracker, self.translate_behaviour)
            else:
                adapter = FakeAssistant(self.tracker, self.assist_behaviour)
            return self._record(resolved, secrets, adapter)

        return {
            ProtocolFamily.OPENAI_REALTIME: build_stt,
            ProtocolFamily.OPENAI_COMPATIBLE_TEXT: build_text,
        }

    def _record(self, resolved, secrets, adapter):
        self.build_calls += 1
        self.built.append(resolved)
        self.tracker.secrets_seen = dict(secrets)
        return adapter

    async def validate(self, service: ServiceConfig, secrets: Optional[dict] = None, **kwargs):
        secrets = {"api_key": SECRET} if secrets is None else secrets
        kwargs.setdefault("timeout_s", 2.0)
        return await validate_service(
            service, secrets, registry=self.registry, builders=self.builders(), **kwargs
        )


def _translated(request: TranslationRequest) -> TranslationResult:
    return TranslationResult(
        translated_text="Xin chao.",
        source_text=request.text,
        source_lang=request.source_lang,
        target_lang=request.target_lang,
    )


def _assisted(request: AssistRequest) -> AssistResult:
    return AssistResult(
        target_message_id=request.target_message.message_id,
        suggested_replies=(),
        clarifying_questions=(),
    )


def service(
    capability: str = "stt", provider_id: str = "fake-stt", version: int = 3, **changes: Any
) -> ServiceConfig:
    refs = {"api_key": CredentialRef(f"{provider_id}-main", "api_key", version)}
    return ServiceConfig(
        service=capability, provider_id=provider_id, credential_refs=refs, **changes
    )


@pytest.fixture
def harness() -> Harness:
    return Harness()


def assert_no_secret(*values: Any) -> None:
    for value in values:
        rendered = value if isinstance(value, str) else json.dumps(value, default=str)
        assert SECRET not in rendered


# ── Strategy selection ───────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("capability", "provider_id", "expected"),
    [
        ("stt", "openai-realtime", HandshakeStrategy),
        ("stt", "gemini-live", HandshakeStrategy),
        ("translation", "openai", TranslationProbeStrategy),
        ("assistant", "openai", AssistantProbeStrategy),
        ("translation", "custom-openai-compatible", TranslationProbeStrategy),
    ],
)
def test_strategy_is_selected_from_the_builtin_manifest(capability, provider_id, expected):
    registry = build_default_registry()
    custom = provider_id.startswith("custom")
    resolved = registry.resolve_service(
        service(
            capability,
            provider_id,
            endpoint="https://llm.fake.invalid/v1" if custom else "",
            model_override="fake-model" if custom else "",
        )
    )
    assert isinstance(select_strategy(resolved), expected)


@pytest.mark.parametrize("capability", ["stt", "translation"])
def test_local_runtimes_have_no_network_strategy(capability):
    resolved = build_default_registry().resolve_service(
        ServiceConfig(service=capability, provider_id="local")
    )
    assert select_strategy(resolved) is None


# ── STT handshake ────────────────────────────────────────────────────────────


async def test_stt_handshake_success_reports_safe_fields(harness):
    config = service()
    secrets = {"api_key": SECRET}
    result = await harness.validate(config, secrets, request_id="val-1")

    assert result.status == "valid"
    assert result.error_code is None
    assert result.request_id == "val-1"
    assert (result.capability, result.provider_id) == ("stt", "fake-stt")
    assert result.fingerprint == config.fingerprint()
    assert result.credential_version == 3
    assert result.latency_ms is not None and result.latency_ms >= 0
    assert result.tested_at > 0
    assert set(result.to_dict()) == RESULT_KEYS
    assert_no_secret(result.to_dict())

    # The builder saw the secret; the caller's dict no longer holds it.
    assert harness.tracker.secrets_seen == {"api_key": SECRET}
    assert secrets == {}
    assert harness.tracker.stopped == 1
    assert harness.tracker.open_sockets == 0


async def test_stt_unhealthy_maps_to_a_code_without_provider_text(harness):
    harness.stt_behaviour = lambda: ProviderHealthCheck(
        status=ProviderHealthStatus.UNHEALTHY,
        message=f"401 for key {SECRET}",
        details={"code": "auth_failed", "retryable": False},
    )
    result = await harness.validate(service())
    assert (result.status, result.error_code) == ("invalid", "auth_failed")
    assert_no_secret(result.to_dict())
    assert harness.tracker.stopped == 1


async def test_stt_unknown_health_code_is_not_echoed(harness):
    harness.stt_behaviour = lambda: ProviderHealthCheck(
        status=ProviderHealthStatus.DEGRADED, details={"code": f"weird {SECRET}"}
    )
    result = await harness.validate(service())
    assert (result.status, result.error_code) == ("invalid", "internal_error")


# ── Text probes ──────────────────────────────────────────────────────────────


async def test_translation_probe_is_one_minimal_bounded_request(harness):
    result = await harness.validate(service("translation", "fake-text"))
    assert result.status == "valid"
    assert harness.tracker.started == 1 and harness.tracker.stopped == 1
    [request] = harness.tracker.requests
    assert request.text == PROBE_TEXT
    assert request.target_lang == "vi"
    [resolved] = harness.built
    assert resolved.options["max_output_tokens"] == PROBE_MAX_OUTPUT_TOKENS


async def test_translation_provider_error_is_normalized(harness):
    def fail(request):
        raise ProviderRuntimeError(
            ProviderError(
                code=ProviderErrorCode.RATE_LIMITED,
                capability=TRANSLATION,
                message=f"slow down {SECRET}",
            )
        )

    harness.translate_behaviour = fail
    result = await harness.validate(service("translation", "fake-text"))
    assert (result.status, result.error_code) == ("invalid", "rate_limited")
    assert_no_secret(result.to_dict())
    assert harness.tracker.stopped == 1


async def test_translation_empty_output_is_an_invalid_response(harness):
    harness.translate_behaviour = lambda request: dataclass_replace_text(_translated(request), " ")
    result = await harness.validate(service("translation", "fake-text"))
    assert (result.status, result.error_code) == ("invalid", "invalid_response")


def dataclass_replace_text(result: TranslationResult, text: str) -> TranslationResult:
    import dataclasses

    return dataclasses.replace(result, translated_text=text)


async def test_assistant_probe_success_and_bad_response(harness):
    result = await harness.validate(service("assistant", "fake-text"))
    assert result.status == "valid"
    [request] = harness.tracker.requests
    assert request.target_message.text == PROBE_TEXT
    assert harness.built[0].options["max_output_tokens"] == PROBE_MAX_OUTPUT_TOKENS

    harness.assist_behaviour = lambda request: {"not": "an AssistResult"}
    result = await harness.validate(service("assistant", "fake-text"))
    assert (result.status, result.error_code) == ("invalid", "invalid_response")
    assert harness.tracker.stopped == 2


# ── Cleanup, timeout, cancellation ───────────────────────────────────────────


async def test_timeout_stops_the_adapter_and_closes_sockets(harness):
    async def hang():
        await asyncio.Event().wait()

    harness.stt_behaviour = hang
    result = await harness.validate(service(), timeout_s=0.05)
    assert (result.status, result.error_code) == ("invalid", "timeout")
    assert harness.tracker.stopped == 1
    assert harness.tracker.open_sockets == 0


async def test_cancellation_still_stops_the_adapter(harness):
    entered = asyncio.Event()

    async def hang():
        entered.set()
        await asyncio.Event().wait()

    harness.stt_behaviour = hang
    secrets = {"api_key": SECRET}
    task = asyncio.create_task(harness.validate(service(), secrets, timeout_s=5.0))
    await asyncio.wait_for(entered.wait(), 1.0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert harness.tracker.stopped == 1
    assert secrets == {}


async def test_unexpected_adapter_exception_is_redacted(harness, caplog):
    caplog.set_level(logging.DEBUG)

    def explode():
        raise RuntimeError(f"socket to wss://x?key={SECRET} failed")

    harness.stt_behaviour = explode
    result = await harness.validate(service())
    assert (result.status, result.error_code) == ("invalid", "internal_error")
    assert harness.tracker.stopped == 1
    assert SECRET not in caplog.text


async def test_stop_failure_does_not_change_the_result(harness):
    tracker = harness.tracker

    async def broken_stop():
        tracker.stopped += 1
        raise RuntimeError(SECRET)

    builders = dict(harness.builders())
    original = builders[ProtocolFamily.OPENAI_REALTIME]

    def build(factory, resolved, secrets):
        adapter = original(factory, resolved, secrets)
        adapter.stop = broken_stop
        return adapter

    builders[ProtocolFamily.OPENAI_REALTIME] = build
    result = await validate_service(
        service(), {"api_key": SECRET}, registry=harness.registry, builders=builders
    )
    assert result.status == "valid"
    assert tracker.stopped == 1


# ── Config and credential failures ───────────────────────────────────────────


async def test_local_provider_is_reported_unvalidated_without_building(harness):
    config = ServiceConfig(service="stt", provider_id="fake-local")
    result = await harness.validate(config, {})
    assert (result.status, result.error_code) == ("unvalidated", ERROR_NOT_SUPPORTED)
    assert result.fingerprint == config.fingerprint()
    assert result.credential_version is None
    assert harness.build_calls == 0


@pytest.mark.parametrize(
    ("secrets", "code"),
    [
        ({}, "missing_credential"),
        ({"api_key": "   "}, "missing_credential"),
        ({"api_key": SECRET, "other": SECRET}, "invalid_request"),
        ({"api_key": 42}, "invalid_request"),
    ],
    ids=["absent", "blank", "extra-slot", "non-string"],
)
async def test_secret_problems_fail_before_any_adapter_exists(harness, secrets, code):
    result = await harness.validate(service(), secrets)
    assert (result.status, result.error_code) == ("invalid", code)
    assert harness.build_calls == 0
    assert secrets == {}
    assert_no_secret(result.to_dict())


async def test_registry_errors_surface_as_codes(harness):
    result = await harness.validate(service(provider_id="nobody"))
    assert (result.status, result.error_code) == ("invalid", "unknown_provider")

    result = await harness.validate(service("assistant", "fake-stt"))
    assert (result.status, result.error_code) == ("invalid", "unsupported_capability")


async def test_missing_adapter_module_reports_factory_unavailable(harness):
    result = await harness.validate(service("translation", "fake-missing"))
    assert (result.status, result.error_code) == ("invalid", "factory_unavailable")


async def test_text_family_without_builder_needs_from_resolved_hook(harness):
    tracker = harness.tracker

    class Factory:
        @staticmethod
        def from_resolved(resolved, secrets):
            tracker.secrets_seen = dict(secrets)
            return FakeTranslation(tracker, _translated)

    harness.registry.factories["fake-text"] = Factory
    result = await validate_service(
        service("translation", "fake-text"),
        {"api_key": SECRET},
        registry=harness.registry,
        builders={},
    )
    assert result.status == "valid"
    assert tracker.secrets_seen == {"api_key": SECRET}

    harness.registry.factories["fake-text"] = object()
    result = await validate_service(
        service("translation", "fake-text"),
        {"api_key": SECRET},
        registry=harness.registry,
        builders={},
    )
    assert (result.status, result.error_code) == ("invalid", "factory_unavailable")


# ── Default family builders (no network: factories are recorders) ────────────


def test_openai_realtime_builder_maps_resolved_settings():
    registry = build_default_registry()
    resolved = registry.resolve_service(
        service("stt", "openai-realtime", options={"language": "vi"})
    )
    captured = {}

    def factory(settings, provider_id):
        captured.update(settings=settings, provider_id=provider_id)
        return "adapter"

    assert validation.build_openai_realtime(factory, resolved, {"api_key": SECRET}) == "adapter"
    settings = captured["settings"]
    assert settings.endpoint == resolved.endpoint
    assert settings.model == resolved.model
    assert settings.api_key == SECRET
    assert settings.languages == ("vi",)
    assert captured["provider_id"] == "openai-realtime"
    assert SECRET not in repr(settings)


def test_gemini_live_builder_maps_resolved_settings():
    resolved = build_default_registry().resolve_service(service("stt", "gemini-live"))
    captured = {}

    def factory(config):
        captured["config"] = config
        return "adapter"

    validation.build_gemini_live(factory, resolved, {"api_key": SECRET})
    config = captured["config"]
    assert (config.endpoint, config.model) == (resolved.endpoint, resolved.model)
    assert config.api_key == SECRET
    assert config.language_hints == ()
    assert config.provider_id == "gemini-live"
    assert SECRET not in repr(config)


# ── Fingerprint / credential version currency ────────────────────────────────


async def test_changed_fingerprint_or_credential_version_reads_unvalidated(harness):
    config = service()
    result = await harness.validate(config)
    assert current_validation_status(result, config) == "valid"

    assert current_validation_status(result, service(endpoint="wss://other.fake.invalid")) == (
        "unvalidated"
    )
    assert current_validation_status(result, service(version=4)) == "unvalidated"
    assert current_validation_status(result, service(provider_id="fake-local")) == "unvalidated"

    tampered = ValidationResult(**{**result.__dict__, "credential_version": 2})
    assert current_validation_status(tampered, config) == "unvalidated"

    failed = ValidationResult(**{**result.__dict__, "status": "invalid"})
    assert current_validation_status(failed, config) == "invalid"
    never = ValidationResult(**{**result.__dict__, "status": "unvalidated"})
    assert current_validation_status(never, config) == "unvalidated"


# ── Request parsing ──────────────────────────────────────────────────────────


def request_doc(**overrides: Any) -> Dict[str, Any]:
    doc = {
        "id": "val-7",
        "capability": "stt",
        "service": service().to_dict(),
        "secrets": {"api_key": SECRET},
    }
    doc.update(overrides)
    return doc


@pytest.mark.parametrize(
    ("raw", "status", "code", "echo_id"),
    [
        ("not json", "unvalidated", "invalid_request", None),
        (json.dumps([SECRET]), "unvalidated", "invalid_request", None),
        (json.dumps(request_doc(id=f"bad id {SECRET}")), "unvalidated", "invalid_request", None),
        (json.dumps(request_doc(extra=SECRET)), "unvalidated", "invalid_request", "val-7"),
        (json.dumps(request_doc(capability="llm")), "unvalidated", "invalid_request", "val-7"),
        (json.dumps(request_doc(secrets=SECRET)), "unvalidated", "invalid_request", "val-7"),
        (
            json.dumps(
                request_doc(service={**service().to_dict(), "options": {"api_key": SECRET}})
            ),
            "invalid",
            "secret_value_rejected",
            "val-7",
        ),
        (
            json.dumps(request_doc(service={**service().to_dict(), "apiKey": SECRET})),
            "invalid",
            "secret_value_rejected",
            "val-7",
        ),
    ],
    ids=[
        "invalid-json",
        "non-object",
        "bad-id",
        "unknown-field",
        "bad-capability",
        "secrets-not-object",
        "secret-in-options",
        "secret-in-service",
    ],
)
async def test_malformed_requests_are_rejected_without_echo(harness, raw, status, code, echo_id):
    result = await validation.run_request(
        raw, registry=harness.registry, builders=harness.builders()
    )
    assert (result.status, result.error_code, result.request_id) == (status, code, echo_id)
    assert_no_secret(result.to_dict())
    assert harness.build_calls == 0


# ── Entrypoint ───────────────────────────────────────────────────────────────


def test_main_writes_exactly_one_safe_line_and_swallows_stray_output(harness, capsys):
    def noisy():
        print(f"debug {SECRET}")
        print(f"debug {SECRET}", file=sys.stderr)
        logging.getLogger("echoflux.providers.fake").error("leak %s", SECRET)
        return ProviderHealthCheck(status=ProviderHealthStatus.HEALTHY)

    harness.stt_behaviour = noisy
    stdin = io.StringIO(json.dumps(request_doc()))
    stdout = io.StringIO()
    exit_code = validation.main(
        stdin,
        stdout,
        registry=harness.registry,
        builders=harness.builders(),
        configure_logging=False,
    )
    assert exit_code == 0
    lines = stdout.getvalue().splitlines()
    assert len(lines) == 1
    result = json.loads(lines[0])
    assert set(result) == RESULT_KEYS
    assert (result["id"], result["status"]) == ("val-7", "valid")
    assert result["fingerprint"] == service().fingerprint()
    captured = capsys.readouterr()
    assert SECRET not in captured.out and SECRET not in captured.err


def test_main_rejects_oversized_requests(harness):
    stdout = io.StringIO()
    oversized = io.StringIO("x" * (validation.MAX_REQUEST_BYTES + 1))
    validation.main(oversized, stdout, registry=harness.registry, configure_logging=False)
    result = json.loads(stdout.getvalue())
    assert (result["status"], result["errorCode"]) == ("unvalidated", "invalid_request")


def _run_python(args: List[str], stdin: str) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if not k.startswith("ECHOFLUX_")}
    env["PYTHONPATH"] = str(ROOT)
    env["ECHOFLUX_DATA_DIR"] = os.environ.get("ECHOFLUX_DATA_DIR", "")
    return subprocess.run(
        [sys.executable, *args],
        input=stdin.encode("utf-8"),
        capture_output=True,
        cwd=str(ROOT),
        env=env,
        timeout=60,
    )


def test_module_entrypoint_process_contract_with_builtin_registry():
    # Local providers never reach the network, so the real registry is safe here.
    doc = {
        "id": "val-local",
        "capability": "stt",
        "service": ServiceConfig(service="stt", provider_id="local").to_dict(),
        "secrets": {},
    }
    proc = _run_python(["-m", "engine.providers.validation"], json.dumps(doc))
    assert proc.returncode == 0
    assert proc.stderr == b""
    lines = proc.stdout.decode("utf-8").splitlines()
    assert len(lines) == 1
    result = json.loads(lines[0])
    assert set(result) == RESULT_KEYS
    assert (result["id"], result["status"], result["errorCode"]) == (
        "val-local",
        "unvalidated",
        ERROR_NOT_SUPPORTED,
    )

    proc = _run_python(["-m", "engine.providers.validation"], f"{{broken {SECRET}")
    assert proc.returncode == 0 and proc.stderr == b""
    result = json.loads(proc.stdout)
    assert (result["status"], result["errorCode"]) == ("unvalidated", "invalid_request")
    assert SECRET.encode() not in proc.stdout


def test_subprocess_with_noisy_fake_adapter_keeps_stdout_and_stderr_clean():
    script = textwrap.dedent(
        f"""
        import logging, sys
        sys.path.insert(0, {str(ROOT / "tests")!r})
        from test_provider_validation import Harness, SECRET
        from engine.providers import validation
        from engine.providers.types import ProviderHealthCheck, ProviderHealthStatus

        harness = Harness()

        def noisy():
            print("stdout", SECRET)
            print("stderr", SECRET, file=sys.stderr)
            logging.getLogger("echoflux.providers.fake").error("log %s", SECRET)
            logging.getLogger("thirdparty").error("log %s", SECRET)
            import warnings
            warnings.warn(SECRET)
            return ProviderHealthCheck(status=ProviderHealthStatus.HEALTHY)

        harness.stt_behaviour = noisy
        sys.exit(validation.main(registry=harness.registry, builders=harness.builders()))
        """
    )
    doc = request_doc(id="val-proc")
    proc = _run_python(["-c", script], json.dumps(doc))
    assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")[-2000:]
    assert proc.stderr == b""
    assert SECRET.encode() not in proc.stdout
    [line] = proc.stdout.decode("utf-8").splitlines()
    result = json.loads(line)
    assert (result["id"], result["status"]) == ("val-proc", "valid")
