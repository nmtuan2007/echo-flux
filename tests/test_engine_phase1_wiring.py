"""Verification tests for Phase 1 EchoFluxEngine start secret & assistant wiring."""

import asyncio
from unittest.mock import AsyncMock, patch

from engine.conversation.repository import ConversationRepository
from engine.main import EchoFluxEngine


def test_extract_start_secrets_scrubs_config_and_routes_by_capability():
    engine = EchoFluxEngine(insecure_dev=True)
    raw_config = {
        "audio.source": "microphone",
        "stt.api_key": "sk-stt-secret",
        "translation.api_key": "sk-trans-secret",
        "assistant.api_key": "sk-assist-secret",
        "services": {"stt": {"providerId": "openai-realtime", "enabled": True}},
    }

    cleaned, flat, per_service = engine._extract_start_secrets(raw_config, None)

    assert "stt.api_key" not in cleaned
    assert "translation.api_key" not in cleaned
    assert "assistant.api_key" not in cleaned
    assert cleaned["audio.source"] == "microphone"
    assert flat == {}
    assert per_service["stt"] == {"api_key": "sk-stt-secret"}
    assert per_service["translation"] == {"api_key": "sk-trans-secret"}
    assert per_service["assistant"] == {"api_key": "sk-assist-secret"}


def test_initialize_pipeline_persists_source_labels_and_existing_conversation_id(tmp_path):
    repo = ConversationRepository(tmp_path / "conv.db")
    engine = EchoFluxEngine(insecure_dev=True, conversation_repo=repo)

    settings = {
        "audio.source": "microphone",
        "vad.enabled": False,
        "conversation_id": "conv-custom-1",
        "conversation_title": "Q4 Review",
        "source_labels": {"mic": "Tuan", "system": "Client"},
    }
    with patch("engine.audio.microphone.MicrophoneInput"):
        engine._initialize_pipeline(settings)

    stored = repo.get_conversation("conv-custom-1")
    assert stored is not None
    assert stored.title == "Q4 Review"
    assert stored.source_labels == {"mic": "Tuan", "system": "Client"}


def test_start_assistant_service_wires_and_scrubs_secrets(tmp_path):
    repo = ConversationRepository(tmp_path / "assist.db")
    engine = EchoFluxEngine(insecure_dev=True, conversation_repo=repo)
    engine._service_secrets = {
        "stt": {},
        "translation": {},
        "assistant": {"api_key": "sk-assist-ephemeral"},
    }

    settings = {
        "services": {
            "assistant": {
                "enabled": True,
                "providerId": "openai",
                "endpoint": "",
                "modelOverride": "",
                "options": {},
                "credentialRefs": {
                    "api_key": {"configId": "openai", "slot": "api_key", "version": 1}
                },
            }
        }
    }

    fake_provider = AsyncMock()
    fake_provider.provider_id = "openai"

    with patch("engine.providers.registry.ProviderRegistry.load_factory", return_value=lambda *a, **kw: fake_provider):
        asyncio.run(engine._start_assistant_service(settings))

    assert engine._assistant_service is not None
    assert engine._summary_service is not None
    assert "assistant" not in engine._service_secrets
    fake_provider.start.assert_awaited_once()

    asyncio.run(engine._start_assistant_service({"services": {"assistant": {"enabled": False}}}))
    assert engine._assistant_service is None
    assert engine._summary_service is None
    fake_provider.stop.assert_awaited_once()
