"""Bounded async client for the Gemini ``generateContent`` text API.

Owns transport, authentication, and request/response shape only. It never builds prompts,
chooses fallback, or knows about sessions; adapters pass fully built messages and the
already-resolved endpoint, model, and key. Nothing here defaults a provider setting: the
recommended endpoint and model live in ``engine.providers.manifests``.

Wire mapping (checked 2026-09-14 against ai.google.dev ``api/generate-content``,
``gemini-api/docs/generate-content/text-generation``, ``structured-output``,
``safety-settings``, and ``api-errors``):

=========================  ================================================================
Operation                  Request
=========================  ================================================================
Completion                 ``POST {endpoint}/models/{model}:generateContent``
Streaming                  ``POST {endpoint}/models/{model}:streamGenerateContent?alt=sse``
Probe                      ``GET {endpoint}/models`` (idempotent, verifies the key)
Authentication             ``x-goog-api-key`` header; the key is never put in the URL
=========================  ================================================================

- ``system``/``developer`` messages become ``systemInstruction``; ``user`` stays ``user``
  and ``assistant`` becomes ``model`` in ``contents``.
- ``temperature`` and ``maxOutputTokens`` go in ``generationConfig``; an optional
  ``GeminiResponseFormat`` adds ``responseMimeType`` and ``responseJsonSchema``.
- Each SSE event is a complete ``GenerateContentResponse``; there is no ``[DONE]`` marker,
  so a stream is complete only once a candidate reports ``finishReason``.
- ``promptFeedback.blockReason`` (prompt blocked, no candidates) and safety-style
  ``finishReason`` values (content withheld) are normalized errors flagged ``blocked``.
  ``thought`` parts are never returned as answer text.

Transport bounds, retry policy, and stream cancellation are inherited unchanged from
``OpenAICompatibleTextClient`` so both text protocols keep identical safety guarantees.
"""

import asyncio
import json
import re
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Optional, Sequence, Tuple
from urllib.parse import quote, urlsplit

from engine.core.logging import get_logger
from engine.providers.text.openai_compatible import (
    OpenAICompatibleTextClient,
    TextCompletion,
    TextConfigError,
    TextEndpointStrategy,
    TextGenerationOptions,
    TextMessage,
    TextStreamChunk,
    TextTransportOptions,
    _validate_base_url,
    classify_http_status,
    config_error_to_provider_error,
    make_text_error,
    sanitize_provider_code,
)
from engine.providers.types import (
    ProviderCapability,
    ProviderError,
    ProviderErrorCode,
    ProviderRuntimeError,
)

logger = get_logger("providers.text.gemini")

GEMINI_MODE = "gemini"
API_KEY_HEADER = "x-goog-api-key"

_MODEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._\-]{0,127}$")
_SYSTEM_ROLES = frozenset({"system", "developer"})
_CONTENT_ROLES = {"user": "user", "assistant": "model"}

# Candidate finish reasons where the content was withheld by a policy filter.
BLOCKED_FINISH_REASONS = frozenset(
    {
        "SAFETY",
        "RECITATION",
        "BLOCKLIST",
        "PROHIBITED_CONTENT",
        "SPII",
        "IMAGE_SAFETY",
        "IMAGE_PROHIBITED_CONTENT",
        "IMAGE_RECITATION",
    }
)
_UNSUPPORTED_LANGUAGE_FINISH = "LANGUAGE"

# Google API status strings (generateContent) and snake_case codes (newer API surfaces).
_PROVIDER_CODE_MAP: Dict[str, ProviderErrorCode] = {
    "invalid_argument": ProviderErrorCode.INVALID_REQUEST,
    "invalid_request": ProviderErrorCode.INVALID_REQUEST,
    "failed_precondition": ProviderErrorCode.INVALID_REQUEST,
    "unauthenticated": ProviderErrorCode.AUTH_FAILED,
    "authentication": ProviderErrorCode.AUTH_FAILED,
    "permission_denied": ProviderErrorCode.AUTH_FAILED,
    "api_key_invalid": ProviderErrorCode.AUTH_FAILED,
    "not_found": ProviderErrorCode.MODEL_UNAVAILABLE,
    "model_not_found": ProviderErrorCode.MODEL_UNAVAILABLE,
    "resource_exhausted": ProviderErrorCode.RATE_LIMITED,
    "rate_limit_exceeded": ProviderErrorCode.RATE_LIMITED,
    "quota_exceeded": ProviderErrorCode.QUOTA_EXCEEDED,
    "internal": ProviderErrorCode.INTERNAL_ERROR,
    "api_error": ProviderErrorCode.INTERNAL_ERROR,
    "unavailable": ProviderErrorCode.NETWORK_UNAVAILABLE,
    "service_unavailable": ProviderErrorCode.NETWORK_UNAVAILABLE,
    "deadline_exceeded": ProviderErrorCode.TIMEOUT,
}


# ── Request options ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class GeminiResponseFormat:
    """Structured output request. The schema is supplied by the caller, never invented here."""

    mime_type: str = "application/json"
    json_schema: Optional[Dict[str, Any]] = None


# ── Endpoint strategy ────────────────────────────────────────────────────────


class GeminiStrategy(TextEndpointStrategy):
    """Gemini API endpoint and ``x-goog-api-key`` authentication.

    ``endpoint`` is the API version root (for example ``.../v1beta``). A query string is
    rejected outright so a key can never ride along in the URL.
    """

    mode = GEMINI_MODE

    def __init__(self, endpoint: str, api_key: str) -> None:
        base = _validate_base_url(endpoint, "endpoint", allow_insecure=False)
        if urlsplit(base).query:
            raise TextConfigError("endpoint")
        # Generation URLs are model-scoped; a pasted ".../models" root would double the path.
        if urlsplit(base).path.rstrip("/").lower().endswith("/models"):
            raise TextConfigError("endpoint")
        if not isinstance(api_key, str) or not api_key.strip():
            raise TextConfigError("api_key")
        if any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in api_key):
            raise TextConfigError("api_key")
        self._base_url = base
        self._api_key = api_key

    def chat_url(self) -> str:
        # Model-independent resource root; used for redacted logging only.
        return f"{self._base_url}/models"

    def probe_url(self) -> Optional[str]:
        return f"{self._base_url}/models"

    def generate_url(self, model: str, *, stream: bool) -> str:
        model_id = self.body_model(model)
        method = "streamGenerateContent?alt=sse" if stream else "generateContent"
        return f"{self._base_url}/models/{quote(model_id, safe='._-')}:{method}"

    def body_model(self, model: str) -> Optional[str]:
        """Validated model id for the URL path; Gemini sends no model in the body."""
        if not isinstance(model, str):
            raise TextConfigError("model")
        candidate = model.strip()
        if candidate.startswith("models/"):
            candidate = candidate[len("models/") :]
        if not _MODEL_RE.match(candidate):
            raise TextConfigError("model")
        return candidate

    def auth_headers(self) -> Dict[str, str]:
        return {API_KEY_HEADER: self._api_key}

    def secret_values(self) -> Tuple[str, ...]:
        return (self._api_key,)


# ── Request shape and response parsing ───────────────────────────────────────


def build_generate_body(
    messages: Sequence[TextMessage],
    generation: TextGenerationOptions,
    response_format: Optional[GeminiResponseFormat] = None,
) -> Dict[str, Any]:
    if not messages:
        raise TextConfigError("messages")
    system_parts: List[Dict[str, str]] = []
    contents: List[Dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, TextMessage) or not isinstance(message.content, str):
            raise TextConfigError("messages")
        if message.role in _SYSTEM_ROLES:
            system_parts.append({"text": message.content})
        elif message.role in _CONTENT_ROLES:
            contents.append(
                {"role": _CONTENT_ROLES[message.role], "parts": [{"text": message.content}]}
            )
        else:
            raise TextConfigError("messages")
    if not contents:
        raise TextConfigError("messages")
    body: Dict[str, Any] = {}
    if system_parts:
        body["systemInstruction"] = {"parts": system_parts}
    body["contents"] = contents

    config: Dict[str, Any] = {}
    temperature = generation.temperature
    if temperature is not None:
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
            raise TextConfigError("temperature")
        if not 0.0 <= float(temperature) <= 2.0:
            raise TextConfigError("temperature")
        config["temperature"] = float(temperature)
    max_tokens = generation.max_output_tokens
    if max_tokens is not None:
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
            raise TextConfigError("max_output_tokens")
        config["maxOutputTokens"] = max_tokens
    if response_format is not None:
        if not isinstance(response_format, GeminiResponseFormat):
            raise TextConfigError("response_format")
        if not isinstance(response_format.mime_type, str) or not response_format.mime_type:
            raise TextConfigError("response_format")
        config["responseMimeType"] = response_format.mime_type
        if response_format.json_schema is not None:
            if not isinstance(response_format.json_schema, dict):
                raise TextConfigError("response_format")
            config["responseJsonSchema"] = response_format.json_schema
    if config:
        body["generationConfig"] = config
    return body


def _count(value: Any) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def parse_usage_metadata(usage: Any) -> Tuple[Optional[int], Optional[int]]:
    """Input is ``promptTokenCount``; output bills ``candidatesTokenCount`` plus thoughts."""
    if not isinstance(usage, dict):
        return None, None
    input_tokens = _count(usage.get("promptTokenCount"))
    candidates = _count(usage.get("candidatesTokenCount"))
    thoughts = _count(usage.get("thoughtsTokenCount"))
    if candidates is None and thoughts is None:
        return input_tokens, None
    return input_tokens, (candidates or 0) + (thoughts or 0)


class _MalformedResponseError(Exception):
    pass


@dataclass(frozen=True)
class _ParsedResponse:
    text: str
    response_id: Optional[str]
    finish_reason: Optional[str]
    block_reason: Optional[str]
    has_candidate: bool
    input_tokens: Optional[int]
    output_tokens: Optional[int]


def _identifier(value: Any) -> Optional[str]:
    return sanitize_provider_code(value)


def parse_generate_response(payload: Any) -> _ParsedResponse:
    """Shape-check one ``GenerateContentResponse``. Policy decisions are left to the caller."""
    if not isinstance(payload, dict):
        raise _MalformedResponseError("payload")
    feedback = payload.get("promptFeedback")
    block_reason = None
    if isinstance(feedback, dict):
        reason = feedback.get("blockReason")
        if reason not in (None, "BLOCK_REASON_UNSPECIFIED"):
            block_reason = _identifier(reason) or "OTHER"
    candidates = payload.get("candidates")
    text = ""
    finish = None
    has_candidate = False
    if candidates is not None:
        if not isinstance(candidates, list):
            raise _MalformedResponseError("candidates")
        if candidates:
            candidate = candidates[0]
            if not isinstance(candidate, dict):
                raise _MalformedResponseError("candidate")
            has_candidate = True
            finish_value = candidate.get("finishReason")
            if finish_value not in (None, "FINISH_REASON_UNSPECIFIED"):
                finish = _identifier(finish_value) or "OTHER"
            text = _candidate_text(candidate.get("content"))
    input_tokens, output_tokens = parse_usage_metadata(payload.get("usageMetadata"))
    return _ParsedResponse(
        text=text,
        response_id=_identifier(payload.get("responseId")),
        finish_reason=finish,
        block_reason=block_reason,
        has_candidate=has_candidate,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def _candidate_text(content: Any) -> str:
    if content is None:
        # Blocked or token-limited candidates may omit content entirely.
        return ""
    if not isinstance(content, dict):
        raise _MalformedResponseError("content")
    parts = content.get("parts")
    if parts is None:
        return ""
    if not isinstance(parts, list):
        raise _MalformedResponseError("parts")
    texts = []
    for part in parts:
        if not isinstance(part, dict) or part.get("thought") is True:
            continue
        if isinstance(part.get("text"), str):
            texts.append(part["text"])
    return "".join(texts)


def _error_provider_code(payload: Any) -> Optional[str]:
    """Prefer a specific ``details[].reason`` (e.g. API_KEY_INVALID) over the status string."""
    if not isinstance(payload, dict):
        return None
    error = payload.get("error")
    if not isinstance(error, dict):
        return None
    details = error.get("details")
    if isinstance(details, list):
        for item in details:
            if isinstance(item, dict) and isinstance(item.get("reason"), str):
                return item["reason"]
    status = error.get("status")
    if isinstance(status, str):
        return status
    code = error.get("code")
    return code if isinstance(code, str) else None


# ── Client ───────────────────────────────────────────────────────────────────


class GeminiTextClient(OpenAICompatibleTextClient):
    """Async Gemini request/stream client. One instance per adapter; close with ``aclose``."""

    def __init__(
        self,
        strategy: GeminiStrategy,
        *,
        capability: ProviderCapability,
        options: Optional[TextTransportOptions] = None,
        http_transport: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        if not isinstance(strategy, GeminiStrategy):
            raise TextConfigError("strategy")
        super().__init__(
            strategy,
            capability=capability,
            options=options,
            http_transport=http_transport,
            **kwargs,
        )
        self._gemini: GeminiStrategy = strategy

    def __repr__(self) -> str:
        return f"GeminiTextClient(mode={self._strategy.mode!r})"

    # ── Public operations ────────────────────────────────────────────────────

    async def complete(  # type: ignore[override]
        self,
        model: str,
        messages: Sequence[TextMessage],
        generation: Optional[TextGenerationOptions] = None,
        response_format: Optional[GeminiResponseFormat] = None,
    ) -> TextCompletion:
        url, body = self._prepare(model, messages, generation, response_format, stream=False)
        began = self._clock()
        try:
            payload = await asyncio.wait_for(
                self._post_json_to(url, body), timeout=self._options.total_timeout_s
            )
        except asyncio.TimeoutError:
            raise self._raise_code(ProviderErrorCode.TIMEOUT) from None
        parsed = self._parse(payload)
        self._raise_if_blocked(parsed, streamed_text=False)
        if not parsed.has_candidate:
            logger.warning("Gemini completion had no candidate: mode=%s", self._strategy.mode)
            raise self._raise_code(ProviderErrorCode.INVALID_RESPONSE)
        return TextCompletion(
            text=parsed.text,
            response_id=parsed.response_id,
            finish_reason=parsed.finish_reason,
            input_tokens=parsed.input_tokens,
            output_tokens=parsed.output_tokens,
            latency_ms=(self._clock() - began) * 1000.0,
        )

    async def stream(  # type: ignore[override]
        self,
        model: str,
        messages: Sequence[TextMessage],
        generation: Optional[TextGenerationOptions] = None,
        response_format: Optional[GeminiResponseFormat] = None,
    ) -> AsyncIterator[TextStreamChunk]:
        """Yield deltas then one ``is_done`` chunk. Closing the iterator closes the response."""
        url, body = self._prepare(model, messages, generation, response_format, stream=True)
        began = self._clock()
        deadline = began + self._options.total_timeout_s
        response = await self._send_with_retry("POST", url, body, stream=True, idempotent=False)
        try:
            await self._raise_for_status(response)
            parts: List[str] = []
            response_id: Optional[str] = None
            finish: Optional[str] = None
            usage: Tuple[Optional[int], Optional[int]] = (None, None)
            async for data in self._iter_sse_data(response, deadline):
                payload = self._decode_event(data)
                if payload is None:
                    continue
                if "error" in payload:
                    self._raise_stream_error(payload)
                parsed = self._parse(payload)
                self._raise_if_blocked(parsed, streamed_text=bool(parts))
                response_id = response_id or parsed.response_id
                if (parsed.input_tokens, parsed.output_tokens) != (None, None):
                    usage = (parsed.input_tokens, parsed.output_tokens)
                if parsed.text:
                    parts.append(parsed.text)
                    yield TextStreamChunk(delta_text=parsed.text)
                if parsed.finish_reason is not None:
                    finish = parsed.finish_reason
            if finish is None:
                raise self._raise_code(ProviderErrorCode.CONNECTION_CLOSED)
            yield TextStreamChunk(
                delta_text="",
                is_done=True,
                completion=TextCompletion(
                    text="".join(parts),
                    response_id=response_id,
                    finish_reason=finish,
                    input_tokens=usage[0],
                    output_tokens=usage[1],
                    latency_ms=(self._clock() - began) * 1000.0,
                ),
            )
        finally:
            await response.aclose()

    # ── Internals ────────────────────────────────────────────────────────────

    def _prepare(
        self,
        model: str,
        messages: Sequence[TextMessage],
        generation: Optional[TextGenerationOptions],
        response_format: Optional[GeminiResponseFormat],
        *,
        stream: bool,
    ) -> Tuple[str, Dict[str, Any]]:
        try:
            url = self._gemini.generate_url(model, stream=stream)
            body = build_generate_body(
                messages, generation or TextGenerationOptions(), response_format
            )
        except TextConfigError as exc:
            raise config_error_to_provider_error(
                exc, self._capability, self._strategy.mode
            ) from None
        return url, body

    async def _post_json_to(self, url: str, body: Dict[str, Any]) -> Any:
        response = await self._send_with_retry("POST", url, body, stream=True, idempotent=False)
        try:
            await self._raise_for_status(response)
            raw = await self._read_bounded(response, self._options.max_response_bytes)
        finally:
            await response.aclose()
        try:
            return json.loads(raw)
        except ValueError:
            raise self._raise_code(ProviderErrorCode.INVALID_RESPONSE) from None

    def _parse(self, payload: Any) -> _ParsedResponse:
        try:
            return parse_generate_response(payload)
        except _MalformedResponseError as exc:
            logger.warning(
                "Gemini response malformed: mode=%s part=%s", self._strategy.mode, exc.args[0]
            )
            raise self._raise_code(ProviderErrorCode.INVALID_RESPONSE) from None

    def _decode_event(self, data: str) -> Optional[Dict[str, Any]]:
        try:
            payload = json.loads(data)
        except ValueError:
            logger.debug("Gemini stream event not JSON: mode=%s", self._strategy.mode)
            return None
        return payload if isinstance(payload, dict) else None

    def _blocked(self, code: ProviderErrorCode, reason: str, stage: str) -> ProviderRuntimeError:
        base = make_text_error(
            code, self._capability, provider_code=reason, mode=self._strategy.mode
        )
        details = dict(base.details or {})
        details["blocked"] = stage
        logger.warning(
            "Gemini content blocked: mode=%s stage=%s reason=%s",
            self._strategy.mode,
            stage,
            reason,
        )
        return ProviderRuntimeError(
            ProviderError(
                code=base.code,
                capability=base.capability,
                message=base.message,
                retryable=False,
                details=details,
            )
        )

    def _raise_if_blocked(self, parsed: _ParsedResponse, *, streamed_text: bool) -> None:
        if parsed.block_reason is not None:
            raise self._blocked(ProviderErrorCode.INVALID_REQUEST, parsed.block_reason, "prompt")
        finish = parsed.finish_reason
        if finish in BLOCKED_FINISH_REASONS:
            # Withheld content is never partially usable, even if earlier deltas arrived.
            raise self._blocked(ProviderErrorCode.INVALID_RESPONSE, finish, "response")
        if finish == _UNSUPPORTED_LANGUAGE_FINISH and not parsed.text and not streamed_text:
            raise self._blocked(ProviderErrorCode.UNSUPPORTED_LANGUAGE, finish, "response")

    def _classify(self, status: Optional[int], provider_code: Optional[str]) -> ProviderErrorCode:
        code = classify_http_status(status) if status is not None else None
        mapped = _PROVIDER_CODE_MAP.get((provider_code or "").lower())
        if code is None:
            return mapped or ProviderErrorCode.INTERNAL_ERROR
        # A specific provider code refines a generic client status, never a server failure.
        if mapped is not None and status is not None and status < 500:
            return mapped
        return code

    def _raise_stream_error(self, payload: Dict[str, Any]) -> None:
        provider_code = sanitize_provider_code(_error_provider_code(payload), self._secrets)
        raise self._raise_code(self._classify(None, provider_code), provider_code=provider_code)

    async def _raise_for_status(self, response: Any) -> None:
        status = response.status_code
        if 200 <= status < 300:
            return
        provider_code: Optional[str] = None
        try:
            raw = await self._read_bounded(
                response, self._options.max_error_body_bytes, truncate=True
            )
            provider_code = sanitize_provider_code(
                _error_provider_code(json.loads(raw)), self._secrets
            )
        except (ProviderRuntimeError, ValueError):
            provider_code = None
        code = self._classify(status, provider_code)
        logger.warning(
            "Gemini request failed: mode=%s status=%d code=%s provider_code=%s",
            self._strategy.mode,
            status,
            code.value,
            provider_code,
        )
        raise self._raise_code(code, http_status=status, provider_code=provider_code)


__all__ = [
    "API_KEY_HEADER",
    "BLOCKED_FINISH_REASONS",
    "GEMINI_MODE",
    "GeminiResponseFormat",
    "GeminiStrategy",
    "GeminiTextClient",
    "build_generate_body",
    "parse_generate_response",
    "parse_usage_metadata",
]
