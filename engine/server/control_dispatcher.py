"""Dedicated dispatcher for Phase B privileged application control commands.

Exposes domain capabilities over the authenticated WebSocket control channel (Role.CONTROL):
1. Context Profiles: profile_create, profile_list, profile_get, profile_update, profile_delete, profile_attach
2. Conversations: conversation_create, conversation_start, conversation_list, conversation_get,
   conversation_rename, conversation_delete, conversation_export, conversation_import, message_update
3. Local Model Packs: pack_status, pack_install, pack_cancel, pack_retry, pack_delete, pack_check_readiness
4. Provider Manifests & Audio Probe: get_provider_manifests, audio_probe
5. Manual Assist, Summary Streaming & Quick Ask: configure_assistant, request_assist,
   request_summary, summary_cancel, assistant_ask, assistant_ask_clear

To protect tests/test_ws_contract.py invariants, this module is completely decoupled from
legacy 'msg_type' matching and 'type'-tagged event broadcasts.
"""

import asyncio
import logging
import threading
from typing import Any, Callable, Dict, Optional, Tuple

from engine.conversation.context_profiles import ContextProfileService
from engine.conversation.models import ConversationValidationError, validate_record_id
from engine.conversation.repository import ConversationRepository
from engine.core.config import Config
from engine.local_models import (
    LocalPackInstaller,
    PackCatalog,
    SystemHardwareProbe,
    builtin_catalog,
)
from engine.server.control_helpers import (
    DispatcherError,
)
from engine.server.control_helpers import (
    is_local_model_ready as _is_local_model_ready,
)
from engine.server.control_helpers import (
    run_summary_stream as _run_summary_stream,
)
from engine.server.control_helpers import (
    stored_assist_payload as _stored_assist_payload,
)
from engine.server.control_helpers import (
    warning_dict as _warning_dict,
)
from engine.server.handlers_conversations import (
    handle_conversation_create,
    handle_conversation_delete,
    handle_conversation_export,
    handle_conversation_get,
    handle_conversation_import,
    handle_conversation_list,
    handle_conversation_rename,
    handle_conversation_start,
    handle_message_update,
)
from engine.server.handlers_packs import (
    handle_audio_probe,
    handle_get_provider_manifests,
    handle_pack_cancel,
    handle_pack_check_readiness,
    handle_pack_delete,
    handle_pack_install,
    handle_pack_status,
)
from engine.server.handlers_profiles import (
    handle_profile_attach,
    handle_profile_create,
    handle_profile_delete,
    handle_profile_get,
    handle_profile_list,
    handle_profile_update,
)
from engine.server.protocol import (
    BootstrapTokens,
    complete_frame,
    encode_frame,
    error_frame,
)

logger = logging.getLogger("server.control_dispatcher")

__all__ = ["ControlCommandDispatcher", "DispatcherError", "_is_local_model_ready"]


class ControlCommandDispatcher:
    """Dispatches privileged control commands for Phase B application APIs."""

    def __init__(
        self,
        engine: Any = None,
        conversation_repo: Optional[ConversationRepository] = None,
        profile_service: Optional[ContextProfileService] = None,
        pack_installer: Optional[LocalPackInstaller] = None,
        catalog: Optional[PackCatalog] = None,
        probe: Optional[Any] = None,
        assistant_service: Optional[Any] = None,
        summary_service: Optional[Any] = None,
    ) -> None:
        self._engine = engine
        self._repo = (
            conversation_repo
            or getattr(engine, "_conversation_repo", None)
            or ConversationRepository()
        )
        self._profiles = profile_service or ContextProfileService(self._repo)
        self._installer = pack_installer
        self._catalog = catalog
        self._probe = probe
        self._assistant_service = assistant_service
        self._summary_service = summary_service

        self._active_tasks: Dict[str, asyncio.Task] = {}
        self._summary_tasks_by_conv: Dict[str, asyncio.Task] = {}
        self._pack_cancels: Dict[str, threading.Event] = {}
        self._pack_progress: Dict[str, Dict[str, Any]] = {}
        self._pack_install_active: Optional[str] = None

        self._handlers: Dict[str, Callable[[Dict[str, Any]], Any]] = {
            # Context Profiles
            "profile_create": lambda p: handle_profile_create(self.profile_service, p),
            "profile_list": lambda p: handle_profile_list(self.profile_service, p),
            "profile_get": lambda p: handle_profile_get(self.profile_service, p),
            "profile_update": lambda p: handle_profile_update(self.profile_service, p),
            "profile_delete": lambda p: handle_profile_delete(self.profile_service, p),
            "profile_attach": lambda p: handle_profile_attach(self.profile_service, p),
            # Conversation Management
            "conversation_create": lambda p: handle_conversation_create(self.conversation_repo, p),
            "conversation_start": lambda p: handle_conversation_start(self.profile_service, p),
            "conversation_list": lambda p: handle_conversation_list(self.conversation_repo, p),
            "conversation_get": lambda p: handle_conversation_get(
                self.conversation_repo, self.profile_service, p
            ),
            "conversation_rename": lambda p: handle_conversation_rename(self.conversation_repo, p),
            "conversation_delete": lambda p: handle_conversation_delete(self.conversation_repo, p),
            "conversation_export": lambda p: handle_conversation_export(self.conversation_repo, p),
            "conversation_import": lambda p: handle_conversation_import(self.conversation_repo, p),
            "message_update": lambda p: handle_message_update(
                self.conversation_repo, self._engine, p
            ),
            # Local Model Packs
            "pack_status": lambda p: handle_pack_status(self, p),
            "pack_install": self._handle_pack_install,
            "pack_cancel": lambda p: handle_pack_cancel(self, p),
            "pack_retry": self._handle_pack_install,
            "pack_delete": lambda p: handle_pack_delete(self, p),
            "pack_check_readiness": lambda p: handle_pack_check_readiness(self, p),
            # Provider Manifests & Audio Input Probe
            "get_provider_manifests": handle_get_provider_manifests,
            "audio_probe": lambda p: handle_audio_probe(self._engine, p),
            # Manual Assist, Streaming Summary & Quick Ask
            "configure_assistant": self._handle_configure_assistant,
            "request_assist": self._handle_request_assist,
            "request_suggestion": self._handle_request_assist,
            "request_summary": self._handle_request_summary,
            "summary_cancel": self._handle_summary_cancel,
            "assistant_ask": self._handle_assistant_ask,
            "assistant_ask_clear": self._handle_assistant_ask_clear,
        }

    @property
    def conversation_repo(self) -> ConversationRepository:
        if self._engine is not None and getattr(self._engine, "_conversation_repo", None):
            return self._engine._conversation_repo
        return self._repo

    @property
    def profile_service(self) -> ContextProfileService:
        if self._engine is not None and getattr(self._engine, "_conversation_repo", None):
            return ContextProfileService(self._engine._conversation_repo)
        return self._profiles

    def _get_catalog(self) -> PackCatalog:
        return self._catalog or builtin_catalog()

    def _get_probe(self) -> Any:
        return self._probe or SystemHardwareProbe()

    def _get_config(self) -> Config:
        cfg = getattr(self._engine, "_app_config", None) if self._engine is not None else None
        return cfg if cfg is not None else Config()

    def _get_installer(self) -> LocalPackInstaller:
        if self._installer is not None:
            return self._installer
        return LocalPackInstaller.from_config(
            self._get_config(), self._get_catalog(), self._get_probe()
        )

    def _get_assistant_service(self) -> Any:
        if self._assistant_service is not None:
            return self._assistant_service
        if self._engine is not None and getattr(self._engine, "_assistant_service", None):
            return self._engine._assistant_service
        return None

    def _get_summary_service(self) -> Any:
        if self._summary_service is not None:
            return self._summary_service
        if self._engine is not None and getattr(self._engine, "_summary_service", None):
            return self._engine._summary_service
        return None

    def can_handle(self, command_name: str) -> bool:
        return command_name in self._handlers

    def extract_command(
        self, frame: Any
    ) -> Optional[Tuple[str, str, Dict[str, Any]]]:
        """Extract (request_id, command_name, payload) if frame is handled by this dispatcher."""
        if not isinstance(frame, dict):
            return None
        req_id = frame.get("id") or frame.get("request_id") or ""
        op = frame.get("op")

        if op == "command":
            cmd = frame.get("command")
            if isinstance(cmd, str) and cmd in self._handlers:
                payload = frame.get("payload") or {}
                return (req_id, cmd, payload if isinstance(payload, dict) else {})

        if op == "request":
            cmd_obj = frame.get("command")
            if isinstance(cmd_obj, dict):
                inner_cmd = cmd_obj.get("command")
                if isinstance(inner_cmd, str) and inner_cmd in self._handlers:
                    payload = cmd_obj.get("payload") or {}
                    return (req_id, inner_cmd, payload if isinstance(payload, dict) else {})
                inner_type = cmd_obj.get("type")
                if (
                    isinstance(inner_type, str)
                    and inner_type in self._handlers
                    and inner_type not in ("request_suggestion", "request_summary")
                ):
                    payload = {k: v for k, v in cmd_obj.items() if k != "type"}
                    return (req_id, inner_type, payload)

        return None

    async def dispatch(
        self,
        request_id: str,
        command_name: str,
        payload: Dict[str, Any],
        websocket: Any,
        tokens: Optional[BootstrapTokens],
    ) -> None:
        handler = self._handlers.get(command_name)
        if not handler:
            await self._send_error(
                websocket, request_id, "unknown_command", f"Unknown control command: {command_name}", tokens
            )
            return

        if command_name == "request_summary":
            task = asyncio.create_task(
                self._dispatch_summary_streaming(request_id, payload, websocket, tokens)
            )
            self._active_tasks[request_id] = task
            cid = payload.get("conversation_id")
            if isinstance(cid, str) and cid:
                self._summary_tasks_by_conv[cid] = task

            def _cleanup_summary_task(done_task: asyncio.Task) -> None:
                self._active_tasks.pop(request_id, None)
                if isinstance(cid, str) and self._summary_tasks_by_conv.get(cid) is done_task:
                    self._summary_tasks_by_conv.pop(cid, None)

            task.add_done_callback(_cleanup_summary_task)
            return

        try:
            result = await handler(payload)
            response = {
                "op": "response",
                "id": request_id,
                "data": result,
                "message": result,
            }
            await websocket.send(encode_frame(response, tokens))
            await websocket.send(encode_frame(complete_frame(request_id), tokens))
        except DispatcherError as err:
            logger.warning("Control command '%s' failed: %s (%s)", command_name, err.code, err.message)
            await self._send_error(websocket, request_id, err.code, err.message, tokens)
        except Exception as e:
            logger.exception("Control command '%s' unexpected error: %s", command_name, e)
            await self._send_error(
                websocket, request_id, "command_failed", "Command execution failed", tokens
            )

    async def _send_error(
        self,
        websocket: Any,
        request_id: str,
        code: str,
        message: str,
        tokens: Optional[BootstrapTokens],
    ) -> None:
        frame = error_frame(code, message, request_id)
        await websocket.send(encode_frame(frame, tokens))
        await websocket.send(encode_frame(complete_frame(request_id), tokens))

    async def _handle_pack_install(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return await handle_pack_install(self, payload)

    async def _handle_configure_assistant(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        secrets = payload.pop("secrets", None)
        service = payload.get("service")
        if not isinstance(service, dict):
            raise DispatcherError("invalid_payload", "service must be an object")
        secret_map: Dict[str, str] = {}
        if isinstance(secrets, dict):
            for k, v in secrets.items():
                if isinstance(k, str) and isinstance(v, str) and v:
                    secret_map[k] = v
        try:
            if self._engine is None or not hasattr(self._engine, "configure_assistant_from_service"):
                raise DispatcherError("assistant_unavailable", "Engine cannot configure assistant")
            configured = await self._engine.configure_assistant_from_service(service, secret_map)
            error_code = (
                None if configured else getattr(self._engine, "_last_assistant_error_code", None)
            )
            return {
                "configured": bool(configured),
                **({"error_code": error_code} if error_code else {}),
            }
        finally:
            secret_map.clear()
            if isinstance(secrets, dict):
                secrets.clear()

    async def _handle_request_assist(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        cid = payload.get("conversation_id")
        mid = payload.get("message_id") or payload.get("entry_id")

        if not cid or not isinstance(cid, str):
            raise DispatcherError("invalid_request", "conversation_id is required")
        if not mid or not isinstance(mid, str):
            raise DispatcherError("invalid_request", "message_id is required")

        service = self._get_assistant_service()
        if service is None:
            raise DispatcherError("assistant_unavailable", "Assistant service is not configured")

        outcome = await service.request_assist(cid, mid)
        if not outcome.ok:
            err = outcome.error
            raise DispatcherError(err.code.value, err.message)

        provider_dict = outcome.provider.to_dict() if outcome.provider else None
        return _stored_assist_payload(cid, mid, outcome, provider=provider_dict)

    async def _dispatch_summary_streaming(
        self,
        request_id: str,
        payload: Dict[str, Any],
        websocket: Any,
        tokens: Optional[BootstrapTokens],
    ) -> None:
        cid = payload.get("conversation_id")
        if not cid or not isinstance(cid, str):
            await self._send_error(websocket, request_id, "invalid_request", "conversation_id is required", tokens)
            return

        try:
            validate_record_id("conversation_id", cid)
        except ConversationValidationError as err:
            await self._send_error(websocket, request_id, "invalid_request", str(err), tokens)
            return

        service = self._get_summary_service()
        if service is None:
            await self._send_error(
                websocket, request_id, "assistant_unavailable", "Summary service is not configured", tokens
            )
            return

        async def _on_chunk(text: str) -> None:
            frame = {
                "op": "response",
                "id": request_id,
                "data": {"chunk": text, "delta_text": text, "done": False},
                "message": {"type": "llm_chunk", "text": text, "conversation_id": cid},
            }
            await websocket.send(encode_frame(frame, tokens))
            if self._engine is not None and hasattr(self._engine, "broadcast_event"):
                await self._engine.broadcast_event(
                    {"type": "llm_chunk", "text": text, "conversation_id": cid}
                )

        async def _on_failure(
            code: str,
            *,
            after_chunks: bool = False,
            message: Optional[str] = None,
            retryable: bool = False,
            provider_code: Optional[str] = None,
            warning: Optional[dict] = None,
        ) -> None:
            msg = (
                "Summary request cancelled"
                if code == "cancelled"
                else (message or "Summary failed unexpectedly")
            )
            await self._send_error(websocket, request_id, code, msg, tokens)

        async def _on_success(outcome: Any, w_payload: Optional[dict]) -> None:
            summary_id = outcome.stored.summary_id if outcome.stored else None
            summary_md = outcome.stored.summary_markdown if outcome.stored else ""
            provider_dict = outcome.provider.to_dict() if outcome.provider else None
            done_frame = {
                "op": "response",
                "id": request_id,
                "data": {
                    "done": True,
                    "conversation_id": cid,
                    "summary_id": summary_id,
                    "summary_markdown": summary_md,
                    "warning": w_payload,
                    "provider": provider_dict,
                    "included_message_count": outcome.included_message_count,
                    "omitted_message_count": outcome.omitted_message_count,
                },
                "message": {
                    "type": "llm_done",
                    "conversation_id": cid,
                    "summary_id": summary_id,
                    "warning": w_payload,
                    "provider": provider_dict,
                    "included_message_count": outcome.included_message_count,
                    "omitted_message_count": outcome.omitted_message_count,
                },
            }
            await websocket.send(encode_frame(done_frame, tokens))
            await websocket.send(encode_frame(complete_frame(request_id), tokens))

        try:
            await _run_summary_stream(service, cid, _on_chunk, _on_failure, _on_success)
        finally:
            self._active_tasks.pop(request_id, None)

    async def _handle_request_summary(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Non-streaming fallback if invoked directly via handler map."""
        cid = payload.get("conversation_id")
        if not cid or not isinstance(cid, str):
            raise DispatcherError("invalid_request", "conversation_id is required")

        service = self._get_summary_service()
        if service is None:
            raise DispatcherError("assistant_unavailable", "Summary service is not configured")

        outcome = await service.request_summary(cid)
        if not outcome.ok:
            raise DispatcherError(outcome.error.code.value, outcome.error.message)

        return {
            "conversation_id": cid,
            "summary_id": outcome.stored.summary_id if outcome.stored else None,
            "summary_markdown": outcome.stored.summary_markdown if outcome.stored else "",
            "warning": _warning_dict(outcome.warning),
            "provider": outcome.provider.to_dict() if outcome.provider else None,
            "included_message_count": outcome.included_message_count,
            "omitted_message_count": outcome.omitted_message_count,
        }

    async def _handle_summary_cancel(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        target_id = payload.get("request_id")
        conversation_id = payload.get("conversation_id")

        if target_id and isinstance(target_id, str):
            task = self._active_tasks.get(target_id)
            if task and not task.done():
                task.cancel()
                return {"cancelled": True, "request_id": target_id}

        cid = conversation_id if isinstance(conversation_id, str) and conversation_id else None
        if cid and cid in self._summary_tasks_by_conv:
            task = self._summary_tasks_by_conv.get(cid)
            if task is not None and not task.done():
                task.cancel()
                return {"cancelled": True, "request_id": target_id}

        if (
            self._engine is not None
            and hasattr(self._engine, "cancel_summary")
            and self._engine.cancel_summary(cid)
        ):
            return {"cancelled": True, "request_id": target_id}

        return {"cancelled": False, "request_id": target_id}

    async def _handle_assistant_ask(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        cid = payload.get("conversation_id")
        question = payload.get("question")
        if not isinstance(cid, str) or not cid:
            raise DispatcherError("invalid_request", "conversation_id is required")
        if not isinstance(question, str) or not question.strip():
            raise DispatcherError("invalid_request", "question is required")

        service = self._get_summary_service()
        if service is None or not hasattr(service, "ask_question"):
            raise DispatcherError("assistant_unavailable", "Assistant service is not configured")

        outcome = await service.ask_question(cid, question)
        if not outcome.ok:
            err = outcome.error
            raise DispatcherError(err.code.value, err.message)

        return {
            "conversation_id": cid,
            "ask_id": getattr(outcome, "ask_id", None),
            "answer_markdown": outcome.answer_markdown,
            "warning": _warning_dict(outcome.warning),
            "provider": outcome.provider.to_dict() if outcome.provider else None,
            "included_message_count": outcome.included_message_count,
            "omitted_message_count": outcome.omitted_message_count,
        }

    async def _handle_assistant_ask_clear(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        cid = payload.get("conversation_id")
        if not isinstance(cid, str) or not cid:
            raise DispatcherError("invalid_request", "conversation_id is required")
        repo = self.conversation_repo
        if not hasattr(repo, "clear_ask_turns"):
            return {"cleared": 0}
        try:
            cleared = await asyncio.to_thread(repo.clear_ask_turns, cid)
        except ConversationValidationError as err:
            raise DispatcherError("invalid_request", str(err)) from None
        return {"cleared": cleared, "conversation_id": cid}
