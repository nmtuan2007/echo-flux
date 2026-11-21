"""Dedicated dispatcher for Phase B privileged application control commands.

Exposes domain capabilities over the authenticated WebSocket control channel (Role.CONTROL):
1. Context Profiles: profile_create, profile_list, profile_get, profile_update, profile_delete, profile_attach
2. Conversations: conversation_create, conversation_start, conversation_list, conversation_get,
   conversation_rename, conversation_delete, conversation_export, conversation_import
3. Local Model Packs: pack_status, pack_install, pack_cancel, pack_retry, pack_check_readiness
4. Provider Manifests: get_provider_manifests
5. ID-Only Manual Assist: request_assist (and request_suggestion alias)
6. Transcript Summary Streaming: request_summary, summary_cancel

To protect tests/test_ws_contract.py invariants, this module is completely decoupled from
legacy 'msg_type' matching and 'type'-tagged event broadcasts.
"""

import asyncio
import contextlib
import json
import logging
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from engine.conversation.context_profiles import ContextProfileService
from engine.conversation.models import (
    ConversationRecord,
    ConversationValidationError,
    ProviderMetadata,
    SessionRecord,
    StoredAssistResult,
    StoredSummary,
    TranscriptMessage,
    import_id_for_browser_history,
    validate_record_id,
)
from engine.conversation.repository import (
    ConversationRepository,
    RecordConflictError,
    RecordNotFoundError,
    new_record_id,
)
from engine.core.config import Config
from engine.local_models import (
    HardwareProfile,
    InstallCancelledError,
    LocalPackInstaller,
    PackCatalog,
    PackInstallError,
    PackKind,
    SystemHardwareProbe,
    builtin_catalog,
    compatibility_issues,
    select_default_pack,
)
from engine.providers.manifests import ManifestField, ProviderManifest
from engine.providers.registry import build_default_registry
from engine.providers.types import AssistQuestion, AssistSuggestion, ProviderCapability
from engine.server.protocol import (
    BootstrapTokens,
    complete_frame,
    encode_frame,
    error_frame,
)

logger = logging.getLogger("server.control_dispatcher")


class DispatcherError(Exception):
    """Safe domain failure with a fixed error code and sanitized display message."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _warning_dict(warning: Any) -> Optional[Dict[str, Any]]:
    if warning is None:
        return None
    return {
        "omitted_message_count": getattr(warning, "omitted_message_count", 0),
        "first_omitted_timestamp_ms": getattr(warning, "first_omitted_timestamp_ms", None),
        "last_omitted_timestamp_ms": getattr(warning, "last_omitted_timestamp_ms", None),
        "reason": getattr(warning, "reason", "context_limit"),
    }


def _serialize_field(f: ManifestField) -> Dict[str, Any]:
    return {
        "key": f.key,
        "label": f.label,
        "field_type": f.field_type.value,
        "default": f.default,
        "required": f.required,
        "advanced": f.advanced,
        "secret": f.secret,
        "choices": list(f.choices),
        "minimum": f.minimum,
        "maximum": f.maximum,
        "description": f.description,
        "applies_to": [c.value for c in f.applies_to],
    }


def _serialize_manifest(m: ProviderManifest) -> Dict[str, Any]:
    return {
        "provider_id": m.provider_id,
        "display_name": m.display_name,
        "capabilities": [c.value for c in m.capabilities],
        "protocol_family": m.protocol_family.value,
        "recommended_endpoint": m.recommended_endpoint,
        "recommended_model": m.recommended_model,
        "endpoint_editable": m.endpoint_editable,
        "requires_endpoint": m.requires_endpoint,
        "model_selectable": m.model_selectable,
        "fields": [_serialize_field(f) for f in m.fields],
        "protocol_profiles": {k: v.value for k, v in m.protocol_profiles.items()},
        "profile_field": m.profile_field,
    }


def _is_local_model_ready(capability: str) -> bool:
    """Checks whether local STT or Translation models exist on disk in models_dir."""
    try:
        models_dir = Config().models_dir
        if not models_dir or not models_dir.exists():
            return False
        cap = (capability or "").lower()
        if cap in ("stt", "speech", "whisper"):
            stt_dirs = list(models_dir.glob("models--Systran--faster-whisper-*")) + list(
                models_dir.glob("models--openai--whisper-*")
            )
            return len(stt_dirs) > 0
        elif cap in ("translation", "translate", "marian", "opus"):
            ct2_dir = models_dir / "ct2"
            if ct2_dir.exists():
                trans_dirs = list(ct2_dir.glob("Helsinki-NLP_opus-mt-*"))
                return len(trans_dirs) > 0
        return False
    except Exception:
        return False


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
        self._pack_cancels: Dict[str, threading.Event] = {}

        self._handlers: Dict[str, Callable[[Dict[str, Any]], Any]] = {
            # Context Profiles
            "profile_create": self._handle_profile_create,
            "profile_list": self._handle_profile_list,
            "profile_get": self._handle_profile_get,
            "profile_update": self._handle_profile_update,
            "profile_delete": self._handle_profile_delete,
            "profile_attach": self._handle_profile_attach,
            # Conversation Management
            "conversation_create": self._handle_conversation_create,
            "conversation_start": self._handle_conversation_start,
            "conversation_list": self._handle_conversation_list,
            "conversation_get": self._handle_conversation_get,
            "conversation_rename": self._handle_conversation_rename,
            "conversation_delete": self._handle_conversation_delete,
            "conversation_export": self._handle_conversation_export,
            "conversation_import": self._handle_conversation_import,
            # Local Model Packs
            "pack_status": self._handle_pack_status,
            "pack_install": self._handle_pack_install,
            "pack_cancel": self._handle_pack_cancel,
            "pack_retry": self._handle_pack_retry,
            "pack_check_readiness": self._handle_pack_check_readiness,
            # Provider Manifests
            "get_provider_manifests": self._handle_get_provider_manifests,
            # Manual Assist
            "request_assist": self._handle_request_assist,
            "request_suggestion": self._handle_request_assist,
            # Streaming Summary & Cancel
            "request_summary": self._handle_request_summary,
            "summary_cancel": self._handle_summary_cancel,
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

    def _get_installer(self) -> LocalPackInstaller:
        if self._installer is not None:
            return self._installer
        config = Config()
        return LocalPackInstaller.from_config(
            config, self._get_catalog(), self._get_probe()
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
            task.add_done_callback(lambda _t: self._active_tasks.pop(request_id, None))
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

    # --- 1. Context Profiles ------------------------------------------------------------------

    async def _handle_profile_create(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        name = payload.get("name")
        content = payload.get("content")
        content_format = payload.get("content_format", "markdown")
        profile_id = payload.get("profile_id")

        if not isinstance(name, str) or not name.strip():
            raise DispatcherError("invalid_payload", "Profile 'name' must be a non-empty string")
        if not isinstance(content, str):
            raise DispatcherError("invalid_payload", "Profile 'content' must be a string")

        try:
            profile = await asyncio.to_thread(
                self.profile_service.create_profile,
                name=name,
                content=content,
                content_format=content_format,
                profile_id=profile_id,
            )
        except RecordConflictError:
            raise DispatcherError("conflict", "duplicate_id") from None
        except ConversationValidationError as err:
            raise DispatcherError("validation_error", str(err)) from None

        res = {
            "profile_id": profile.profile_id,
            "name": profile.name,
            "content": profile.content,
            "content_format": profile.content_format,
            "revision": profile.revision,
            "created_at_ms": profile.created_at_ms,
            "updated_at_ms": profile.updated_at_ms,
        }
        return {"profile": res, **res}

    async def _handle_profile_list(self, _payload: Dict[str, Any]) -> Dict[str, Any]:
        profiles = await asyncio.to_thread(self.profile_service.list_profiles)
        return {
            "profiles": [
                {
                    "profile_id": p.profile_id,
                    "name": p.name,
                    "content_format": p.content_format,
                    "revision": p.revision,
                    "content_bytes": p.content_bytes,
                    "updated_at_ms": p.updated_at_ms,
                }
                for p in profiles
            ]
        }

    async def _handle_profile_get(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        profile_id = payload.get("profile_id")
        if not isinstance(profile_id, str) or not profile_id:
            raise DispatcherError("invalid_payload", "profile_id is required")

        try:
            profile = await asyncio.to_thread(self.profile_service.get_profile, profile_id)
        except ConversationValidationError as err:
            raise DispatcherError("invalid_payload", str(err)) from None

        if profile is None:
            raise DispatcherError("not_found", f"Profile '{profile_id}' not found")

        res = {
            "profile_id": profile.profile_id,
            "name": profile.name,
            "content": profile.content,
            "content_format": profile.content_format,
            "revision": profile.revision,
            "created_at_ms": profile.created_at_ms,
            "updated_at_ms": profile.updated_at_ms,
        }
        return {"profile": res, **res}

    async def _handle_profile_update(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        profile_id = payload.get("profile_id")
        if not isinstance(profile_id, str) or not profile_id:
            raise DispatcherError("invalid_payload", "profile_id is required")

        name = payload.get("name")
        content = payload.get("content")
        content_format = payload.get("content_format")
        expected_revision = payload.get("expected_revision")

        try:
            profile = await asyncio.to_thread(
                self.profile_service.update_profile,
                profile_id=profile_id,
                name=name,
                content=content,
                content_format=content_format,
                expected_revision=expected_revision,
            )
        except RecordNotFoundError:
            raise DispatcherError("not_found", f"Profile '{profile_id}' not found") from None
        except RecordConflictError:
            raise DispatcherError("conflict", "stale_revision") from None
        except ConversationValidationError as err:
            raise DispatcherError("validation_error", str(err)) from None

        res = {
            "profile_id": profile.profile_id,
            "name": profile.name,
            "content": profile.content,
            "content_format": profile.content_format,
            "revision": profile.revision,
            "created_at_ms": profile.created_at_ms,
            "updated_at_ms": profile.updated_at_ms,
        }
        return {"profile": res, **res}

    async def _handle_profile_delete(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        profile_id = payload.get("profile_id")
        if not isinstance(profile_id, str) or not profile_id:
            raise DispatcherError("invalid_payload", "profile_id is required")

        try:
            deleted = await asyncio.to_thread(self.profile_service.delete_profile, profile_id)
        except ConversationValidationError as err:
            raise DispatcherError("invalid_payload", str(err)) from None

        return {"deleted": deleted, "profile_id": profile_id}

    async def _handle_profile_attach(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        conversation_id = payload.get("conversation_id")
        profile_ids = payload.get("profile_ids")
        if not isinstance(conversation_id, str) or not conversation_id:
            raise DispatcherError("invalid_payload", "conversation_id is required")
        if not isinstance(profile_ids, list):
            raise DispatcherError("invalid_payload", "profile_ids must be a list")

        try:
            await asyncio.to_thread(
                self.profile_service.set_attached_profiles,
                conversation_id=conversation_id,
                profile_ids=profile_ids,
            )
        except RecordNotFoundError:
            raise DispatcherError("not_found", "Conversation or profile not found") from None
        except ConversationValidationError as err:
            raise DispatcherError("validation_error", str(err)) from None

        return {
            "conversation_id": conversation_id,
            "attached_profile_ids": list(profile_ids),
        }

    # --- 2. Conversations ---------------------------------------------------------------------

    async def _handle_conversation_create(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        cid = payload.get("conversation_id") or new_record_id()
        title = payload.get("title") or f"Conversation {cid[:8]}"
        source_labels = payload.get("source_labels") or {"mic": "Me", "system": "Others"}
        import_id = payload.get("import_id")
        now_ms = time.time() * 1000.0

        record = ConversationRecord(
            conversation_id=cid,
            title=title,
            created_at_ms=now_ms,
            updated_at_ms=now_ms,
            source_labels=source_labels,
            import_id=import_id,
        )
        try:
            created = await asyncio.to_thread(self.conversation_repo.create_conversation, record)
        except RecordConflictError:
            raise DispatcherError("conflict", "duplicate_id") from None
        except ConversationValidationError as err:
            raise DispatcherError("validation_error", str(err)) from None

        conv_dict = {
            "conversation_id": created.conversation_id,
            "title": created.title,
            "created_at_ms": created.created_at_ms,
            "updated_at_ms": created.updated_at_ms,
            "source_labels": dict(created.source_labels),
            "import_id": created.import_id,
        }
        return {"conversation": conv_dict, **conv_dict}

    async def _handle_conversation_start(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        conversation_id = payload.get("conversation_id")
        if not isinstance(conversation_id, str) or not conversation_id:
            raise DispatcherError("invalid_payload", "conversation_id is required")

        session_id = payload.get("session_id") or new_record_id()
        profile_ids = payload.get("profile_ids")
        providers = payload.get("providers") or {}

        prov_meta = {}
        if isinstance(providers, dict):
            for svc, pinfo in providers.items():
                if isinstance(pinfo, dict):
                    prov_meta[svc] = ProviderMetadata(
                        provider_id=pinfo.get("provider_id", "local"),
                        model=pinfo.get("model"),
                    )
                elif isinstance(pinfo, str):
                    prov_meta[svc] = ProviderMetadata(provider_id=pinfo)

        now_ms = time.time() * 1000.0
        session_record = SessionRecord(
            session_id=session_id,
            conversation_id=conversation_id,
            started_at_ms=now_ms,
            providers=prov_meta,
        )

        try:
            snapshots = await asyncio.to_thread(
                self.profile_service.start_session, session_record, profile_ids
            )
        except RecordNotFoundError:
            raise DispatcherError("not_found", "Conversation or profile not found") from None
        except RecordConflictError:
            raise DispatcherError("conflict", "duplicate_session_id") from None
        except ConversationValidationError as err:
            raise DispatcherError("validation_error", str(err)) from None

        return {
            "session_id": session_id,
            "conversation_id": conversation_id,
            "started_at_ms": now_ms,
            "providers": {k: v.to_dict() for k, v in prov_meta.items()},
            "snapshots_count": len(snapshots),
        }

    async def _handle_conversation_list(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        conversations = await asyncio.to_thread(self.conversation_repo.list_conversations)
        offset = payload.get("offset")
        limit = payload.get("limit")

        if isinstance(offset, int) and offset > 0:
            conversations = conversations[offset:]
        if isinstance(limit, int) and limit > 0:
            conversations = conversations[:limit]

        return {
            "conversations": [
                {
                    "conversation_id": c.conversation_id,
                    "title": c.title,
                    "created_at_ms": c.created_at_ms,
                    "updated_at_ms": c.updated_at_ms,
                    "source_labels": dict(c.source_labels),
                    "import_id": c.import_id,
                }
                for c in conversations
            ]
        }

    async def _handle_conversation_get(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        conversation_id = payload.get("conversation_id")
        if not isinstance(conversation_id, str) or not conversation_id:
            raise DispatcherError("invalid_payload", "conversation_id is required")

        include_messages = bool(payload.get("include_messages", False))

        try:
            conversation = await asyncio.to_thread(
                self.conversation_repo.get_conversation, conversation_id
            )
        except ConversationValidationError as err:
            raise DispatcherError("invalid_payload", str(err)) from None

        if conversation is None:
            raise DispatcherError("not_found", f"Conversation '{conversation_id}' not found")

        sessions = await asyncio.to_thread(self.conversation_repo.list_sessions, conversation_id)
        attached = await asyncio.to_thread(
            self.profile_service.attached_profile_ids, conversation_id
        )

        messages_list = []
        if include_messages:
            msgs = await asyncio.to_thread(self.conversation_repo.list_messages, conversation_id)
            messages_list = [
                {
                    "message_id": m.message_id,
                    "sequence": m.sequence,
                    "session_id": m.session_id,
                    "source": m.source,
                    "text": m.text,
                    "translation": m.translation,
                    "is_final": m.is_final,
                    "revision": m.revision,
                    "timestamp_ms": m.timestamp_ms,
                }
                for m in msgs
            ]

        conv_dict = {
            "conversation_id": conversation.conversation_id,
            "title": conversation.title,
            "created_at_ms": conversation.created_at_ms,
            "updated_at_ms": conversation.updated_at_ms,
            "source_labels": dict(conversation.source_labels),
            "import_id": conversation.import_id,
        }
        return {
            "conversation": conv_dict,
            "sessions": [
                {
                    "session_id": s.session_id,
                    "started_at_ms": s.started_at_ms,
                    "ended_at_ms": s.ended_at_ms,
                    "providers": {k: v.to_dict() for k, v in s.providers.items()},
                }
                for s in sessions
            ],
            "attached_profile_ids": attached,
            "messages": messages_list,
            **conv_dict,
        }

    async def _handle_conversation_rename(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        conversation_id = payload.get("conversation_id")
        title = payload.get("title")
        if not isinstance(conversation_id, str) or not conversation_id:
            raise DispatcherError("invalid_payload", "conversation_id is required")
        if not isinstance(title, str) or not title.strip():
            raise DispatcherError("invalid_payload", "title is required")

        try:
            updated = await asyncio.to_thread(
                self.conversation_repo.rename_conversation, conversation_id, title
            )
        except RecordNotFoundError:
            raise DispatcherError("not_found", f"Conversation '{conversation_id}' not found") from None
        except ConversationValidationError as err:
            raise DispatcherError("validation_error", str(err)) from None

        conv_dict = {
            "conversation_id": updated.conversation_id,
            "title": updated.title,
            "created_at_ms": updated.created_at_ms,
            "updated_at_ms": updated.updated_at_ms,
            "source_labels": dict(updated.source_labels),
            "import_id": updated.import_id,
        }
        return {"conversation": conv_dict, **conv_dict}

    async def _handle_conversation_delete(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        conversation_id = payload.get("conversation_id")
        if not isinstance(conversation_id, str) or not conversation_id:
            raise DispatcherError("invalid_payload", "conversation_id is required")

        try:
            deleted = await asyncio.to_thread(
                self.conversation_repo.delete_conversation, conversation_id
            )
        except ConversationValidationError as err:
            raise DispatcherError("invalid_payload", str(err)) from None

        return {"deleted": deleted, "conversation_id": conversation_id}

    async def _handle_conversation_export(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        conversation_id = payload.get("conversation_id")
        if not isinstance(conversation_id, str) or not conversation_id:
            raise DispatcherError("invalid_payload", "conversation_id is required")

        try:
            exported = await asyncio.to_thread(
                self.conversation_repo.export_conversation, conversation_id
            )
        except RecordNotFoundError:
            raise DispatcherError("not_found", f"Conversation '{conversation_id}' not found") from None
        except ConversationValidationError as err:
            raise DispatcherError("invalid_payload", str(err)) from None

        if "conversation_id" not in exported:
            exported["conversation_id"] = conversation_id
        return {"export": exported, "conversation_id": conversation_id, **exported}

    async def _handle_conversation_import(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        export_data = payload.get("export_data") or payload.get("data") or payload
        if not isinstance(export_data, dict):
            raise DispatcherError("invalid_payload", "export_data must be an object")

        conv_dict = export_data.get("conversation") if isinstance(export_data.get("conversation"), dict) else export_data
        cid = export_data.get("conversation_id") or conv_dict.get("conversation_id")
        if not cid:
            raise DispatcherError("invalid_payload", "conversation_id is required in export_data")

        import_id = export_data.get("import_id") or conv_dict.get("import_id") or import_id_for_browser_history(cid)
        record = ConversationRecord(
            conversation_id=cid,
            title=conv_dict.get("title") or export_data.get("title", f"Imported {cid[:8]}"),
            created_at_ms=conv_dict.get("created_at_ms") or export_data.get("created_at_ms", time.time() * 1000.0),
            updated_at_ms=conv_dict.get("updated_at_ms") or export_data.get("updated_at_ms", time.time() * 1000.0),
            source_labels=conv_dict.get("source_labels") or export_data.get("source_labels", {"mic": "Me", "system": "Others"}),
            import_id=import_id,
        )

        messages: List[TranscriptMessage] = []
        for m in export_data.get("messages", []):
            messages.append(
                TranscriptMessage(
                    message_id=m["message_id"],
                    source=m.get("source", "mic"),
                    text=m.get("text", ""),
                    timestamp_ms=m.get("timestamp_ms", 0.0),
                    is_final=bool(m.get("is_final", True)),
                    revision=int(m.get("revision", 1)),
                    translation=m.get("translation"),
                    session_id=None,
                    start_ms=m.get("start_ms"),
                    end_ms=m.get("end_ms"),
                    detected_language=m.get("detected_language"),
                    provider_event_id=m.get("provider_event_id"),
                )
            )

        assist_results: List[StoredAssistResult] = []
        for a in export_data.get("assist_results", []):
            prov = a.get("provider")
            replies_raw = a.get("suggested_replies") or ()
            questions_raw = a.get("clarifying_questions") or ()
            if not replies_raw and not questions_raw and "payload" in a:
                raw_payload = a.get("payload")
                if isinstance(raw_payload, str):
                    try:
                        raw_payload = json.loads(raw_payload)
                    except Exception:
                        raw_payload = {}
                if isinstance(raw_payload, dict):
                    replies_raw = raw_payload.get("suggested_replies") or ()
                    questions_raw = raw_payload.get("clarifying_questions") or ()

            parsed_replies = []
            for r in replies_raw:
                if isinstance(r, AssistSuggestion):
                    parsed_replies.append(r)
                elif isinstance(r, dict):
                    parsed_replies.append(
                        AssistSuggestion(
                            strategy=r.get("strategy", "Direct"),
                            text=r.get("text", ""),
                            reasoning=r.get("reasoning"),
                        )
                    )

            parsed_questions = []
            for q in questions_raw:
                if isinstance(q, AssistQuestion):
                    parsed_questions.append(q)
                elif isinstance(q, dict):
                    parsed_questions.append(
                        AssistQuestion(
                            question=q.get("question", ""),
                            purpose=q.get("purpose"),
                        )
                    )

            assist_results.append(
                StoredAssistResult(
                    assist_id=a["assist_id"],
                    conversation_id=cid,
                    target_message_id=a["target_message_id"],
                    suggested_replies=tuple(parsed_replies),
                    clarifying_questions=tuple(parsed_questions),
                    created_at_ms=float(a.get("created_at_ms", time.time() * 1000.0)),
                    provider=(
                        prov
                        if isinstance(prov, ProviderMetadata)
                        else ProviderMetadata.from_dict(prov)
                        if isinstance(prov, dict)
                        else None
                    ),
                    provider_event_id=a.get("provider_event_id"),
                    input_tokens=a.get("input_tokens"),
                    output_tokens=a.get("output_tokens"),
                    latency_ms=a.get("latency_ms"),
                )
            )

        summaries: List[StoredSummary] = []
        for s in export_data.get("summaries", []):
            prov = s.get("provider")
            summaries.append(
                StoredSummary(
                    summary_id=s["summary_id"],
                    conversation_id=cid,
                    summary_markdown=s.get("summary_markdown", ""),
                    created_at_ms=s.get("created_at_ms", time.time() * 1000.0),
                    bullet_points=tuple(s.get("bullet_points", ())),
                    decisions=tuple(s.get("decisions", ())),
                    action_items=tuple(s.get("action_items", ())),
                    risks=tuple(s.get("risks", ())),
                    provider=ProviderMetadata.from_dict(prov) if prov else None,
                )
            )

        try:
            outcome = await asyncio.to_thread(
                self.conversation_repo.import_conversation,
                record=record,
                messages=messages,
                assist_results=assist_results,
                summaries=summaries,
            )
        except ConversationValidationError as err:
            raise DispatcherError("invalid_payload", str(err)) from None

        return {"conversation_id": outcome.conversation_id, "imported": outcome.imported}

    # --- 3. Local Model Packs -----------------------------------------------------------------

    async def _handle_pack_status(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        pack_id = payload.get("pack_id")
        catalog = self._get_catalog()
        installer = self._get_installer()
        probe = self._get_probe()
        profile: HardwareProfile = await asyncio.to_thread(probe.probe)

        if pack_id:
            pack = catalog.get(pack_id)
            if pack is None:
                raise DispatcherError("not_found", f"Pack '{pack_id}' not found in catalog")
            packs_to_check = [pack]
        else:
            packs_to_check = list(catalog.packs)

        status_list = []
        for p in packs_to_check:
            active = await asyncio.to_thread(installer.active, p.pack_id)
            installed = await asyncio.to_thread(installer.installed_versions, p.pack_id)
            issues = compatibility_issues(p, profile)
            local_ready = _is_local_model_ready(p.kind.value)
            is_ready = (active is not None and len(issues) == 0) or local_ready
            is_installed = bool(installed) or (active is not None) or local_ready
            status_list.append(
                {
                    "pack_id": p.pack_id,
                    "version": p.version,
                    "kind": p.kind.value,
                    "display_name": p.display_name,
                    "size_bytes": p.size_bytes,
                    "tiers": [t.value for t in p.tiers],
                    "active_version": active.version if active else ("1.0.0" if local_ready else None),
                    "installed_versions": installed if installed else (["1.0.0"] if local_ready else []),
                    "is_active": active is not None or local_ready,
                    "is_ready": is_ready,
                    "installed": is_installed,
                    "compatibility_issues": [] if local_ready else issues,
                    "is_compatible": len(issues) == 0 or local_ready,
                }
            )

        hw_info = {
            "cpu_cores": profile.cpu_cores,
            "ram_gb": round(profile.ram_gb, 2),
            "platform": profile.platform,
            "architecture": profile.architecture,
            "has_gpu": profile.gpu is not None,
        }

        if pack_id and status_list:
            return {"pack": status_list[0], "hardware": hw_info, **status_list[0]}
        return {"packs": status_list, "hardware": hw_info}

    async def _handle_pack_install(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        pack_id = payload.get("pack_id")
        if not isinstance(pack_id, str) or not pack_id:
            raise DispatcherError("invalid_payload", "pack_id is required")
        version = payload.get("version")

        lowered = pack_id.lower()
        models_dir = Config().models_dir
        if "whisper" in lowered or lowered in ("base", "tiny", "small", "medium", "large"):
            size = "base"
            if "tiny" in lowered:
                size = "tiny"
            elif "small" in lowered:
                size = "small"
            elif "medium" in lowered:
                size = "medium"
            elif "large" in lowered:
                size = "large-v3"

            from faster_whisper.utils import download_model

            out_path = await asyncio.to_thread(download_model, size, output_dir=str(models_dir))
            return {
                "installed": True,
                "pack_id": pack_id,
                "version": "1.0.0",
                "path": str(out_path),
                "active": True,
            }

        if "marian" in lowered or "opus" in lowered:
            from engine.translation.marian_backend import MarianBackend

            backend = MarianBackend()
            await asyncio.to_thread(
                backend.load_model,
                {"translation.source_lang": "en", "translation.target_lang": "vi"},
            )
            out_path = str(backend._ct2_model_path or models_dir)
            with contextlib.suppress(Exception):
                backend.unload_model()
            return {
                "installed": True,
                "pack_id": pack_id,
                "version": "1.0.0",
                "path": out_path,
                "active": True,
            }

        installer = self._get_installer()
        cancel_event = threading.Event()
        self._pack_cancels[pack_id] = cancel_event

        try:
            installed = await asyncio.to_thread(
                installer.install,
                pack_id=pack_id,
                version=version,
                cancel=cancel_event,
            )
            return {
                "installed": True,
                "pack_id": installed.pack_id,
                "version": installed.version,
                "path": str(installed.path),
                "active": True,
            }
        except InstallCancelledError:
            raise DispatcherError("cancelled", f"Install of pack '{pack_id}' cancelled") from None
        except PackInstallError as err:
            raise DispatcherError(err.code, err.detail or err.code) from None
        finally:
            self._pack_cancels.pop(pack_id, None)

    async def _handle_pack_cancel(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        pack_id = payload.get("pack_id")
        if not isinstance(pack_id, str) or not pack_id:
            raise DispatcherError("invalid_payload", "pack_id is required")

        cancel_event = self._pack_cancels.get(pack_id)
        if cancel_event:
            cancel_event.set()
            return {"cancelled": True, "pack_id": pack_id}
        return {"cancelled": False, "pack_id": pack_id}

    async def _handle_pack_retry(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return await self._handle_pack_install(payload)

    async def _handle_pack_check_readiness(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        capability = payload.get("capability", "stt")
        pack_id = payload.get("pack_id")
        installer = self._get_installer()
        catalog = self._get_catalog()
        probe = self._get_probe()
        profile: HardwareProfile = await asyncio.to_thread(probe.probe)

        kind = PackKind.STT if capability.lower() == "stt" else PackKind.TRANSLATION
        is_local = _is_local_model_ready(capability)

        if pack_id:
            pack = catalog.get(pack_id)
            if pack is None:
                raise DispatcherError("not_found", f"Pack '{pack_id}' not found")
            issues = compatibility_issues(pack, profile)
            active = await asyncio.to_thread(installer.active, pack.pack_id)
            ready = (active is not None and len(issues) == 0) or is_local
            installed = (active is not None) or is_local
            tier_val = pack.tiers[0].value if pack.tiers else "minimal"
            return {
                "capability": capability,
                "pack_id": pack.pack_id,
                "ready": ready,
                "issues": [] if ready else issues,
                "installed": installed,
                "tier": tier_val,
            }

        selection = select_default_pack(catalog, kind, profile)
        pack = selection.pack
        if pack is None:
            return {
                "capability": capability,
                "pack_id": None,
                "ready": is_local,
                "issues": [] if is_local else ["no_compatible_pack"],
                "installed": is_local,
                "tier": "minimal",
                "reason": selection.reason,
            }

        active = await asyncio.to_thread(installer.active, pack.pack_id)
        ready = (active is not None) or is_local
        installed = (active is not None) or is_local
        tier_val = selection.tier.value if selection.tier else "minimal"
        return {
            "capability": capability,
            "pack_id": pack.pack_id,
            "ready": ready,
            "issues": [],
            "installed": installed,
            "tier": tier_val,
            "reason": selection.reason,
        }

    # --- 4. Provider Manifests ----------------------------------------------------------------

    async def _handle_get_provider_manifests(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        capability = payload.get("capability")
        registry = build_default_registry()
        cap_enum = None
        if capability:
            try:
                cap_enum = ProviderCapability(capability.lower())
            except ValueError:
                raise DispatcherError("invalid_payload", f"Unknown capability '{capability}'") from None

        if cap_enum:
            manifests = registry.providers_for(cap_enum)
        else:
            with registry._lock:
                manifests = tuple(sorted(registry._manifests.values(), key=lambda m: m.provider_id))

        manifests_list = [_serialize_manifest(m) for m in manifests]
        return {"manifests": manifests_list}

    # --- 5. Manual Assist ---------------------------------------------------------------------

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

        stored = outcome.stored
        replies = [
            {"strategy": r.strategy, "text": r.text, "reasoning": r.reasoning}
            for r in stored.suggested_replies
        ]
        questions = [
            {"question": q.question, "purpose": q.purpose}
            for q in stored.clarifying_questions
        ]
        options = [
            r.get("text") if isinstance(r, dict) else getattr(r, "text", str(r))
            for r in replies
        ]
        provider_dict = outcome.provider.to_dict() if outcome.provider else None

        return {
            "type": "suggestion_result",
            "conversation_id": cid,
            "message_id": mid,
            "entry_id": mid,
            "assist_id": stored.assist_id,
            "options": options,
            "suggested_replies": replies,
            "clarifying_questions": questions,
            "provider": provider_dict,
            "context_window": {
                "message_count": outcome.context_message_count,
                "omitted_message_count": outcome.omitted_message_count,
                "profile_count": outcome.profile_count,
                "omitted_profile_count": outcome.omitted_profile_count,
            },
        }

    # --- 6. Transcript Summary Streaming ------------------------------------------------------

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
                "data": {
                    "chunk": text,
                    "delta_text": text,
                    "done": False,
                },
                "message": {
                    "type": "llm_chunk",
                    "text": text,
                    "conversation_id": cid,
                },
            }
            await websocket.send(encode_frame(frame, tokens))

        try:
            outcome = await service.request_summary(cid, on_chunk=_on_chunk)
        except asyncio.CancelledError:
            await self._send_error(websocket, request_id, "cancelled", "Summary request cancelled", tokens)
            raise
        except Exception as e:
            logger.exception("Summary streaming unexpected error: %s", e)
            await self._send_error(websocket, request_id, "internal_error", "Summary failed unexpectedly", tokens)
            return
        finally:
            self._active_tasks.pop(request_id, None)

        if not outcome.ok:
            err = outcome.error
            await self._send_error(websocket, request_id, err.code.value, err.message, tokens)
            return

        w_payload = _warning_dict(outcome.warning)
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
        if not target_id or not isinstance(target_id, str):
            return {"cancelled": False}

        task = self._active_tasks.get(target_id)
        if task and not task.done():
            task.cancel()
            return {"cancelled": True, "request_id": target_id}
        return {"cancelled": False, "request_id": target_id}
