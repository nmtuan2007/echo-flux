"""Control dispatcher handlers for Context Profiles."""

import asyncio
from typing import Any, Dict

from engine.conversation.context_profiles import ContextProfileService
from engine.conversation.models import ConversationValidationError
from engine.conversation.repository import RecordConflictError, RecordNotFoundError
from engine.server.control_helpers import DispatcherError


def _serialize_profile(profile: Any) -> Dict[str, Any]:
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


async def handle_profile_create(
    profile_service: ContextProfileService, payload: Dict[str, Any]
) -> Dict[str, Any]:
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
            profile_service.create_profile,
            name=name,
            content=content,
            content_format=content_format,
            profile_id=profile_id,
        )
    except RecordConflictError:
        raise DispatcherError("conflict", "duplicate_id") from None
    except ConversationValidationError as err:
        raise DispatcherError("validation_error", str(err)) from None

    return _serialize_profile(profile)


async def handle_profile_list(
    profile_service: ContextProfileService, _payload: Dict[str, Any]
) -> Dict[str, Any]:
    profiles = await asyncio.to_thread(profile_service.list_profiles)
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


async def handle_profile_get(
    profile_service: ContextProfileService, payload: Dict[str, Any]
) -> Dict[str, Any]:
    profile_id = payload.get("profile_id")
    if not isinstance(profile_id, str) or not profile_id:
        raise DispatcherError("invalid_payload", "profile_id is required")

    try:
        profile = await asyncio.to_thread(profile_service.get_profile, profile_id)
    except ConversationValidationError as err:
        raise DispatcherError("invalid_payload", str(err)) from None

    if profile is None:
        raise DispatcherError("not_found", f"Profile '{profile_id}' not found")

    return _serialize_profile(profile)


async def handle_profile_update(
    profile_service: ContextProfileService, payload: Dict[str, Any]
) -> Dict[str, Any]:
    profile_id = payload.get("profile_id")
    if not isinstance(profile_id, str) or not profile_id:
        raise DispatcherError("invalid_payload", "profile_id is required")

    try:
        profile = await asyncio.to_thread(
            profile_service.update_profile,
            profile_id=profile_id,
            name=payload.get("name"),
            content=payload.get("content"),
            content_format=payload.get("content_format"),
            expected_revision=payload.get("expected_revision"),
        )
    except RecordNotFoundError:
        raise DispatcherError("not_found", f"Profile '{profile_id}' not found") from None
    except RecordConflictError:
        raise DispatcherError("conflict", "stale_revision") from None
    except ConversationValidationError as err:
        raise DispatcherError("validation_error", str(err)) from None

    return _serialize_profile(profile)


async def handle_profile_delete(
    profile_service: ContextProfileService, payload: Dict[str, Any]
) -> Dict[str, Any]:
    profile_id = payload.get("profile_id")
    if not isinstance(profile_id, str) or not profile_id:
        raise DispatcherError("invalid_payload", "profile_id is required")

    try:
        deleted = await asyncio.to_thread(profile_service.delete_profile, profile_id)
    except ConversationValidationError as err:
        raise DispatcherError("invalid_payload", str(err)) from None

    return {"deleted": deleted, "profile_id": profile_id}


async def handle_profile_attach(
    profile_service: ContextProfileService, payload: Dict[str, Any]
) -> Dict[str, Any]:
    conversation_id = payload.get("conversation_id")
    profile_ids = payload.get("profile_ids")
    if not isinstance(conversation_id, str) or not conversation_id:
        raise DispatcherError("invalid_payload", "conversation_id is required")
    if not isinstance(profile_ids, list):
        raise DispatcherError("invalid_payload", "profile_ids must be a list")

    try:
        await asyncio.to_thread(
            profile_service.set_attached_profiles,
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
