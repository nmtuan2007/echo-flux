"""Control dispatcher handlers for Conversations and transcript messages."""

import asyncio
import json
import time
from typing import Any, Dict, List

from engine.conversation.context_profiles import ContextProfileService
from engine.conversation.models import (
    ConversationRecord,
    ConversationValidationError,
    ProviderMetadata,
    SessionRecord,
    StoredAskTurn,
    StoredAssistResult,
    StoredSummary,
    TranscriptMessage,
    import_id_for_browser_history,
)
from engine.conversation.repository import (
    ConversationRepository,
    RecordConflictError,
    RecordNotFoundError,
    new_record_id,
)
from engine.providers.types import AssistQuestion, AssistSuggestion
from engine.server.control_helpers import DispatcherError


def _conversation_dict(conv: ConversationRecord) -> Dict[str, Any]:
    return {
        "conversation_id": conv.conversation_id,
        "title": conv.title,
        "created_at_ms": conv.created_at_ms,
        "updated_at_ms": conv.updated_at_ms,
        "source_labels": dict(conv.source_labels),
        "import_id": conv.import_id,
    }


async def handle_conversation_create(
    repo: ConversationRepository, payload: Dict[str, Any]
) -> Dict[str, Any]:
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
        created = await asyncio.to_thread(repo.create_conversation, record)
    except RecordConflictError:
        raise DispatcherError("conflict", "duplicate_id") from None
    except ConversationValidationError as err:
        raise DispatcherError("validation_error", str(err)) from None

    conv_dict = _conversation_dict(created)
    return {"conversation": conv_dict, **conv_dict}


async def handle_conversation_start(
    profile_service: ContextProfileService, payload: Dict[str, Any]
) -> Dict[str, Any]:
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
            profile_service.start_session, session_record, profile_ids
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


async def handle_conversation_list(
    repo: ConversationRepository, payload: Dict[str, Any]
) -> Dict[str, Any]:
    raw_query = payload.get("search_query")
    search_query = raw_query.strip() if isinstance(raw_query, str) and raw_query.strip() else None
    if hasattr(repo, "list_conversations_with_stats"):
        try:
            items = await asyncio.to_thread(repo.list_conversations_with_stats, search_query)
        except TypeError:
            items = await asyncio.to_thread(repo.list_conversations_with_stats)
    else:
        convs = await asyncio.to_thread(repo.list_conversations)
        items = [(c, 0, False) for c in convs]

    offset = payload.get("offset")
    limit = payload.get("limit")

    if isinstance(offset, int) and offset > 0:
        items = items[offset:]
    if isinstance(limit, int) and limit > 0:
        items = items[:limit]

    return {
        "conversations": [
            {
                **_conversation_dict(c),
                "message_count": msg_count,
                "has_summary": has_summary,
            }
            for c, msg_count, has_summary in items
        ]
    }


async def handle_conversation_get(
    repo: ConversationRepository,
    profile_service: ContextProfileService,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    conversation_id = payload.get("conversation_id")
    if not isinstance(conversation_id, str) or not conversation_id:
        raise DispatcherError("invalid_payload", "conversation_id is required")

    include_messages = bool(payload.get("include_messages", False))

    try:
        conversation = await asyncio.to_thread(repo.get_conversation, conversation_id)
    except ConversationValidationError as err:
        raise DispatcherError("invalid_payload", str(err)) from None

    if conversation is None:
        raise DispatcherError("not_found", f"Conversation '{conversation_id}' not found")

    sessions = await asyncio.to_thread(repo.list_sessions, conversation_id)
    attached = await asyncio.to_thread(profile_service.attached_profile_ids, conversation_id)
    latest_summary = (
        await asyncio.to_thread(repo.latest_summary, conversation_id)
        if hasattr(repo, "latest_summary")
        else None
    )
    ask_turns_list: List[Dict[str, Any]] = []
    if hasattr(repo, "list_ask_turns"):
        turns = await asyncio.to_thread(repo.list_ask_turns, conversation_id)
        ask_turns_list = [
            {
                "ask_id": t.ask_id,
                "question": t.question,
                "answer_markdown": t.answer_markdown,
                "created_at_ms": t.created_at_ms,
                "provider_id": t.provider.provider_id if t.provider else None,
            }
            for t in turns
        ]

    messages_list = []
    assist_results_list = []
    if include_messages:
        msgs = await asyncio.to_thread(repo.list_messages, conversation_id)
        labels = dict(conversation.source_labels)
        messages_list = [
            {
                "message_id": m.message_id,
                "sequence": m.sequence,
                "session_id": m.session_id,
                "source": m.source,
                "speaker_label": labels.get(m.source),
                "text": m.text,
                "translation": m.translation,
                "is_final": m.is_final,
                "revision": m.revision,
                "timestamp_ms": m.timestamp_ms,
                "start_ms": m.start_ms,
                "end_ms": m.end_ms,
            }
            for m in msgs
        ]
        if hasattr(repo, "list_assist_results"):
            assists = await asyncio.to_thread(repo.list_assist_results, conversation_id)
            assist_results_list = [
                {
                    "assist_id": a.assist_id,
                    "target_message_id": a.target_message_id,
                    "suggested_replies": [
                        {
                            "strategy": r.strategy,
                            "text": r.text,
                            **({"reasoning": r.reasoning} if r.reasoning else {}),
                        }
                        for r in a.suggested_replies
                    ],
                    "clarifying_questions": [
                        {
                            "question": q.question,
                            **({"purpose": q.purpose} if q.purpose else {}),
                        }
                        for q in a.clarifying_questions
                    ],
                    "provider_id": a.provider.provider_id if a.provider else None,
                    "created_at_ms": a.created_at_ms,
                }
                for a in assists
            ]

    summary_dict = (
        {
            "summary_id": latest_summary.summary_id,
            "summary_markdown": latest_summary.summary_markdown,
            "created_at_ms": latest_summary.created_at_ms,
        }
        if latest_summary is not None
        else None
    )

    conv_dict = _conversation_dict(conversation)
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
        "assist_results": assist_results_list,
        "ask_turns": ask_turns_list,
        "summary": summary_dict,
        **conv_dict,
    }


async def handle_conversation_rename(
    repo: ConversationRepository, payload: Dict[str, Any]
) -> Dict[str, Any]:
    conversation_id = payload.get("conversation_id")
    title = payload.get("title")
    if not isinstance(conversation_id, str) or not conversation_id:
        raise DispatcherError("invalid_payload", "conversation_id is required")
    if not isinstance(title, str) or not title.strip():
        raise DispatcherError("invalid_payload", "title is required")

    try:
        updated = await asyncio.to_thread(repo.rename_conversation, conversation_id, title)
    except RecordNotFoundError:
        raise DispatcherError("not_found", f"Conversation '{conversation_id}' not found") from None
    except ConversationValidationError as err:
        raise DispatcherError("validation_error", str(err)) from None

    conv_dict = _conversation_dict(updated)
    return {"conversation": conv_dict, **conv_dict}


async def handle_conversation_delete(
    repo: ConversationRepository, payload: Dict[str, Any]
) -> Dict[str, Any]:
    conversation_id = payload.get("conversation_id")
    if not isinstance(conversation_id, str) or not conversation_id:
        raise DispatcherError("invalid_payload", "conversation_id is required")

    try:
        deleted = await asyncio.to_thread(repo.delete_conversation, conversation_id)
    except ConversationValidationError as err:
        raise DispatcherError("invalid_payload", str(err)) from None

    return {"deleted": deleted, "conversation_id": conversation_id}


async def handle_conversation_export(
    repo: ConversationRepository, payload: Dict[str, Any]
) -> Dict[str, Any]:
    conversation_id = payload.get("conversation_id")
    if not isinstance(conversation_id, str) or not conversation_id:
        raise DispatcherError("invalid_payload", "conversation_id is required")

    try:
        exported = await asyncio.to_thread(repo.export_conversation, conversation_id)
    except RecordNotFoundError:
        raise DispatcherError("not_found", f"Conversation '{conversation_id}' not found") from None
    except ConversationValidationError as err:
        raise DispatcherError("invalid_payload", str(err)) from None

    if "conversation_id" not in exported:
        exported["conversation_id"] = conversation_id
    return {"export": exported, "conversation_id": conversation_id, **exported}


async def handle_conversation_import(
    repo: ConversationRepository, payload: Dict[str, Any]
) -> Dict[str, Any]:
    export_data = payload.get("export_data") or payload.get("data") or payload
    if not isinstance(export_data, dict):
        raise DispatcherError("invalid_payload", "export_data must be an object")

    conv_dict = (
        export_data.get("conversation")
        if isinstance(export_data.get("conversation"), dict)
        else export_data
    )
    cid = export_data.get("conversation_id") or conv_dict.get("conversation_id")
    if not cid:
        raise DispatcherError("invalid_payload", "conversation_id is required in export_data")

    import_id = (
        export_data.get("import_id")
        or conv_dict.get("import_id")
        or import_id_for_browser_history(cid)
    )
    record = ConversationRecord(
        conversation_id=cid,
        title=conv_dict.get("title") or export_data.get("title", f"Imported {cid[:8]}"),
        created_at_ms=conv_dict.get("created_at_ms")
        or export_data.get("created_at_ms", time.time() * 1000.0),
        updated_at_ms=conv_dict.get("updated_at_ms")
        or export_data.get("updated_at_ms", time.time() * 1000.0),
        source_labels=conv_dict.get("source_labels")
        or export_data.get("source_labels", {"mic": "Me", "system": "Others"}),
        import_id=import_id,
    )

    messages: List[TranscriptMessage] = [
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
        for m in export_data.get("messages", [])
    ]

    assist_results: List[StoredAssistResult] = []
    for idx, a in enumerate(export_data.get("assist_results", [])):
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

        parsed_replies = [
            r
            if isinstance(r, AssistSuggestion)
            else AssistSuggestion(
                strategy=r.get("strategy", "Direct"),
                text=r.get("text", ""),
                reasoning=r.get("reasoning"),
            )
            for r in replies_raw
            if isinstance(r, (AssistSuggestion, dict))
        ]
        parsed_questions = [
            q
            if isinstance(q, AssistQuestion)
            else AssistQuestion(
                question=q.get("question", ""),
                purpose=q.get("purpose"),
            )
            for q in questions_raw
            if isinstance(q, (AssistQuestion, dict))
        ]

        assist_results.append(
            StoredAssistResult(
                assist_id=a.get("assist_id") or a.get("id") or f"migrated-assist-{idx}",
                conversation_id=cid,
                target_message_id=a.get("target_message_id")
                or a.get("entry_id")
                or f"migrated-msg-{idx}",
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
    for idx, s in enumerate(export_data.get("summaries", [])):
        prov = s.get("provider")
        summaries.append(
            StoredSummary(
                summary_id=s.get("summary_id") or s.get("id") or f"migrated-sum-{idx}",
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

    ask_turns: List[StoredAskTurn] = []
    for idx, t in enumerate(export_data.get("ask_turns", [])):
        if not isinstance(t, dict) or not t.get("question") or not t.get("answer_markdown"):
            continue
        prov = t.get("provider")
        ask_turns.append(
            StoredAskTurn(
                ask_id=t.get("ask_id") or f"migrated-ask-{idx}",
                conversation_id=cid,
                question=str(t["question"]),
                answer_markdown=str(t["answer_markdown"]),
                created_at_ms=float(t.get("created_at_ms", time.time() * 1000.0)),
                provider=ProviderMetadata.from_dict(prov) if isinstance(prov, dict) else None,
            )
        )

    try:
        outcome = await asyncio.to_thread(
            repo.import_conversation,
            record=record,
            messages=messages,
            assist_results=assist_results,
            summaries=summaries,
            ask_turns=ask_turns,
        )
    except ConversationValidationError as err:
        raise DispatcherError("invalid_payload", str(err)) from None

    return {"conversation_id": outcome.conversation_id, "imported": outcome.imported}


async def handle_message_update(
    repo: ConversationRepository, engine: Any, payload: Dict[str, Any]
) -> Dict[str, Any]:
    cid = payload.get("conversation_id")
    mid = payload.get("message_id")
    text = payload.get("text")
    source = payload.get("source")

    if not isinstance(cid, str) or not cid:
        raise DispatcherError("invalid_payload", "conversation_id is required")
    if not isinstance(mid, str) or not mid:
        raise DispatcherError("invalid_payload", "message_id is required")
    if text is None and source is None:
        raise DispatcherError("invalid_payload", "text or source is required")
    if text is not None and (not isinstance(text, str) or not text.strip()):
        raise DispatcherError("invalid_payload", "text must be a non-empty string")
    if source is not None and source not in ("mic", "system", "both"):
        raise DispatcherError("invalid_payload", "source must be mic, system, or both")

    try:
        updated = await asyncio.to_thread(
            repo.update_message,
            cid,
            mid,
            text=text,
            source=source,
        )
    except RecordNotFoundError:
        raise DispatcherError("not_found", "Message or conversation not found") from None
    except ConversationValidationError as err:
        raise DispatcherError("validation_error", str(err)) from None

    new_translation = None
    if text is not None and engine is not None and hasattr(engine, "retranslate_message"):
        new_translation = await engine.retranslate_message(cid, mid, updated.text)

    return {
        "conversation_id": cid,
        "message_id": updated.message_id,
        "text": updated.text,
        "source": updated.source,
        **({"translation": new_translation} if new_translation else {}),
    }
