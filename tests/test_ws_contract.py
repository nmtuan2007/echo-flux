"""Pins the engine<->UI WebSocket contract across all five places it is expressed.

The engine and the UI have no shared schema, so a message type added on one side and
forgotten on the other fails silently at runtime. Five surfaces must agree:

    1. what the engine sends      (websocket_server.py, main.py)
    2. what the engine handles    (websocket_server.py)
    3. what the UI sends          (engineStore.ts)
    4. what the UI handles        (engineStore.ts)
    5. what the docs claim        (.agent/rules/03-architecture.md)

This is regex over source, not a parser, so every pattern below is deliberately narrow
and all of it lives in this one module. When a test here fails, check the extraction
before assuming the code is wrong.
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SERVER = ROOT / "engine" / "server" / "websocket_server.py"
MAIN = ROOT / "engine" / "main.py"
STORE = ROOT / "apps" / "desktop" / "src" / "store" / "engineStore.ts"
TRANSCRIPT_SLICE = ROOT / "apps" / "desktop" / "src" / "store" / "transcriptSlice.ts"
RULES = ROOT / ".agent" / "rules" / "03-architecture.md"


def _read(path: Path) -> str:
    assert path.exists(), f"Contract surface missing: {path}"
    return path.read_text(encoding="utf-8")


def _python_sent_types(source: str) -> set:
    """Types the engine emits: the value slice after a "type": key.

    The slice stops at the first comma/brace/newline so that sibling keys are not
    swept up, but stays wide enough to catch the ternary in main.py's _enqueue_asr_result
    ('"partial" if not asr_result.is_final else "final"'), which yields two types.

    Note '"model_type":' does not match: the pattern requires a quote directly before
    'type'.
    """
    found = set()
    for match in re.finditer(r'"type":\s*([^,\n}]+)', source):
        found.update(re.findall(r'"(\w+)"', match.group(1)))
    return found


def _python_handled_types(source: str) -> set:
    return set(re.findall(r'msg_type\s*==\s*"(\w+)"', source))


def _ts_sent_types(source: str) -> set:
    """Types the UI sends: the first `type: "..."` inside each JSON.stringify(...).

    Scoping to JSON.stringify keeps TypeScript type annotations out -- notably
    `type: "asr" | "translation"` in the store's interface declarations, which is a
    parameter type, not a message.
    """
    found = set()
    for match in re.finditer(r"JSON\.stringify\(", source):
        window = source[match.end() : match.end() + 400]
        first = re.search(r'\btype:\s*"(\w+)"', window)
        if first:
            found.add(first.group(1))
    return found


def _ts_handled_types(source: str) -> set:
    return set(re.findall(r'case\s+"(\w+)":', source))


def _documented_engine_to_ui(source: str) -> set:
    return set(re.findall(r'\{"type":\s*"(\w+)"', source))


def _documented_ui_to_engine(source: str) -> set:
    # Table rows of the form: | `start` | ... |
    return set(re.findall(r"^\|\s*`(\w+)`\s*\|", source, re.MULTILINE))


@pytest.fixture(scope="module")
def surfaces():
    server_src = _read(SERVER)
    main_src = _read(MAIN)
    store_src = _read(STORE)
    rules_src = _read(RULES)

    data = {
        "engine_sends": _python_sent_types(server_src) | _python_sent_types(main_src),
        "engine_handles": _python_handled_types(server_src),
        "ui_sends": _ts_sent_types(store_src),
        "ui_handles": _ts_handled_types(store_src),
        "documented_out": _documented_engine_to_ui(rules_src),
        "documented_in": _documented_ui_to_engine(rules_src),
    }

    # Guard against a regex silently matching nothing and turning every
    # assertion below into a vacuous pass.
    for name, values in data.items():
        assert values, f"Extraction for '{name}' found nothing -- the regex has rotted."
    return data


def _diff(label_a, set_a, label_b, set_b) -> str:
    return (
        f"\n  only in {label_a}: {sorted(set_a - set_b)}"
        f"\n  only in {label_b}: {sorted(set_b - set_a)}"
    )


def test_every_message_the_engine_sends_is_handled_by_the_ui(surfaces):
    sends, handles = surfaces["engine_sends"], surfaces["ui_handles"]
    assert sends == handles, "Engine/UI message drift." + _diff(
        "engine sends", sends, "UI handles", handles
    )


def test_every_command_the_ui_sends_is_handled_by_the_engine(surfaces):
    sends, handles = surfaces["ui_sends"], surfaces["engine_handles"]
    assert sends == handles, "UI/engine command drift." + _diff(
        "UI sends", sends, "engine handles", handles
    )


def test_documented_engine_to_ui_matches_implementation(surfaces):
    documented, actual = surfaces["documented_out"], surfaces["engine_sends"]
    assert documented == actual, f"{RULES.name} is out of sync with the engine." + _diff(
        "docs", documented, "code", actual
    )


def test_documented_ui_to_engine_matches_implementation(surfaces):
    documented, actual = surfaces["documented_in"], surfaces["engine_handles"]
    assert documented == actual, f"{RULES.name} is out of sync with the server." + _diff(
        "docs", documented, "code", actual
    )


def test_core_transcription_types_are_present(surfaces):
    # A sanity floor: if these vanish, the extraction broke rather than the contract.
    for required in ("partial", "final", "translation_update", "status", "error"):
        assert required in surfaces["engine_sends"]
    for required in ("start", "stop"):
        assert required in surfaces["engine_handles"]


def test_unknown_command_is_answered_with_an_error(surfaces):
    # The server must reply to unrecognized types rather than drop them.
    assert "Unknown type" in _read(SERVER)
    assert "error" in surfaces["engine_sends"]


# ---------------------------------------------------------------------------------------------
# Realtime STT session payload extensions (EF-016)
#
# The session adds no message types. It extends `partial`/`final` with metadata keys and reports
# provider transitions on `status` with new status values. The extension keys are a closed,
# named set: the engine may emit nothing outside it, and the store may consume these keys (and
# the legacy documented keys) but nothing else. The Translation session (EF-019) extends
# `translation_update` and `status` the same way. `stt_*` and `translation_*` status values are
# never lifecycle transitions in the store.
# ---------------------------------------------------------------------------------------------

ENGINE_SOURCES = (
    MAIN,
    SERVER,
    ROOT / "engine" / "server" / "protocol.py",
    *sorted((ROOT / "engine" / "session").glob("*.py")),
)
STORE_DIR = ROOT / "apps" / "desktop" / "src" / "store"

# Pinned so a new message type cannot slip in by editing code and docs together.
DOCUMENTED_ENGINE_TO_UI = {
    "partial",
    "final",
    "translation_update",
    "status",
    "error",
    "audio_level",
    "devices_list",
    "suggestion_result",
    "llm_chunk",
    "llm_done",
    "test_connection_result",
}
DOCUMENTED_UI_TO_ENGINE = {
    "start",
    "stop",
    "list_devices",
    "request_suggestion",
    "request_summary",
    "test_connection",
}

LEGACY_TRANSCRIPT_KEYS = {
    "type",
    "text",
    "translation",
    "is_final",
    "timestamp",
    "source",
    "entry_id",
    "translation_backend",
}
LEGACY_STATUS_KEYS = {"type", "status"}
LEGACY_TRANSLATION_UPDATE_KEYS = {
    "type",
    "entry_id",
    "source_text",
    "translation",
    "timestamp",
    "is_final",
    "translation_backend",
}

ADDITIVE_TRANSCRIPT_KEYS = {
    "revision",
    "item_id",
    "epoch",
    "stt_provider",
    "language",
    "retracted",
    "duplicate_of",
    "continuity",
    "conversation_id",
    "message_id",
}
ADDITIVE_STATUS_KEYS = {
    "source",
    "stt_provider",
    "previous_stt_provider",
    "reason",
    "stt_state",
    "replay",
    "continuity",
}
# Translation session (EF-019); allowed before the engine emits them.
ADDITIVE_TRANSLATION_METADATA_KEYS = {
    "translation_provider",
    "translation_fallback",
    "source_lang",
    "target_lang",
    "detected_source_lang",
}
ADDITIVE_TRANSLATION_UPDATE_KEYS = ADDITIVE_TRANSLATION_METADATA_KEYS | {
    "revision",
    "is_final",
    "item_id",
    "source",
    "translation_error",
}
ADDITIVE_TRANSLATION_STATUS_KEYS = {
    "translation_provider",
    "previous_translation_provider",
    "reason",
    "translation_state",
}
# New `status` values. None of them is a lifecycle transition.
STT_STATUS_VALUES = {"stt_reconnecting", "stt_reconnected", "stt_fallback", "stt_failed"}
TRANSLATION_STATUS_VALUES = {"translation_fallback", "translation_error", "translation_failed"}
# Already integrated by EF-022 (engine/server/protocol.py builders).
ADDITIVE_SUGGESTION_RESULT_KEYS = {
    "conversation_id",
    "message_id",
    "assist_id",
    "suggested_replies",
    "clarifying_questions",
    "provider",
    "context_window",
    "error_code",
    "retryable",
    "provider_code",
}
ADDITIVE_SUMMARY_KEYS = {
    "conversation_id",
    "summary_id",
    "warning",
    "included_message_count",
    "omitted_message_count",
    "is_error",
    "provider",
    "error_code",
    "error",
    "retryable",
    "provider_code",
}

# Emitted keys per message type that the engine side of this contract closes over.
# A partial may carry its caption's translation, with the translation metadata.
ENGINE_ALLOWED_KEYS = {
    "partial": LEGACY_TRANSCRIPT_KEYS
    | ADDITIVE_TRANSCRIPT_KEYS
    | ADDITIVE_TRANSLATION_METADATA_KEYS,
    "final": LEGACY_TRANSCRIPT_KEYS | ADDITIVE_TRANSCRIPT_KEYS,
    "status": LEGACY_STATUS_KEYS | ADDITIVE_STATUS_KEYS | ADDITIVE_TRANSLATION_STATUS_KEYS,
    "translation_update": LEGACY_TRANSLATION_UPDATE_KEYS | ADDITIVE_TRANSLATION_UPDATE_KEYS,
}

APPROVED_STORE_ADDITIVE_KEYS = (
    ADDITIVE_TRANSCRIPT_KEYS
    | ADDITIVE_TRANSLATION_UPDATE_KEYS
    | ADDITIVE_SUGGESTION_RESULT_KEYS
    | ADDITIVE_SUMMARY_KEYS
)


def _documented_payload_keys(rules_src: str) -> dict:
    """Keys of each documented engine->UI message, from its `{"type": "x", ...}` line."""
    keys: dict = {}
    for line in rules_src.splitlines():
        match = re.match(r'-\s*`\{"type":\s*"(\w+)"', line.strip())
        if match:
            keys.setdefault(match.group(1), set()).update(re.findall(r'"(\w+)":', line))
    return keys


# ── Engine emitters (AST, so nested payload dicts and formatting do not matter) ──────────


def _const_str(node):
    import ast

    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _type_values(node) -> set:
    import ast

    if isinstance(node, ast.IfExp):
        return _type_values(node.body) | _type_values(node.orelse)
    value = _const_str(node)
    return {value} if value is not None else set()


def _returned_dict_keys(tree) -> dict:
    """{function name: keys of the dict literals it returns}, for `**helper()` and
    `.update(helper())` resolution."""
    import ast

    returns: dict = {}
    for func in ast.walk(tree):
        if isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for node in ast.walk(func):
                if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict):
                    returns.setdefault(func.name, set()).update(_dict_keys(node.value))
    return returns


def _expanded_keys(node, returns: dict) -> set:
    """Keys a dict-valued expression contributes: a literal, or a call to a local helper."""
    import ast

    if isinstance(node, ast.Dict):
        return _dict_keys(node, returns)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in returns:
        return set(returns[node.func.id])
    return {"<dynamic>"}


def _dict_keys(node, returns=None) -> set:
    keys = set()
    for key, value in zip(node.keys, node.values, strict=True):
        if key is None:
            keys |= _expanded_keys(value, returns or {})
            continue
        name = _const_str(key)
        keys.add(name if name is not None else "<dynamic>")
    return keys


def _engine_emitted_keys(source: str) -> dict:
    """{message type: [top-level key sets]} for every dict literal carrying a "type" key.

    Keys added afterwards in the same function through `var["k"] = ...` or
    `var.update(...)` count as part of that message, as do `**helper()` expansions of a
    module-local helper returning a dict literal. Anything unresolvable is `<dynamic>`.
    """
    import ast

    found: dict = {}
    tree = ast.parse(source)
    returns = _returned_dict_keys(tree)
    for func in ast.walk(tree):
        if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        bound = {}
        for node in ast.walk(func):
            if isinstance(node, (ast.Assign, ast.AnnAssign)) and isinstance(node.value, ast.Dict):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if isinstance(target, ast.Name):
                        bound[id(node.value)] = target.id
        for node in ast.walk(func):
            if not isinstance(node, ast.Dict):
                continue
            type_nodes = [
                v for k, v in zip(node.keys, node.values, strict=True) if _const_str(k) == "type"
            ]
            if not type_nodes:
                continue
            keys = _dict_keys(node, returns)
            var = bound.get(id(node))
            if var is not None:
                for extra in ast.walk(func):
                    if isinstance(extra, ast.Assign):
                        for target in extra.targets:
                            if (
                                isinstance(target, ast.Subscript)
                                and isinstance(target.value, ast.Name)
                                and target.value.id == var
                            ):
                                keys.add(_const_str(target.slice) or "<dynamic>")
                    if (
                        isinstance(extra, ast.Call)
                        and isinstance(extra.func, ast.Attribute)
                        and extra.func.attr == "update"
                        and isinstance(extra.func.value, ast.Name)
                        and extra.func.value.id == var
                    ):
                        for arg in extra.args:
                            keys |= _expanded_keys(arg, returns)
            for message_type in _type_values(type_nodes[0]):
                found.setdefault(message_type, []).append(keys)
    return found


# ── Store key reads (regex over TypeScript; every form of reading a key is caught) ──────

_TS_IDENT = r"[A-Za-z_$][\w$]*"


def _ts_receivers(source: str) -> set:
    """`message` plus every identifier annotated as an EngineMessage."""
    return {"message"} | set(re.findall(rf"({_TS_IDENT})\??\s*:\s*EngineMessage\b", source))


def _balanced_open(source: str, close_index: int) -> int:
    depth = 0
    for index in range(close_index, -1, -1):
        if source[index] == "}":
            depth += 1
        elif source[index] == "{":
            depth -= 1
            if depth == 0:
                return index
    return -1


def _destructured_keys(body: str) -> set:
    keys, depth, part = set(), 0, ""
    for char in body + ",":
        if char in "{[(":
            depth += 1
        elif char in "}])":
            depth -= 1
        if char == "," and depth == 0:
            entry = part.strip()
            part = ""
            if not entry:
                continue
            if entry.startswith("..."):
                keys.add("<rest>")
                continue
            if entry.startswith("["):
                keys.add("<computed>")
                continue
            name = re.split(r"[:=]", entry, maxsplit=1)[0].strip().strip("\"'`")
            keys.add(name if re.fullmatch(r"\w+", name) else "<computed>")
        else:
            part += char
    return keys


def _ts_code_only(source: str) -> str:
    """Blank comments and prose string contents, preserving offsets.

    Prose such as "one finalized message. Sends ids only" must not read as `message.Sends`.
    Identifier-like string literals ("revision", "started") and template literals, whose
    `${...}` holes are code, are kept.
    """
    out, i, n = list(source), 0, len(source)
    while i < n:
        two = source[i : i + 2]
        if two == "//":
            end = source.find("\n", i)
            end = n if end == -1 else end
            out[i:end] = " " * (end - i)
            i = end
        elif two == "/*":
            end = source.find("*/", i + 2)
            end = n if end == -1 else end + 2
            out[i:end] = [c if c == "\n" else " " for c in source[i:end]]
            i = end
        elif source[i] in "\"'`":
            quote, j = source[i], i + 1
            while j < n and source[j] != quote and not (quote != "`" and source[j] == "\n"):
                j += 2 if source[j] == "\\" else 1
            if quote != "`" and not re.fullmatch(r"\w*", source[i + 1 : j]):
                out[i + 1 : j] = " " * (j - i - 1)
            i = j + 1
        else:
            i += 1
    return "".join(out)


def _store_key_reads(source: str) -> set:
    """Top-level message keys a store module reads, by any syntax.

    Caught: `m.k`, `m?.k`, `(m as T).k`, `m["k"]`, destructuring (including nested patterns and
    parameter destructuring), `"k" in m`, and any call passing `m` alongside a string-literal
    key (`field(m, "k")`). Reads the scanner cannot name -- computed `m[expr]`, spreads, rest
    patterns, `Object.keys(m)` -- are reported as `<computed>`, `<spread>`, `<rest>`,
    `<enumerate>`, so they can never pass as approved.
    """
    reads = set()
    source = _ts_code_only(source)
    for receiver in _ts_receivers(source):
        r = rf"(?<![\w$.]){re.escape(receiver)}(?![\w$])"
        reads.update(re.findall(rf"{r}\s*(?:\?\.|\.)\s*({_TS_IDENT})", source))
        reads.update(re.findall(rf"\(\s*{r}\s+as\s+[^()]*\)\s*(?:\?\.|\.)\s*({_TS_IDENT})", source))
        for match in re.finditer(rf"{r}\s*(?:\?\.)?\[\s*([^\]]*)\]", source):
            literal = re.fullmatch(r"([\"'`])(\w+)\1", match.group(1).strip())
            reads.add(literal.group(2) if literal else "<computed>")
        reads.update(re.findall(rf"[\"'`](\w+)[\"'`]\s+in\s+{r}", source))
        if re.search(rf"\.\.\.\s*\(?\s*{re.escape(receiver)}(?![\w$])", source):
            reads.add("<spread>")
        if re.search(rf"Object\s*\.\s*\w+\s*\(\s*{r}\s*[,)]", source):
            reads.add("<enumerate>")
        destructuring = rf"\}}\s*(?:=\s*\(?\s*{r}|:\s*EngineMessage\b)"
        for match in re.finditer(destructuring, source):
            opened = _balanced_open(source, match.start())
            if opened != -1:
                reads |= _destructured_keys(source[opened + 1 : match.start()])
        for call in re.finditer(rf"{_TS_IDENT}\s*\(([^()]*)\)", source):
            args = [arg.strip() for arg in call.group(1).split(",")]
            if any(re.fullmatch(rf"{r}(?:\s+as\s+.+)?", arg) for arg in args):
                for arg in args:
                    literal = re.fullmatch(r"([\"'`])(\w+)\1", arg)
                    if literal:
                        reads.add(literal.group(2))
    return reads


def _store_logic_sources() -> dict:
    files = [
        path
        for path in sorted(STORE_DIR.glob("*.ts"))
        if not path.name.endswith(".test.ts") and path.name != "types.ts"
    ]
    assert STORE in files and TRANSCRIPT_SLICE in files
    return {path.name: _read(path) for path in files}


def _allowed_store_reads(documented_keys: dict) -> set:
    legacy = set().union(*documented_keys.values())
    return legacy | APPROVED_STORE_ADDITIVE_KEYS


def _unapproved_store_reads(source: str, documented_keys: dict) -> set:
    return _store_key_reads(source) - _allowed_store_reads(documented_keys)


def _status_comparisons(source: str) -> tuple:
    """(status values compared against, uses of `.status` that are not such a comparison)."""
    compared, other = set(), []
    source = _ts_code_only(source)
    for receiver in _ts_receivers(source):
        r = rf"(?<![\w$.]){re.escape(receiver)}(?![\w$])"
        status = rf"{r}\s*(?:\?\.|\.)\s*status\b|{r}\s*(?:\?\.)?\[\s*[\"'`]status[\"'`]\s*\]"
        for match in re.finditer(status, source):
            after = re.match(r"\s*[!=]==?\s*[\"'`](\w+)[\"'`]", source[match.end() :])
            before = re.search(r"[\"'`](\w+)[\"'`]\s*[!=]==?\s*$", source[: match.start()])
            if after:
                compared.add(after.group(1))
            elif before:
                compared.add(before.group(1))
            else:
                other.append(source[match.start() : match.end() + 40])
    return compared, other


def _session_transcript_event(**overrides):
    from engine.session.session_orchestrator import STTTranscriptEvent

    values = {
        "stream_id": "system",
        "is_final": True,
        "text": "hello",
        "item_key": "system:1:local:system:0",
        "revision": 2,
        "epoch_id": 1,
        "provider_id": "local",
        "language": "en",
    }
    values.update(overrides)
    return STTTranscriptEvent(**values)


@pytest.fixture(scope="module")
def documented_keys():
    keys = _documented_payload_keys(_read(RULES))
    assert set(keys) == DOCUMENTED_ENGINE_TO_UI, "Payload-line extraction has rotted."
    return keys


def _stt_status_event(kind, **overrides):
    from engine.session.session_orchestrator import STTStatusEvent

    values = {
        "stream_id": "mic",
        "status": kind,
        "provider_id": "local",
        "previous_provider_id": "openai-realtime",
        "reason": "timeout",
        "state": "local_active",
    }
    values.update(overrides)
    return STTStatusEvent(**values)


def test_session_adds_no_message_types_and_the_documented_table_is_unchanged(
    surfaces, documented_keys
):
    assert surfaces["documented_out"] == DOCUMENTED_ENGINE_TO_UI
    assert surfaces["documented_in"] == DOCUMENTED_UI_TO_ENGINE
    assert surfaces["ui_handles"] == DOCUMENTED_ENGINE_TO_UI
    assert surfaces["ui_sends"] == DOCUMENTED_UI_TO_ENGINE

    # The documented legacy payloads are the ones this contract extends.
    assert documented_keys["final"] == LEGACY_TRANSCRIPT_KEYS
    assert documented_keys["partial"] == LEGACY_TRANSCRIPT_KEYS - {"entry_id"}
    assert documented_keys["status"] == LEGACY_STATUS_KEYS
    assert documented_keys["translation_update"] == LEGACY_TRANSLATION_UPDATE_KEYS


def test_session_transcript_payloads_keep_legacy_keys_and_a_closed_additive_set(surfaces):
    from engine.main import _stt_transcript_message

    final = _stt_transcript_message(_session_transcript_event(), "e-1.5", "marian")
    partial = _stt_transcript_message(_session_transcript_event(is_final=False), None, None)

    assert set(final) == LEGACY_TRANSCRIPT_KEYS | ADDITIVE_TRANSCRIPT_KEYS
    assert set(partial) == (LEGACY_TRANSCRIPT_KEYS - {"entry_id", "translation_backend"}) | (
        ADDITIVE_TRANSCRIPT_KEYS
    )

    # Legacy meaning is unchanged: translation always arrives later, finals carry entry_id.
    assert (final["type"], final["is_final"], final["translation"]) == ("final", True, None)
    assert (final["text"], final["source"], final["entry_id"]) == ("hello", "system", "e-1.5")
    assert final["translation_backend"] == "marian"
    assert isinstance(final["timestamp"], float)
    assert (partial["type"], partial["is_final"], partial["translation"]) == (
        "partial",
        False,
        None,
    )
    assert {final["type"], partial["type"]} <= surfaces["documented_out"]


def test_session_status_payloads_keep_legacy_keys_and_a_closed_additive_set(surfaces):
    from engine.main import _stt_status_message
    from engine.session.session_orchestrator import STTStatusKind

    kinds = list(STTStatusKind)
    assert {kind.value for kind in kinds} == STT_STATUS_VALUES
    for kind in kinds:
        for extra in ({}, {"replay_start_ms": 1.0, "replay_end_ms": 2.0, "gap": (1.0, 2.0)}):
            message = _stt_status_message(_stt_status_event(kind, **extra))
            assert set(message) == LEGACY_STATUS_KEYS | ADDITIVE_STATUS_KEYS
            assert (message["type"], message["status"]) == ("status", kind.value)
            assert message["type"] in surfaces["documented_out"]


def test_engine_emits_no_unlisted_keys_on_extended_messages():
    emitted: dict = {}
    for path in ENGINE_SOURCES:
        for message_type, key_sets in _engine_emitted_keys(_read(path)).items():
            if message_type in ENGINE_ALLOWED_KEYS:
                emitted.setdefault(message_type, []).extend(key_sets)

    # Non-vacuous: every extended message has at least one emitter in the scanned sources.
    assert set(emitted) == set(ENGINE_ALLOWED_KEYS), "Emitter extraction has rotted."
    for message_type, key_sets in emitted.items():
        for keys in key_sets:
            assert "type" in keys
            unlisted = keys - ENGINE_ALLOWED_KEYS[message_type]
            assert not unlisted, f"`{message_type}` emits unlisted keys: {sorted(unlisted)}"

    # Every legacy key is still emitted.
    assert any(keys >= LEGACY_TRANSCRIPT_KEYS for keys in emitted["final"])
    assert any(keys >= LEGACY_TRANSCRIPT_KEYS - {"entry_id"} for keys in emitted["partial"])
    assert any(keys >= LEGACY_TRANSLATION_UPDATE_KEYS for keys in emitted["translation_update"])
    assert any(keys == LEGACY_STATUS_KEYS for keys in emitted["status"])
    status_values = set(re.findall(r'"status":\s*"(\w+)"', _read(SERVER)))
    assert {"started", "stopped"} <= status_values


def test_store_reads_only_legacy_and_approved_additive_keys(documented_keys):
    sources = _store_logic_sources()
    reads = {name: _store_key_reads(src) for name, src in sources.items()}
    # Non-vacuous: the transcript handlers must be visible to the scanner.
    assert {"text", "entry_id", "translation", "source"} <= reads[TRANSCRIPT_SLICE.name]
    assert "status" in reads[STORE.name]

    allowed = _allowed_store_reads(documented_keys)
    unapproved = {name: sorted(keys - allowed) for name, keys in reads.items() if keys - allowed}
    assert not unapproved, f"Store reads keys outside the approved contract: {unapproved}"


def test_store_keys_finals_by_the_legacy_entry_id():
    source = _read(TRANSCRIPT_SLICE)
    aliases = {"entry_id"}
    for receiver in _ts_receivers(source):
        r = re.escape(receiver)
        aliases.update(
            re.findall(
                rf"(?:const|let|var)\s+({_TS_IDENT})\s*=\s*\(?\s*{r}\s*"
                rf"(?:\?\.\s*|\.\s*|\[\s*[\"'`])entry_id\b",
                source,
            )
        )
    id_values = re.findall(r"(?<![\w$])id\s*:\s*([^,\n}]+)", source)
    keyed = [value for value in id_values if any(re.search(rf"\b{a}\b", value) for a in aliases)]
    assert keyed, "No transcript entry takes its id from entry_id."
    for value in id_values:
        assert not re.search(r"item_?[iI]d|epoch|revision", value), (
            f"Transcript entry id derived from session metadata: {value.strip()}"
        )


def test_session_status_values_are_a_closed_set_apart_from_lifecycle():
    import importlib.util

    assert not (STT_STATUS_VALUES | TRANSLATION_STATUS_VALUES) & {"started", "stopped"}
    # The Translation session may not be integrated yet; once it is, its values are pinned.
    if importlib.util.find_spec("engine.session.translation_orchestrator") is not None:
        from engine.session.translation_orchestrator import TranslationStatusKind

        assert {kind.value for kind in TranslationStatusKind} == TRANSLATION_STATUS_VALUES

    # Every status value the engine emits as a literal or enum is lifecycle or listed.
    emitted = set()
    for path in ENGINE_SOURCES:
        emitted |= set(re.findall(r'"status":\s*"(\w+)"', _read(path)))
    assert emitted <= {"started", "stopped"}


def test_store_treats_only_started_and_stopped_as_lifecycle_status():
    compared, uses = set(), []
    for name, source in _store_logic_sources().items():
        values, other = _status_comparisons(source)
        compared |= values
        uses += [f"{name}: {use.strip()}" for use in other]
        for value in STT_STATUS_VALUES | TRANSLATION_STATUS_VALUES:
            literal = rf"[\"'`]{value}[\"'`]"
            assert not re.search(literal, source), f"{name} routes session status `{value}`."

    assert compared == {"started", "stopped"}
    assert not uses, f"`status` read outside a started/stopped comparison: {uses}"


# ── Scanner self-tests: in-test fixtures, no files ──────────────────────────────────────

_READ_FORMS = {
    "dot": "const v = message.{key};",
    "optional": "const v = message?.{key};",
    "cast": "const v = (message as EngineMessage).{key};",
    "bracket": 'const v = message["{key}"];',
    "optional_bracket": "const v = message?.['{key}'];",
    "annotated_param": "function f(evt: EngineMessage) {{ return evt.{key}; }}",
    "destructure": "const {{ {key}: v }} = message;",
    "nested_destructure": "const {{ {key}: {{ start_ms }} }} = message;",
    "param_destructure": "function f({{ text, {key} }}: EngineMessage) {{ return text; }}",
    "helper": 'function f(evt: EngineMessage) {{ return field(evt, "{key}"); }}',
    "helper_key_first": 'const v = pick("{key}", message as EngineMessage);',
    "in_operator": 'const has = "{key}" in message;',
}


@pytest.mark.parametrize("form", sorted(_READ_FORMS))
@pytest.mark.parametrize("key", sorted(ADDITIVE_TRANSCRIPT_KEYS | ADDITIVE_TRANSLATION_UPDATE_KEYS))
def test_scanner_accepts_each_approved_key_in_every_read_form(documented_keys, form, key):
    snippet = _READ_FORMS[form].format(key=key)
    assert key in _store_key_reads(snippet)
    assert _unapproved_store_reads(snippet, documented_keys) == set()


@pytest.mark.parametrize("form", sorted(_READ_FORMS))
def test_scanner_rejects_an_unapproved_key_in_every_read_form(documented_keys, form):
    snippet = _READ_FORMS[form].format(key="stt_debug_token")
    assert _unapproved_store_reads(snippet, documented_keys) == {"stt_debug_token"}


@pytest.mark.parametrize(
    ("snippet", "marker"),
    [
        ("const v = message[key];", "<computed>"),
        ("const copy = { ...message };", "<spread>"),
        ("const { text, ...rest } = message;", "<rest>"),
        ("for (const k of Object.keys(message)) {}", "<enumerate>"),
    ],
)
def test_scanner_rejects_reads_it_cannot_name(documented_keys, snippet, marker):
    assert marker in _unapproved_store_reads(snippet, documented_keys)


def test_scanner_ignores_comments_and_prose_but_not_code(documented_keys):
    prose = (
        "// Manual Assist on one finalized message. Sends ids only.\n"
        "/* the message.secret_key is never read */\n"
        'const text = "Assist needs this message. Live transcripts";\n'
    )
    assert _store_key_reads(prose) == set()
    code = prose + 'const v = `${message.stt_debug_token}`; const w = message["revision"];'
    assert _unapproved_store_reads(code, documented_keys) == {"stt_debug_token"}
    assert "revision" in _store_key_reads(code)


def test_scanner_ignores_other_objects_and_flags_non_lifecycle_status_use():
    assert _store_key_reads("const v = error.message.secret; other.stt_debug;") == set()
    assert _status_comparisons('if (message.status === "stt_fallback") {}')[0] == {"stt_fallback"}
    assert _status_comparisons("set({ phase: message.status });")[1]
