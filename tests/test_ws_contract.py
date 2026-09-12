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
