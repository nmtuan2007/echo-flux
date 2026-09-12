"""Pins the configuration precedence chain documented in .agent/rules/01-python-engine.md.

Order (later wins): defaults -> config.json -> .env -> environment variables.
"""

import os


def test_defaults_are_available_without_any_files(make_config):
    config = make_config()
    assert config.get("engine.port") == 8765
    assert config.get("engine.host") == "127.0.0.1"
    assert config.get("asr.model_size") == "small"
    assert config.get("vad.enabled") is True
    assert config.get("translation.enabled") is False


def test_missing_key_returns_default_argument(make_config):
    config = make_config()
    assert config.get("nope.not.here") is None
    assert config.get("nope.not.here", "fallback") == "fallback"


def test_config_file_overrides_defaults(make_config):
    config = make_config({"engine": {"port": 9999}})
    assert config.get("engine.port") == 9999
    # Sibling keys in the same section must survive the merge.
    assert config.get("engine.host") == "127.0.0.1"


def test_config_file_accepts_dotted_keys(make_config):
    # _deep_merge routes any key containing "." through set().
    config = make_config({"asr.model_size": "large-v3"})
    assert config.get("asr.model_size") == "large-v3"


def test_dotenv_overrides_config_file(make_config):
    config = make_config(
        {"engine": {"port": 9999}},
        dotenv="ECHOFLUX_PORT=7777\n",
    )
    assert config.get("engine.port") == 7777


def test_environment_variable_wins_over_dotenv(make_config, monkeypatch):
    # _load_dotenv refuses to clobber an existing os.environ key, so a real
    # environment variable outranks the .env file.
    monkeypatch.setenv("ECHOFLUX_PORT", "6666")
    config = make_config(dotenv="ECHOFLUX_PORT=7777\n")
    assert config.get("engine.port") == 6666


def test_full_precedence_chain(make_config, monkeypatch):
    monkeypatch.setenv("ECHOFLUX_LANGUAGE", "ja")
    config = make_config(
        {"asr": {"model_size": "medium", "language": "fr"}},
        dotenv="ECHOFLUX_LANGUAGE=de\nECHOFLUX_MODEL_SIZE=tiny\n",
    )
    assert config.get("asr.language") == "ja"  # environment beats .env
    assert config.get("asr.model_size") == "tiny"  # .env beats config.json
    assert config.get("asr.device") == "auto"  # default survives


def test_dotenv_strips_quotes_and_ignores_comments(make_config):
    config = make_config(
        dotenv='# a comment\n\nECHOFLUX_HOST="0.0.0.0"\nnot_a_pair\n',
    )
    assert config.get("engine.host") == "0.0.0.0"


def test_env_values_are_cast_to_declared_types(make_config):
    config = make_config(
        dotenv=(
            "ECHOFLUX_PORT=1234\n"
            "ECHOFLUX_VAD_THRESHOLD=0.75\n"
            "ECHOFLUX_TRANSLATION_ENABLED=true\n"
            "ECHOFLUX_VAD_ENABLED=false\n"
        )
    )
    assert config.get("engine.port") == 1234
    assert isinstance(config.get("engine.port"), int)
    assert config.get("vad.threshold") == 0.75
    assert config.get("translation.enabled") is True
    assert config.get("vad.enabled") is False


def test_uncastable_env_value_is_ignored_not_fatal(make_config):
    # _apply_env_overrides swallows ValueError/TypeError and keeps the default.
    config = make_config(dotenv="ECHOFLUX_PORT=not-a-number\n")
    assert config.get("engine.port") == 8765


def test_set_creates_intermediate_nodes(make_config):
    config = make_config()
    config.set("brand.new.branch", 42)
    assert config.get("brand.new.branch") == 42


def test_directories_resolve_under_data_dir(make_config, isolated_env):
    config = make_config()
    assert config.data_dir == isolated_env
    assert config.models_dir == isolated_env / "models"
    assert config.logs_dir == isolated_env / "logs"
    # Config creates them eagerly on construction.
    assert config.models_dir.is_dir()
    assert config.logs_dir.is_dir()


def test_models_dir_env_override(tmp_path, monkeypatch, make_config):
    override = tmp_path / "elsewhere"
    monkeypatch.setenv("ECHOFLUX_MODELS_DIR", str(override))
    config = make_config()
    assert config.models_dir == override


def test_save_then_reload_round_trips(make_config, tmp_path):
    from engine.core.config import Config

    config = make_config()
    config.set("asr.model_size", "large-v3")
    config.save()

    reloaded = Config(
        config_path=str(tmp_path / "config.json"),
        env_path=str(tmp_path / ".env"),
    )
    assert reloaded.get("asr.model_size") == "large-v3"


def test_config_does_not_read_the_real_project_dotenv(make_config):
    # Guards the fixture itself: if this ever fails, tests are reading real secrets.
    config = make_config()
    assert config.get("llm.api_key") == ""
    assert "ECHOFLUX_LLM_API_KEY" not in os.environ
