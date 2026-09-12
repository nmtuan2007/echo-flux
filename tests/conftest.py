"""Shared fixtures.

The isolation fixture here is load-bearing. ``Config.__init__`` calls ``_load_dotenv()``
on both the project ``.env`` and ``~/.echoflux/.env``, writing into ``os.environ``, and it
mkdirs its data/models/logs directories on construction. Without isolation a test run
reads the developer's real settings, creates real directories, and leaks environment
state into every subsequent test in the session.
"""

import json
import os
from pathlib import Path
from typing import Optional

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch):
    """Point Config at a throwaway data dir and strip inherited ECHOFLUX_* vars."""
    for key in list(os.environ):
        if key.startswith("ECHOFLUX_"):
            monkeypatch.delenv(key, raising=False)

    data_dir = tmp_path / "data"
    monkeypatch.setenv("ECHOFLUX_DATA_DIR", str(data_dir))
    return data_dir


@pytest.fixture
def make_config(tmp_path):
    """Build a Config with explicit paths, never the real project .env.

    ``Config`` defaults ``env_path`` to ``Path.cwd() / ".env"``, so tests that omit it
    would silently pick up the developer's real credentials.
    """
    from engine.core.config import Config

    def _make(config_data: Optional[dict] = None, dotenv: Optional[str] = None):
        config_path = tmp_path / "config.json"
        if config_data is not None:
            config_path.write_text(json.dumps(config_data), encoding="utf-8")

        env_path = tmp_path / ".env"
        # Always pass a path; an absent file makes _load_dotenv a no-op.
        if dotenv is not None:
            env_path.write_text(dotenv, encoding="utf-8")

        return Config(config_path=str(config_path), env_path=str(env_path))

    return _make


@pytest.fixture(scope="session")
def project_root():
    return PROJECT_ROOT
