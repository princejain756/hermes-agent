"""Regression for #146870: a group/user allowlist edit to the owning profile's config.yaml
takes effect on the running Telegram adapter — no gateway restart, no reconnect.

Two stale layers made the persisted file and the effective policy disagree: the adapter's
construction-time ``config.extra`` and the env value its own YAML bridge wrote once and never
revisited. The refresh re-derives both from the file, while an env value the bridge did not
write (an explicit override) keeps winning.
"""
import asyncio
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms import _shared


def _make_adapter(extra, *, owner_profile=None):
    from plugins.platforms.telegram.adapter import TelegramAdapter

    adapter = object.__new__(TelegramAdapter)
    adapter.platform = Platform.TELEGRAM
    adapter.config = PlatformConfig(enabled=True, token="fake-token", extra=dict(extra))
    adapter._owner_profile = owner_profile
    adapter._bot = SimpleNamespace(id=999, username="test_bot")
    adapter._message_handler = AsyncMock()
    adapter._pending_text_batches = {}
    adapter._pending_text_batch_tasks = {}
    adapter._text_batch_delay_seconds = 0.01
    adapter._text_batch_split_delay_seconds = 0.01
    adapter._mention_patterns = adapter._compile_mention_patterns()
    adapter._forum_lock = asyncio.Lock()
    adapter._forum_command_registered = set()
    adapter._active_sessions = {}
    adapter._pending_messages = {}
    return adapter


def _make_message(text="hello", *, from_user_id=111, chat_id=-222, chat_type="group"):
    return SimpleNamespace(
        message_id=42,
        text=text,
        caption=None,
        entities=[],
        caption_entities=[],
        message_thread_id=None,
        is_topic_message=False,
        chat=SimpleNamespace(id=chat_id, type=chat_type, title="Test", is_forum=False),
        from_user=SimpleNamespace(id=from_user_id, full_name="Test User", first_name="Test"),
        reply_to_message=None,
        date=None,
        location=None,
        photo=None,
        video=None,
        audio=None,
        voice=None,
        document=None,
        sticker=None,
        media_group_id=None,
    )


@pytest.fixture
def telegram_home(tmp_path, monkeypatch):
    """An isolated profile home holding a config.yaml the adapter's gate reads live."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("TELEGRAM_ALLOWED_CHATS", raising=False)
    monkeypatch.delenv("TELEGRAM_GROUP_ALLOWED_CHATS", raising=False)
    bridged = dict(_shared._YAML_BRIDGED_ENV)
    try:
        yield home
    finally:
        _shared._YAML_BRIDGED_ENV.clear()
        _shared._YAML_BRIDGED_ENV.update(bridged)


def _write_config(home, body):
    (home / "config.yaml").write_text(body, encoding="utf-8")


def test_added_group_chat_admitted_without_restart(telegram_home):
    _write_config(telegram_home, "telegram:\n  allowed_chats: [-100]\n")
    adapter = _make_adapter({"allowed_chats": ["-100"]})

    assert adapter._should_process_message(_make_message()) is False

    _write_config(telegram_home, "telegram:\n  allowed_chats: [-100, -222]\n")

    assert adapter._should_process_message(_make_message()) is True


def test_bridged_env_value_updated_not_kept_stale(telegram_home):
    """The env rung outranks extra — a stale bridged write must move with the file too."""
    _write_config(telegram_home, "telegram:\n  allowed_chats: [-100]\n")
    _shared.yaml_env_setter()("TELEGRAM_ALLOWED_CHATS", "-100")  # what boot's bridge did
    adapter = _make_adapter({})

    assert adapter._should_process_message(_make_message()) is False

    _write_config(telegram_home, "telegram:\n  allowed_chats: [-222]\n")

    assert adapter._should_process_message(_make_message()) is True
    assert os.environ["TELEGRAM_ALLOWED_CHATS"] == "-222"


def test_removed_allowlist_key_retracts_bridged_env_and_extra(telegram_home):
    _write_config(telegram_home, "telegram:\n  allowed_chats: [-100]\n")
    _shared.yaml_env_setter()("TELEGRAM_ALLOWED_CHATS", "-100")
    adapter = _make_adapter({"allowed_chats": ["-100"]})
    assert adapter._should_process_message(_make_message()) is False

    _write_config(telegram_home, "telegram:\n  require_mention: false\n")

    # An empty allowlist is unrestricted, so the previously denied chat now passes the gate.
    assert adapter._should_process_message(_make_message()) is True
    assert os.environ.get("TELEGRAM_ALLOWED_CHATS") is None
    assert "allowed_chats" not in adapter.config.extra


def test_explicit_env_override_survives_the_refresh(telegram_home):
    """An env value the bridge did not write is an explicit override and keeps precedence."""
    _write_config(telegram_home, "telegram:\n  allowed_chats: [-100, -222]\n")
    os.environ["TELEGRAM_ALLOWED_CHATS"] = "-333"  # user-set, never bridge-recorded
    adapter = _make_adapter({})

    assert adapter._should_process_message(_make_message()) is False
    assert os.environ["TELEGRAM_ALLOWED_CHATS"] == "-333"
    assert adapter._should_process_message(_make_message(chat_id=-333)) is True


def test_unparseable_config_keeps_the_current_policy(telegram_home):
    _write_config(telegram_home, "telegram:\n  allowed_chats: [-100]\n")
    adapter = _make_adapter({"allowed_chats": ["-100"]})
    assert adapter._should_process_message(_make_message()) is False

    _write_config(telegram_home, "telegram: [unclosed\n")

    assert adapter._should_process_message(_make_message()) is False
    assert "-100" in adapter._telegram_allowed_chats()
