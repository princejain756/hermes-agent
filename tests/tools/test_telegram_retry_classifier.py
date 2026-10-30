"""Telegram send-retry classification (#146922).

``_telegram_retry_delay`` must classify structured transport status — ``retry_after``,
``status_code``/``error_code``/``response.status`` attributes, explicit "HTTP 502"-shaped text —
not bare status digits anywhere in an exception message. A parse error whose text happens to
carry ``502`` inside a URL path is a permanent failure for the rich-text attempt and must fall
back to plain text immediately, not wait out a transient backoff first.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from tools.send_message_senders import _send_telegram_message_with_retry, _telegram_retry_delay


def test_parse_error_digits_do_not_select_retry():
    exc = Exception("can't parse entities at byte offset 10: https://example.com/502/bad-tag")
    assert _telegram_retry_delay(exc, 0) is None


def test_bare_status_digits_do_not_select_retry():
    exc = Exception("offset 502 in the entity list")
    assert _telegram_retry_delay(exc, 0) is None


def test_structured_status_selects_retry():
    exc = Exception("upstream failed")
    exc.response = SimpleNamespace(status_code=502)
    assert _telegram_retry_delay(exc, 0) == 1.0
    assert _telegram_retry_delay(exc, 1) == 2.0


def test_status_attr_429_selects_retry():
    exc = Exception("flood")
    exc.status_code = 429
    assert _telegram_retry_delay(exc, 0) == 1.0


def test_permanent_status_does_not_select_retry():
    exc = Exception("bad request")
    exc.status_code = 400
    assert _telegram_retry_delay(exc, 0) is None


def test_status_shaped_text_selects_retry():
    exc = Exception("HTTP 503 from upstream")
    assert _telegram_retry_delay(exc, 0) == 1.0


def test_retry_after_attribute_wins():
    exc = Exception("flood wait")
    exc.retry_after = 7
    assert _telegram_retry_delay(exc, 0) == 7.0


def test_timeout_is_never_retried():
    exc = Exception("Request timed out")
    assert _telegram_retry_delay(exc, 0) is None


def test_phrase_marker_selects_retry():
    exc = Exception("Bad Gateway")
    assert _telegram_retry_delay(exc, 0) == 1.0


def test_retry_loop_does_not_delay_on_parse_error_digits(monkeypatch):
    """End-to-end: a parse error carrying ``502`` in its text surfaces immediately — no backoff sleep."""
    sleeps: list[float] = []
    async def _sleep(delay):
        sleeps.append(delay)
    monkeypatch.setattr(asyncio, "sleep", _sleep)
    bot = MagicMock()
    exc = Exception("can't parse entities at byte offset 10: https://example.com/502/bad-tag")
    bot.send_message = AsyncMock(side_effect=exc)
    with pytest.raises(Exception, match="can't parse entities"):
        asyncio.run(_send_telegram_message_with_retry(bot, chat_id=1, text="hi"))
    assert sleeps == []
    assert bot.send_message.await_count == 1
