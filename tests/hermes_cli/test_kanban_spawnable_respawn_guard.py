"""Tests that spawnability probes respect check_respawn_guard (#159275)."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

from gateway.kanban_watchers_dispatcher import _KanbanDispatcher, _resolve_dispatcher_settings
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _move_to_review_with_auth_failure(conn, title: str, assignee: str) -> str:
    tid = kb.create_task(conn, title=title, assignee="worker")
    kb.claim_task(conn, tid)
    run_id = kb.get_task(conn, tid).current_run_id
    assert kb.request_review(
        conn,
        tid,
        summary="PR ready for review",
        reviewer=assignee,
        expected_run_id=run_id,
    )
    conn.execute(
        "UPDATE tasks SET last_failure_error = ? WHERE id = ?",
        ("PR acceptance auth: GitHub refused the acceptance read", tid),
    )
    conn.commit()
    return tid


def test_has_spawnable_review_skips_blocker_auth_guarded_task(
    kanban_home: Path, all_assignees_spawnable, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A review card parked by blocker_auth must not trip the stuck watchdog."""
    monkeypatch.setenv("HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS", "0")
    with kbc.connect() as conn:
        tid = _move_to_review_with_auth_failure(conn, "guarded review", "reviewer")
        assert kbd.check_respawn_guard(conn, tid, lane="review") == "blocker_auth"
        assert kbd.has_spawnable_review(conn) is False
        res = kbd.dispatch_once(conn, dry_run=True)
        assert res.spawned == []
        assert res.respawn_guarded == [(tid, "blocker_auth")]

    dispatcher = _KanbanDispatcher(kb, _resolve_dispatcher_settings({}, kb))
    assert dispatcher.ready_nonempty() is False


def test_has_spawnable_review_true_when_unguarded_peer_shares_assignee(
    kanban_home: Path, all_assignees_spawnable, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One guarded card must not hide an unguarded card with the same assignee."""
    monkeypatch.setenv("HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS", "0")
    with kbc.connect() as conn:
        guarded = _move_to_review_with_auth_failure(conn, "guarded review", "reviewer")
        ready_review = kb.create_task(conn, title="healthy review", assignee="worker")
        kb.claim_task(conn, ready_review)
        run_id = kb.get_task(conn, ready_review).current_run_id
        assert kb.request_review(
            conn,
            ready_review,
            summary="Clean PR",
            reviewer="reviewer",
            expected_run_id=run_id,
        )
        assert kbd.check_respawn_guard(conn, guarded, lane="review") == "blocker_auth"
        assert kbd.check_respawn_guard(conn, ready_review, lane="review") is None
        assert kbd.has_spawnable_review(conn) is True


def test_has_spawnable_ready_skips_respawn_guarded_and_respects_profiles_fallback(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both guarded ready cards and profiles-unavailable fallback check the guard."""
    monkeypatch.setenv("HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS", "0")
    with kbc.connect() as conn:
        guarded = kb.create_task(conn, title="guarded ready", assignee="default")
        unguarded = kb.create_task(conn, title="queued ready", assignee="default")
        conn.execute(
            "UPDATE tasks SET last_failure_error = ? WHERE id = ?",
            ("401 authentication failed", guarded),
        )
        conn.execute("UPDATE tasks SET status = 'todo' WHERE id = ?", (unguarded,))
        conn.commit()
        monkeypatch.setitem(sys.modules, "hermes_cli.profiles", None)
        assert kbd.check_respawn_guard(conn, guarded, lane="ready") == "blocker_auth"
        assert kbd.has_spawnable_ready(conn) is False

        conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (unguarded,))
        conn.commit()
        assert kbd.has_spawnable_ready(conn) is True


def test_has_spawnable_review_keeps_normal_review_handoff_spawnable(
    kanban_home: Path, all_assignees_spawnable, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A normal review handoff with a PR URL comment stays spawnable in review."""
    monkeypatch.setenv("HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS", "0")
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="pr review", assignee="worker")
        kb.claim_task(conn, tid)
        run_id = kb.get_task(conn, tid).current_run_id
        kb.add_comment(conn, tid, "worker", "Opened https://github.com/org/repo/pull/42")
        assert kb.request_review(
            conn,
            tid,
            summary="Ready for review",
            reviewer="reviewer",
            expected_run_id=run_id,
        )
        # Ensure ended_at is strictly in the past relative to check_respawn_guard's window.
        conn.execute("UPDATE task_runs SET ended_at = ? WHERE task_id = ?", (int(time.time()) - 5, tid))
        conn.commit()
        assert kbd.check_respawn_guard(conn, tid, lane="review") is None
        assert kbd.has_spawnable_review(conn) is True
