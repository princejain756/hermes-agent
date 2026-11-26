"""session.interrupt on a compute host that is gone (hermes-agent 82603).

A dead host (broken pipe, respawn exhausted) can never deliver the ``turn.end`` that clears
``running``. Stop used to return 5019 and leave the session busy forever, so every later
``prompt.submit`` queued with no way back short of a backend restart. Stop now recovers the
session unless the supervisor still owes it a completion or a successor turn owns it.
"""

from __future__ import annotations

import threading

from tui_gateway import server


class _DeadComputeHost:
    """Supervisor whose interrupt() raises and which owes a completion only for ``pending_sids``."""

    def __init__(self, pending_sids=()):
        self._pending_sids = set(pending_sids)

    def interrupt(self, sid, *, request_id=None):
        raise RuntimeError("compute host is not running")

    def has_pending_turn(self, sid):
        return sid in self._pending_sids


def _isolated_session(sid, monkeypatch, host, **extra):
    session = {
        "agent": None,
        "agent_ready": threading.Event(),
        "session_key": "session-key",
        "history": [],
        "history_lock": threading.Lock(),
        "running": True,
        "_compute_host_active": True,
        **extra,
    }
    server._sessions[sid] = session
    monkeypatch.setattr(server, "_load_cfg", lambda: {"dashboard": {"turn_isolation": True}})
    monkeypatch.setattr(server, "_get_compute_host_supervisor", lambda _cfg=None: host)
    return session


def _interrupt(sid):
    try:
        return server.handle_request({"id": "1", "method": "session.interrupt", "params": {"session_id": sid}})
    finally:
        server._sessions.pop(sid, None)


def test_dead_host_stop_clears_the_stuck_turn(monkeypatch):
    session = _isolated_session(
        "iso-dead", monkeypatch, _DeadComputeHost(),
        queued_prompt={"text": "stale next", "transport": None},
        inflight_turn={"user": "hi", "assistant": "", "streaming": True},
    )

    resp = _interrupt("iso-dead")

    assert resp.get("result") == {"status": "interrupted", "turn_isolation": True}, resp
    assert session["running"] is False
    assert session["_turn_cancel_requested"] is True
    assert session.get("queued_prompt") is None
    assert session.get("inflight_turn") is None


def test_a_pending_host_completion_keeps_its_own_teardown(monkeypatch):
    session = _isolated_session(
        "iso-pending", monkeypatch, _DeadComputeHost(pending_sids=("iso-pending",)),
        inflight_turn={"user": "hi", "assistant": "", "streaming": True},
    )

    resp = _interrupt("iso-pending")

    assert resp.get("result") == {"status": "interrupted", "turn_isolation": True}, resp
    assert session["running"] is True
    assert session["_turn_cancel_requested"] is True
    assert session.get("inflight_turn") is not None


def test_after_teardown_running_belongs_to_the_drain(monkeypatch):
    # The crash waiter already cleared inflight; a successor drain may own `running` for its queued prompt.
    session = _isolated_session("iso-post", monkeypatch, _DeadComputeHost(), inflight_turn=None)

    resp = _interrupt("iso-post")

    assert resp.get("result") == {"status": "interrupted", "turn_isolation": True}, resp
    assert session["running"] is True
    assert session["_turn_cancel_requested"] is True


def test_a_successor_turn_that_replaced_inflight_is_not_clobbered(monkeypatch):
    old_inflight = {"user": "old", "assistant": "", "streaming": True}
    new_inflight = {"user": "new", "assistant": "", "streaming": True}
    host = _DeadComputeHost()
    session = _isolated_session("iso-race", monkeypatch, host, inflight_turn=old_inflight)
    cleared_pending = []
    cancelled_approvals = []
    resolved_approvals = []
    monkeypatch.setattr(server, "_clear_pending", lambda s: cleared_pending.append(s))
    monkeypatch.setattr(
        server,
        "_announce_cancelled_gateway_approvals",
        lambda s, reason, session_id=None: cancelled_approvals.append((reason, session_id)),
    )
    import tools.approval as approval_mod
    monkeypatch.setattr(
        approval_mod,
        "resolve_gateway_approval",
        lambda key, decision, resolve_all=False: resolved_approvals.append((key, decision, resolve_all)),
    )

    def interrupt_and_swap(sid, *, request_id=None):
        # prompt.submit replaces inflight after history_lock is released, before submit_turn registers it.
        with session["history_lock"]:
            session["inflight_turn"] = new_inflight
            session["_turn_cancel_requested"] = False
            session["queued_prompt"] = {"text": "keep me", "transport": None}
            session["queued_prompts"] = [{"text": "keep me", "transport": None}]
            session["_queued_prompt_generation"] = 3
        raise RuntimeError("compute host is not running")

    host.interrupt = interrupt_and_swap

    resp = _interrupt("iso-race")

    assert resp.get("result") == {"status": "interrupted", "turn_isolation": True}, resp
    assert session["running"] is True
    assert session.get("inflight_turn") is new_inflight
    assert session["_turn_cancel_requested"] is False
    assert session["queued_prompt"] == {"text": "keep me", "transport": None}
    assert session["queued_prompts"] == [{"text": "keep me", "transport": None}]
    assert session["_queued_prompt_generation"] == 3
    assert cleared_pending == []
    assert cancelled_approvals == []
    assert resolved_approvals == []


def test_steered_turn_still_clears_on_dead_host_recovery(monkeypatch):
    host = _DeadComputeHost()
    session = _isolated_session("iso-steered", monkeypatch, host)
    server._start_inflight_turn(session, "initial")
    cleared_pending = []
    monkeypatch.setattr(server, "_clear_pending", lambda s: cleared_pending.append(s))

    def interrupt_and_steer(sid, *, request_id=None):
        with session["history_lock"]:
            server._record_inflight_correction(session, "redirected")
        raise RuntimeError("compute host is not running")

    host.interrupt = interrupt_and_steer

    resp = _interrupt("iso-steered")

    assert resp.get("result") == {"status": "interrupted", "turn_isolation": True}, resp
    assert session["running"] is False
    assert session.get("inflight_turn") is None
    assert session["_turn_cancel_requested"] is True
    assert cleared_pending == ["iso-steered"]


def test_idle_capture_followed_by_new_turn_does_not_clobber_successor(monkeypatch):
    host = _DeadComputeHost()
    session = _isolated_session("iso-idle-race", monkeypatch, host, running=False, inflight_turn=None)
    cleared_pending = []
    monkeypatch.setattr(server, "_clear_pending", lambda s: cleared_pending.append(s))

    def interrupt_and_start(sid, *, request_id=None):
        with session["history_lock"]:
            server._start_inflight_turn(session, "fresh")
            session["running"] = True
            session["_turn_cancel_requested"] = False
            session["queued_prompt"] = {"text": "queued", "transport": None}
        raise RuntimeError("compute host is not running")

    host.interrupt = interrupt_and_start

    resp = _interrupt("iso-idle-race")

    assert resp.get("result") == {"status": "interrupted", "turn_isolation": True}, resp
    assert session["running"] is True
    assert session["_turn_cancel_requested"] is False
    assert session["queued_prompt"] == {"text": "queued", "transport": None}
    assert cleared_pending == []


def test_inflight_snapshot_hides_turn_token():
    session = {"inflight_turn": None}
    server._start_inflight_turn(session, "hello")
    assert "_turn_token" in session["inflight_turn"]
    snap = server._inflight_snapshot(session)
    assert snap is not None
    assert "_turn_token" not in snap


def test_supervisor_reports_pending_turns_per_session():
    from tui_gateway.host_supervisor import HostSupervisor

    supervisor = HostSupervisor.__new__(HostSupervisor)  # __init__ would spawn a child
    supervisor._lock = threading.RLock()
    supervisor._pending_turns = {"req-a": ("sid-a", None), "req-b": ("sid-b", None)}

    assert supervisor.has_pending_turn("sid-a") is True
    assert supervisor.has_pending_turn("sid-c") is False
    supervisor._pending_turns.pop("req-a")
    assert supervisor.has_pending_turn("sid-a") is False
