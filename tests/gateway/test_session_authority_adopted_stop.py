"""A Stop after the turn adopted its agent reaches that agent even if the cache no longer holds it."""
import asyncio
from types import SimpleNamespace

import pytest

from tests.gateway.test_session_authority_cancel_settlement import ACTOR, REF, _authority, _submit, _wire_turn_agent
from tests.gateway.test_session_authority_preadopt_stop import Agent


@pytest.mark.asyncio
async def test_stop_between_adoption_and_promotion_reaches_the_adopted_agent_after_cache_eviction(tmp_path, monkeypatch):
    """adopt_agent ran, the route's running slot is still the claim's pending sentinel, and the
    agent cache entry was evicted (pressure / idle sweep): the Stop must still interrupt the agent
    this generation adopted, not resolve nobody through the cache and latch nothing."""
    from gateway import session_finite

    db, authority = _authority(tmp_path, monkeypatch)
    fresh = Agent()
    cache = {}
    authority.runner._cached_agent_for = cache.get
    adopted, stopped = asyncio.Event(), asyncio.Event()
    ran = {}

    async def execute(authority, ref, row):
        cache['route'] = fresh
        _wire_turn_agent(authority, row['generation'], fresh)
        cache.pop('route')  # evicted before the running slot is promoted
        adopted.set()
        await asyncio.wait_for(stopped.wait(), 5)
        ran[row['request_id']] = fresh.interrupted
        return 'done'
    monkeypatch.setattr(session_finite, 'execute_finite_admission', execute)

    with db:
        await _submit(authority, 'adopted')
        await asyncio.wait_for(adopted.wait(), 5)
        await authority.interrupt(ACTOR, REF, authority._handle(REF).execution_generation)
        stopped.set()
        await asyncio.wait_for(authority.sessions['s'].task, 5)
        assert ran['adopted'] is True, 'the Stop reached nobody: the adopted agent ran on'
        assert not authority.adopted, 'settlement must drop the adopted agent reference'


@pytest.mark.asyncio
async def test_retire_stop_between_adoption_and_promotion_reaches_the_adopted_agent(tmp_path, monkeypatch):
    """The same window on the retire/shutdown path: stop_authority_turns(in_process=True) used to
    re-resolve the running slot (still the pending sentinel) and latch a generation adopt_agent had
    already consumed, so the adopted agent ran on through the profile stop."""
    from gateway import session_finite
    from gateway.run_runtime import stop_authority_turns

    db, authority = _authority(tmp_path, monkeypatch)
    fresh = Agent()
    authority.runner._cached_agent_for = {}.get
    adopted, stopped = asyncio.Event(), asyncio.Event()
    ran = {}

    async def execute(authority, ref, row):
        _wire_turn_agent(authority, row['generation'], fresh)
        adopted.set()
        await asyncio.wait_for(stopped.wait(), 5)
        ran[row['request_id']] = fresh.interrupted
        return 'done'
    monkeypatch.setattr(session_finite, 'execute_finite_admission', execute)

    with db:
        await _submit(authority, 'adopted')
        await asyncio.wait_for(adopted.wait(), 5)
        stop_authority_turns(authority, in_process=True)
        stopped.set()
        await asyncio.wait_for(authority.sessions['s'].task, 5)
        assert ran['adopted'] is True, 'the profile Stop reached nobody: the adopted agent ran on'


def test_stop_authority_turns_handles_adopted_managed_worker_open_and_closing():
    """During managed shutdown, execute_managed.finally pops _managed_workers before worker.close()
    sets worker.closed while authority.adopted still holds that worker. Re-polling stop_authority_turns
    must signal an open worker without TypeError and treat an already-closing worker as a no-op."""
    from gateway.run_runtime import stop_authority_turns
    from gateway.session_managed_worker import ManagedWorker

    worker = ManagedWorker(SimpleNamespace())
    live = SimpleNamespace(route='r', event_stream=SimpleNamespace(execution={'execution_generation': 1}))
    authority = SimpleNamespace(
        runner=SimpleNamespace(_running_agents={}),
        sessions={'s': live},
        adopted={'s': (1, worker)},
        pending_stops={},
        _managed_workers={},
        _managed_spawning={},
    )

    assert stop_authority_turns(authority, in_process=True) == 1
    assert worker.stop.is_set() is True
    assert worker.commands.get_nowait() == {'type': 'stop'}

    closing_worker = ManagedWorker(SimpleNamespace())
    closing_worker.closed.set()
    authority.adopted['s'] = (1, closing_worker)

    assert stop_authority_turns(authority, in_process=True) == 1


def test_stop_authority_turns_isolates_failing_agent_interrupt():
    """A raising interrupt on one session must not abort interrupting remaining active sessions."""
    from gateway.run_runtime import stop_authority_turns

    class _FailingAgent:
        def hard_interrupt(self, message=None, **_kwargs):
            raise RuntimeError('boom')

    healthy = Agent()
    authority = SimpleNamespace(
        runner=SimpleNamespace(_running_agents={}),
        sessions={
            's1': SimpleNamespace(route='r1', event_stream=SimpleNamespace(execution={'execution_generation': 1})),
            's2': SimpleNamespace(route='r2', event_stream=SimpleNamespace(execution={'execution_generation': 1})),
        },
        adopted={'s1': (1, _FailingAgent()), 's2': (1, healthy)},
        pending_stops={},
        _managed_workers={},
        _managed_spawning={},
    )

    stop_authority_turns(authority, in_process=True)
    assert healthy.interrupted is True
