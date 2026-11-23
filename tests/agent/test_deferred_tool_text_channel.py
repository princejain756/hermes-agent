"""Regressions for #159166: deferred tools emitted in text-channel markup route or get corrective retries."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from persistence_assembly import open_persistence, profile_scope
from run_agent import AIAgent

_DEAD_LOCAL = "http://127.0.0.1:9"


def _response(content="", finish_reason="stop", tool_calls=None):
    choice = SimpleNamespace(
        message=SimpleNamespace(content=content, tool_calls=tool_calls),
        finish_reason=finish_reason,
        index=0,
    )
    return SimpleNamespace(id="chatcmpl-test", choices=[choice], model="test/model", usage=None)


def _call(name, args, call_id):
    return SimpleNamespace(id=call_id, type="function", function=SimpleNamespace(name=name, arguments=json.dumps(args)))


@pytest.fixture
def terminal_loop(tmp_path, monkeypatch):
    """Real AIAgent with terminal + deferred process_manage behind tool_call bridge; provider is scripted."""
    for var in ("HTTPS_PROXY", "HTTP_PROXY", "https_proxy", "http_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.setenv(var, _DEAD_LOCAL)
    monkeypatch.setenv("NO_PROXY", "")
    monkeypatch.setenv("no_proxy", "")
    monkeypatch.setattr("agent.title_generator.maybe_auto_title", lambda *a, **k: None)
    monkeypatch.setattr("agent.title_generator.start_title_upgrade", lambda *a, **k: None)
    monkeypatch.chdir(tmp_path)
    db = open_persistence(profile_scope(home=tmp_path))
    with patch("agent.process_bootstrap.OpenAI"), patch("agent.model_metadata.fetch_model_metadata", return_value={}):
        agent = AIAgent(
            api_key="test-key",
            base_url=f"{_DEAD_LOCAL}/v1",
            model="test/model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            enabled_toolsets=["terminal"],
            persistence=db,
            session_id="sess-deferred-text",
        )
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent.save_trajectories = False
    try:
        yield agent
    finally:
        db.close()


@pytest.mark.parametrize(
    ("text_payload", "expected_args"),
    [
        (
            "<think>Need to check the background command.</think>\n"
            "<tool_call>process_manage"
            "<arg_key>action</arg_key><arg_value>wait</arg_value>"
            "<arg_key>session_id</arg_key><arg_value>proc_abc</arg_value>"
            "<arg_key>timeout</arg_key><arg_value>30</arg_value>"
            "</tool_call>",
            {"action": "wait", "session_id": "proc_abc", "timeout": 30},
        ),
        (
            '<tool_call>\n{"name": "process_manage", "arguments": {"action": "poll", "session_id": "proc_def"}}\n</tool_call>',
            {"action": "poll", "session_id": "proc_def"},
        ),
        (
            "<tool_call>process<arg_key>action</arg_key><arg_value>poll</arg_value>"
            "<arg_key>session_id</arg_key><arg_value>proc_legacy</arg_value></tool_call>",
            {"action": "poll", "session_id": "proc_legacy"},
        ),
    ],
    ids=["glm-xml", "json-xml", "legacy-alias-xml"],
)
def test_text_channel_deferred_tool_call_routes_through_bridge(terminal_loop, monkeypatch, text_payload, expected_args):
    agent = terminal_loop
    assert "tool_call" in agent.valid_tool_names
    assert "process_manage" not in agent.valid_tool_names

    dispatched = []

    def fake_handle(fn_name, fn_args, _task_id=None, **_kwargs):
        dispatched.append((fn_name, fn_args))
        if fn_name == "terminal":
            return json.dumps({"status": "running", "session_id": expected_args["session_id"]})
        if fn_name == "process_manage":
            return json.dumps({"status": "exited", "exit_code": 0})
        raise AssertionError(f"Unexpected tool call: {fn_name}")

    monkeypatch.setattr("model_tools.handle_function_call", fake_handle)
    script = [
        _response(finish_reason="tool_calls", tool_calls=[_call("terminal", {"command": "sleep 1", "background": True}, "call_t1")]),
        _response(content=text_payload, finish_reason="stop", tool_calls=None),
        _response(content="Background command finished cleanly.", finish_reason="stop"),
    ]
    agent.client = MagicMock()
    agent.client.chat.completions.create.side_effect = lambda **_kw: script.pop(0)

    result = agent.run_conversation("Start sleep and poll it.")

    assert result["completed"] is True
    assert result["final_response"] == "Background command finished cleanly."
    assert result.get("turn_exit_reason") != "empty_response_exhausted"
    assert dispatched == [
        ("terminal", {"command": "sleep 1", "background": True}),
        ("process_manage", expected_args),
    ]
    bridge_rows = [
        tc
        for m in result["messages"]
        if m.get("role") == "assistant"
        for tc in (m.get("tool_calls") or [])
        if tc.get("function", {}).get("name") == "tool_call"
    ]
    assert len(bridge_rows) == 1
    assert json.loads(bridge_rows[0]["function"]["arguments"]) == {
        "calls": [{"name": "process_manage", "arguments": expected_args}]
    }


def test_text_channel_parser_ignores_offered_tools_thinking_and_code_fences(terminal_loop):
    from agent.turn_response_intake import normalize_model_response

    agent = terminal_loop
    samples = [
        # Already-offered tool (not deferred) must stay text.
        '<tool_call>terminal<arg_key>command</arg_key><arg_value>date</arg_value></tool_call>',
        # Unknown tool not in scoped deferred set must stay text.
        '<tool_call>not_a_real_tool<arg_key>x</arg_key><arg_value>1</arg_value></tool_call>',
        # Fenced markdown example must stay text.
        '```xml\n<tool_call>process_manage<arg_key>action</arg_key><arg_value>list</arg_value></tool_call>\n```',
        # Markup inside <think> reasoning block must not execute as a call.
        '<think><tool_call>process_manage<arg_key>action</arg_key><arg_value>list</arg_value></tool_call></think>\nDone.',
        # Quoted markup in prose must stay inert so discussing logs or docs cannot execute calls.
        'The server log printed <tool_call>process_manage<arg_key>action</arg_key><arg_value>kill</arg_value><arg_key>session_id</arg_key><arg_value>proc_1</arg_value></tool_call> before failing.',
        # Quoted markup inside inline code must stay inert.
        'Example payload: `<tool_call>process_manage<arg_key>action</arg_key><arg_value>kill</arg_value><arg_key>session_id</arg_key><arg_value>proc_1</arg_value></tool_call>`',
        # Reasoning + prose before a quoted call must also stay inert.
        '<think>Checking example.</think>\nQuoted line:\n<tool_call>process_manage<arg_key>action</arg_key><arg_value>kill</arg_value></tool_call>',
        # Fenced markdown before an unfenced call is non-whitespace reply text, so keep it inert.
        '```xml\n<tool_call>process_manage<arg_key>action</arg_key><arg_value>list</arg_value></tool_call>\n```\n<tool_call>process_manage<arg_key>action</arg_key><arg_value>kill</arg_value></tool_call>',
    ]
    for sample in samples:
        verdict = normalize_model_response(
            agent, response=_response(content=sample), messages=[], api_messages=[], conversation_history=[],
            api_call_count=1, api_duration=0.0, api_start_time=0.0, api_request_id="r1",
            effective_task_id="t1", turn_id="t1",
        )
        assert not verdict.assistant_message.tool_calls, sample
        assert verdict.assistant_message.content == sample


def test_truncated_text_deferred_markup_gets_corrective_retry_hint(terminal_loop, monkeypatch):
    agent = terminal_loop
    monkeypatch.setattr("agent.turn_empty_response.interruptible_backoff_sleep", lambda *_a, **_k: None)

    seen_last_messages = []
    dispatched = []

    def fake_handle(fn_name, fn_args, _task_id=None, **_kwargs):
        dispatched.append((fn_name, fn_args))
        if fn_name == "terminal":
            return json.dumps({"status": "running", "session_id": "proc_cut"})
        if fn_name == "process_manage":
            return json.dumps({"status": "exited"})
        raise AssertionError(f"Unexpected tool: {fn_name}")

    monkeypatch.setattr("model_tools.handle_function_call", fake_handle)
    script = [
        _response(finish_reason="tool_calls", tool_calls=[_call("terminal", {"command": "sleep 1", "background": True}, "call_t1")]),
        # Unclosed/truncated XML block naming process_manage: stripped to empty by strip_think_blocks.
        _response(content="<tool_call>process_manage<arg_key>action</arg_key><arg_value>poll", finish_reason="stop"),
        # Model recovers cleanly once hinted to use structured tool_call.
        _response(
            finish_reason="tool_calls",
            tool_calls=[_call("tool_call", {"calls": [{"name": "process_manage", "arguments": {"action": "poll", "session_id": "proc_cut"}}]}, "call_bridge")],
        ),
        _response(content="Recovered cleanly.", finish_reason="stop"),
    ]

    def fake_create(**kwargs):
        seen_last_messages.append(kwargs["messages"][-1]["content"])
        return script.pop(0)

    agent.client = MagicMock()
    agent.client.chat.completions.create.side_effect = fake_create

    result = agent.run_conversation("Start sleep and poll it.")

    assert result["completed"] is True
    assert result["final_response"] == "Recovered cleanly."
    # Call index 2 (after the truncated markup attempt) must include the corrective hint naming tool_call and process_manage.
    assert len(seen_last_messages) == 4
    assert "tool_call" in seen_last_messages[2]
    assert "process_manage" in seen_last_messages[2]
