"""Regression tests for context-length cache invalidation on /model switch (#159799)."""

from __future__ import annotations

import contextlib
from unittest.mock import MagicMock, patch

import httpx
import pytest

from agent import bedrock_adapter as ba
from agent import model_metadata as mm
from agent.agent_init import _enforce_minimum_context
from agent.context_compressor import ContextCompressor
from hermes_cli.model_switch import switch_model
from run_agent import AIAgent


@pytest.fixture(autouse=True)
def _isolated_context_caches(tmp_path, monkeypatch):
    """Isolate HERMES_HOME and reset in-process context caches around each test."""
    home = tmp_path / "hermes"
    home.mkdir()
    (home / "config.yaml").write_text("model: my-local-model\n", encoding="utf-8")
    (home / ".env").write_text("", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    mm._LOCAL_CTX_PROBE_CACHE.clear()
    mm._endpoint_model_metadata_cache.clear()
    mm._endpoint_model_metadata_cache_time.clear()
    mm._endpoint_probe_path_cache.clear()
    mm._BEDROCK_PROBE_FAILURE_CACHE.clear()
    yield home
    mm._LOCAL_CTX_PROBE_CACHE.clear()
    mm._endpoint_model_metadata_cache.clear()
    mm._endpoint_model_metadata_cache_time.clear()
    mm._endpoint_probe_path_cache.clear()
    mm._BEDROCK_PROBE_FAILURE_CACHE.clear()


def _install_local_models_mock(monkeypatch, window_ref: dict[str, int]):
    """Stub HTTP transport so /models returns window_ref['value'] for my-local-model."""
    def _stream(url, **_kwargs):
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status = MagicMock()
        resp.read = MagicMock()
        resp.json.return_value = {
            "data": [{"id": "my-local-model", "max_model_len": window_ref["value"]}],
        }
        return contextlib.nullcontext(resp)

    monkeypatch.setattr(mm.model_metadata_http, "stream", _stream)
    monkeypatch.setattr(mm, "detect_local_server_type", lambda *a, **k: None)


def _make_local_agent(base_url: str, window: int) -> AIAgent:
    agent = AIAgent.__new__(AIAgent)
    agent.model = "my-local-model"
    agent.provider = "custom"
    agent.base_url = base_url
    agent.api_key = "local-test"
    agent.api_mode = "chat_completions"
    agent.client = MagicMock()
    agent.quiet_mode = True
    agent._config_context_length = None
    agent.context_compressor = ContextCompressor(
        model="my-local-model",
        threshold_percent=0.50,
        base_url=base_url,
        api_key="local-test",
        provider="custom",
        quiet_mode=True,
        config_context_length=window,
    )
    agent._primary_runtime = {}
    return agent


def test_switch_model_invalidates_endpoint_metadata_memo(monkeypatch):
    """Reselecting a local model after server restart must not keep the stale 300s /models memo."""
    base_url = "http://127.0.0.1:8080/v1"
    window = {"value": 32_768}
    _install_local_models_mock(monkeypatch, window)

    assert mm.get_model_context_length("my-local-model", base_url=base_url, provider="custom") == 32_768

    window["value"] = 65_536
    with (
        patch("hermes_cli.models_validate.validate_requested_model", return_value={"accepted": True, "persist": True, "recognized": True, "message": ""}),
        patch("hermes_cli.model_switch.get_model_capabilities", return_value=None),
        patch("hermes_cli.model_switch.get_model_info", return_value=None),
        patch("hermes_cli.model_picker_usage.large_model_notice", return_value=""),
    ):
        res = switch_model(
            "my-local-model",
            current_provider="custom",
            current_model="my-local-model",
            current_base_url=base_url,
            current_api_key="local-test",
        )
    assert res.success is True
    assert mm.get_model_context_length("my-local-model", base_url=base_url, provider="custom") == 65_536


def test_switch_model_invalidates_local_ctx_probe_cache(monkeypatch):
    """Reselecting a local model clears _LOCAL_CTX_PROBE_CACHE so the 30s probe memo re-probes."""
    base_url = "http://127.0.0.1:8080/v1"
    window = {"value": 32_768}

    def _empty_stream(url, **_kwargs):
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status = MagicMock()
        resp.read = MagicMock()
        resp.json.return_value = {"data": []}
        return contextlib.nullcontext(resp)

    real_client = httpx.Client

    def _mock_client(**_kw):
        def _handle(_request):
            return httpx.Response(200, json={"data": [{"id": "my-local-model", "max_model_len": window["value"]}]})
        return real_client(transport=httpx.MockTransport(_handle))

    monkeypatch.setattr(mm.model_metadata_http, "stream", _empty_stream)
    monkeypatch.setattr(mm, "detect_local_server_type", lambda *a, **k: None)
    monkeypatch.setattr(httpx, "Client", _mock_client)

    assert mm.get_model_context_length("my-local-model", base_url=base_url, provider="custom") == 32_768
    assert ("my-local-model", base_url) in mm._LOCAL_CTX_PROBE_CACHE

    window["value"] = 65_536
    with (
        patch("hermes_cli.models_validate.validate_requested_model", return_value={"accepted": True, "persist": True, "recognized": True, "message": ""}),
        patch("hermes_cli.model_switch.get_model_capabilities", return_value=None),
        patch("hermes_cli.model_switch.get_model_info", return_value=None),
        patch("hermes_cli.model_picker_usage.large_model_notice", return_value=""),
    ):
        res = switch_model(
            "my-local-model",
            current_provider="custom",
            current_model="my-local-model",
            current_base_url=base_url,
            current_api_key="local-test",
        )
    assert res.success is True
    assert mm.get_model_context_length("my-local-model", base_url=base_url, provider="custom") == 65_536


def test_switch_model_invalidates_persisted_context_and_disk_memos(monkeypatch):
    """Switching drops stale context_length_cache.yaml, endpoint disk memo, and ollama_num_ctx disk memo."""
    base_url = "http://127.0.0.1:8080/v1"
    mm.save_context_length("my-local-model", base_url, 131_072)
    mm._endpoint_disk_cache_put(base_url, {"my-local-model": {"id": "my-local-model", "context_length": 131_072}})
    mm._local_probe_disk_put("ollama_num_ctx", "http://127.0.0.1:8080|my-local-model", 131_072)

    with (
        patch("hermes_cli.models_validate.validate_requested_model", return_value={"accepted": True, "persist": True, "recognized": True, "message": ""}),
        patch("hermes_cli.model_switch.get_model_capabilities", return_value=None),
        patch("hermes_cli.model_switch.get_model_info", return_value=None),
        patch("hermes_cli.model_picker_usage.large_model_notice", return_value=""),
    ):
        res = switch_model(
            "my-local-model",
            current_provider="custom",
            current_model="my-local-model",
            current_base_url=base_url,
            current_api_key="local-test",
        )

    assert res.success is True
    assert mm.get_cached_context_length("my-local-model", base_url) is None
    assert mm._endpoint_disk_cache_get(base_url) is None
    assert mm._local_probe_disk_get("ollama_num_ctx", "http://127.0.0.1:8080|my-local-model") is None


def test_failed_switch_keeps_existing_cached_context_length():
    """A rejected model switch must not evict the active model's cached context length."""
    base_url = "http://127.0.0.1:8080/v1"
    mm.save_context_length("my-local-model", base_url, 131_072)
    memo_key = mm._endpoint_memo_key(base_url, "")
    mm._endpoint_model_metadata_cache[memo_key] = {"my-local-model": {"context_length": 131_072}}
    mm._endpoint_model_metadata_cache_time[memo_key] = 1e12

    with patch(
        "hermes_cli.models_validate.validate_requested_model",
        return_value={"accepted": False, "persist": False, "recognized": False, "message": "not found"},
    ):
        res = switch_model(
            "missing-model",
            current_provider="custom",
            current_model="my-local-model",
            current_base_url=base_url,
            current_api_key="local-test",
        )

    assert res.success is False
    assert mm.get_cached_context_length("my-local-model", base_url) == 131_072
    assert memo_key in mm._endpoint_model_metadata_cache


def test_live_agent_switch_model_clears_stale_probe_memos(monkeypatch):
    """Direct agent.switch_model() must invalidate stale endpoint/probe memos before sizing compressor."""
    base_url = "http://127.0.0.1:8080/v1"
    window = {"value": 32_768}
    _install_local_models_mock(monkeypatch, window)

    assert mm.get_model_context_length(
        "my-local-model", base_url=base_url, api_key="local-test", provider="custom",
    ) == 32_768
    agent = _make_local_agent(base_url, 32_768)
    agent._create_openai_client = lambda *_args, **_kwargs: MagicMock()

    window["value"] = 65_536
    agent.switch_model("my-local-model", "custom", api_key="local-test", base_url=base_url)
    assert agent.context_compressor.context_length == 65_536
    agent._ollama_num_ctx = None
    _enforce_minimum_context(agent)


def test_invalidation_clears_endpoint_memo_when_file_lock_times_out(monkeypatch):
    """In-memory _endpoint_model_metadata_cache eviction must run even if the file lock times out."""
    import hermes_cli.auth as auth_module

    base_url = "https://a.example/v1"
    memo_key = mm._endpoint_memo_key(base_url, "k")
    mm._endpoint_model_metadata_cache[memo_key] = {"model-a": {"context_length": 32_768}}
    mm._endpoint_model_metadata_cache_time[memo_key] = 1e12

    @contextlib.contextmanager
    def _wedged_lock(*_args, **_kwargs):
        raise TimeoutError("Timed out waiting for lock")
        yield

    monkeypatch.setattr(auth_module, "_file_lock", _wedged_lock)
    mm._invalidate_cached_context_length("model-a", base_url)

    assert memo_key not in mm._endpoint_model_metadata_cache
    assert memo_key not in mm._endpoint_model_metadata_cache_time


def test_switch_does_not_drop_confirmed_bedrock_context(monkeypatch):
    """Switching involving Bedrock must preserve bedrock_confirmed_v1 cache entries."""
    monkeypatch.setattr(ba, "resolve_bedrock_region", lambda: "us-east-1")
    model = "global.xai.grok-4.6"
    mm.save_provider_context_length(model, "", 500_000, provider="bedrock")
    assert mm.get_cached_context_length(model, "bedrock://", bedrock_confirmed=True) == 500_000

    with (
        patch("hermes_cli.models_validate.validate_requested_model", return_value={"accepted": True, "persist": True, "recognized": True, "message": ""}),
        patch("hermes_cli.model_switch.get_model_capabilities", return_value=None),
        patch("hermes_cli.model_switch.get_model_info", return_value=None),
        patch("hermes_cli.model_picker_usage.large_model_notice", return_value=""),
        patch("hermes_cli.model_switch._Switch.resolve_runtime", lambda self, **_kw: None),
    ):
        res = switch_model(
            model,
            current_provider="bedrock",
            current_model=model,
        )
    assert res.success is True
    assert mm.get_cached_context_length(model, "bedrock://", bedrock_confirmed=True) == 500_000
