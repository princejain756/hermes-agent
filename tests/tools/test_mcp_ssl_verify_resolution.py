"""MCP HTTP transports must not hand a raw ``verify=True`` to httpx (#146924).

After ``hermes pm install`` rotates ``environments/<hash>``, a long-running runtime's
``certifi.where()`` still points at the deleted bundle, so ``AsyncHTTPTransport(verify=True)``
raises ``FileNotFoundError`` at construction and every MCP HTTP server wedges in
``connect_timeout``. ``_resolve_mcp_ssl_verify`` routes the configured ``ssl_verify`` through
``agent.ssl_verify.resolve_httpx_verify`` — the same repair every other httpx client gets — so
the dead path resolves to the platform store context instead.
"""

from __future__ import annotations

import ssl

import httpx
import pytest

from agent import ssl_verify
from tools.mcp_tool_transport import _resolve_mcp_ssl_verify


def test_dead_certifi_bundle_resolves_platform_context(monkeypatch):
    """The repro: certifi's frozen bundle path is gone. Raw ``verify=True`` still raises;
    the resolved verify builds the transport cleanly off the platform store."""
    def _dead():
        raise FileNotFoundError(2, "No such file or directory")

    monkeypatch.setattr(ssl_verify, "certifi_bundle_path", lambda: None)
    monkeypatch.setattr("certifi.where", _dead)
    with pytest.raises(FileNotFoundError):
        httpx.AsyncHTTPTransport(verify=True)  # the pre-fix MCP path

    resolved = _resolve_mcp_ssl_verify({}, "https://mcp.example.com/mcp")
    assert isinstance(resolved, ssl.SSLContext)
    httpx.AsyncHTTPTransport(verify=resolved)


def test_healthy_certifi_keeps_verify_true(monkeypatch):
    """No rotation damage → the resolution is the same ``True`` the config carried."""
    monkeypatch.setattr(ssl_verify, "certifi_bundle_path", lambda: "/live/cacert.pem")
    assert _resolve_mcp_ssl_verify({"ssl_verify": True}, "https://mcp.example.com/mcp") is True


def test_ssl_verify_false_keeps_insecure_opt_out(monkeypatch):
    monkeypatch.setattr(ssl_verify, "certifi_bundle_path", lambda: "/live/cacert.pem")
    assert _resolve_mcp_ssl_verify({"ssl_verify": False}, "http://mcp.local/mcp") is False


def test_ssl_verify_path_passes_through_verbatim(tmp_path):
    """An explicit CA bundle path is httpx's own input — it never touches certifi."""
    ca = tmp_path / "ca.pem"
    ca.write_text("dummy")
    assert _resolve_mcp_ssl_verify({"ssl_verify": str(ca)}, "https://mcp.example.com/mcp") == str(ca)
