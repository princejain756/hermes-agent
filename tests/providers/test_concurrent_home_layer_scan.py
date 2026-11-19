"""Concurrent first lookups for a profile home keep its ``$HERMES_HOME`` provider plugin
registered and leave ``providers._discovering`` false (#157976).
"""

from __future__ import annotations

import contextvars
import sys
import threading
from pathlib import Path

import pytest

from hermes_constants import reset_hermes_home_override, set_hermes_home_override


def _install_plugin(home: Path, name: str, *, extra_sleep: float = 0.02, reenter: bool = False) -> None:
    plugin = home / "plugins" / name
    plugin.mkdir(parents=True, exist_ok=True)
    (plugin / "plugin.yaml").write_text(f"name: {name}\nkind: model-provider\nversion: 0.1.0\n", encoding="utf-8")
    reenter_line = "import providers; providers.list_providers()\n" if reenter else ""
    (plugin / "__init__.py").write_text(
        "import time\n"
        "from providers import register_provider\n"
        "from providers.base import ProviderProfile\n"
        f"{reenter_line}"
        f"time.sleep({extra_sleep})\n"
        f"register_provider(ProviderProfile(name={name!r}, aliases=('{name}-alias',), "
        f"base_url='process://{name}', auth_type='external_process'))\n",
        encoding="utf-8",
    )


@pytest.fixture
def isolated_layers(monkeypatch):
    import providers

    monkeypatch.setattr(providers, "_REGISTRY", dict(providers._REGISTRY))
    monkeypatch.setattr(providers, "_ALIASES", dict(providers._ALIASES))
    monkeypatch.setattr(providers, "_PROVIDER_LIST_CACHE", None)
    monkeypatch.setattr(providers, "_HOME_LAYERS", {}, raising=False)
    yield
    for mod in [m for m in sys.modules if m.startswith("_hermes_user_provider")]:
        del sys.modules[mod]


def test_concurrent_first_lookups_keep_provider_registered_and_clear_discovering(tmp_path, isolated_layers):
    import providers
    from hermes_cli.auth import resolve_provider

    home = tmp_path / "profile"
    _install_plugin(home, "demo-race")
    providers._discover_providers()

    n_threads = 6
    barrier = threading.Barrier(n_threads)
    resolved: list[str] = []
    errors: list[BaseException] = []
    lock = threading.Lock()

    def worker() -> None:
        token = set_hermes_home_override(home)
        try:
            barrier.wait(timeout=10)
            out = resolve_provider("demo-race")
            with lock:
                resolved.append(out)
        except BaseException as exc:  # pragma: no cover - surfaced in assertion
            with lock:
                errors.append(exc)
        finally:
            reset_hermes_home_override(token)

    threads = [threading.Thread(target=contextvars.copy_context().run, args=(worker,)) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=15)

    assert not errors
    assert resolved == ["demo-race"] * n_threads
    assert providers._discovering is False

    token = set_hermes_home_override(home)
    try:
        layer = providers._bound_home_layer()[0]
        assert "demo-race" in layer.registry
        assert layer.aliases.get("demo-race-alias") == "demo-race"
        assert providers.get_provider_profile("demo-race-alias") is layer.registry["demo-race"]
    finally:
        reset_hermes_home_override(token)


def test_skipped_preexisting_module_does_not_wipe_layer_on_empty_staging(tmp_path, isolated_layers):
    import providers
    from providers.plugin_reload import import_into_layer, tree_fingerprint

    home = tmp_path / "profile-preexisting"
    _install_plugin(home, "demo-preexisting", extra_sleep=0.0)
    plugin_dir = home / "plugins" / "demo-preexisting"
    providers._discover_providers()

    token = set_hermes_home_override(home)
    try:
        assert providers.get_provider_profile("demo-preexisting") is not None
        layer, _home, key = providers._bound_home_layer()
        base_name = providers._user_module_name(plugin_dir, key)
        fp = tree_fingerprint(plugin_dir)
        mod_name = f"{base_name}_{fp}"
        assert mod_name in sys.modules
        # Simulate a layer rescan where trees was cleared while sys.modules already holds mod_name.
        layer.trees.pop(base_name, None)
        reg_token = providers._REGISTRATION_TARGET.set(layer)
        try:
            import_into_layer(plugin_dir, "user", base_name)
        finally:
            providers._REGISTRATION_TARGET.reset(reg_token)
        assert "demo-preexisting" in layer.registry
    finally:
        reset_hermes_home_override(token)


def test_reentrant_list_providers_inside_plugin_import_does_not_deadlock(tmp_path, isolated_layers):
    import providers

    home = tmp_path / "profile-reenter"
    _install_plugin(home, "demo-reenter", extra_sleep=0.0, reenter=True)
    providers._discover_providers()

    token = set_hermes_home_override(home)
    try:
        assert providers.get_provider_profile("demo-reenter") is not None
        assert providers._discovering is False
    finally:
        reset_hermes_home_override(token)
