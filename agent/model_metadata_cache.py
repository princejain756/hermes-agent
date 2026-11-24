"""``context_length_cache.yaml`` writes and their cross-process serialization.

Every agent (CLI, gateway sessions, cron jobs, subagents) persists probed context windows with a
load -> mutate -> write cycle; unserialized, one writer rewrites a stale snapshot and drops the
other's entry (or resurrects an invalidated, too-large length).
"""

import contextlib
import functools
import logging
import threading

from utils import atomic_json_write, atomic_yaml_write

logger = logging.getLogger(__name__)

_CONTEXT_CACHE_LOCK_HOLDER = threading.local()
_CONTEXT_CACHE_LOCK_TIMEOUT_SECONDS = 5.0


def _write_context_cache(cache: dict[str, int], bedrock_confirmed: dict | None = None) -> None:
    """Atomic write (a truncating write killed mid-dump leaves a partial file that
    _load_context_cache() swallows as {}, wiping EVERY cached length). Raises on failure."""
    from agent.model_metadata import _get_context_cache_path  # facade seam: tests patch the path there
    document = {"context_lengths": cache}
    if bedrock_confirmed:
        document["bedrock_confirmed_v1"] = bedrock_confirmed
    atomic_yaml_write(_get_context_cache_path(), document)


def locked_context_cache(fn):
    """Run ``fn``'s load -> mutate -> write cycle under the cache file's cross-process lock.

    The cache is best-effort: when the lock cannot be taken in time the update is skipped, never raised.
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        from agent.model_metadata import _get_context_cache_path  # facade seam: tests patch the path there
        from hermes_cli.auth import _file_lock
        path = _get_context_cache_path()
        lock_path = path.with_name(path.name + ".lock")
        with contextlib.ExitStack() as stack:
            try:
                stack.enter_context(_file_lock(
                    lock_path, _CONTEXT_CACHE_LOCK_HOLDER, _CONTEXT_CACHE_LOCK_TIMEOUT_SECONDS,
                    f"Timed out waiting for the context length cache lock ({lock_path})"))
            except (OSError, TimeoutError):
                logger.debug("Skipping context length cache update", exc_info=True)
                return None
            return fn(*args, **kwargs)
    return wrapper


@locked_context_cache
def drop_context_cache_rows(stale_keys: set[str], label: str) -> None:
    """Remove every key shape of one cached entry (and its Bedrock provenance) under the cache lock.

    Only the file update is locked: callers evict their in-process probe caches unconditionally."""
    from agent.model_metadata import _load_context_cache_document  # facade seam: tests patch it there
    document = _load_context_cache_document()
    cache = document.get("context_lengths") or {}
    confirmed = document.get("bedrock_confirmed_v1") or {}
    confirmed = confirmed if isinstance(confirmed, dict) else {}
    if not any(k in cache or k in confirmed for k in stale_keys):
        return
    for k in stale_keys:
        cache.pop(k, None)
        confirmed.pop(k, None)
    try:
        _write_context_cache(cache, confirmed)
    except Exception:
        logger.debug("Failed to invalidate context length cache entry %s", label, exc_info=True)


def _drop_disk_memo_keys(path, predicate, what: str) -> None:
    """Remove matching keys from a JSON disk memo file. Best-effort."""
    if not path.exists():
        return
    try:
        from agent.model_metadata import _load_json_dict

        data = _load_json_dict(path)
        kept = {k: v for k, v in data.items() if not predicate(k)}
        if len(kept) != len(data):
            atomic_json_write(path, kept, indent=0, separators=(",", ":"))
    except (OSError, TypeError, ValueError) as exc:
        logger.debug("Failed to drop %s: %s", what, exc)


def invalidate_context_length_caches(model: str, base_url: str) -> None:
    """Evict in-memory and disk context-length caches for ``model`` at ``base_url``."""
    from agent import model_metadata as mm
    from agent.model_metadata_memo_bounds import MEMO_LOCK
    from hermes_constants import hermes_home_key

    key = mm._context_cache_key(model, base_url)
    bare = mm._strip_provider_prefix(model)
    raw_stripped = (base_url or "").rstrip("/")
    stripped = mm._normalize_base_url(base_url)
    for url_key in {raw_stripped, stripped}:
        mm._LOCAL_CTX_PROBE_CACHE.pop((bare, url_key), None)
        mm._LOCAL_CTX_PROBE_CACHE.pop(("ollama_show", bare, url_key), None)
    if stripped:
        with MEMO_LOCK:
            for memo_key in [k for k in mm._endpoint_model_metadata_cache if k[0] == stripped]:
                mm._endpoint_model_metadata_cache.pop(memo_key, None)
                mm._endpoint_model_metadata_cache_time.pop(memo_key, None)
    for memo_key in list(mm._BEDROCK_PROBE_FAILURE_CACHE):
        if memo_key[:2] == (hermes_home_key(), raw_stripped) and memo_key[2] in (model, bare):
            mm._BEDROCK_PROBE_FAILURE_CACHE.pop(memo_key, None)
    if stripped:
        _drop_disk_memo_keys(
            mm._get_endpoint_metadata_cache_path(),
            lambda k: k == stripped or k.startswith(f"{stripped} ["),
            "endpoint model metadata disk cache",
        )
    if stripped and bare:
        target = f"ollama_num_ctx:{mm._server_root(stripped)}|{bare}"
        _drop_disk_memo_keys(
            mm._local_probe_disk_cache_path(),
            lambda k: k == target,
            "local probe disk cache",
        )
    stale_keys = {key, f"{model}@{base_url}", f"{key}/", f"{model}@{stripped}"}
    if bare != model:
        bare_key = mm._context_cache_key(bare, stripped)
        stale_keys.update({bare_key, f"{bare}@{base_url}", f"{bare_key}/"})
    drop_context_cache_rows(stale_keys, key)


def invalidate_switch_routes(routes) -> None:
    """Clear memoized context windows for non-Bedrock ``(model, base_url, provider)`` routes."""
    try:
        from agent.model_metadata import _invalidate_cached_context_length, _is_bedrock_context

        seen: set[tuple[str, str]] = set()
        for model, base_url, provider in routes:
            model_name = (model or "").strip()
            norm_url = (base_url or "").strip().rstrip("/")
            if not model_name or (model_name, norm_url) in seen:
                continue
            seen.add((model_name, norm_url))
            if _is_bedrock_context(norm_url, provider or ""):
                continue
            _invalidate_cached_context_length(model_name, norm_url)
    except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
        logger.debug("Failed to invalidate switched context cache: %s", exc)
