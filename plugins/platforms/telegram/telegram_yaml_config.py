"""Telegram config.yaml → env/``PlatformConfig.extra`` bridge and the live group-gate refresh
(split from ``plugins.platforms.telegram.adapter``)."""
from typing import Any


def _apply_yaml_config(yaml_cfg: dict, telegram_cfg: dict) -> dict | None:
    """Translate config.yaml telegram: keys into TELEGRAM_* env vars and PlatformConfig.extra. Env vars
    take precedence over YAML. Returns extras to merge into PlatformConfig.extra, or None.

    Implements the apply_yaml_config_fn contract (#24849). Mirrors the legacy telegram_cfg block from
    gateway/config.py::load_gateway_config().
    """
    import json as _json
    from gateway.platforms._shared import yaml_env_setter
    extras: dict = {}
    # Under multiplex a secondary profile's settings must NOT hit the process-global env (first-writer-wins
    # would pin them for every profile, #72348); yaml_env_setter skips the write under its scope and the
    # values flow via extra/secret scope instead.
    _set_env = yaml_env_setter()

    def _bridge_lower(key: str, env: str) -> None:
        if key in telegram_cfg:
            extras.setdefault(key, telegram_cfg[key])
            _set_env(env, str(telegram_cfg[key]).lower())

    def _bridge_gate(key: str, env: str, value: Any, *, seed_extra: bool = False) -> None:
        """CSV allowlist gate: list → comma-joined; env write skipped under multiplex secret scope."""
        if value is None:
            return
        if seed_extra:
            extras.setdefault(key, value)
        _set_env(env, value)

    for _topic_key in ("disable_topic_auto_rename", "dm_topic_auto_transfer"):
        if _topic_key in telegram_cfg:
            extras.setdefault(_topic_key, telegram_cfg[_topic_key])
    _effective_rm = telegram_cfg.get("require_mention", yaml_cfg.get("require_mention"))
    if _effective_rm is not None:
        _set_env("TELEGRAM_REQUIRE_MENTION", str(_effective_rm).lower())
    if "mention_patterns" in telegram_cfg:
        _set_env("TELEGRAM_MENTION_PATTERNS", _json.dumps(telegram_cfg["mention_patterns"]))
    for key, env in (
        ("exclusive_bot_mentions", "TELEGRAM_EXCLUSIVE_BOT_MENTIONS"), ("allow_bots", "TELEGRAM_ALLOW_BOTS"),
        ("bots_require_mention", "TELEGRAM_BOTS_REQUIRE_MENTION"),
        ("guest_mode", "TELEGRAM_GUEST_MODE", ), ("observe_unmentioned_group_messages", "TELEGRAM_OBSERVE_UNMENTIONED_GROUP_MESSAGES")):
        _bridge_lower(key, env)
    # No extras seed for allowed_chats / allowed_topics / group_allowed_chats: the shared-key loop already
    # bridges them with their original type and this merge would clobber it.
    for key, env, seed in (
        ("free_response_chats", "TELEGRAM_FREE_RESPONSE_CHATS", True), ("free_response_topics", "TELEGRAM_FREE_RESPONSE_TOPICS", False),
        ("allowed_chats", "TELEGRAM_ALLOWED_CHATS", False), ("allowed_topics", "TELEGRAM_ALLOWED_TOPICS", False),
        ("ignored_threads", "TELEGRAM_IGNORED_THREADS", True)):
        _bridge_gate(key, env, telegram_cfg.get(key), seed_extra=seed)
    _bridge_lower("reactions", "TELEGRAM_REACTIONS")
    if "proxy_url" in telegram_cfg:
        # Seeded into extra so ``_build_ptb_requests`` keeps a secondary's route without the env bridge.
        extras.setdefault("proxy_url", str(telegram_cfg["proxy_url"]).strip())
        _set_env("TELEGRAM_PROXY", str(telegram_cfg["proxy_url"]).strip())
    _telegram_extra = telegram_cfg.get("extra") if isinstance(telegram_cfg.get("extra"), dict) else {}
    _telegram_rtm = telegram_cfg["reply_to_mode"] if "reply_to_mode" in telegram_cfg else _telegram_extra.get("reply_to_mode")
    if _telegram_rtm is not None:
        _set_env("TELEGRAM_REPLY_TO_MODE", "off" if _telegram_rtm is False else str(_telegram_rtm).lower())
    _bridge_gate("allow_from", "TELEGRAM_ALLOWED_USERS", telegram_cfg.get("allow_from"))
    _bridge_gate("group_allow_from", "TELEGRAM_GROUP_ALLOWED_USERS", telegram_cfg.get("group_allow_from") or _telegram_extra.get("group_allow_from"))
    _bridge_gate(
        "group_allowed_chats", "TELEGRAM_GROUP_ALLOWED_CHATS",
        telegram_cfg.get("group_allowed_chats") or _telegram_extra.get("group_allowed_chats"))
    for _key in ("guest_mode", "disable_link_previews", "observe_unmentioned_group_messages", "free_response_topics"):
        if _key in telegram_cfg:
            extras.setdefault(_key, telegram_cfg[_key])
    # Pass through telegram-specific extra keys but EXCLUDE generic shared-config keys: _merge_platform_map
    # already applied top-level-over-nested precedence and re-emitting them via dict.update() would undo it.
    _GENERIC_MERGE_KEYS = {
        "reply_prefix", "reply_in_thread", "reply_to_mode", "unauthorized_dm_behavior", "notice_delivery",
        "require_mention", "channel_skill_bindings", "channel_prompts", "gateway_restart_notification", "allow_from",
        "allow_admin_from", "dm_policy", "group_policy"}
    for _k, _v in _telegram_extra.items():
        if _k not in _GENERIC_MERGE_KEYS:
            extras.setdefault(_k, _v)
    return extras or None


# YAML allowlist keys the group gates read, mapped to the env names _apply_yaml_config bridged.
_GROUP_GATE_REFRESH_SPEC = {
    "allowed_chats": "TELEGRAM_ALLOWED_CHATS",
    "group_allowed_chats": "TELEGRAM_GROUP_ALLOWED_CHATS",
    "allow_from": "TELEGRAM_ALLOWED_USERS",
    "group_allow_from": "TELEGRAM_GROUP_ALLOWED_USERS",
}


def refresh_telegram_group_gate(adapter) -> None:
    """Let an added-or-removed group/user allowlist entry take effect on the running adapter:
    re-reads the owning profile's ``telegram:`` keys from config.yaml when the file changed
    (#146870). Explicit env overrides keep precedence; a bad read keeps the current policy."""
    from gateway.platforms._shared import refresh_yaml_gate
    refresh_yaml_gate(
        adapter, section="telegram", spec=_GROUP_GATE_REFRESH_SPEC, attr="_group_gate_sig")
