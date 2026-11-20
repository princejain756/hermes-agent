import json
from pathlib import Path
import pytest

from tools import kanban_tools as kt
from hermes_cli import kanban as kanban_cli
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli.plugins import get_plugin_manager


@pytest.fixture
def isolated_kanban(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "test-worker")
    monkeypatch.delenv("HERMES_SESSION_ID", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    # Setup an installed local skill: 'code-review'
    local_skill = home / "skills" / "code-review"
    local_skill.mkdir(parents=True, exist_ok=True)
    (local_skill / "SKILL.md").write_text("---\nname: code-review\ndescription: local review skill\n---\n")

    # Setup an installed plugin skill: 'shared-skills:blog-poster'
    plugin_skill_dir = tmp_path / "plugins" / "shared-skills" / "blog-poster"
    plugin_skill_dir.mkdir(parents=True, exist_ok=True)
    plugin_skill_md = plugin_skill_dir / "SKILL.md"
    plugin_skill_md.write_text("---\nname: blog-poster\ndescription: shared blog poster\n---\n")

    pm = get_plugin_manager()
    pm._plugin_skills["shared-skills:blog-poster"] = {
        "plugin_key": "shared-skills",
        "bare_name": "blog-poster",
        "path": plugin_skill_md,
        "description": "shared blog poster",
    }

    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    return home


def test_kanban_create_refuses_bare_name_with_qualified_suggestion(isolated_kanban: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_mock")
    # Calling with bare name 'blog-poster' where installed skill is 'shared-skills:blog-poster'
    out = kt._handle_create({
        "title": "post blog",
        "assignee": "writer",
        "skills": ["blog-poster"],
    })
    res = json.loads(out)
    assert res.get("error") is not None
    assert "skill 'blog-poster' does not resolve" in res["error"]
    assert "did you mean 'shared-skills:blog-poster'?" in res["error"]


def test_kanban_create_refuses_typo_with_suggestion(isolated_kanban: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_mock")
    out = kt._handle_create({
        "title": "review code",
        "assignee": "reviewer",
        "skills": ["code-reveiw"],
    })
    res = json.loads(out)
    assert res.get("error") is not None
    assert "skill 'code-reveiw' does not resolve" in res["error"]
    assert "did you mean 'code-review'?" in res["error"]


def test_kanban_create_refuses_unknown_skill_without_suggestion(isolated_kanban: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_mock")
    out = kt._handle_create({
        "title": "do work",
        "assignee": "worker",
        "skills": ["completely-nonexistent-skill-xyz"],
    })
    res = json.loads(out)
    assert res.get("error") is not None
    assert "skill 'completely-nonexistent-skill-xyz' does not resolve" in res["error"]
    assert "did you mean" not in res["error"]


def test_kanban_create_accepts_valid_qualified_and_local_skills(isolated_kanban: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_mock")
    out = kt._handle_create({
        "title": "post and review",
        "assignee": "worker",
        "skills": ["shared-skills:blog-poster", "code-review"],
    })
    res = json.loads(out)
    assert res.get("error") is None
    assert res.get("task_id") is not None
    assert res.get("ok") is True


def test_kanban_cli_create_refuses_unresolvable_skill(isolated_kanban: Path):
    out = kanban_cli.run_slash("create 'cli task' --assignee worker --skill blog-poster")
    assert "skill 'blog-poster' does not resolve" in out
    assert "did you mean 'shared-skills:blog-poster'?" in out


def test_dispatcher_claim_refuses_unresolvable_skill_and_blocks_card(isolated_kanban: Path, monkeypatch: pytest.MonkeyPatch):
    import hermes_cli.profiles as profmod
    monkeypatch.setattr(profmod, "profile_exists", lambda name: True)

    spawned_tasks = []
    def fake_spawn(task, workspace):
        spawned_tasks.append(task.id)
        return 9999

    # Directly create a task with validate_skills=False to simulate a legacy card
    with kbc.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="legacy card with bad skill",
            assignee="worker",
            skills=["blog-poster"],
            validate_skills=False,
        )
        # Verify task is ready
        task = kb.get_task(conn, task_id)
        assert task.status == "ready"

        # Dispatcher runs tick
        result = kbd.dispatch_once(conn, spawn_fn=fake_spawn)

        # Worker was NOT spawned
        assert task_id not in spawned_tasks

        # Task is blocked and rejection event recorded
        task_after = kb.get_task(conn, task_id)
        assert task_after.status == "blocked"

        events = kb.list_events(conn, task_id)
        rejected = [e for e in events if e.kind == "claim_rejected"]
        assert len(rejected) >= 1
        assert rejected[-1].payload.get("reason") == "unresolvable_skills"
        error_msg = rejected[-1].payload.get("error", "")
        assert "skill 'blog-poster' does not resolve" in error_msg
        assert "did you mean 'shared-skills:blog-poster'?" in error_msg
