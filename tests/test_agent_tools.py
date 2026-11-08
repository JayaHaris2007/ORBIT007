"""Tests for the native agent tool registry, validation, and executors."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from agent.confirm import ConfirmationManager
from agent.deps import AgentDeps
from agent.tools import (
    TOOLS,
    ToolContext,
    resolve_entity,
    validate_and_execute_tool,
)
from memory import MemoryStore


@pytest.fixture
def temp_store(tmp_path: Path):
    db_file = tmp_path / "test_agent_tools.db"
    store = MemoryStore(db_file)
    store.initialize()
    return store


@pytest.fixture
def tool_context(temp_store):
    deps = MagicMock(spec=AgentDeps)
    deps.memory_store = temp_store
    deps.local_timezone = timezone.utc

    with temp_store.connect() as conn:
        now_iso = datetime.now(timezone.utc).isoformat()
        conn.execute("INSERT OR IGNORE INTO conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                     ("test-chat", "Test Chat", now_iso, now_iso))
        conn.execute("INSERT OR IGNORE INTO conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                     ("default", "Default Chat", now_iso, now_iso))

    # Hook real memory_store into reminder and learning hooks
    def _create_rem(title, when, recurrence=None, chat_id=None):
        if "garbage" in when.lower() or "invalid" in when.lower():
            raise ValueError(f"Invalid time: {when}")
        run_at = "2026-11-02T10:00:00Z"
        row = temp_store.create_reminder(title, run_at, recurrence=recurrence, chat_id=chat_id or "default")
        return {"id": row["id"], "title": row["text"], "remind_at": row["run_at"], "recurrence": row.get("recurrence")}

    deps.create_reminder_unified = _create_rem
    deps.cancel_reminder_unified = lambda r_id: temp_store.cancel_reminder(r_id)
    deps.parse_reminder_time = lambda s: datetime.now(timezone.utc) if "garbage" not in s.lower() else None
    deps.live_reminders = lambda: [
        {"id": r["id"], "title": r["text"], "remind_at": r["run_at"], "recurrence": r.get("recurrence")}
        for r in temp_store.list_reminders()
    ]
    deps.learning_schema_initialized = MagicMock(return_value=False)
    deps.learning_overall_progress = MagicMock(return_value={"completed_topics": 2, "topic_count": 5, "completion_percent": 40})
    deps.learning_next_topics = MagicMock(return_value=[{"name": "Asyncio"}])
    deps.run_learning_hook = lambda fn: fn()
    deps.suggest_learning_for_task = MagicMock()
    deps.web_search = MagicMock(return_value=[
        {"title": "Python Docs", "domain": "python.org", "content": "Official Python documentation", "url": "https://python.org"}
    ])

    return ToolContext(
        deps=deps,
        chat_id="test-chat",
        channel="web",
        confirm_mgr=ConfirmationManager(),
        ui_actions=[],
        has_called_web_search=False,
    )


# --- Schema and Validation Tests ---

def test_tool_count_and_descriptions():
    assert len(TOOLS) == 18
    for t in TOOLS:
        words = t.description.split()
        assert len(words) <= 40, f"Description for {t.name} exceeds 40 words ({len(words)})"
        schema = t.parameters
        assert schema.get("type") == "object"
        assert "properties" in schema


def test_validation_unknown_tool(tool_context):
    res = validate_and_execute_tool("fake_nonexistent_tool", {}, tool_context)
    assert res["status"] == "error"
    assert "Unknown tool" in res["message"]


def test_validation_missing_required_key(tool_context):
    res = validate_and_execute_tool("create_task", {}, tool_context)
    assert res["status"] == "error"
    assert "Missing required parameter 'title'" in res["message"]


def test_validation_wrong_type(tool_context):
    res = validate_and_execute_tool("create_task", {"title": 12345}, tool_context)
    # Int converted/handled or accepted as string
    assert res["status"] == "ok"

    res_bad = validate_and_execute_tool("update_task", {"task": "1", "completed": "not_a_bool"}, tool_context)
    # In our validator, strings not in (true, 1, yes) become false, but invalid type check
    assert res_bad["status"] in ("ok", "not_found")


def test_validation_bad_enum(tool_context):
    res = validate_and_execute_tool("create_task", {"title": "Task 1", "priority": "ultra-mega"}, tool_context)
    assert res["status"] == "error"
    assert "priority" in res["message"]


def test_validation_strips_unknown_keys(tool_context):
    res = validate_and_execute_tool("create_task", {"title": "Task Clean", "malicious_injected_key": "drop database"}, tool_context)
    assert res["status"] == "ok"
    assert res["task"]["title"] == "Task Clean"


# --- Task Tools ---

def test_task_tools_lifecycle(tool_context):
    # 1. Create task
    c_res = validate_and_execute_tool("create_task", {"title": "Write unit tests", "priority": "high"}, tool_context)
    assert c_res["status"] == "ok"
    task_id = c_res["task"]["id"]
    assert any(a.get("type") == "refresh" and "tasks" in a.get("sections", []) for a in tool_context.ui_actions)

    # 2. List tasks
    l_res = validate_and_execute_tool("list_tasks", {"status": "pending"}, tool_context)
    assert l_res["status"] == "ok"
    assert any(t["id"] == task_id for t in l_res["tasks"])

    # 3. Update task
    u_res = validate_and_execute_tool("update_task", {"task": str(task_id), "completed": True}, tool_context)
    assert u_res["status"] == "ok"
    assert u_res["task"]["completed"] is True

    # 4. Delete task requires confirmation
    d_res = validate_and_execute_tool("delete_task", {"task": str(task_id)}, tool_context)
    assert d_res["status"] == "needs_confirmation"
    assert d_res["tool"] == "delete_task"


# --- Habit Tools ---

def test_habit_tools_lifecycle(tool_context):
    c_res = validate_and_execute_tool("create_habit", {"name": "Read 20 pages", "daily_goal": 1}, tool_context)
    assert c_res["status"] == "ok"
    habit_id = c_res["habit"]["id"]

    l_res = validate_and_execute_tool("list_habits", {}, tool_context)
    assert l_res["status"] == "ok"
    assert any(h["id"] == habit_id for h in l_res["habits"])

    comp_res = validate_and_execute_tool("complete_habit", {"habit": "Read 20 pages", "completed": True}, tool_context)
    assert comp_res["status"] == "ok"
    assert comp_res["habit"]["completed"] is True

    del_res = validate_and_execute_tool("delete_habit", {"habit": str(habit_id)}, tool_context)
    assert del_res["status"] == "needs_confirmation"


# --- Reminder Tools ---

def test_reminder_tools(tool_context):
    # Happy path
    res = validate_and_execute_tool("create_reminder", {"title": "stretch", "when": "in 1 minute"}, tool_context)
    assert res["status"] == "ok"
    assert res["reminder"]["title"] == "stretch"

    # Garbage time
    bad_res = validate_and_execute_tool("create_reminder", {"title": "stretch", "when": "garbage time yesterday"}, tool_context)
    assert bad_res["status"] == "error"

    # List reminders
    list_res = validate_and_execute_tool("list_reminders", {}, tool_context)
    assert list_res["status"] == "ok"
    assert len(list_res["reminders"]) >= 1

    # Delete reminder requires confirmation
    rem_id = res["reminder"]["id"]
    del_res = validate_and_execute_tool("delete_reminder", {"reminder": str(rem_id)}, tool_context)
    assert del_res["status"] == "needs_confirmation"


# --- Memory Tools ---

def test_memory_tools(tool_context):
    s_res = validate_and_execute_tool("save_memory", {"content": "I like dark roast coffee"}, tool_context)
    assert s_res["status"] == "ok"
    mem_id = s_res["memory"]["id"]

    search_res = validate_and_execute_tool("search_memory", {"query": "coffee"}, tool_context)
    assert search_res["status"] == "ok"
    assert len(search_res["results"]) >= 1

    del_res = validate_and_execute_tool("delete_memory", {"memory": str(mem_id)}, tool_context)
    assert del_res["status"] == "needs_confirmation"


# --- Web Search and Prompt Injection Guard ---

def test_web_search_and_injection_guard(tool_context):
    # Web search succeeds
    w_res = validate_and_execute_tool("web_search", {"query": "Python release"}, tool_context)
    assert w_res["status"] == "ok"
    assert len(w_res["results"]) <= 5
    assert tool_context.has_called_web_search is True

    # In the same turn, destructive tools are refused
    tool_context.deps.memory_store.create_task("Danger task")
    d_res = validate_and_execute_tool("delete_task", {"task": "Danger task"}, tool_context)
    assert d_res["status"] == "error"
    assert "Destructive operations are forbidden in the same turn as web search" in d_res["message"]


def test_web_search_not_configured(tool_context):
    tool_context.deps.web_search.side_effect = RuntimeError("Tavily API key not configured")
    res = validate_and_execute_tool("web_search", {"query": "test"}, tool_context)
    assert res["status"] == "error"
    assert "Web search is not configured" in res["message"]


# --- Navigation Tool ---

def test_navigate_tool(tool_context):
    tool_context.channel = "web"
    res = validate_and_execute_tool("navigate", {"page": "habits"}, tool_context)
    assert res["status"] == "ok"
    assert res["route"] == "/habits"
    assert any(a.get("type") == "navigate" and a.get("route") == "/habits" for a in tool_context.ui_actions)

    # Disallowed channel
    tool_context.channel = "discord"
    bad_res = validate_and_execute_tool("navigate", {"page": "habits"}, tool_context)
    assert bad_res["status"] == "error"
    assert "Navigation is only available in the web app" in bad_res["message"]


# --- Entity Resolution Tests ---

def test_resolve_entity_exact_and_fuzzy():
    items = [
        {"id": 1, "title": "Buy oat milk"},
        {"id": 2, "title": "Buy whole milk"},
        {"id": 3, "title": "Wash the sports car"},
    ]

    # Exact ID
    res, err = resolve_entity(1, items, "title")
    assert err is None
    assert res["id"] == 1

    # Exact title
    res, err = resolve_entity("Wash the sports car", items, "title")
    assert err is None
    assert res["id"] == 3

    # Ambiguous match (both "Buy oat milk" and "Buy whole milk" score closely for "milk")
    res, err = resolve_entity("milk", items, "title")
    assert res is None
    assert err["status"] == "ambiguous"
    assert len(err["candidates"]) >= 2

    # Weak match (< 70)
    res, err = resolve_entity("completely unrelated gibberish xyz", items, "title")
    assert res is None
    assert err["status"] == "not_found"


# --- List Cap at 15 ---

def test_list_tasks_capped_at_15(tool_context):
    for i in range(25):
        tool_context.deps.memory_store.create_task(f"Task #{i}")
    res = validate_and_execute_tool("list_tasks", {"status": "all"}, tool_context)
    assert res["status"] == "ok"
    assert len(res["tasks"]) == 15
