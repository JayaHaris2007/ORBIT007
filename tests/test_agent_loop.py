"""Tests for the multi-turn agent tool loop, scripted Ollama mocking, and rules-first ordering."""

from __future__ import annotations

import io
import json
import urllib.request
from unittest.mock import MagicMock, patch

import pytest
from starlette.testclient import TestClient

import app
from agent.confirm import ConfirmationManager
from agent.deps import AgentDeps
from agent.loop import AgentResult, run_agent_loop


def _fake_http_response(payload: dict) -> MagicMock:
    raw = json.dumps(payload).encode("utf-8")
    mock_resp = MagicMock()
    mock_resp.read.return_value = raw
    mock_resp.__enter__.return_value = mock_resp
    return mock_resp


@pytest.fixture
def dummy_deps():
    deps = MagicMock(spec=AgentDeps)
    deps.local_timezone = app.LOCAL_TIMEZONE
    deps.memory_store = MagicMock()
    deps.memory_store.create_task = MagicMock(return_value={"id": 1, "title": "Buy milk", "priority": "medium"})
    deps.create_reminder_unified = MagicMock(return_value={"id": "rem-1", "title": "Call John", "remind_at": "2026-11-02T10:00:00Z"})
    deps.run_learning_hook = lambda fn: None
    deps.live_reminders = MagicMock(return_value=[])
    deps.generate_response = MagicMock(return_value="Plain LLM fallback reply.")
    deps.token_budget = MagicMock(return_value=128)
    return deps


def test_loop_single_tool_call_then_final_text(dummy_deps):
    confirm_mgr = ConfirmationManager()

    # Round 1: Model calls create_task
    resp_round_1 = {
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "function": {"name": "create_task", "arguments": {"title": "Buy milk"}},
                }
            ],
        },
        "prompt_eval_count": 120,
        "eval_count": 15,
        "eval_duration": 300000000,
    }
    # Round 2: Model gives final reply
    resp_round_2 = {
        "message": {
            "role": "assistant",
            "content": "<think>Thinking...</think>I have created the task 'Buy milk' for you.",
        },
        "prompt_eval_count": 180,
        "eval_count": 20,
        "eval_duration": 400000000,
    }

    mock_urlopen = MagicMock(side_effect=[
        _fake_http_response(resp_round_1),
        _fake_http_response(resp_round_2),
    ])

    with patch("urllib.request.urlopen", mock_urlopen):
        result = run_agent_loop(
            prompt="add task buy milk",
            chat_id="test-chat",
            history=[],
            memories=[],
            channel="web",
            deps=dummy_deps,
            confirm_mgr=confirm_mgr,
        )

    assert result.provider == "agent-tools"
    assert "<think>" not in result.reply
    assert "I have created the task 'Buy milk' for you." in result.reply
    assert any(a.get("type") == "refresh" and "tasks" in a.get("sections", []) for a in result.ui_actions)
    dummy_deps.memory_store.create_task.assert_called_once_with("Buy milk", priority="medium", due_at=None)


def test_loop_two_chained_tool_calls(dummy_deps):
    confirm_mgr = ConfirmationManager()

    # Round 1: Model calls create_task
    r1 = {
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "call_1", "function": {"name": "create_task", "arguments": {"title": "Finish report"}}}
            ],
        }
    }
    # Round 2: Model calls create_reminder
    r2 = {
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "call_2", "function": {"name": "create_reminder", "arguments": {"title": "Review report", "when": "tomorrow 9am"}}}
            ],
        }
    }
    # Round 3: Final response
    r3 = {
        "message": {
            "role": "assistant",
            "content": "Added the task and scheduled your reminder.",
        }
    }

    mock_urlopen = MagicMock(side_effect=[
        _fake_http_response(r1),
        _fake_http_response(r2),
        _fake_http_response(r3),
    ])

    with patch("urllib.request.urlopen", mock_urlopen):
        result = run_agent_loop(
            prompt="create task and reminder",
            chat_id="test-chat",
            history=[],
            memories=[],
            channel="web",
            deps=dummy_deps,
            confirm_mgr=confirm_mgr,
        )

    assert result.reply == "Added the task and scheduled your reminder."
    dummy_deps.memory_store.create_task.assert_called_once()
    dummy_deps.create_reminder_unified.assert_called_once()


def test_loop_unknown_tool_then_corrected(dummy_deps):
    confirm_mgr = ConfirmationManager()

    # Round 1: Model calls unknown tool "make_todo"
    r1 = {
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "call_bad", "function": {"name": "make_todo", "arguments": {"text": "Clean desk"}}}
            ],
        }
    }
    # Round 2: Model corrects to "create_task"
    r2 = {
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "call_good", "function": {"name": "create_task", "arguments": {"title": "Clean desk"}}}
            ],
        }
    }
    # Round 3: Final answer
    r3 = {
        "message": {"role": "assistant", "content": "Done, Clean desk task is added."}
    }

    mock_urlopen = MagicMock(side_effect=[
        _fake_http_response(r1),
        _fake_http_response(r2),
        _fake_http_response(r3),
    ])

    with patch("urllib.request.urlopen", mock_urlopen):
        result = run_agent_loop(
            prompt="clean desk",
            chat_id="test-chat",
            history=[],
            memories=[],
            channel="web",
            deps=dummy_deps,
            confirm_mgr=confirm_mgr,
        )

    assert result.reply == "Done, Clean desk task is added."


def test_loop_max_rounds_ends_with_honest_message(dummy_deps):
    confirm_mgr = ConfirmationManager()

    # Script 4 rounds of endless tool calls
    endless_round = {
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "call_loop", "function": {"name": "list_tasks", "arguments": {}}}
            ],
        }
    }

    mock_urlopen = MagicMock(side_effect=[
        _fake_http_response(endless_round),
        _fake_http_response(endless_round),
        _fake_http_response(endless_round),
        _fake_http_response(endless_round),
    ])

    with patch("urllib.request.urlopen", mock_urlopen):
        result = run_agent_loop(
            prompt="endless",
            chat_id="test-chat",
            history=[],
            memories=[],
            channel="web",
            deps=dummy_deps,
            confirm_mgr=confirm_mgr,
        )

    assert "I couldn't finish" in result.reply


def test_orbit_tools_zero_bypasses_agent_loop(monkeypatch):
    monkeypatch.setenv("ORBIT_TOOLS", "0")
    monkeypatch.setattr(app, "ensure_model_loaded", lambda: None)
    monkeypatch.setattr(app, "generate_response", lambda *args, **kwargs: "Bypassed tool loop reply")

    res = app.respond_with_tools("hello", "chat-1", [], [], "web")
    assert res.provider == "ollama"
    assert res.reply == "Bypassed tool loop reply"
    assert res.ui_actions == []


def test_rules_first_exact_command_bypasses_ollama(monkeypatch):
    mock_urlopen = MagicMock()
    monkeypatch.setattr("urllib.request.urlopen", mock_urlopen)

    client = TestClient(app.app)
    # "list tasks" matches rules first in _handle_local_command
    resp = client.post("/api/generate", json={"prompt": "list tasks"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["provider"] == "local-task-manager"
    # Ollama was never called!
    mock_urlopen.assert_not_called()


def test_ui_actions_in_ws_and_chat_payload(monkeypatch):
    client = TestClient(app.app)
    monkeypatch.setattr(app, "ensure_model_loaded", lambda: None)

    # Mock respond_with_tools to return an AgentResult with ui_actions
    fake_result = AgentResult(
        reply="Navigating you to habits.",
        ui_actions=[{"type": "navigate", "page": "habits", "route": "/habits"}, {"type": "refresh", "sections": ["habits"]}],
        provider="agent-tools",
    )
    monkeypatch.setattr(app, "respond_with_tools", lambda *args, **kwargs: fake_result)

    # Mock _handle_local_command to return None so the request goes to tools/LLM
    monkeypatch.setattr(app, "_handle_local_command", lambda *args, **kwargs: None)

    # 1. Test POST /api/generate
    res_gen = client.post("/api/generate", json={"prompt": "free-form navigation request to habits"})
    assert res_gen.status_code == 200
    data_gen = res_gen.json()
    assert data_gen["navigate"] == "habits"
    assert len(data_gen["ui_actions"]) == 2

    # 2. Test POST /api/chat
    res_chat = client.post("/api/chat", json={"message": "free-form navigation request to habits", "session_id": "default"})
    assert res_chat.status_code == 200
    data_chat = res_chat.json()
    assert len(data_chat["ui_actions"]) == 2
