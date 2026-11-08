"""Tests for confirmation manager and destructive action lifecycle."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from agent.confirm import ConfirmationManager
from agent.deps import AgentDeps


@pytest.fixture
def dummy_deps():
    deps = MagicMock(spec=AgentDeps)
    deps.memory_store = MagicMock()
    deps.cancel_reminder_unified = MagicMock(return_value=True)
    return deps


def test_request_confirmation_stores_pending():
    mgr = ConfirmationManager(timeout_seconds=120.0)
    target = {"id": 10, "title": "Buy milk"}
    res = mgr.request_confirmation(
        chat_id="chat-1",
        tool_name="delete_task",
        target=target,
        summary="Delete the task 'Buy milk'?",
    )
    assert res["status"] == "needs_confirmation"
    assert res["tool"] == "delete_task"
    assert res["target_id"] == 10

    pending = mgr.get_pending("chat-1")
    assert pending is not None
    assert pending.tool_name == "delete_task"
    assert pending.target["id"] == 10


def test_confirmation_affirmative_executes_task(dummy_deps):
    mgr = ConfirmationManager(timeout_seconds=120.0)
    target = {"id": 42, "title": "Important Task"}
    mgr.request_confirmation("chat-1", "delete_task", target, "Delete Important Task?")

    handled, reply, actions = mgr.check_and_handle("chat-1", "yes please", dummy_deps)
    assert handled is True
    assert "Important Task" in reply
    assert "deleted" in reply
    dummy_deps.memory_store.delete_task.assert_called_once_with(42)
    assert any(a.get("type") == "refresh" and "tasks" in a.get("sections", []) for a in actions)

    # Pending action is cleared
    assert mgr.get_pending("chat-1") is None


def test_confirmation_negative_cancels(dummy_deps):
    mgr = ConfirmationManager(timeout_seconds=120.0)
    target = {"id": 42, "title": "Important Task"}
    mgr.request_confirmation("chat-1", "delete_task", target, "Delete Important Task?")

    handled, reply, actions = mgr.check_and_handle("chat-1", "no don't", dummy_deps)
    assert handled is True
    assert "cancelled" in reply.lower()
    dummy_deps.memory_store.delete_task.assert_not_called()
    assert mgr.get_pending("chat-1") is None


def test_confirmation_unrelated_message_discards_pending(dummy_deps):
    mgr = ConfirmationManager(timeout_seconds=120.0)
    target = {"id": 42, "title": "Important Task"}
    mgr.request_confirmation("chat-1", "delete_task", target, "Delete Important Task?")

    handled, reply, actions = mgr.check_and_handle("chat-1", "what's the weather in Tokyo?", dummy_deps)
    assert handled is False
    assert reply == ""
    dummy_deps.memory_store.delete_task.assert_not_called()
    assert mgr.get_pending("chat-1") is None


def test_confirmation_expires_after_120_seconds(monkeypatch, dummy_deps):
    mgr = ConfirmationManager(timeout_seconds=120.0)
    target = {"id": 42, "title": "Important Task"}
    current_time = 1000.0
    monkeypatch.setattr(time, "time", lambda: current_time)

    mgr.request_confirmation("chat-1", "delete_task", target, "Delete Important Task?")

    # Advance time past 120s
    current_time = 1125.0
    handled, reply, actions = mgr.check_and_handle("chat-1", "yes", dummy_deps)
    assert handled is False
    dummy_deps.memory_store.delete_task.assert_not_called()
    assert mgr.get_pending("chat-1") is None


def test_destructive_delete_habit_confirmation(dummy_deps):
    mgr = ConfirmationManager()
    target = {"id": 5, "name": "Morning Yoga"}
    mgr.request_confirmation("chat-1", "delete_habit", target, "Delete habit?")

    handled, reply, actions = mgr.check_and_handle("chat-1", "confirm", dummy_deps)
    assert handled is True
    assert "Morning Yoga" in reply
    dummy_deps.memory_store.delete_habit.assert_called_once_with(5)


def test_destructive_delete_reminder_confirmation(dummy_deps):
    mgr = ConfirmationManager()
    target = {"id": "rem-123", "title": "Doctor Appointment"}
    mgr.request_confirmation("chat-1", "delete_reminder", target, "Delete reminder?")

    handled, reply, actions = mgr.check_and_handle("chat-1", "yep", dummy_deps)
    assert handled is True
    assert "Doctor Appointment" in reply
    dummy_deps.cancel_reminder_unified.assert_called_once_with("rem-123")


def test_destructive_delete_memory_confirmation(dummy_deps):
    mgr = ConfirmationManager()
    target = {"id": 8, "content": "I like dark mode"}
    mgr.request_confirmation("chat-1", "delete_memory", target, "Delete memory?")

    handled, reply, actions = mgr.check_and_handle("chat-1", "do it", dummy_deps)
    assert handled is True
    assert "dark mode" in reply
    dummy_deps.memory_store.delete_memory.assert_called_once_with(8)

