"""Characterization tests for chat endpoint, task creation/completion, and intent parser."""

import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient

import app
from tests.test_memory_and_context import FakeTokenizer


def test_characterization_task_create_via_chat(temp_db):
    client = TestClient(app.app)
    chat = temp_db.create_chat()
    chat_id = chat["id"]

    response = client.post(
        "/api/generate",
        json={"prompt": "add task Buy milk priority high", "chat_id": chat_id},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["provider"] == "local-task-manager"
    assert "Buy milk" in data["response"]

    # Verify task was created in database
    tasks = temp_db.list_tasks(include_completed=False)
    assert len(tasks) == 1
    assert tasks[0]["title"] == "Buy milk"
    assert tasks[0]["priority"] == "high"


def test_characterization_task_complete_via_chat(temp_db):
    client = TestClient(app.app)
    chat = temp_db.create_chat()
    chat_id = chat["id"]

    # Create task first
    task = temp_db.create_task(title="Finish homework", priority="medium")
    task_id = task["id"]

    response = client.post(
        "/api/generate",
        json={"prompt": f"complete task {task_id}", "chat_id": chat_id},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["provider"] == "local-task-manager"

    # Verify task is completed
    tasks = temp_db.list_tasks(include_completed=True)
    completed_task = next(t for t in tasks if t["id"] == task_id)
    assert completed_task["completed"] == 1


def test_characterization_llm_chat_fallback(temp_db, monkeypatch):
    client = TestClient(app.app)
    chat = temp_db.create_chat()
    chat_id = chat["id"]

    mock_gen = MagicMock(return_value="Mocked LLM reply for characterization test.")
    monkeypatch.setattr(app, "generate_response", mock_gen)
    monkeypatch.setattr(app, "tokenizer", FakeTokenizer())

    response = client.post(
        "/api/generate",
        json={"prompt": "Tell me a story about coding", "chat_id": chat_id},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["provider"] == "local-qwen"
    assert data["response"] == "Mocked LLM reply for characterization test."
    mock_gen.assert_called_once()
