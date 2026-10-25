import os
from pathlib import Path
import pytest
from unittest.mock import MagicMock

# Ensure semantic memory doesn't download sentence-transformer weights during unit tests
os.environ["AI_ENABLE_SEMANTIC_MEMORY"] = "0"

import app
from memory import MemoryStore


@pytest.fixture
def temp_db(tmp_path):
    """Provide a fresh, isolated MemoryStore in a temporary directory."""
    db_file = tmp_path / "test_assistant.db"
    store = MemoryStore(db_file)
    return store


@pytest.fixture(autouse=True)
def isolate_app_environment(temp_db, monkeypatch):
    """Ensure all app.py operations use the temporary SQLite database and never touch assistant.db."""
    monkeypatch.setattr(app, "DATABASE_PATH", temp_db.database_path)
    monkeypatch.setattr(app, "memory_store", temp_db)

    # Mock the APScheduler instance to avoid spawning unneeded background threads during testing
    mock_scheduler = MagicMock()
    monkeypatch.setattr(app, "reminder_scheduler", mock_scheduler)

    # Mock ensure_model_loaded so tests never attempt to load Qwen3 into VRAM or CPU
    monkeypatch.setattr(app, "ensure_model_loaded", MagicMock())

    return temp_db
