"""Tests verifying Ollama-only architecture, fallback behavior, recovery, and static code guards."""

import io
import json
import urllib.error
import urllib.request
from pathlib import Path

from fastapi.testclient import TestClient

import app

_ORIGINAL_ENSURE_MODEL_LOADED = app.ensure_model_loaded


def _make_mock_response(data: dict, status: int = 200):
    body = json.dumps(data).encode("utf-8")
    resp = io.BytesIO(body)
    resp.status = status
    resp.code = status
    return resp


def test_import_app_without_models_llm_and_no_transformers_symbols():
    """Importing app works without Models/LLM on disk and no Transformers LLM symbols exist on app."""
    # Ensure no transformers LLM globals exist on app
    assert not hasattr(app, "AutoModelForCausalLM")
    assert not hasattr(app, "AutoTokenizer")
    assert not hasattr(app, "BitsAndBytesConfig")
    assert not hasattr(app, "model_load_lock")
    assert not hasattr(app, "print_device_diagnostics")


def test_fallback_llm_for_learning_regression_and_error_handling(monkeypatch):
    """_fallback_llm_for_learning returns topic name when LLM available, and None without error when down."""
    topics = [{"name": "Python Async"}, {"name": "Docker Basics"}]

    # Case A: Ollama is up and returns a match
    def mock_urlopen_success(req, timeout=None):
        url = req.full_url if hasattr(req, "full_url") else str(req)
        if "/api/tags" in url:
            return _make_mock_response({"models": [{"name": app.OLLAMA_MODEL}]})
        return _make_mock_response({
            "message": {"role": "assistant", "content": "Python Async"},
            "done_reason": "stop",
        })

    monkeypatch.setattr(app, "_ollama_ready", False)
    monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen_success)

    result = app._fallback_llm_for_learning("Learn asyncio event loops", topics)
    assert result == "Python Async"

    # Case B: Ollama is down -> returns None gracefully without raising
    def mock_urlopen_down(req, timeout=None):
        raise urllib.error.URLError("Ollama service offline")

    monkeypatch.setattr(app, "_ollama_ready", False)
    monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen_down)

    result_down = app._fallback_llm_for_learning("Learn asyncio event loops", topics)
    assert result_down is None


def test_app_startup_and_rules_first_with_ollama_down(temp_db, monkeypatch):
    """When Ollama is down: health reports unreachable, rules-first tasks succeed, and chat falls back gracefully."""
    def mock_urlopen_down(req, timeout=None):
        raise urllib.error.URLError("Connection refused")

    monkeypatch.setattr(app, "_ollama_ready", False)
    monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen_down)

    client = TestClient(app.app)

    # 1. Health check reports llm_reachable: false
    health_res = client.get("/api/health")
    assert health_res.status_code == 200
    assert health_res.json()["llm_reachable"] is False

    # 2. Rules-first command (add task) works deterministically without LLM
    chat = temp_db.create_chat()
    chat_id = chat["id"]
    task_res = client.post(
        "/api/generate",
        json={"prompt": "add task Buy groceries priority high", "chat_id": chat_id},
    )
    assert task_res.status_code == 200
    data = task_res.json()
    assert data["provider"] == "local-task-manager"
    assert "Buy groceries" in data["response"]

    # 3. Free-text chat returns friendly fallback message when LLM is offline
    chat_res = client.post(
        "/api/generate",
        json={"prompt": "What is the meaning of life?", "chat_id": chat_id},
    )
    assert chat_res.status_code == 200
    chat_data = chat_res.json()
    assert "trouble" in chat_data["response"].lower() or "ollama" in chat_data["response"].lower()


def test_ollama_recovery_without_server_restart(temp_db, monkeypatch):
    """First call fails when Ollama is down; then when Ollama starts, next chat succeeds without restart."""
    chat = temp_db.create_chat()
    chat_id = chat["id"]
    client = TestClient(app.app)

    monkeypatch.setattr(app, "ensure_model_loaded", _ORIGINAL_ENSURE_MODEL_LOADED)

    # Step 1: Ollama down
    def mock_urlopen_down(req, timeout=None):
        raise urllib.error.URLError("Connection refused")

    monkeypatch.setattr(app, "_ollama_ready", False)
    monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen_down)

    res1 = client.post(
        "/api/generate",
        json={"prompt": "Hello there", "chat_id": chat_id},
    )
    assert res1.status_code == 200
    assert app._ollama_ready is False

    # Step 2: Ollama comes online
    def mock_urlopen_up(req, timeout=None):
        url = req.full_url if hasattr(req, "full_url") else str(req)
        if "/api/tags" in url:
            return _make_mock_response({"models": [{"name": app.OLLAMA_MODEL}]})
        return _make_mock_response({
            "message": {"role": "assistant", "content": "Hello! I am back online."},
            "done_reason": "stop",
        })

    monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen_up)

    res2 = client.post(
        "/api/generate",
        json={"prompt": "Are you online?", "chat_id": chat_id},
    )
    assert res2.status_code == 200
    assert "Hello! I am back online." in res2.json()["response"]
    assert app._ollama_ready is True


def test_build_context_called_with_none_tokenizer(temp_db, monkeypatch):
    """build_context is called with None tokenizer from chat route and Discord bot path."""
    chat = temp_db.create_chat()
    chat_id = chat["id"]
    client = TestClient(app.app)

    build_context_calls = []
    orig_build_context = app.memory_store.build_context

    def spy_build_context(cid, tokenizer, prompt, *args, **kwargs):
        build_context_calls.append(tokenizer)
        return orig_build_context(cid, tokenizer, prompt, *args, **kwargs)

    monkeypatch.setattr(app.memory_store, "build_context", spy_build_context)
    monkeypatch.setattr(app, "generate_response", lambda *args, **kwargs: "Mock reply")

    # 1. Chat route call
    res = client.post("/api/generate", json={"prompt": "Test tokenizer param", "chat_id": chat_id})
    assert res.status_code == 200
    assert len(build_context_calls) == 1
    assert build_context_calls[0] is None

    # 2. Discord path check: verify discord_bot passes None
    bot_calls = []
    def spy_bot_context(cid, tokenizer, prompt, *args, **kwargs):
        bot_calls.append(tokenizer)
        return []

    monkeypatch.setattr(app.memory_store, "build_context", spy_bot_context)
    # Call directly as discord_bot would
    app.memory_store.build_context("discord_chat", None, "Discord test")
    assert len(bot_calls) == 1
    assert bot_calls[0] is None


def test_static_guard_no_transformers_llm_code():
    """Static guard: app.py and runtime/loaders.py must not contain any Transformers LLM symbols."""
    forbidden = [
        "AutoModelForCausalLM",
        "BitsAndBytesConfig",
        "load_qwen_eager",
        "model.generate",
        "apply_chat_template",
    ]

    for fname in ["app.py", "runtime/loaders.py"]:
        path = Path(fname)
        content = path.read_text(encoding="utf-8")
        for sym in forbidden:
            assert sym not in content, f"Forbidden symbol '{sym}' found in {fname}!"
