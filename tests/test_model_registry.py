"""Unit and characterization tests for the central model registry and single-process lifecycle."""

from __future__ import annotations

import asyncio
import os
import socket
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import WebSocketDisconnect

from runtime.guard import acquire_model_process_lock, release_model_process_lock
from runtime.models import ModelNotReady, ModelRegistry


def test_registry_lifecycle_and_invariants():
    """Verify registration, retrieval, failure, and strict non-loading invariant."""
    reg = ModelRegistry()

    # 1. Before loading, item is not ready and get() raises ModelNotReady
    assert not reg.is_ready("test_model")
    with pytest.raises(ModelNotReady) as exc:
        reg.get("test_model")
    assert "pending" in str(exc.value)

    # 2. Registering an object marks it ready
    dummy_model = {"weights": [1, 2, 3]}
    reg.register("test_model", dummy_model)
    assert reg.is_ready("test_model")
    assert reg.get("test_model") is dummy_model

    # 3. Marking a model failed stores the error and get() raises ModelNotReady with detail
    reg.fail("failed_model", "CUDA out of memory error")
    assert not reg.is_ready("failed_model")
    assert reg.status["failed_model"] == "failed"
    assert "CUDA out of memory" in reg.errors["failed_model"]
    with pytest.raises(ModelNotReady) as exc:
        reg.get("failed_model")
    assert "failed" in str(exc.value)
    assert "CUDA out of memory" in str(exc.value)


def test_process_guard_collision_detection():
    """Verify that a second process attempting to acquire the lock is rejected."""
    # Use an ephemeral high test port to avoid touching the production port in tests
    test_port = 49123

    # Release any prior state
    release_model_process_lock()

    sock1 = acquire_model_process_lock(port=test_port)
    assert sock1 is not None

    # Simulating a second process trying to bind to the same port
    sock2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    with pytest.raises(OSError):
        sock2.bind(("127.0.0.1", test_port))
    sock2.close()

    # Cleanup
    release_model_process_lock()


def test_offline_environment_variables_configured():
    """Verify offline huggingface flags are set appropriately."""
    if os.environ.get("SPIDEY_ALLOW_HUB") != "1":
        assert os.environ.get("HF_HUB_OFFLINE") == "1"
        assert os.environ.get("TRANSFORMERS_OFFLINE") == "1"


def test_websocket_disconnect_clean_exit():
    """Verify that websocket.disconnect message cleanly terminates loop without raising RuntimeError."""
    from voice.router import websocket_voice_endpoint

    mock_ws = AsyncMock()
    # First message: disconnect message from client
    mock_ws.receive = AsyncMock(return_value={"type": "websocket.disconnect"})
    mock_ws.accept = AsyncMock()

    # Run the endpoint and confirm it exits cleanly without re-calling receive()
    asyncio.run(websocket_voice_endpoint(mock_ws))
    assert mock_ws.receive.call_count == 1
