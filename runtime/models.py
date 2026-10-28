"""Process-wide central model registry holding loaded model objects and status.

Strict invariant: get(name) NEVER triggers loading. Loading belongs solely to the
lifespan startup sequence.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

log = logging.getLogger("spidey.models")


class ModelNotReady(RuntimeError):
    """Raised when an operation requests a model that is not loaded or has failed."""
    pass


class ModelRegistry:
    """Thread-safe registry for single-process eager model instances."""

    def __init__(self) -> None:
        self._items: dict[str, Any] = {}
        self.status: dict[str, str] = {}  # name -> "pending" | "ready" | "failed"
        self.errors: dict[str, str] = {}  # name -> short error string
        self.metrics: dict[str, dict[str, Any]] = {}  # name -> {device, load_time, vram_mb}
        self._lock = threading.Lock()

    def register(self, name: str, obj: Any, metrics: dict[str, Any] | None = None) -> None:
        """Register a successfully loaded and warmed-up model object."""
        with self._lock:
            self._items[name] = obj
            self.status[name] = "ready"
            if name in self.errors:
                del self.errors[name]
            if metrics:
                self.metrics[name] = metrics
            log.info("Model '%s' registered successfully (status=ready).", name)

    def fail(self, name: str, err: Any) -> None:
        """Record that a model failed to load or warm up."""
        with self._lock:
            self.status[name] = "failed"
            self.errors[name] = str(err)[:300]
            log.error("Model '%s' marked as failed: %s", name, self.errors[name])

    def get(self, name: str) -> Any:
        """Retrieve a loaded model object. NEVER loads anything lazily."""
        with self._lock:
            obj = self._items.get(name)
            if obj is None:
                st = self.status.get(name, "pending")
                err = self.errors.get(name, "")
                err_detail = f": {err}" if err else ""
                raise ModelNotReady(f"Model '{name}' is not ready ({st}{err_detail}).")
            return obj

    def is_ready(self, name: str) -> bool:
        """Check whether a model is loaded and ready for inference."""
        with self._lock:
            return self.status.get(name) == "ready"

    def get_status_summary(self) -> dict[str, str]:
        """Return a copy of the model status mapping for health endpoints."""
        with self._lock:
            return dict(self.status)

    def get_error_summary(self) -> dict[str, str]:
        """Return a copy of the model errors mapping."""
        with self._lock:
            return dict(self.errors)


REGISTRY = ModelRegistry()
