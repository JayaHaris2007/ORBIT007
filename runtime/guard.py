"""Single-instance process lock guard for S.P.I.D.E.Y models.

Ensures that exactly ONE process can load models into GPU/CPU memory at any time.
If a second process attempts to start, it is refused immediately.
"""

from __future__ import annotations

import logging
import os
import socket
import subprocess
import sys

log = logging.getLogger("spidey.guard")

SPIDEY_MODEL_GUARD_PORT = 47654
_GUARD_SOCKET: socket.socket | None = None


def get_port_owner_pid(port: int) -> int | None:
    """Find the process ID currently listening on the specified localhost port."""
    try:
        if sys.platform == "win32":
            output = subprocess.check_output(
                f'netstat -ano | findstr ":{port} "',
                shell=True,
                text=True,
                stderr=subprocess.DEVNULL,
            )
            for line in output.strip().splitlines():
                parts = line.split()
                if len(parts) >= 5 and "LISTENING" in line:
                    return int(parts[-1])
    except Exception:
        pass
    return None


def acquire_model_process_lock(port: int = SPIDEY_MODEL_GUARD_PORT) -> socket.socket:
    """Acquire the exclusive model process lock on localhost.

    Raises SystemExit if another process already holds the lock.
    """
    global _GUARD_SOCKET
    if _GUARD_SOCKET is not None:
        return _GUARD_SOCKET

    guard = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        guard.bind(("127.0.0.1", port))
        guard.listen(1)
        _GUARD_SOCKET = guard
        log.info("Acquired single-instance model lock on port %d (PID %d).", port, os.getpid())
        return guard
    except OSError as err:
        guard.close()
        owner_pid = get_port_owner_pid(port)
        if owner_pid:
            pid_msg = (
                f"Models are already loaded by process [PID {owner_pid}]. "
                f"To stop the existing instance, run: taskkill /PID {owner_pid} /F (or Stop-Process -Id {owner_pid})."
            )
        else:
            pid_msg = "Models are already loaded by another process. Cannot start duplicate instance."
        log.error("Process lock collision on port %d: %s", port, err)
        print(f"\n[FATAL] {pid_msg}\n")
        raise SystemExit(pid_msg) from err


def release_model_process_lock() -> None:
    """Release the process lock socket."""
    global _GUARD_SOCKET
    if _GUARD_SOCKET is not None:
        try:
            _GUARD_SOCKET.close()
        except Exception:
            pass
        _GUARD_SOCKET = None
