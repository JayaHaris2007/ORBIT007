"""Client session state machine and transient buffer lifecycle for voice interactions."""

from __future__ import annotations

import asyncio
import enum
import threading
from typing import Any

from . import config


class VState(str, enum.Enum):
    IDLE = "idle"
    LISTENING = "listening"
    TRANSCRIBING = "transcribing"
    THINKING = "thinking"
    SPEAKING = "speaking"


class VoiceSession:
    """Represents one active voice client connection (browser tab or Discord voice)."""

    def __init__(self, session_id: str = "default") -> None:
        self.session_id = session_id
        self.state = VState.IDLE
        self.cancel = threading.Event()
        self.pcm = bytearray()  # Raw mic audio: 16 kHz mono s16le, memory only
        self.partial_text = ""
        self.tasks: set[asyncio.Task[Any]] = set()
        self.tts_queue: asyncio.Queue[tuple[str, bytes] | None] = asyncio.Queue()
        self.lock = asyncio.Lock()

    def reset(self) -> None:
        """STOP: cancel in-flight work and wipe every transient buffer of the current turn."""
        self.cancel.set()  # Tells STT/LLM/TTS threads to quit at the next check
        for t in list(self.tasks):
            if not t.done():
                t.cancel()
        self.tasks.clear()
        self.pcm = bytearray()  # Drop mic audio
        self.partial_text = ""
        while not self.tts_queue.empty():
            try:
                self.tts_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        config.purge_voice_tmp()  # Purge any transient temp files
        self.cancel = threading.Event()  # Fresh token for the next turn
        self.state = VState.IDLE


_SESSIONS: dict[str, VoiceSession] = {}
_SESSIONS_LOCK = threading.Lock()


def get_voice_session(session_id: str = "default") -> VoiceSession:
    """Retrieve or create a VoiceSession for a client ID."""
    with _SESSIONS_LOCK:
        if session_id not in _SESSIONS:
            _SESSIONS[session_id] = VoiceSession(session_id)
        return _SESSIONS[session_id]


def remove_voice_session(session_id: str) -> None:
    """Reset and remove a client's VoiceSession."""
    with _SESSIONS_LOCK:
        session = _SESSIONS.pop(session_id, None)
        if session:
            session.reset()


def reset_all_sessions() -> None:
    """Reset all active voice sessions."""
    with _SESSIONS_LOCK:
        for session in _SESSIONS.values():
            session.reset()

