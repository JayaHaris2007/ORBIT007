"""Keep Discord voice receive alive when an Opus packet cannot be decoded."""

from __future__ import annotations

import logging
import threading

import discord.opus
from discord.opus import OpusError

log = logging.getLogger("spidey.voice.discord")

_SILENT_FRAME = b"\x00" * 3840
_WARN_AFTER = 50
_lock = threading.Lock()
_state = {"consecutive": 0, "total": 0, "warned": False}
_original = None


def install() -> None:
    global _original
    if _original is not None:
        return
    _original = discord.opus.Decoder.decode

    def safe_decode(self, *args, **kwargs):
        try:
            pcm = _original(self, *args, **kwargs)
        except OpusError as exc:
            with _lock:
                _state["consecutive"] += 1
                _state["total"] += 1
                warn = (
                    _state["consecutive"] >= _WARN_AFTER
                    and not _state["warned"]
                )
                if warn:
                    _state["warned"] = True
            if warn:
                log.warning(
                    "Incoming Discord voice is undecodable; check discord.py "
                    "/ voice_recv / DAVE versions"
                )
            log.debug("Skipping undecodable Opus packet: %s", exc)
            return _SILENT_FRAME
        with _lock:
            _state["consecutive"] = 0
            _state["warned"] = False
        return pcm

    discord.opus.Decoder.decode = safe_decode


def uninstall() -> None:
    global _original
    if _original is not None:
        discord.opus.Decoder.decode = _original
        _original = None
    with _lock:
        _state.update(consecutive=0, total=0, warned=False)


def stats() -> dict[str, int | bool]:
    with _lock:
        return dict(_state)
