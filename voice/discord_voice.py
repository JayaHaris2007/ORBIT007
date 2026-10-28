"""Discord voice call bridge for TEXT IN, VOICE OUT operation."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Any

import discord
import numpy as np
import torch
import torchaudio

from . import config, sentences, text_clean, tts

log = logging.getLogger("spidey.voice.discord")

if not discord.opus.is_loaded():
    try:
        discord.opus._load_default()
    except Exception as e:
        log.warning("Could not auto-load discord opus: %s", e)

FRAME_BYTES = 3840  # 20 ms of 48 kHz, 2 channels, 16-bit


def to_discord_pcm(wav24k: np.ndarray) -> bytes:
    """Resample 24 kHz mono float32 audio to 48 kHz stereo s16le PCM bytes."""
    if len(wav24k) == 0:
        return b""
    t = torch.from_numpy(wav24k).unsqueeze(0)
    t = torchaudio.functional.resample(t, 24000, 48000).squeeze(0).clamp(-1, 1)
    mono = (t.numpy() * 32767).astype("<i2")
    return np.column_stack([mono, mono]).tobytes()  # Interleaved L, R


class QueuePCMSource(discord.AudioSource):
    """Continuous stream AudioSource fed from synthesized PCM chunks with silence padding."""

    def __init__(self) -> None:
        self._buf = bytearray()
        self._lock = threading.Lock()

    def feed(self, pcm: bytes) -> None:
        with self._lock:
            self._buf.extend(pcm)

    def clear(self) -> None:
        with self._lock:
            self._buf.clear()

    def pending_ms(self) -> float:
        with self._lock:
            return len(self._buf) / FRAME_BYTES * 20.0

    def read(self) -> bytes:
        with self._lock:
            if len(self._buf) >= FRAME_BYTES:
                chunk = bytes(self._buf[:FRAME_BYTES])
                del self._buf[:FRAME_BYTES]
                return chunk
            if self._buf:  # Pad the final partial frame
                chunk = bytes(self._buf) + b"\x00" * (FRAME_BYTES - len(self._buf))
                self._buf.clear()
                return chunk
        return b"\x00" * FRAME_BYTES  # Silence keeps the player alive

    def is_opus(self) -> bool:
        return False


class DiscordVoiceCall:
    """Encapsulates active call state for a single Discord guild."""

    def __init__(
        self,
        guild_id: int,
        voice_client: Any,
        text_channel_id: int,
        source: QueuePCMSource,
    ) -> None:
        self.guild_id = guild_id
        self.voice_client = voice_client
        self.text_channel_id = text_channel_id
        self.source = source
        self.voice_enabled = True
        self.cancel = threading.Event()
        self.last_active_time = time.perf_counter()


class DiscordVoiceManager:
    """Manages active Discord voice connections, speech synthesis queues, and auto-leave."""

    def __init__(self) -> None:
        self.calls: dict[int, DiscordVoiceCall] = {}
        self._lock = threading.Lock()

    def get_call(self, guild_id: int) -> DiscordVoiceCall | None:
        with self._lock:
            return self.calls.get(guild_id)

    def is_call_active(self, guild_id: int) -> bool:
        with self._lock:
            return guild_id in self.calls

    async def join_call(
        self, message: discord.Message
    ) -> tuple[bool, str]:
        """Join the voice channel of the message author and link the text channel."""
        if message.guild is None:
            return False, "Voice calls can only be started in a server channel, not in DMs."

        author = message.author
        voice_state = getattr(author, "voice", None)
        if not voice_state or not getattr(voice_state, "channel", None):
            return False, "You need to be in a voice channel first for me to join."

        target_channel = author.voice.channel
        guild_id = message.guild.id

        # Check existing call
        call = self.get_call(guild_id)
        if call:
            if call.voice_client.channel.id == target_channel.id:
                call.text_channel_id = message.channel.id
                return True, f"Already in {target_channel.name}. Linked to this channel."
            await self.leave_call(guild_id)

        try:
            vc = await target_channel.connect()
            source = QueuePCMSource()
            vc.play(source)

            call = DiscordVoiceCall(
                guild_id=guild_id,
                voice_client=vc,
                text_channel_id=message.channel.id,
                source=source,
            )
            with self._lock:
                self.calls[guild_id] = call

            log.info("Joined voice channel %s in guild %s", target_channel.name, guild_id)
            return True, f"Joined **{target_channel.name}**. I'll speak responses in voice here!"
        except Exception as e:
            err_msg = str(e)
            log.exception("Failed to connect to Discord voice: %s", e)
            if "dave" in err_msg.lower() or "encryption" in err_msg.lower():
                import davey
                return False, (
                    f"Discord voice encryption error: {err_msg} "
                    f"(discord.py {discord.__version__}, davey {davey.__version__})"
                )
            return False, f"Could not join voice channel: {err_msg}"

    async def leave_call(self, guild_id: int) -> str:
        with self._lock:
            call = self.calls.pop(guild_id, None)

        if not call:
            return "No active voice call in this server."

        call.cancel.set()
        call.source.clear()
        try:
            if call.voice_client.is_connected():
                await call.voice_client.disconnect(force=True)
        except Exception as e:
            log.warning("Error disconnecting voice client: %s", e)

        log.info("Left voice channel in guild %s", guild_id)
        return "Left voice channel."

    def stop_speaking(self, guild_id: int) -> str:
        call = self.get_call(guild_id)
        if not call:
            return "No active voice call."
        call.cancel.set()
        call.source.clear()
        call.cancel = threading.Event()
        return "Stopped speaking."

    def set_voice_enabled(self, guild_id: int, enabled: bool) -> str:
        call = self.get_call(guild_id)
        if not call:
            return "No active voice call."
        call.voice_enabled = enabled
        status = "enabled" if enabled else "disabled"
        return f"Voice output is now {status}."

    def is_user_allowed(self, user_id: int) -> bool:
        if not config.DISCORD_VOICE_ALLOWED_USER_IDS:
            return True
        return user_id in config.DISCORD_VOICE_ALLOWED_USER_IDS

    async def speak_text(
        self, guild_id: int, user_id: int, text: str
    ) -> None:
        """Clean, truncate, split, synthesize and stream reply to the active voice channel."""
        call = self.get_call(guild_id)
        if not call or not call.voice_enabled:
            return

        if not self.is_user_allowed(user_id):
            return

        call.last_active_time = time.perf_counter()

        # Enforce spoken length cap
        spoken_text = text.strip()
        if len(spoken_text) > config.DISCORD_VOICE_MAX_CHARS:
            cutoff = spoken_text[: config.DISCORD_VOICE_MAX_CHARS].rfind(" ")
            if cutoff <= 0:
                cutoff = config.DISCORD_VOICE_MAX_CHARS
            spoken_text = (
                spoken_text[:cutoff].strip()
                + ". The rest of the answer is in the chat."
            )

        cleaned = text_clean.clean_text_for_speech(spoken_text)
        chunk_list = sentences.split_sentences(
            cleaned, max_chars=config.MAX_SENTENCE_CHARS
        )
        if not chunk_list:
            return

        tts_engine = tts.get_tts_engine()
        if not tts_engine.available:
            await asyncio.to_thread(tts_engine.load)
        if not tts_engine.available:
            return

        for chunk in chunk_list:
            if call.cancel.is_set():
                break
            try:
                audio = await asyncio.to_thread(
                    tts_engine.synthesize, chunk, call.cancel
                )
                if audio is None or len(audio) == 0 or call.cancel.is_set():
                    break
                pcm_bytes = to_discord_pcm(audio)
                call.source.feed(pcm_bytes)
            except Exception as e:
                log.exception("Error synthesizing Discord speech chunk: %s", e)
                break


_DISCORD_VOICE_MANAGER: DiscordVoiceManager | None = None


def get_discord_voice_manager() -> DiscordVoiceManager:
    global _DISCORD_VOICE_MANAGER
    if _DISCORD_VOICE_MANAGER is None:
        _DISCORD_VOICE_MANAGER = DiscordVoiceManager()
    return _DISCORD_VOICE_MANAGER
