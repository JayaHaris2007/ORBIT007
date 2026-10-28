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


def from_discord_pcm(pcm48k_stereo: bytes) -> bytes:
    """Convert 48 kHz stereo s16le PCM bytes from Discord to 16 kHz mono s16le PCM bytes for STT."""
    if not pcm48k_stereo:
        return b""
    arr = np.frombuffer(pcm48k_stereo, dtype="<i2").reshape(-1, 2)
    mono = (arr[:, 0].astype(np.float32) + arr[:, 1].astype(np.float32)) / (2.0 * 32768.0)
    t = torch.from_numpy(mono).unsqueeze(0)
    res = torchaudio.functional.resample(t, 48000, 16000).squeeze(0).clamp(-1, 1)
    return (res.numpy() * 32767).astype("<i2").tobytes()


def _calculate_discord_pcm_energy(pcm48k_stereo: bytes) -> float:
    """Calculate RMS energy from 48 kHz stereo PCM bytes."""
    if not pcm48k_stereo:
        return 0.0
    arr = np.frombuffer(pcm48k_stereo, dtype="<i2")
    if len(arr) == 0:
        return 0.0
    return float(np.sqrt(np.mean(arr.astype(np.float64) ** 2)))


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


try:
    from discord.ext import voice_recv
    _VOICE_RECV_AVAILABLE = True
except ImportError:
    voice_recv = None
    _VOICE_RECV_AVAILABLE = False


class DiscordVoiceCall:
    """Encapsulates active call state for a single Discord guild."""

    def __init__(
        self,
        guild_id: int,
        voice_client: Any,
        text_channel_id: int,
        source: QueuePCMSource,
        sink: Any | None = None,
    ) -> None:
        self.guild_id = guild_id
        self.voice_client = voice_client
        self.text_channel_id = text_channel_id
        self.source = source
        self.sink = sink
        self.voice_enabled = True
        self.is_speaking = False
        self.cancel = threading.Event()
        self.last_active_time = time.perf_counter()
if _VOICE_RECV_AVAILABLE and voice_recv is not None:
    class DiscordVoiceAudioSink(voice_recv.AudioSink):
        """Audio sink receiving user voice packets from Discord and triggering STT on silence."""

        def __init__(
            self,
            call: DiscordVoiceCall,
            bot: Any,
            loop: asyncio.AbstractEventLoop,
            manager: DiscordVoiceManager,
        ) -> None:
            super().__init__()
            self.call = call
            self.bot = bot
            self.loop = loop
            self.manager = manager
            self.buffers: dict[int, dict[str, Any]] = {}
            self._lock = threading.Lock()

        def wants_opus(self) -> bool:
            return False

        def write(self, user: discord.Member | discord.User | None, data: voice_recv.VoiceData) -> None:
            if user is None or getattr(user, "bot", False):
                return
            if not self.manager.is_user_allowed(user.id):
                return

            pcm = data.pcm
            if not pcm:
                return

            now = time.perf_counter()
            energy = _calculate_discord_pcm_energy(pcm)

            # Barge-in: if assistant is speaking and user speaks loud enough, cancel assistant speech
            if getattr(self.call, "is_speaking", False) and energy >= config.LIVE_BARGE_IN_THRESHOLD:
                self.call.cancel.set()
                self.call.source.clear()
                self.call.cancel = threading.Event()
                log.info("Barge-in: interrupted assistant speech due to voice input from %s", user.display_name)

            user_id = user.id
            with self._lock:
                if user_id not in self.buffers:
                    self.buffers[user_id] = {
                        "pcm": bytearray(),
                        "speech_detected": False,
                        "speech_start_time": 0.0,
                        "last_speech_time": 0.0,
                        "user": user,
                    }
                buf = self.buffers[user_id]
                buf["user"] = user

            if energy >= config.VAD_ENERGY_THRESHOLD:
                with self._lock:
                    if not buf["speech_detected"]:
                        buf["speech_detected"] = True
                        buf["speech_start_time"] = now
                    buf["last_speech_time"] = now
                    buf["pcm"].extend(pcm)
            elif buf["speech_detected"]:
                with self._lock:
                    buf["pcm"].extend(pcm)
                    silence_ms = (now - buf["last_speech_time"]) * 1000.0
                    speech_ms = (buf["last_speech_time"] - buf["speech_start_time"]) * 1000.0
                    if silence_ms >= config.VAD_TRAILING_SILENCE_MS:
                        if speech_ms >= config.VAD_MIN_SPEECH_MS:
                            raw_pcm = bytes(buf["pcm"])
                            buf["pcm"].clear()
                            buf["speech_detected"] = False
                            buf["speech_start_time"] = 0.0
                            buf["last_speech_time"] = 0.0
                            pcm16k = from_discord_pcm(raw_pcm)
                            if pcm16k:
                                asyncio.run_coroutine_threadsafe(
                                    self._process_utterance(user, pcm16k),
                                    self.loop,
                                )
                        else:
                            buf["pcm"].clear()
                            buf["speech_detected"] = False

        async def _process_utterance(self, user: discord.Member | discord.User, pcm16k: bytes) -> None:
            try:
                from runtime.models import REGISTRY
                if REGISTRY.is_ready("stt"):
                    stt_engine = REGISTRY.get("stt")
                else:
                    from .stt import get_stt
                    stt_engine = get_stt()

                text = await asyncio.to_thread(stt_engine.transcribe, pcm16k)
                if not text or not text.strip():
                    return
                log.info("Transcribed voice from %s in guild %s: '%s'", user.display_name, self.call.guild_id, text)
                await self.bot.respond_to_voice_utterance(
                    self.call.guild_id, user, self.call.text_channel_id, text.strip()
                )
            except Exception as e:
                log.exception("Error processing Discord voice utterance from %s: %s", user.display_name, e)

        def cleanup(self) -> None:
            with self._lock:
                self.buffers.clear()
else:
    class DiscordVoiceAudioSink:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass



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
            if _VOICE_RECV_AVAILABLE and voice_recv is not None:
                try:
                    vc = await target_channel.connect(cls=voice_recv.VoiceRecvClient)
                except Exception as err:
                    log.warning("VoiceRecvClient connect failed (%s); falling back to VoiceClient", err)
                    vc = await target_channel.connect()
            else:
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

            # Attach receive sink if voice_recv is available
            if _VOICE_RECV_AVAILABLE and voice_recv is not None and isinstance(vc, voice_recv.VoiceRecvClient):
                try:
                    import discord_bot
                    raw_sink = DiscordVoiceAudioSink(call, discord_bot.bot, asyncio.get_running_loop(), self)
                    sink = voice_recv.SilenceGeneratorSink(raw_sink)
                    vc.listen(sink)
                    call.sink = sink
                    log.info("Attached Discord voice receive sink on channel %s", target_channel.name)
                except Exception as err:
                    log.warning("Could not attach voice receive sink: %s", err)

            log.info("Joined voice channel %s in guild %s", target_channel.name, guild_id)
            return True, f"Joined **{target_channel.name}**. I'm listening and speaking in voice!"
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
            if hasattr(call.voice_client, "is_listening") and call.voice_client.is_listening():
                call.voice_client.stop_listening()
            if hasattr(call.voice_client, "is_playing") and call.voice_client.is_playing():
                call.voice_client.stop()
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

        call.is_speaking = True
        try:
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
        finally:
            call.is_speaking = False


_DISCORD_VOICE_MANAGER: DiscordVoiceManager | None = None


def get_discord_voice_manager() -> DiscordVoiceManager:
    global _DISCORD_VOICE_MANAGER
    if _DISCORD_VOICE_MANAGER is None:
        _DISCORD_VOICE_MANAGER = DiscordVoiceManager()
    return _DISCORD_VOICE_MANAGER
