"""Discord voice call bridge for TEXT IN, VOICE OUT operation."""

from __future__ import annotations

import asyncio
import ctypes.util
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

import discord
import numpy as np

from . import config, sentences, text_clean, tts

log = logging.getLogger("spidey.voice.discord")


def _ensure_opus_loaded() -> bool:
    """Ensure discord.opus is loaded so raw PCM AudioSources encode properly."""
    if discord.opus.is_loaded():
        return True
    try:
        discord.opus._load_default()
    except Exception as e:
        log.debug("discord.opus._load_default() did not load opus: %s", e)
    if discord.opus.is_loaded():
        return True

    for p in sys.path:
        for dll_name in ("libopus-0.x64.dll", "libopus-0.x86.dll", "opus.dll"):
            candidate = Path(p) / "discord" / "bin" / dll_name
            if candidate.exists():
                try:
                    discord.opus.load_opus(str(candidate))
                    log.info("Loaded discord opus library from %s", candidate)
                    return True
                except Exception as err:
                    log.debug("Failed loading opus from %s: %s", candidate, err)

    for lib_name in ("opus", "libopus.so.0", "libopus.0.dylib", "libopus-0.x64.dll"):
        found = ctypes.util.find_library(lib_name) or lib_name
        try:
            discord.opus.load_opus(found)
            if discord.opus.is_loaded():
                log.info("Loaded system opus library (%s)", found)
                return True
        except Exception:
            continue

    log.warning("discord.opus could not be loaded; Discord voice output may fail!")
    return False


_ensure_opus_loaded()

try:
    from . import voice_recv_patch
    voice_recv_patch.install()
except Exception as e:
    log.debug("voice_recv_patch not loaded: %s", e)

FRAME_BYTES = 3840  # 20 ms of 48 kHz, 2 channels, 16-bit


def to_discord_pcm(wav24k: np.ndarray, normalize: bool = False) -> bytes:
    """Resample 24 kHz mono float32 audio to 48 kHz stereo s16le PCM bytes without torch/torchaudio."""
    if wav24k is None or len(wav24k) == 0:
        return b""
    arr = np.asarray(wav24k, dtype=np.float32).reshape(-1)
    if len(arr) == 0:
        return b""
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=-1.0)

    if normalize:
        peak = float(np.max(np.abs(arr)))
        if 0.002 < peak < 0.65:
            arr = arr * (0.85 / peak)

    # Exact 2x upsample from 24 kHz to 48 kHz via linear interpolation
    n = len(arr)
    up48k = np.empty(n * 2, dtype=np.float32)
    up48k[0::2] = arr
    if n > 1:
        up48k[1:-1:2] = 0.5 * (arr[:-1] + arr[1:])
    up48k[-1] = arr[-1]

    up48k = np.clip(up48k, -1.0, 1.0)
    mono = (up48k * 32767.0).astype("<i2")
    return np.column_stack([mono, mono]).tobytes()  # Interleaved L, R


def from_discord_pcm(pcm48k_stereo: bytes) -> bytes:
    """Convert 48 kHz stereo s16le PCM bytes from Discord to 16 kHz mono s16le PCM bytes for STT."""
    if not pcm48k_stereo:
        return b""
    arr = np.frombuffer(pcm48k_stereo, dtype="<i2").reshape(-1, 2)
    mono = (arr[:, 0].astype(np.float32) + arr[:, 1].astype(np.float32)) / (2.0 * 32768.0)
    # Exact 3:1 decimation from 48 kHz to 16 kHz with 3-tap anti-aliasing average
    n = len(mono)
    out_len = n // 3
    if out_len == 0:
        return b""
    trimmed = mono[: out_len * 3].reshape(out_len, 3)
    res = np.clip(trimmed.mean(axis=1), -1.0, 1.0)
    return (res * 32767.0).astype("<i2").tobytes()


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


class UtterancePCMSource(discord.AudioSource):
    """Finite PCM AudioSource for a single spoken utterance so Discord toggles Speaking state properly."""

    def __init__(self, pcm: bytes, cancel: threading.Event | None = None) -> None:
        self._buf = memoryview(pcm)
        self._pos = 0
        self._len = len(pcm)
        self._cancel = cancel

    def read(self) -> bytes:
        if self._cancel is not None and self._cancel.is_set():
            return b""
        if self._pos >= self._len:
            return b""
        end = self._pos + FRAME_BYTES
        chunk = bytes(self._buf[self._pos : end])
        self._pos = end
        if len(chunk) < FRAME_BYTES:
            chunk = chunk + b"\x00" * (FRAME_BYTES - len(chunk))
        return chunk

    def is_opus(self) -> bool:
        return False


try:
    from discord.ext import voice_recv
    _VOICE_RECV_AVAILABLE = True
except ImportError:
    voice_recv = None
    _VOICE_RECV_AVAILABLE = False

# By default, operate in strict TEXT IN (via server chat), VOICE OUT (TTS-only) mode.
# Set DISCORD_VOICE_INPUT=1 only if experimental Discord voice-in (STT) is desired.
_ENABLE_DISCORD_VOICE_INPUT = os.environ.get("DISCORD_VOICE_INPUT", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


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
        self.speak_lock = asyncio.Lock()
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

            # Barge-in: if assistant is speaking and user speaks loud enough, cancel assistant speech
            if getattr(self.call, "is_speaking", False) and energy >= config.LIVE_BARGE_IN_THRESHOLD:
                self.call.cancel.set()
                self.call.source.clear()
                self.call.cancel = threading.Event()
                log.info("Barge-in: interrupted assistant speech due to voice input from %s (energy=%.1f)", user.display_name, energy)

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

    async def join_voice_channel(
        self,
        target_channel: Any,
        text_channel_id: int,
        guild_id: int | None = None,
    ) -> tuple[bool, str]:
        if guild_id is None:
            guild_id = getattr(getattr(target_channel, "guild", None), "id", None)

        if guild_id is not None:
            call = self.get_call(guild_id)
            if call:
                if (
                    getattr(getattr(call, "voice_client", None), "channel", None)
                    and call.voice_client.channel.id == target_channel.id
                    and hasattr(call.voice_client, "is_connected")
                    and call.voice_client.is_connected()
                ):
                    call.text_channel_id = text_channel_id
                    return True, f"Already in **{getattr(target_channel, 'name', 'voice')}**. Send any message here and I'll speak the reply in voice!"
                await self.leave_call(guild_id)

        try:
            _ensure_opus_loaded()
            if _ENABLE_DISCORD_VOICE_INPUT and _VOICE_RECV_AVAILABLE and voice_recv is not None:
                try:
                    vc = await target_channel.connect(cls=voice_recv.VoiceRecvClient)
                except Exception as err:
                    log.warning("VoiceRecvClient connect failed (%s); falling back to VoiceClient", err)
                    vc = await target_channel.connect()
            else:
                try:
                    vc = await target_channel.connect(self_deaf=True)
                except TypeError:
                    vc = await target_channel.connect()

            source = QueuePCMSource()
            vc.play(source)

            call = DiscordVoiceCall(
                guild_id=guild_id if guild_id is not None else 0,
                voice_client=vc,
                text_channel_id=text_channel_id,
                source=source,
            )
            if guild_id is not None:
                with self._lock:
                    self.calls[guild_id] = call

            if _ENABLE_DISCORD_VOICE_INPUT and _VOICE_RECV_AVAILABLE and voice_recv is not None and isinstance(vc, voice_recv.VoiceRecvClient):
                try:
                    import discord_bot
                    raw_sink = DiscordVoiceAudioSink(call, discord_bot.bot, asyncio.get_running_loop(), self)
                    sink = voice_recv.SilenceGeneratorSink(raw_sink)
                    vc.listen(sink)
                    call.sink = sink
                    log.info("Attached Discord voice receive sink on channel %s", getattr(target_channel, "name", "voice"))
                except Exception as err:
                    log.warning("Could not attach voice receive sink: %s", err)

            channel_name = getattr(target_channel, "name", "voice channel")
            log.info(
                "Joined voice channel %s in guild %s (mode=%s)",
                channel_name,
                guild_id,
                "voice-in+out" if _ENABLE_DISCORD_VOICE_INPUT else "TTS-only text-in/voice-out",
            )
            if guild_id is not None:
                async def _delayed_join_greeting(gid: int, cname: str) -> None:
                    await asyncio.sleep(0.6)
                    c = self.get_call(gid)
                    if c and c.voice_enabled and not c.is_speaking:
                        await self.speak_text(gid, 0, f"Hello! Connected to {cname} and ready.")

                asyncio.create_task(_delayed_join_greeting(guild_id, channel_name))
            return True, f"Joined **{channel_name}**. Type your messages in this channel and I will speak the replies in voice!"
        except Exception as e:
            err_msg = str(e)
            log.exception("Failed to connect to Discord voice: %s", e)
            if "dave" in err_msg.lower() or "encryption" in err_msg.lower():
                try:
                    import davey
                    davey_ver = getattr(davey, "__version__", "unknown")
                except Exception:
                    davey_ver = "missing"
                return False, (
                    f"Discord voice encryption error: {err_msg} "
                    f"(discord.py {discord.__version__}, davey {davey_ver})"
                )
            return False, f"Could not join voice channel: {err_msg}"

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
        return await self.join_voice_channel(target_channel, message.channel.id, guild_id=message.guild.id)

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
        try:
            if call.voice_client and hasattr(call.voice_client, "is_playing") and call.voice_client.is_playing():
                call.voice_client.stop()
        except Exception as e:
            log.debug("Error stopping voice_client playback: %s", e)
        call.cancel = threading.Event()
        return "Stopped speaking."

    def set_voice_enabled(self, guild_id: int, enabled: bool) -> str:
        call = self.get_call(guild_id)
        if not call:
            return "No active voice call."
        call.voice_enabled = enabled
        if not enabled:
            self.stop_speaking(guild_id)
        status = "enabled" if enabled else "disabled"
        return f"Voice output is now {status}."

    def is_user_allowed(self, user_id: int | str) -> bool:
        if not user_id or user_id == 0 or str(user_id) == "0":
            return True
        if not config.DISCORD_VOICE_ALLOWED_USER_IDS:
            return True
        try:
            return int(user_id) in config.DISCORD_VOICE_ALLOWED_USER_IDS
        except (ValueError, TypeError):
            return False

    async def _play_pcm_utterance(
        self,
        call: DiscordVoiceCall,
        pcm_bytes: bytes,
        cancel_event: threading.Event,
    ) -> None:
        """Play a finite PCM buffer via VoiceClient.play() so Discord receives Speaking(True)->PCM->Speaking(False)."""
        if not pcm_bytes or cancel_event.is_set():
            return

        vc = call.voice_client
        if vc is None:
            return
        if hasattr(vc, "is_connected") and not vc.is_connected():
            log.warning("VoiceClient for guild %s is no longer connected; skipping playback.", call.guild_id)
            return

        # Keep call.source updated for any external inspection/tests
        call.source.clear()
        call.source.feed(pcm_bytes)

        loop = asyncio.get_running_loop()
        done_event = asyncio.Event()

        def _after_playback(err: Exception | None) -> None:
            if err:
                log.error("Discord AudioPlayer error in guild %s: %s", call.guild_id, err)
            call.source.clear()
            try:
                loop.call_soon_threadsafe(done_event.set)
            except RuntimeError:
                pass

        # Stop any idle QueuePCMSource or previous stream so vc.play() sends a fresh Speaking(True) opcode
        try:
            if hasattr(vc, "is_playing") and vc.is_playing():
                vc.stop()
                await asyncio.sleep(0.03)
        except Exception as stop_err:
            log.debug("Stopping previous AudioPlayer before utterance: %s", stop_err)

        utterance_source = UtterancePCMSource(pcm_bytes, cancel=cancel_event)
        try:
            vc.play(utterance_source, after=_after_playback)
        except Exception as play_err:
            log.exception("vc.play() failed in guild %s: %s", call.guild_id, play_err)
            return

        duration_sec = len(pcm_bytes) / float(FRAME_BYTES * 50)
        timeout_sec = max(5.0, duration_sec + 5.0)
        try:
            await asyncio.wait_for(done_event.wait(), timeout=timeout_sec)
        except asyncio.TimeoutError:
            log.warning("Timed out waiting for Discord utterance playback (%.1fs) in guild %s", duration_sec, call.guild_id)
            try:
                if hasattr(vc, "is_playing") and vc.is_playing():
                    vc.stop()
            except Exception:
                pass

    async def speak_text(
        self, guild_id: int, user_id: int | str, text: str
    ) -> None:
        """Clean, truncate, split, synthesize and play reply to the active Discord voice channel."""
        call = self.get_call(guild_id)
        if not call or not call.voice_enabled:
            log.warning("speak_text skipped: call=%s, voice_enabled=%s", call, getattr(call, "voice_enabled", False))
            return

        if not self.is_user_allowed(user_id):
            log.warning("speak_text skipped: user %r not in allowlist %r", user_id, config.DISCORD_VOICE_ALLOWED_USER_IDS)
            return

        call.last_active_time = time.perf_counter()

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
            log.debug("speak_text: no chunks after cleaning text %r", spoken_text[:40])
            return

        _ensure_opus_loaded()
        tts_engine = tts.get_tts_engine()
        if not tts_engine.available or (config.TTS_BACKEND != "sapi" and not (tts_engine.phonon and tts_engine.phonon.ready)):
            await asyncio.to_thread(tts_engine.load)
        if not tts_engine.available:
            log.error("speak_text failed: TTS engine is not available")
            return

        # Interrupt any currently playing utterance when a new user reply starts speaking
        if call.is_speaking:
            call.cancel.set()
            try:
                if call.voice_client and hasattr(call.voice_client, "is_playing") and call.voice_client.is_playing():
                    call.voice_client.stop()
            except Exception:
                pass

        async with call.speak_lock:
            call.cancel = threading.Event()
            cancel_token = call.cancel
            call.is_speaking = True
            log.info(
                "Synthesizing and speaking %d chunk(s) for guild %s via %s...",
                len(chunk_list),
                guild_id,
                tts_engine.backend,
            )
            try:
                for chunk in chunk_list:
                    if cancel_token.is_set() or not call.voice_enabled:
                        log.info("speak_text cancelled before chunk.")
                        break
                    try:
                        log.info("Synthesizing TTS for guild %s: %r", guild_id, chunk[:60])
                        audio = await asyncio.to_thread(
                            tts_engine.synthesize, chunk, cancel_token
                        )
                        if audio is None or len(audio) == 0 or cancel_token.is_set():
                            continue
                        pcm_bytes = to_discord_pcm(audio, normalize=True)
                        if not pcm_bytes:
                            continue
                        log.info(
                            "Playing %d PCM bytes (%.2fs) to Discord in guild %s",
                            len(pcm_bytes),
                            len(pcm_bytes) / float(FRAME_BYTES * 50),
                            guild_id,
                        )
                        await self._play_pcm_utterance(call, pcm_bytes, cancel_token)
                        log.info("Finished speaking chunk in guild %s", guild_id)
                    except Exception as e:
                        log.exception("Error synthesizing/playing Discord speech chunk: %s", e)
                        break
            finally:
                call.is_speaking = False


_DISCORD_VOICE_MANAGER: DiscordVoiceManager | None = None


def get_discord_voice_manager() -> DiscordVoiceManager:
    global _DISCORD_VOICE_MANAGER
    if _DISCORD_VOICE_MANAGER is None:
        _DISCORD_VOICE_MANAGER = DiscordVoiceManager()
    return _DISCORD_VOICE_MANAGER
