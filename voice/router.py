"""FastAPI router and WebSocket gateway for the voice module."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import struct
import time
from typing import Any, Callable, Coroutine

from fastapi import APIRouter, Body, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from . import config, text_clean, sentences, session, tts, stt

log = logging.getLogger("spidey.voice.router")

router = APIRouter(prefix="/api/voice", tags=["voice"])
ws_router = APIRouter(tags=["voice"])

# Persistent chat handler registered by app.py
_CHAT_HANDLER: Callable[[str, str | None], Coroutine[Any, Any, dict[str, Any]]] | None = None


def set_chat_handler(
    handler: Callable[[str, str | None], Coroutine[Any, Any, dict[str, Any]]] | None
) -> None:
    """Register the chat pipeline function from app.py."""
    global _CHAT_HANDLER
    _CHAT_HANDLER = handler


def get_chat_handler() -> Callable[[str, str | None], Coroutine[Any, Any, dict[str, Any]]] | None:
    return _CHAT_HANDLER


_in_memory_speak_enabled: bool = False
_live_enabled_flag: bool = False


def _read_speak_setting() -> bool:
    try:
        from learning import progress, schema
        if not schema.is_initialized():
            import app
            if getattr(app, "DATABASE_PATH", None):
                schema.init(app.DATABASE_PATH)
        val = progress.get_setting("voice_speak_enabled", None)
        if val is not None:
            return val == "1"
        return _in_memory_speak_enabled
    except Exception:
        return _in_memory_speak_enabled


def _write_speak_setting(enabled: bool) -> None:
    global _in_memory_speak_enabled
    _in_memory_speak_enabled = enabled
    try:
        from learning import progress, schema
        if not schema.is_initialized():
            import app
            if getattr(app, "DATABASE_PATH", None):
                schema.init(app.DATABASE_PATH)
        progress.set_setting("voice_speak_enabled", "1" if enabled else "0")
    except Exception:
        pass


@router.get("/settings")
def get_voice_settings() -> dict[str, Any]:
    """Retrieve voice system status and persistent configurations."""
    engine = tts.get_tts_engine()
    engine.preload_background()

    # Mask any file paths in error strings
    safe_error = None
    if engine.error:
        safe_error = engine.error.replace("\\", "/").split("/")[-1]

    # Engine is ready if available or ready to load without unrecoverable error
    engine_ready = engine.available or (engine.error is None)

    return {
        "speak_enabled": _read_speak_setting(),
        "live_enabled": _live_enabled_flag,
        "engine_ready": engine_ready,
        "is_loading": getattr(engine, "is_loading", False),
        "available": engine.available,
        "device": engine.device,
        "dtype": engine.dtype,
        "error": safe_error,
        "allow_cloud_stt": config.STT_ALLOW_CLOUD,
        "stt_engine": config.STT_ENGINE,
    }


@router.put("/settings")
def update_voice_settings(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """Update speak and live mode toggles."""
    global _live_enabled_flag
    if not isinstance(payload, dict):
        raise HTTPException(400, "Request body must be a JSON object.")

    if "speak_enabled" in payload:
        speak_val = bool(payload["speak_enabled"])
        _write_speak_setting(speak_val)

    if "live_enabled" in payload:
        _live_enabled_flag = bool(payload["live_enabled"])

    return get_voice_settings()


@router.post("/tts")
def synthesize_tts_chunk(payload: dict[str, Any] = Body(...)) -> Response:
    """Synthesize a single short text chunk into WAV audio."""
    if not isinstance(payload, dict):
        raise HTTPException(400, "Request body must be a JSON object.")

    text = payload.get("text")
    if not isinstance(text, str) or not text.strip():
        raise HTTPException(400, "'text' must be a non-empty string.")

    cleaned = text_clean.clean_text_for_speech(text)
    if not cleaned:
        raise HTTPException(400, "Text contains no speakable characters.")

    engine = tts.get_tts_engine()
    if not engine.available:
        engine.load()

    if not engine.available:
        raise HTTPException(503, "OmniVoice TTS engine is not available.")

    audio_array = engine.synthesize(cleaned)
    if audio_array is None or len(audio_array) == 0:
        raise HTTPException(500, "TTS synthesis failed or was cancelled.")

    wav_bytes = tts.audio_to_wav_bytes(audio_array, sample_rate=config.SAMPLE_RATE)
    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


@router.post("/stop")
def stop_voice_turn(payload: dict[str, Any] | None = Body(default=None)) -> dict[str, str]:
    """Cancel any in-flight voice synthesis, transcribing, or playback."""
    session_id = "default"
    if payload and isinstance(payload, dict) and "session_id" in payload:
        session_id = str(payload["session_id"])
    voice_session = session.get_voice_session(session_id)
    voice_session.reset()
    return {"status": "stopped"}


def _calculate_pcm_energy(pcm_bytes: bytes) -> float:
    """Compute Root Mean Square (RMS) energy of 16-bit mono PCM samples."""
    if len(pcm_bytes) < 2:
        return 0.0
    num_samples = len(pcm_bytes) // 2
    # Unpack 16-bit signed integers
    fmt = f"<{num_samples}h"
    try:
        samples = struct.unpack(fmt, pcm_bytes[: num_samples * 2])
    except struct.error:
        return 0.0
    sum_squares = sum(s * s for s in samples)
    return math.sqrt(sum_squares / max(1, num_samples))


@ws_router.websocket("/ws/voice")
async def websocket_voice_endpoint(websocket: WebSocket) -> None:
    """WebSocket gateway for continuous mic streaming, STT, and live conversational turns."""
    await websocket.accept()
    client_id = f"ws-{id(websocket)}"
    voice_sess = session.get_voice_session(client_id)
    stt_engine = stt.get_stt()
    tts_engine = tts.get_tts_engine()

    mode = "listen"
    chat_id: str | None = None
    is_speaking = False
    speech_detected = False
    speech_start_time = 0.0
    silence_start_time = 0.0

    async def send_state(st: session.VState) -> None:
        voice_sess.state = st
        try:
            await websocket.send_text(json.dumps({"type": "state", "value": st.value}))
        except Exception:
            pass

    async def speak_sentences(reply_text: str) -> None:
        nonlocal is_speaking
        cleaned = text_clean.clean_text_for_speech(reply_text)
        sentences_list = sentences.split_sentences(cleaned, max_chars=config.MAX_SENTENCE_CHARS)
        if not sentences_list:
            return
        if mode != "live" and not _read_speak_setting():
            return

        if not tts_engine.available:
            await asyncio.to_thread(tts_engine.load)
        if not tts_engine.available:
            return

        is_speaking = True
        await send_state(session.VState.SPEAKING)

        for seq, sent in enumerate(sentences_list):
            if voice_sess.cancel.is_set():
                break
            audio_array = await asyncio.to_thread(
                tts_engine.synthesize, sent, voice_sess.cancel
            )
            if audio_array is None or voice_sess.cancel.is_set():
                break

            wav_data = tts.audio_to_wav_bytes(audio_array, sample_rate=config.SAMPLE_RATE)
            header = json.dumps({"type": "audio", "seq": seq, "sentence": sent})
            try:
                await websocket.send_text(header)
                await websocket.send_bytes(wav_data)
            except Exception:
                break

        is_speaking = False

    async def process_utterance() -> None:
        nonlocal speech_detected, is_speaking
        speech_detected = False
        pcm_data = bytes(voice_sess.pcm)
        voice_sess.pcm.clear()

        if len(pcm_data) < 3200:  # Less than 100 ms of audio
            await send_state(session.VState.LISTENING if mode == "live" else session.VState.IDLE)
            return

        # 1. Transcribe
        await send_state(session.VState.TRANSCRIBING)
        try:
            transcript = await stt_engine.transcribe_async(pcm_data)
        except Exception as e:
            await websocket.send_text(json.dumps({"type": "error", "message": str(e)}))
            await send_state(session.VState.LISTENING if mode == "live" else session.VState.IDLE)
            return

        if not transcript.strip() or voice_sess.cancel.is_set():
            await send_state(session.VState.LISTENING if mode == "live" else session.VState.IDLE)
            return

        await websocket.send_text(json.dumps({"type": "transcript", "text": transcript}))

        # 2. Think / Generate reply
        await send_state(session.VState.THINKING)
        handler = get_chat_handler()
        reply_text = ""
        if handler:
            try:
                chat_res = await handler(transcript, chat_id)
                reply_text = chat_res.get("response", "")
            except Exception as e:
                await websocket.send_text(json.dumps({"type": "error", "message": str(e)}))
                await send_state(session.VState.LISTENING if mode == "live" else session.VState.IDLE)
                return

        if not reply_text.strip() or voice_sess.cancel.is_set():
            await send_state(session.VState.LISTENING if mode == "live" else session.VState.IDLE)
            return

        await websocket.send_text(json.dumps({"type": "reply", "text": reply_text}))

        # 3. Speak
        await speak_sentences(reply_text)

        # 4. Finish turn
        if mode == "live" and not voice_sess.cancel.is_set():
            await send_state(session.VState.LISTENING)
        else:
            await send_state(session.VState.IDLE)

    try:
        while True:
            msg = await websocket.receive()
            if "text" in msg:
                try:
                    payload = json.loads(msg["text"])
                except Exception:
                    continue

                msg_type = payload.get("type")
                if msg_type == "start":
                    mode = payload.get("mode", "listen")
                    chat_id = payload.get("chat_id")
                    voice_sess.reset()
                    speech_detected = False
                    await send_state(session.VState.LISTENING)

                elif msg_type == "stop":
                    voice_sess.reset()
                    is_speaking = False
                    speech_detected = False
                    await send_state(session.VState.IDLE)

                elif msg_type == "interrupt":
                    voice_sess.reset()
                    is_speaking = False
                    speech_detected = False
                    await send_state(session.VState.LISTENING if mode == "live" else session.VState.IDLE)

            elif "bytes" in msg:
                pcm_chunk = msg["bytes"]
                now = time.perf_counter()
                energy = _calculate_pcm_energy(pcm_chunk)

                # Check barge-in during assistant speaking
                if is_speaking:
                    if energy >= config.LIVE_BARGE_IN_THRESHOLD:
                        voice_sess.reset()
                        is_speaking = False
                        speech_detected = True
                        speech_start_time = now
                        voice_sess.pcm.extend(pcm_chunk)
                        await send_state(session.VState.LISTENING)
                    continue

                if voice_sess.state != session.VState.LISTENING:
                    continue

                voice_sess.pcm.extend(pcm_chunk)

                if energy >= config.VAD_ENERGY_THRESHOLD:
                    if not speech_detected:
                        speech_detected = True
                        speech_start_time = now
                    silence_start_time = 0.0
                elif speech_detected:
                    if silence_start_time == 0.0:
                        silence_start_time = now
                    elif (now - silence_start_time) * 1000 >= config.VAD_TRAILING_SILENCE_MS:
                        # Ensure minimum speech length was satisfied
                        if (now - speech_start_time) * 1000 >= config.VAD_MIN_SPEECH_MS:
                            task = asyncio.create_task(process_utterance())
                            voice_sess.tasks.add(task)
                        else:
                            speech_detected = False
                            voice_sess.pcm.clear()

    except WebSocketDisconnect:
        pass
    except Exception as e:
        log.exception("Voice WebSocket error: %s", e)
    finally:
        session.remove_voice_session(client_id)
