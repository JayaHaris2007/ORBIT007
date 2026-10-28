"""OmniVoice TTS engine wrapper with lazy loading, device fallback, and thread-safe cancellation."""

from __future__ import annotations

import io
import logging
import os
import threading
import time
from typing import Any

import numpy as np
import soundfile as sf
import torch

from . import config

log = logging.getLogger("spidey.voice.tts")
SR = config.SAMPLE_RATE


class OmniVoiceEngine:
    """Manages OmniVoice model loading and sentence-by-sentence synthesis."""

    def __init__(
        self,
        model_id: str = config.TTS_MODEL_ID,
        device: str = config.TTS_DEVICE,
        dtype: str = config.TTS_DTYPE,
        voice_kwargs: dict[str, Any] | None = None,
        shared_lock: threading.Lock | None = None,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.voice_kwargs = voice_kwargs or {}
        self._model = None
        self._load_lock = threading.Lock()
        self.infer_lock = shared_lock or threading.Lock()
        self.available: bool = False
        self.error: str | None = None

        # Resolve voice reference or voice design if not explicitly provided
        if not self.voice_kwargs:
            ref_path = config.TTS_REF_AUDIO
            if ref_path and os.path.isfile(ref_path):
                self.voice_kwargs = {"ref_audio": ref_path, "speed": config.TTS_SPEED}
                log.info("OmniVoice configured with reference audio: %s", ref_path)
            else:
                self.voice_kwargs = {
                    "instruct": config.TTS_INSTRUCT,
                    "steps": config.TTS_STEPS,
                    "speed": config.TTS_SPEED,
                }
                log.info("OmniVoice configured with voice design instruct: %s", config.TTS_INSTRUCT)

    def load(self) -> None:
        """Lazily load the model on the configured device with CPU fallback."""
        with self._load_lock:
            if self._model is not None:
                return
            try:
                from omnivoice import OmniVoice
            except Exception as e:
                self.error = repr(e)
                self.available = False
                log.exception("omnivoice import failed; voice disabled")
                return

            attempts = [(self.device, self.dtype)]
            if self.device.startswith("cuda"):
                attempts.append(("cpu", "float32"))  # Automatic fallback, e.g. on CUDA OOM

            for dev, dt in attempts:
                try:
                    t0 = time.perf_counter()
                    torch_dtype = getattr(torch, dt, torch.float32)
                    self._model = OmniVoice.from_pretrained(
                        self.model_id, device_map=dev, dtype=torch_dtype
                    )
                    load_time = time.perf_counter() - t0
                    vram_mb = 0.0
                    if torch.cuda.is_available() and dev.startswith("cuda"):
                        vram_mb = torch.cuda.memory_allocated() / (1024 * 1024)
                    self.device, self.dtype, self.available, self.error = (
                        dev,
                        dt,
                        True,
                        None,
                    )
                    log.info(
                        "OmniVoice loaded successfully on %s (%s) in %.2fs (VRAM: %.1f MB)",
                        dev,
                        dt,
                        load_time,
                        vram_mb,
                    )
                    return
                except Exception as e:
                    self.error = repr(e)
                    log.exception("OmniVoice load failed on %s (%s)", dev, dt)
                    if dev.startswith("cuda") and torch.cuda.is_available():
                        torch.cuda.empty_cache()

            self.available = False

    def reload_on_cpu(self) -> bool:
        """Switch from CUDA to CPU float32 if CUDA runs out of memory during inference."""
        with self._load_lock:
            try:
                from omnivoice import OmniVoice

                if self.device.startswith("cuda") and torch.cuda.is_available():
                    del self._model
                    self._model = None
                    torch.cuda.empty_cache()

                self._model = OmniVoice.from_pretrained(
                    self.model_id, device_map="cpu", dtype=torch.float32
                )
                self.device, self.dtype, self.available, self.error = (
                    "cpu",
                    "float32",
                    True,
                    None,
                )
                log.warning("OmniVoice reloaded on CPU (float32) after CUDA failure")
                return True
            except Exception as e:
                self.error = repr(e)
                self.available = False
                log.exception("OmniVoice reload on CPU failed")
                return False

    def synthesize(
        self, text: str, cancel: threading.Event | None = None
    ) -> np.ndarray | None:
        """Synthesize a single sentence chunk into 24 kHz mono float32 audio."""
        if not self.available or (cancel and cancel.is_set()):
            return None

        # Check chunk length safety
        if len(text) > config.MAX_SENTENCE_CHARS:
            text = text[: config.MAX_SENTENCE_CHARS].rsplit(" ", 1)[0]

        with self.infer_lock:
            if cancel and cancel.is_set():
                return None
            try:
                audio = self._model.generate(text=text, **self.voice_kwargs)
            except (torch.cuda.OutOfMemoryError, RuntimeError) as err:
                log.warning("Synthesis error on %s: %s; clearing cache and retrying", self.device, err)
                if self.device.startswith("cuda") and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if cancel and cancel.is_set():
                    return None
                try:
                    audio = self._model.generate(text=text, **self.voice_kwargs)
                except Exception as retry_err:
                    log.error("Retry failed on %s: %s; falling back to CPU", self.device, retry_err)
                    if self.reload_on_cpu():
                        if cancel and cancel.is_set():
                            return None
                        audio = self._model.generate(text=text, **self.voice_kwargs)
                    else:
                        return None
            except Exception as err:
                log.exception("Unexpected error during synthesis: %s", err)
                return None

        if cancel and cancel.is_set():
            return None

        return np.asarray(audio[0], dtype=np.float32)


def audio_to_wav_bytes(audio_array: np.ndarray, sample_rate: int = SR) -> bytes:
    """Convert float32 numpy audio array to 16-bit PCM WAV bytes in memory."""
    buf = io.BytesIO()
    sf.write(buf, audio_array, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


_ENGINE: OmniVoiceEngine | None = None
_ENGINE_LOCK = threading.Lock()


def get_tts_engine() -> OmniVoiceEngine:
    """Return the global TTS engine singleton."""
    global _ENGINE
    with _ENGINE_LOCK:
        if _ENGINE is None:
            _ENGINE = OmniVoiceEngine()
        return _ENGINE


def set_tts_engine(engine: OmniVoiceEngine | None) -> None:
    """Override the global TTS engine (useful for testing or shared lock wiring)."""
    global _ENGINE
    with _ENGINE_LOCK:
        _ENGINE = engine
