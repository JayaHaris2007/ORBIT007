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
        self.is_loading: bool = False
        self.error: str | None = None

        # Configure exclusively with voice design instruct ("describe one")
        if not self.voice_kwargs:
            self.voice_kwargs = {
                "instruct": config.TTS_INSTRUCT,
                "steps": config.TTS_STEPS,
                "speed": config.TTS_SPEED,
            }
            log.info("OmniVoice configured with voice design instruct: %s", config.TTS_INSTRUCT)

    def set_instruct(self, instruct: str, speed: float | None = None) -> None:
        """Update voice design tone dynamically."""
        self.voice_kwargs = {
            "instruct": instruct.strip(),
            "steps": config.TTS_STEPS,
            "speed": speed if speed is not None else config.TTS_SPEED,
        }
        log.info("OmniVoice voice design updated to: %s (speed=%.2f)", instruct, self.voice_kwargs["speed"])

    def preload_background(self) -> None:
        """No-op when server eager loading is active; preserved for backward compatibility."""
        if self._model is not None or self.available:
            return

    def _load_eager(self) -> None:
        """Perform the actual model weight loading once at startup."""
        with self._load_lock:
            if self._model is not None:
                return
            self.is_loading = True
            try:
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
            finally:
                self.is_loading = False

    def load(self) -> None:
        """Thin wrapper preserving backward compatibility. Does NOT trigger lazy loads in conversation."""
        if self._model is not None or self.available:
            return
        try:
            from runtime.models import REGISTRY
            if REGISTRY.is_ready("tts"):
                reg_engine = REGISTRY.get("tts")
                self._model = reg_engine._model
                self.device = reg_engine.device
                self.dtype = reg_engine.dtype
                self.available = reg_engine.available
                self.error = reg_engine.error
                return
            elif REGISTRY.status.get("tts") == "failed":
                self.available = False
                self.error = REGISTRY.errors.get("tts", "Initialization failed")
                return
        except Exception:
            pass
        # Fallback for standalone test runners without server lifespan
        self._load_eager()

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
        if _ENGINE is not None and _ENGINE.available:
            return _ENGINE
        try:
            from runtime.models import REGISTRY
            if REGISTRY.is_ready("tts"):
                reg = REGISTRY.get("tts")
                _ENGINE = reg
                return _ENGINE
        except Exception:
            pass
        if _ENGINE is None:
            _ENGINE = OmniVoiceEngine()
        return _ENGINE


def set_tts_engine(engine: OmniVoiceEngine | None) -> None:
    """Override the global TTS engine (useful for testing or shared lock wiring)."""
    global _ENGINE
    with _ENGINE_LOCK:
        _ENGINE = engine
        try:
            from runtime.models import REGISTRY
            if engine is not None:
                REGISTRY.register("tts", engine)
        except Exception:
            pass
