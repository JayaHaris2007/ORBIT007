"""CPU-only Gradium Phonon engine backed by the ptts package."""

from __future__ import annotations

import io
import logging
import threading
import time
from typing import Any

import numpy as np
import soundfile as sf

log = logging.getLogger("spidey.voice.tts")
SAMPLE_RATE = 24000


class PhononEngine:
    """Load Phonon once and serialize CPU synthesis requests."""

    def __init__(self, model: str, lang: str, quant: str, voice: str) -> None:
        self.model = model
        self.lang = lang
        self.quant = quant
        self.voice = voice
        self.load_time = 0.0
        self.error: str | None = None
        self._tts: Any | None = None
        self._lock = threading.Lock()

    def load(self) -> "PhononEngine":
        import ptts

        started = time.perf_counter()
        try:
            # ptts 0.6 names its 8-bit format q8_0; retain the requested q8
            # configuration name for compatibility with the model instructions.
            quant = "q8_0" if self.quant == "q8" else self.quant
            self._tts = ptts.TTS(self.model, device="cpu", lang=self.lang, quant=quant)
            # Validate voice and fall back if unavailable
            if hasattr(self._tts, "synth"):
                try:
                    self._tts.synth("warmup", voice=self.voice)
                except Exception as voice_err:
                    voices = list(getattr(self._tts, "voices", []))
                    fallback = getattr(voices[0], "name", voices[0]) if voices else "default"
                    log.warning("Phonon voice %r not available (%s); using %r", self.voice, voice_err, fallback)
                    self.voice = fallback
            elif hasattr(self._tts, "voices"):
                voices = list(self._tts.voices)
                names = [getattr(item, "name", item) for item in voices]
                if names and self.voice not in names:
                    self.voice = names[0]
        except Exception as exc:
            self.error = str(exc)
            raise
        self.load_time = time.perf_counter() - started
        log.info(
            "Phonon ready (%s, %s, CPU) voice=%s in %.2fs",
            self.model, self.quant, self.voice, self.load_time,
        )
        return self

    def warm_up(self) -> None:
        """Warm up the phonon engine with a short token."""
        try:
            self.synth_pcm("warmup")
        except Exception as exc:
            log.warning("Phonon warm-up exception (non-fatal): %s", exc)

    @property
    def ready(self) -> bool:
        return self._tts is not None

    def synth_pcm(self, text: str, cancel: threading.Event | None = None) -> np.ndarray | None:
        if cancel and cancel.is_set():
            return None
        text = text.strip()
        if not text:
            return None
        with self._lock:
            if cancel and cancel.is_set():
                return None
            pcm = self._tts.synth(text, voice=self.voice)
        if cancel and cancel.is_set():
            return None
        return np.asarray(pcm, dtype=np.float32)

    def synth_wav(self, text: str) -> bytes:
        pcm = self.synth_pcm(text)
        if pcm is None:
            return b""
        buf = io.BytesIO()
        sf.write(buf, pcm, SAMPLE_RATE, format="WAV", subtype="PCM_16")
        return buf.getvalue()

    def stream_pcm(self, text: str, cancel: threading.Event | None = None):
        if cancel and cancel.is_set():
            return
        text = text.strip()
        if not text:
            return
        with self._lock:
            if cancel and cancel.is_set():
                return
            stream_obj = self._tts.stream(text, voice=self.voice)
            if hasattr(stream_obj, "__enter__"):
                with stream_obj as audio:
                    for chunk in audio:
                        if cancel and cancel.is_set():
                            break
                        yield np.asarray(chunk, dtype=np.float32)
            else:
                for chunk in stream_obj:
                    if cancel and cancel.is_set():
                        break
                    yield np.asarray(chunk, dtype=np.float32)

