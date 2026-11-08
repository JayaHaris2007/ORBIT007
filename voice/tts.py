"""Text-to-speech service with CPU Phonon and Windows SAPI fallback."""

from __future__ import annotations

import io
import logging
import os
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from . import config
from .phonon_engine import PhononEngine

log = logging.getLogger("spidey.voice.tts")
SR = config.SAMPLE_RATE
_warned: set[str] = set()


def _warn_once(key: str, message: str, *args: Any) -> None:
    if key not in _warned:
        _warned.add(key)
        log.warning(message, *args)


def audio_to_wav_bytes(audio_array: np.ndarray, sample_rate: int = SR) -> bytes:
    """Convert float32 numpy audio array to 16-bit PCM WAV bytes in memory."""
    buf = io.BytesIO()
    sf.write(buf, np.asarray(audio_array, dtype=np.float32), sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


class WindowsSapiEngine:
    """Windows-native fallback that does not import torch or Phonon."""

    def __init__(self) -> None:
        self.device = "windows-sapi"
        self.dtype = "pcm"
        self.available = os.name == "nt"
        self.is_loading = False
        self.error: str | None = None if self.available else "Windows SAPI is only available on Windows."
        self.voice = ""
        self.load_time = 0.0
        self.infer_lock = threading.Lock()

    def load(self) -> None:
        return

    def synthesize(self, text: str, cancel: threading.Event | None = None) -> np.ndarray | None:
        if not self.available or (cancel and cancel.is_set()):
            return None

        output_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", dir=config.TMP_DIR, delete=False) as output_file:
                output_path = output_file.name
            import base64

            encode = lambda value: base64.b64encode(value.encode("utf-8")).decode("ascii")
            script = (
                "Add-Type -AssemblyName System.Speech; "
                "$e=New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                f"$t=[Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('{encode(text)}')); "
                f"$p=[Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('{encode(output_path)}')); "
                "$e.SetOutputToWaveFile($p); $e.Speak($t); $e.Dispose()"
            )
            result = subprocess.run(
                ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
                capture_output=True, text=True, timeout=30, check=False,
            )
            if result.returncode != 0:
                raise RuntimeError(result.stderr.strip() or "Windows SAPI synthesis failed.")
            audio, file_sr = sf.read(output_path, dtype="float32")
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            audio = np.asarray(audio, dtype=np.float32)
            if file_sr and file_sr != SR and len(audio) > 1:
                new_len = max(1, int(round(len(audio) * SR / float(file_sr))))
                x_old = np.linspace(0.0, 1.0, len(audio), endpoint=False)
                x_new = np.linspace(0.0, 1.0, new_len, endpoint=False)
                audio = np.interp(x_new, x_old, audio).astype(np.float32)
            return audio
        except (OSError, RuntimeError, subprocess.SubprocessError, ValueError) as err:
            self.error = str(err)
            log.error("Windows SAPI synthesis failed: %s", err)
            return None
        finally:
            if output_path:
                Path(output_path).unlink(missing_ok=True)


class TTSService:
    """Phonon-first TTS service with automatic SAPI fallback."""

    def __init__(self, phonon: PhononEngine | None = None, sapi: WindowsSapiEngine | None = None) -> None:
        self.phonon = phonon
        self.sapi = sapi or WindowsSapiEngine()
        self.error: str | None = phonon.error if phonon else None
        self.load_time = phonon.load_time if phonon else 0.0
        self.is_loading = False
        self.last_backend: str | None = None
        self.last_error: str | None = None

    @property
    def available(self) -> bool:
        return bool(self.phonon and self.phonon.ready) or self.sapi.available

    @property
    def device(self) -> str:
        return "cpu" if self.phonon and self.phonon.ready else self.sapi.device

    @property
    def dtype(self) -> str:
        return "float32" if self.phonon and self.phonon.ready else self.sapi.dtype

    @property
    def voice(self) -> str:
        return self.phonon.voice if self.phonon and self.phonon.ready else self.sapi.voice

    @property
    def backend(self) -> str:
        return "phonon" if self.phonon and self.phonon.ready and config.TTS_BACKEND != "sapi" else "sapi"

    def load(self) -> None:
        """Eagerly load Phonon if configured and not already loaded."""
        if config.TTS_BACKEND != "sapi" and (self.phonon is None or not self.phonon.ready):
            try:
                from .phonon_engine import PhononEngine
                self.phonon = PhononEngine(
                    model=config.PHONON_MODEL,
                    lang=config.PHONON_LANG,
                    quant=config.PHONON_QUANT,
                    voice=config.PHONON_VOICE,
                ).load()
                self.error = None
                self.load_time = self.phonon.load_time
            except Exception as exc:
                self.error = str(exc)
                log.exception("Phonon TTS failed to load; using SAPI fallback: %s", exc)

    def synthesize(self, text: str, cancel: threading.Event | None = None) -> np.ndarray | None:
        if config.TTS_BACKEND != "sapi":
            if self.phonon is None or not self.phonon.ready:
                self.load()
            if self.phonon and self.phonon.ready:
                try:
                    audio = self.phonon.synth_pcm(text, cancel=cancel)
                    if audio is None:
                        return None
                    self.last_backend = "phonon"
                    self.last_error = None
                    return audio
                except Exception as exc:
                    self.error = str(exc)
                    self.last_error = str(exc)
                    _warn_once(type(exc).__name__, "Phonon synthesis failed (%s); using SAPI.", exc)
        audio = self.sapi.synthesize(text, cancel)
        if audio is not None:
            self.last_backend = "sapi"
        return audio

    def stream_pcm(self, text: str, cancel: threading.Event | None = None):
        """Yield float32 audio chunks in real time if Phonon is active, else synthesize whole."""
        if config.TTS_BACKEND != "sapi":
            if self.phonon is None or not self.phonon.ready:
                self.load()
            if self.phonon and self.phonon.ready:
                try:
                    self.last_backend = "phonon"
                    self.last_error = None
                    for chunk in self.phonon.stream_pcm(text, cancel=cancel):
                        yield chunk
                    return
                except Exception as exc:
                    self.error = str(exc)
                    self.last_error = str(exc)
                    _warn_once(type(exc).__name__, "Phonon streaming failed (%s); using fallback.", exc)
        audio = self.synthesize(text, cancel=cancel)
        if audio is not None and len(audio) > 0:
            yield audio


_ENGINE: TTSService | None = None
_ENGINE_LOCK = threading.Lock()


def get_tts_engine() -> TTSService:
    """Return the process-wide TTS service singleton."""
    global _ENGINE
    with _ENGINE_LOCK:
        if _ENGINE is not None:
            return _ENGINE
        try:
            from runtime.models import REGISTRY
            if REGISTRY.is_ready("tts"):
                _ENGINE = REGISTRY.get("tts")
                return _ENGINE
        except Exception:
            pass
        _ENGINE = TTSService()
        _ENGINE.load()
        return _ENGINE


def set_tts_engine(engine: TTSService | None) -> None:
    """Override the global TTS service for testing or application integration."""
    global _ENGINE
    with _ENGINE_LOCK:
        _ENGINE = engine
        if engine is not None:
            try:
                from runtime.models import REGISTRY
                REGISTRY.register("tts", engine)
            except Exception:
                log.exception("Could not register overridden TTS service.")
