"""Speech-to-Text adapter using SpeechRecognition with offline default and cloud safeguards."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from . import config

log = logging.getLogger("spidey.voice.stt")

# Categorization of SpeechRecognition engines
OFFLINE_ENGINES = {"whisper", "whisper_local", "sphinx", "vosk"}
CLOUD_ENGINES = {
    "google",
    "google_cloud",
    "bing",
    "houndify",
    "ibm",
    "amazon",
    "azure",
}


def list_available_engines() -> dict[str, list[str]]:
    """Inspect SpeechRecognition to find installed recognizers."""
    try:
        import speech_recognition as sr
        methods = [n for n in dir(sr.Recognizer) if n.startswith("recognize_")]
        clean_names = [m.replace("recognize_", "") for m in methods]
        offline = [n for n in clean_names if n in OFFLINE_ENGINES]
        cloud = [n for n in clean_names if n in CLOUD_ENGINES or n not in OFFLINE_ENGINES]
        return {"offline": offline, "cloud": cloud}
    except ImportError:
        return {"offline": ["whisper_local", "sphinx"], "cloud": ["google"]}


class STT:
    """Wrapper around speech_recognition.Recognizer with cloud restrictions and thread-safety."""

    def __init__(
        self,
        engine: str = config.STT_ENGINE,
        language: str = config.STT_LANGUAGE,
        allow_cloud: bool = config.STT_ALLOW_CLOUD,
        model: str = config.STT_MODEL,
    ) -> None:
        self.engine = engine
        self.language = language
        self.allow_cloud = allow_cloud
        self.model = model

    def transcribe(self, pcm16k: bytes) -> str:
        """Transcribe 16 kHz mono 16-bit PCM audio. Synchronous; call via asyncio.to_thread."""
        if not pcm16k:
            return ""

        try:
            import speech_recognition as sr
        except ImportError as e:
            raise RuntimeError("SpeechRecognition package is not installed.") from e

        # Ensure cloud guard is enforced
        is_cloud = self.engine in CLOUD_ENGINES or "google" in self.engine
        if is_cloud and not self.allow_cloud:
            raise RuntimeError(
                f"Cloud STT engine '{self.engine}' is disabled. "
                f"Set STT_ALLOW_CLOUD=true in .env to allow cloud transcription."
            )

        audio = sr.AudioData(bytes(pcm16k), 16000, 2)  # 16 kHz, 16-bit
        recognizer = sr.Recognizer()

        if self.engine in {"whisper_local", "whisper"}:
            return recognizer.recognize_whisper(
                audio, model=self.model, language=self.language
            ).strip()

        if self.engine == "google":
            return recognizer.recognize_google(audio, language=self.language).strip()

        if hasattr(recognizer, f"recognize_{self.engine}"):
            method = getattr(recognizer, f"recognize_{self.engine}")
            return str(method(audio)).strip()

        raise RuntimeError(
            f"STT engine '{self.engine}' is not supported or not available."
        )

    async def transcribe_async(self, pcm16k: bytes) -> str:
        """Non-blocking transcription executed in a worker thread."""
        return await asyncio.to_thread(self.transcribe, pcm16k)


_STT_INSTANCE: STT | None = None


def get_stt() -> STT:
    global _STT_INSTANCE
    if _STT_INSTANCE is None:
        _STT_INSTANCE = STT()
    return _STT_INSTANCE
