"""Configuration constants for the voice module."""

from __future__ import annotations

import os
import shutil
from pathlib import Path


def _int(name: str, default: int, minimum: int = 1) -> int:
    try:
        value = int(os.environ.get(name, "").strip())
    except ValueError:
        return default
    return value if value >= minimum else default


def _float(name: str, default: float, minimum: float = 0.0) -> float:
    try:
        value = float(os.environ.get(name, "").strip())
    except ValueError:
        return default
    return value if value >= minimum else default


def _bool(name: str, default: bool = False) -> bool:
    val = os.environ.get(name, "").strip().lower()
    if not val:
        return default
    return val in {"1", "true", "yes", "on"}


def _parse_id_allowlist(name: str) -> set[int]:
    raw_value = os.environ.get(name, "")
    values: set[int] = set()
    for item in raw_value.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            values.add(int(item))
        except ValueError:
            pass
    return values


# Directory paths
PACKAGE_DIR = Path(__file__).resolve().parent
ROOT_DIR = PACKAGE_DIR.parent
TMP_DIR = PACKAGE_DIR / "tmp"
TMP_DIR.mkdir(parents=True, exist_ok=True)


def purge_voice_tmp() -> None:
    """Purge temporary voice files, preserving .gitignore and .gitkeep."""
    if not TMP_DIR.exists():
        return
    for item in TMP_DIR.iterdir():
        if item.name in {".gitignore", ".gitkeep"}:
            continue
        try:
            if item.is_file() or item.is_symlink():
                item.unlink(missing_ok=True)
            elif item.is_dir():
                shutil.rmtree(item, ignore_errors=True)
        except Exception:
            pass


# TTS Configuration
TTS_BACKEND = os.environ.get("TTS_BACKEND", "phonon").strip().lower() or "phonon"
PHONON_MODEL = os.environ.get("PHONON_MODEL", "hf://gradium/phonon-31m-en").strip()
PHONON_QUANT = os.environ.get("PHONON_QUANT", "q8").strip() or "q8"
PHONON_LANG = os.environ.get("PHONON_LANG", "en").strip() or "en"
PHONON_VOICE = os.environ.get("SPIDEY_TTS_VOICE", "Easton").strip() or "Easton"
TTS_CPU_THREADS = _int("TTS_CPU_THREADS", 4)
TTS_SHARE_QWEN_LOCK = _bool("TTS_SHARE_QWEN_LOCK", False)
MAX_SENTENCE_CHARS = _int("VOICE_MAX_SENTENCE_CHARS", 220)
SAMPLE_RATE = 24000

# STT Configuration
STT_ENGINE = os.environ.get("STT_ENGINE", "whisper_local").strip() or "whisper_local"
STT_ALLOW_CLOUD = _bool("STT_ALLOW_CLOUD", True)
STT_LANGUAGE = os.environ.get("STT_LANGUAGE", "en").strip() or "en"
STT_MODEL = os.environ.get("STT_MODEL", "base").strip() or "base"

# VAD Configuration
VAD_ENERGY_THRESHOLD = _float("VAD_ENERGY_THRESHOLD", 500.0)
VAD_TRAILING_SILENCE_MS = _int("VAD_TRAILING_SILENCE_MS", 700)
VAD_MIN_SPEECH_MS = _int("VAD_MIN_SPEECH_MS", 250)
LIVE_BARGE_IN_THRESHOLD = _float("LIVE_BARGE_IN_THRESHOLD", 1500.0)
LIVE_BARGE_IN_MIN_SPEECH_MS = _int("LIVE_BARGE_IN_MIN_SPEECH_MS", 300)

# Discord Voice Configuration
DISCORD_VOICE_ALLOWED_USER_IDS = _parse_id_allowlist(
    "DISCORD_VOICE_ALLOWED_USER_IDS"
) or _parse_id_allowlist("AI_DISCORD_ALLOWED_USER_IDS")
DISCORD_VOICE_MAX_CHARS = _int("DISCORD_VOICE_MAX_CHARS", 500)
DISCORD_CALL_IDLE_TIMEOUT_SECONDS = _int("DISCORD_CALL_IDLE_TIMEOUT_SECONDS", 600)
