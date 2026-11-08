"""Smoke test for Phonon TTS and its Windows SAPI fallback."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import soundfile as sf
from voice import config
from runtime.loaders import load_phonon_eager


def run_selfcheck() -> dict[str, Any]:
    print("=== Phonon TTS Self-Check Smoke Test ===")
    out_dir = config.TMP_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "selfcheck.wav"

    test_sentence = "Hello! S.P.I.D.E.Y. voice system is online and ready."
    print(f"Target text: {test_sentence!r}")
    print(f"Configured backend: {config.TTS_BACKEND}, voice: {config.PHONON_VOICE}")
    t0 = time.perf_counter()
    engine, load_time = load_phonon_eager()

    if not engine.available:
        print(f"[FAIL] TTS failed to load: {engine.error}")
        return {"status": "failed", "error": engine.error}

    print(f"[LOADED] Backend: {engine.backend}, voice: {engine.voice}, in {load_time:.2f}s")

    t1 = time.perf_counter()
    audio = engine.synthesize(test_sentence)
    gen_time = time.perf_counter() - t1

    if audio is None or len(audio) == 0:
        print("[FAIL] Synthesis produced empty audio")
        return {"status": "failed", "error": "empty audio"}

    audio_duration = len(audio) / config.SAMPLE_RATE
    rtf = gen_time / max(0.001, audio_duration)

    sf.write(str(out_path), audio, config.SAMPLE_RATE)
    print(f"[SYNTHESIS] Generated {audio_duration:.2f}s audio in {gen_time:.2f}s (RTF: {rtf:.3f})")
    print(f"[OUTPUT] Written to {out_path} ({os.path.getsize(out_path)} bytes)")

    return {
        "status": "success",
        "backend": engine.backend,
        "voice": engine.voice,
        "sample_rate": config.SAMPLE_RATE,
        "device": engine.device,
        "dtype": engine.dtype,
        "load_time_s": round(load_time, 2),
        "gen_time_s": round(gen_time, 2),
        "audio_s": round(audio_duration, 2),
        "rtf": round(rtf, 3),
    }


if __name__ == "__main__":
    result = run_selfcheck()
    print("Result summary:", result)
