"""Smoke test for OmniVoice TTS engine loading and single-sentence synthesis."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import soundfile as sf
import torch

from voice import config, tts


def run_selfcheck() -> dict[str, Any]:
    print("=== OmniVoice Self-Check Smoke Test ===")
    out_dir = config.TMP_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "selfcheck.wav"

    test_sentence = "Hello! S.P.I.D.E.Y. voice system is online and ready."
    print(f"Target text: {test_sentence!r}")
    print(f"Configured device: {config.TTS_DEVICE}, dtype: {config.TTS_DTYPE}")

    engine = tts.OmniVoiceEngine(
        model_id=config.TTS_MODEL_ID,
        device=config.TTS_DEVICE,
        dtype=config.TTS_DTYPE,
    )

    t0 = time.perf_counter()
    engine.load()
    load_time = time.perf_counter() - t0

    if not engine.available:
        print(f"[FAIL] OmniVoice failed to load: {engine.error}")
        return {"status": "failed", "error": engine.error}

    print(f"[LOADED] Device used: {engine.device} ({engine.dtype}) in {load_time:.2f}s")
    if engine.device != config.TTS_DEVICE:
        print(f"[FALLBACK] Requested {config.TTS_DEVICE} but engine fell back to {engine.device}. Reason: {engine.error}")

    if torch.cuda.is_available():
        vram_mb = torch.cuda.memory_allocated() / (1024 * 1024)
        peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        print(f"[VRAM] Current allocated: {vram_mb:.1f} MB, Peak: {peak_mb:.1f} MB")
    else:
        print("[VRAM] CUDA not available; running on CPU")

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
        "device": engine.device,
        "dtype": engine.dtype,
        "load_time_s": round(load_time, 2),
        "gen_time_s": round(gen_time, 2),
        "audio_s": round(audio_duration, 2),
        "rtf": round(rtf, 3),
        "vram_allocated_mb": round(vram_mb, 1) if torch.cuda.is_available() else 0,
    }


if __name__ == "__main__":
    result = run_selfcheck()
    print("Result summary:", result)
