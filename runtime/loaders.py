"""Eager, single-pass model loading and warm-up sequence for S.P.I.D.E.Y server startup."""

from __future__ import annotations

import logging
import os
import threading
import time
import warnings
from typing import Any

import torch

from .models import REGISTRY

log = logging.getLogger("spidey.loaders")

# Filter benign pydub warning about missing system ffmpeg
warnings.filterwarnings("ignore", category=RuntimeWarning, module="pydub.utils")


def get_gpu_memory_gib() -> tuple[float, float]:
    """Return (allocated_gib, free_gib) for cuda:0, or (0.0, 0.0) if unavailable."""
    if not torch.cuda.is_available():
        return 0.0, 0.0
    try:
        allocated = torch.cuda.memory_allocated(0) / (1024**3)
        total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        free = total - allocated
        return round(allocated, 2), round(free, 2)
    except Exception:
        return 0.0, 0.0


def load_phonon_eager() -> tuple[Any, float]:
    """Load the CPU-only Phonon engine and retain SAPI as a Windows fallback."""
    from voice import config
    from voice.phonon_engine import PhononEngine
    from voice.tts import TTSService, WindowsSapiEngine

    started = time.perf_counter()
    sapi = WindowsSapiEngine()
    phonon = None
    if config.TTS_BACKEND != "sapi":
        try:
            phonon = PhononEngine(
                model=config.PHONON_MODEL,
                lang=config.PHONON_LANG,
                quant=config.PHONON_QUANT,
                voice=config.PHONON_VOICE,
            ).load()
        except Exception as err:
            log.exception("Phonon TTS failed to initialize; using SAPI fallback: %s", err)
    service = TTSService(phonon=phonon, sapi=sapi)
    load_time = time.perf_counter() - started
    return service, load_time


def load_embedder_eager(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> tuple[Any, float]:
    """Load sentence-transformers embedder on CPU, warm up, and register."""
    from sentence_transformers import SentenceTransformer

    t0 = time.perf_counter()
    log.info("Loading sentence-transformers embedder '%s' on CPU...", model_name)
    offline = os.environ.get("HF_HUB_OFFLINE") == "1"
    embedder = SentenceTransformer(
        model_name,
        device="cpu",
        local_files_only=offline,
    )

    # Warm-up encode
    _ = embedder.encode("warmup", normalize_embeddings=True)
    load_time = time.perf_counter() - t0
    log.info("Sentence embedder ready in %.2fs (device=cpu).", load_time)
    return embedder, load_time


def load_stt_eager() -> Any:
    """Initialize speech recognition adapter and verify availability."""
    from voice.stt import get_stt

    stt_engine = get_stt()
    log.info("STT engine ready: engine=%s, cloud_allowed=%s.", stt_engine.engine, stt_engine.allow_cloud)
    return stt_engine


def check_ollama_running(url: str = "http://127.0.0.1:11434", model_name: str = "qwen3.5:4b", timeout: float = 2.0) -> bool:
    """Check if Ollama server is running and has the requested model available."""
    import json
    import urllib.request
    try:
        req = urllib.request.Request(f"{url.rstrip('/')}/api/tags", headers={"User-Agent": "spidey/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            models = [m.get("name", "") for m in data.get("models", [])]
            return any(
                m == model_name
                or m == f"{model_name}:latest"
                or (":" not in model_name and m.startswith(f"{model_name}:"))
                or m.split(":")[0] == model_name.split(":")[0]
                for m in models
            )
    except Exception:
        return False



def wait_for_ollama(url: str, model_name: str, max_wait: float = 10.0, poll_interval: float = 1.5) -> bool:
    """
    Wait up to `max_wait` seconds for Ollama to become reachable at server startup.
    Polls every `poll_interval` seconds. Returns True if Ollama is up and the model
    is available within the window, False otherwise (caller falls back to local Qwen).
    """
    deadline = time.perf_counter() + max_wait
    while time.perf_counter() < deadline:
        if check_ollama_running(url, model_name, timeout=2.0):
            return True
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            break
        time.sleep(min(poll_interval, remaining))
    return False


def load_all_models_eager(
    generation_lock: Any | None = None,
) -> None:
    """Load all models sequentially once at server startup. Never crashes the server."""
    log.info("=== BEGINNING SINGLE-PROCESS EAGER MODEL INITIALIZATION ===")
    total_start = time.perf_counter()

    # 1. LLM (Ollama only)
    ollama_url = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434").strip()
    ollama_model = os.environ.get("OLLAMA_MODEL", "qwen3.5:4b").strip()

    log.info("Checking for Ollama at %s (model=%s)...", ollama_url, ollama_model)
    ollama_active = wait_for_ollama(ollama_url, ollama_model, max_wait=10.0)

    if ollama_active:
        log.info(
            "Detected running Ollama at %s with model '%s'. Using Ollama backend.",
            ollama_url,
            ollama_model,
        )
        REGISTRY.register(
            "llm",
            "ollama",
            metrics={"backend": "ollama", "model": ollama_model, "load_time": 0.0, "device": "ollama"},
        )
        try:
            import app
            app._ollama_ready = True
        except Exception:
            pass
    else:
        err_msg = (
            f"Ollama is not running at {ollama_url} or model '{ollama_model}' not found. "
            f"Start Ollama with `ollama serve` and pull the model with `ollama pull {ollama_model}`."
        )
        log.warning(err_msg)
        REGISTRY.fail("llm", err_msg)
        try:
            import app
            app._ollama_ready = False
        except Exception:
            pass

    # 2. Phonon TTS (CPU) with Windows SAPI fallback
    try:
        tts_engine, load_time = load_phonon_eager()
        if os.environ.get("TTS_BACKEND", "phonon").strip().lower() != "sapi" and tts_engine.phonon is None:
            REGISTRY.fail("tts", tts_engine.error or "Phonon failed to initialize.")
            log.warning("Windows SAPI fallback remains available after Phonon failure.")
        else:
            REGISTRY.register(
                "tts",
                tts_engine,
                metrics={
                    "backend": tts_engine.backend,
                    "voice": tts_engine.voice,
                    "sample_rate": 24000,
                    "load_time": load_time,
                    "device": tts_engine.device,
                },
            )
            if tts_engine.phonon is not None and hasattr(tts_engine.phonon, "warm_up"):
                threading.Thread(
                    target=tts_engine.phonon.warm_up,
                    daemon=True,
                    name="phonon-warmup",
                ).start()
    except Exception as err:
        log.exception("Eager load failed for Phonon TTS: %s", err)
        REGISTRY.fail("tts", err)

    # 3. SentenceTransformer Embedder
    #
    # This stack also loads native torch/torchaudio components. On Windows,
    # an incompatible binary can terminate the process before Python can
    # handle an exception, so keep it opt-in.
    eager_embedder = os.environ.get("SPIDEY_EAGER_EMBEDDER", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if not eager_embedder:
        message = (
            "SentenceTransformer eager loading is disabled. Set "
            "SPIDEY_EAGER_EMBEDDER=1 after verifying the installed torch build."
        )
        log.warning(message)
        REGISTRY.fail("embedder", message)
    else:
        try:
            embedder, load_time = load_embedder_eager()
            REGISTRY.register("embedder", embedder, metrics={"load_time": load_time})
        except Exception as err:
            log.exception("Eager load failed for SentenceTransformer: %s", err)
            REGISTRY.fail("embedder", err)

    # 4. STT Engine
    try:
        stt_engine = load_stt_eager()
        REGISTRY.register("stt", stt_engine)
    except Exception as err:
        log.exception("Eager load failed for STT: %s", err)
        REGISTRY.fail("stt", err)

    total_time = time.perf_counter() - total_start
    alloc_gib, free_gib = get_gpu_memory_gib()
    status_map = REGISTRY.get_status_summary()

    log.info(
        "All models loaded in %.2fs: llm=%s tts=%s embedder=%s stt=%s | GPU allocated %.2f GiB, free %.2f GiB",
        total_time,
        status_map.get("llm", "unknown"),
        status_map.get("tts", "unknown"),
        status_map.get("embedder", "unknown"),
        status_map.get("stt", "unknown"),
        alloc_gib,
        free_gib,
    )
