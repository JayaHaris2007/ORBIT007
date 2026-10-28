"""Eager, single-pass model loading and warm-up sequence for S.P.I.D.E.Y server startup."""

from __future__ import annotations

import logging
import os
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
        reserved = torch.cuda.memory_reserved(0) / (1024**3)
        total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        free = total - allocated
        return round(allocated, 2), round(free, 2)
    except Exception:
        return 0.0, 0.0


def check_local_model_files(model_dir: str) -> None:
    """Verify local model weights exist before offline loading."""
    offline = os.environ.get("HF_HUB_OFFLINE") == "1"
    if not os.path.exists(model_dir):
        msg = (
            f"Local model directory '{model_dir}' not found. "
            "If downloading for the first time, start with SPIDEY_ALLOW_HUB=1."
        )
        if offline:
            raise FileNotFoundError(msg)
        log.warning(msg)


def load_qwen_eager(
    model_dir: str,
    gpu_limit: str = "5GiB",
    cpu_limit: str = "6GiB",
    generation_lock: Any | None = None,
) -> tuple[Any, Any]:
    """Load Qwen3-1.7B in 4-bit NF4 quantization once, warm up, and register."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.perf_counter()
    check_local_model_files(model_dir)

    log.info("Loading Qwen tokenizer from %s...", model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)

    cuda_available = torch.cuda.is_available()
    if cuda_available:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs = {
            "dtype": "auto",
            "device_map": "auto",
            "max_memory": {0: gpu_limit, "cpu": cpu_limit},
            "quantization_config": quantization_config,
        }
    else:
        model_kwargs = {
            "device_map": "cpu",
            "dtype": torch.float32,
        }

    log.info("Loading Qwen causal language model (cuda=%s)...", cuda_available)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        low_cpu_mem_usage=True,
        local_files_only=True,
        **model_kwargs,
    )

    # 1-token warm-up to compile CUDA graph and initialize buffers
    log.info("Warming up Qwen model (max_new_tokens=1)...")
    prompt = "Hello"
    inputs = tokenizer([prompt], return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    if generation_lock is not None:
        with generation_lock:
            with torch.inference_mode():
                model.generate(**inputs, max_new_tokens=1)
    else:
        with torch.inference_mode():
            model.generate(**inputs, max_new_tokens=1)

    load_time = time.perf_counter() - t0
    alloc_gib, _ = get_gpu_memory_gib()
    log.info(
        "Qwen LLM ready in %.2fs (device=%s, VRAM allocated=%.2f GiB).",
        load_time,
        getattr(model, "device", "cpu"),
        alloc_gib,
    )
    return model, tokenizer, load_time


def load_omnivoice_eager() -> tuple[Any, float]:
    """Load OmniVoice once, warm up with one short synthesis, and register."""
    from voice.tts import OmniVoiceEngine

    t0 = time.perf_counter()
    engine = OmniVoiceEngine()
    engine._load_eager()

    if not engine.available:
        raise RuntimeError(f"OmniVoice failed to initialize: {engine.error}")

    # Warm up with one short phrase (discard audio)
    log.info("Warming up OmniVoice TTS engine...")
    _ = engine.synthesize("Hi.")

    load_time = time.perf_counter() - t0
    alloc_gib, _ = get_gpu_memory_gib()
    log.info(
        "OmniVoice TTS ready in %.2fs (device=%s, dtype=%s, VRAM allocated=%.2f GiB).",
        load_time,
        engine.device,
        engine.dtype,
        alloc_gib,
    )
    return engine, load_time


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


def load_all_models_eager(
    qwen_dir: str,
    gpu_limit: str = "5GiB",
    cpu_limit: str = "6GiB",
    generation_lock: Any | None = None,
) -> None:
    """Load all models sequentially once at server startup. Never crashes the server."""
    log.info("=== BEGINNING SINGLE-PROCESS EAGER MODEL INITIALIZATION ===")
    total_start = time.perf_counter()

    # 1. Qwen LLM
    try:
        model, tokenizer, load_time = load_qwen_eager(
            qwen_dir, gpu_limit=gpu_limit, cpu_limit=cpu_limit, generation_lock=generation_lock
        )
        REGISTRY.register("llm", (model, tokenizer), metrics={"load_time": load_time})
    except Exception as err:
        log.exception("Eager load failed for Qwen LLM: %s", err)
        REGISTRY.fail("llm", err)

    # 2. OmniVoice TTS
    try:
        tts_engine, load_time = load_omnivoice_eager()
        REGISTRY.register("tts", tts_engine, metrics={"load_time": load_time})
    except Exception as err:
        log.exception("Eager load failed for OmniVoice TTS: %s", err)
        REGISTRY.fail("tts", err)

    # 3. SentenceTransformer Embedder
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
