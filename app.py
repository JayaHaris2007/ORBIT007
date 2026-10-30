import logging
import secrets
import os
import sys
import argparse
import asyncio
import difflib
import re
import threading
import time
import uuid
import warnings
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

# Ensure `import app` inside discord_bot.py shares this exact module instance when run as __main__
sys.modules.setdefault("app", sys.modules[__name__])

# Filter benign pydub warning about missing system ffmpeg
warnings.filterwarnings("ignore", category=RuntimeWarning, module="pydub.utils")

# Set Hugging Face offline flags unless explicitly opted out with SPIDEY_ALLOW_HUB=1
if os.environ.get("SPIDEY_ALLOW_HUB") != "1":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

try:
    from dotenv import load_dotenv
except ImportError:  # Optional during partial installs.
    load_dotenv = None
else:
    load_dotenv(Path(__file__).resolve().parent / ".env")

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import dateparser
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger
from apscheduler.triggers.interval import IntervalTrigger

CPU_THREADS = 4
CPU_INTEROP_THREADS = 1
GPU_MEMORY_LIMIT = "5GiB"
CPU_MEMORY_LIMIT = "6GiB"
MAX_PROMPT_CHARS = 2000
MAX_NEW_TOKENS = 512
DATABASE_PATH = Path(os.environ.get("ORBIT_DB_DIR", Path(__file__).resolve().parent)) / "assistant.db"

from transformers import AutoModelForCausalLM, AutoTokenizer
import uvicorn
import assistant_tone
import json
from typing import Any, Optional, Dict, List
from pydantic import BaseModel
from fastapi import Body, FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from db import (
    init_db,
    DB_PATH,
    create_reminder,
    list_reminders,
    update_reminder_status,
    delete_reminder,
    save_message,
    get_recent_messages,
)

BASE_DIR = Path(__file__).resolve().parent
TTS_DIR = BASE_DIR / "TTS"
TTS_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_OUTPUT_PATH = TTS_DIR / "audio.wav"
from memory import LOCAL_TIMEZONE, MAX_RETRIEVED_MEMORIES, MemoryStore
from learning import roadmaps as learning_roadmaps, schema as learning_schema
from learning import classifier as learning_classifier
from learning import nudges as learning_nudges
from learning import progress as learning_progress
from learning.router import router as learning_router
from research.router import router as research_router
from voice.router import (
    router as voice_router,
    ws_router as voice_ws_router,
    set_chat_handler as set_voice_chat_handler,
)
from voice.config import purge_voice_tmp
from research.search_engine import (
    SearchFailed,
    SearchNotConfigured,
    search as web_search,
)

MODEL_DIR = Path(__file__).resolve().parent / "Models" / "LLM"
if not (MODEL_DIR / "config.json").is_file():
    raise FileNotFoundError(f"Local model files were not found in {MODEL_DIR}")

tokenizer = None
model = None
cuda_available = False
load_seconds = None
model_load_lock = threading.Lock()


def ensure_model_loaded() -> None:
    global tokenizer, model, cuda_available, load_seconds
    if model is not None and tokenizer is not None:
        return
    with model_load_lock:
        if model is not None and tokenizer is not None:
            return
        from runtime.models import REGISTRY
        if REGISTRY.is_ready("llm"):
            model, tokenizer = REGISTRY.get("llm")
            cuda_available = torch.cuda.is_available()
            metrics = REGISTRY.metrics.get("llm", {})
            load_seconds = metrics.get("load_time", 0.0)
            return
        if REGISTRY.status.get("llm") == "failed":
            raise RuntimeError(f"Qwen LLM failed to load during startup: {REGISTRY.errors.get('llm')}")

        torch.set_num_threads(CPU_THREADS)
        try:
            torch.set_num_interop_threads(CPU_INTEROP_THREADS)
        except RuntimeError as error:
            if "after parallel work has started" not in str(error):
                raise
        cuda_available = torch.cuda.is_available()
        if not cuda_available and os.environ.get("AI_ALLOW_CPU_LLM") != "1":
            raise RuntimeError(
                "CUDA is unavailable. Install a CUDA-enabled PyTorch build, or "
                "set AI_ALLOW_CPU_LLM=1 to explicitly allow slow CPU mode."
            )
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_DIR, local_files_only=True
        )
        if cuda_available:
            try:
                from transformers import BitsAndBytesConfig
                import bitsandbytes
            except ImportError as error:
                raise RuntimeError(
                    "4-bit CUDA loading requires the installed bitsandbytes package."
                ) from error
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            model_load_kwargs = {
                "dtype": "auto",
                "device_map": "auto",
                "max_memory": {0: GPU_MEMORY_LIMIT, "cpu": CPU_MEMORY_LIMIT},
                "quantization_config": quantization_config,
            }
        else:
            model_load_kwargs = {
                "device_map": "cpu",
                "dtype": torch.float32,
            }
        load_started = time.perf_counter()
        try:
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_DIR,
                low_cpu_mem_usage=True,
                local_files_only=True,
                **model_load_kwargs,
            )
        except torch.cuda.OutOfMemoryError as error:
            raise RuntimeError(
                "The quantized model does not fit within the 5 GiB GPU and 6 GiB "
                "RAM placement limits. Close other GPU applications and try again."
            ) from error
        load_seconds = time.perf_counter() - load_started
        from runtime.models import REGISTRY
        REGISTRY.register("llm", (model, tokenizer))
        print_device_diagnostics()


def print_device_diagnostics() -> None:
    device_map = getattr(model, "hf_device_map", {}) if model else {}
    distribution = {"cuda": 0, "cpu": 0, "disk": 0, "other": 0}
    for device in device_map.values():
        device_name = str(device)
        if device_name.startswith("cuda"):
            distribution["cuda"] += 1
        elif device_name == "cpu":
            distribution["cpu"] += 1
        elif device_name == "disk":
            distribution["disk"] += 1
        else:
            distribution["other"] += 1
    print("\n--- MODEL DEVICE MAP ---")
    print(device_map)
    print("Device-map distribution:", distribution)
    print("Model loading time:", f"{load_seconds:.2f}s" if load_seconds else "n/a")
    print("\n--- CUDA STATUS ---")
    print("CUDA available:", cuda_available)
    if cuda_available:
        print("GPU:", torch.cuda.get_device_name(0))
        print(
            "GPU memory allocated:",
            round(torch.cuda.memory_allocated(0) / 1024**3, 2),
            "GiB",
        )
        print(
            "GPU memory reserved:",
            round(torch.cuda.memory_reserved(0) / 1024**3, 2),
            "GiB",
        )
    if distribution["disk"]:
        print("WARNING: disk offload is active.")


generation_lock = threading.Lock()
memory_store = MemoryStore(DATABASE_PATH)
reminder_scheduler = BackgroundScheduler(timezone="Asia/Kolkata")

DEFAULT_SYSTEM_PROMPT = (
    "You are S.P.I.D.E.Y., Jaya's personal AI assistant, technical co-pilot, "
    "and everyday companion. Talk naturally like a smart, friendly, attentive "
    "person. Be confident, practical, curious, and occasionally witty. Use "
    "natural contractions, simple English, and a conversational tone. "
    "Never sound robotic or overly formal. "

    "Keep routine replies concise, but explain technical topics in depth when "
    "Jaya asks. Avoid canned praise, unnecessary introductions, repetitive "
    "conclusions, and narrating simple actions. Respond directly to what "
    "Jaya actually says and understand short follow-ups using conversation "
    "context. "

    "Jaya is a Computer Science student, developer, and technology enthusiast "
    "who enjoys building real-world projects, exploring AI, and experimenting "
    "with new technologies. Adapt explanations to his technical knowledge. "
    "Help with programming, debugging, web development, databases, AI models, "
    "system architecture, and software engineering. Prefer practical solutions, "
    "working code, and clear explanations over unnecessary theory. "

    "Act as a development partner, not just a chatbot. Help plan, build, "
    "debug, optimize, and improve projects. Respect the existing tech stack "
    "and architecture unless a change is genuinely beneficial. Explain "
    "important trade-offs and recommend the best approach when appropriate. "
    "Never claim code was tested or executed unless it actually was. "

    "Help manage tasks, todos, habits, reminders, schedules, learning goals, "
    "research, and everyday activities using available application tools. "
    "Use conversation context and connected data whenever available. Local "
    "habit, task, and reminder details shown in the conversation are available "
    "to you; never ask for information that was already provided. Ask one "
    "focused question if essential information is missing. Never claim an "
    "action succeeded unless the application confirms it. "

    "Support Jaya's learning and growth as a developer. Explain difficult "
    "concepts clearly, connect new ideas to existing knowledge, correct "
    "mistakes respectfully, and suggest useful next steps when appropriate. "
    "Be encouraging without giving empty praise. Challenge ideas constructively "
    "when doing so helps achieve a better result. "

    "Use saved notes and conversation history to understand the user, "
    "preferences, and ongoing projects. Saved notes may describe either Jaya "
    "or you. When a note describes Jaya, speak about him as 'you' and never "
    "present his experiences, preferences, or plans as your own. Never invent "
    "personal facts, memories, or previous decisions. Use personal context "
    "only when it is relevant. "

    "Provide accurate, useful, and honest answers. Distinguish facts from "
    "assumptions and acknowledge uncertainty. Web search runs through the "
    "application, not inside your reply. If Jaya wants live web results, tell "
    "him to send 'web search <topic>'. Never claim or imply that you searched, "
    "are searching, or that a search failed unless the application provides "
    "actual search results. "

    "Be action-oriented, resourceful, reliable, and technically capable. "
    "Don't overcomplicate simple requests or repeat instructions already "
    "received. Use light humor naturally without forcing it. Think through "
    "problems carefully, but communicate clearly and directly. "

    "Your identity is S.P.I.D.E.Y. You are Jaya's dependable personal assistant, "
    "technical co-pilot, and companion in learning and development. Help him "
    "think clearly, build better projects, solve problems, and get things done. "

    "You cannot create, complete, or delete tasks, habits, or reminders "
    "yourself. Only the application can. Never say you did. "

    "Answer in English unless Jaya requests another language."
)


def _push_reminder_alert(reminder: dict[str, Any], content: str) -> None:
    if MAIN_LOOP is None or not ws_manager.active:
        return
    async def _go() -> None:
        try:
            audio_url = await generate_tts_url(content)
            await ws_manager.broadcast({
                "type": "reminder_alert",
                "reminder_id": reminder["id"],
                "text": content,
                "audio_url": audio_url,
                "reminders": _live_reminders(),
            })
        except Exception as exc:
            logging.getLogger("orbit007").warning("Failed in _push_reminder_alert: %s", exc)
    try:
        asyncio.run_coroutine_threadsafe(_go(), MAIN_LOOP)
    except Exception as exc:
        logging.getLogger("orbit007").warning("Failed to schedule _push_reminder_alert: %s", exc)


def _run_reminder(reminder_id: str) -> None:
    reminder = memory_store.get_reminder(reminder_id)
    if not reminder or not reminder["active"] or not reminder["chat_id"]:
        return
    # Written locally so web and Discord both get the same warm, instant nudge.
    content = assistant_tone.reminder_fired(
        reminder["text"], name=_user_name(_user_fact_memories())
    )
    memory_store.add_notification(reminder["chat_id"], reminder_id, content)
    try:
        _push_reminder_alert(reminder, content)
    except Exception as exc:
        logging.getLogger("orbit007").warning("Push reminder alert error: %s", exc)
    if not reminder.get("recurrence"):
        memory_store.cancel_reminder(reminder_id)


def _recommend_next_task(
    chat_id: str, completed_task_title: str | None = None
) -> None:
    """Use the LLM to recommend the next task, delivered as a notification."""
    if not chat_id:
        return
    tasks = memory_store.list_tasks(include_completed=False)
    current_time = datetime.now().astimezone().strftime("%d %B %Y, %I:%M %p")

    if not tasks:
        if completed_task_title:
            llm_prompt = (
                "Write one short, celebratory message as S.P.I.D.E.Y, a warm personal "
                "assistant. The user just finished their last remaining task. "
                "Congratulate them and suggest they relax or plan new goals. "
                "Do not mention AI, prompts, or instructions. Return only the "
                "message text, no quotes.\n"
                f"Completed task: {completed_task_title}\n"
                f"Current time: {current_time}"
            )
        else:
            return
    else:
        task_list = "\n".join(
            f"- [{t['priority']}] {t['title']}" for t in tasks[:5]
        )
        if completed_task_title:
            llm_prompt = (
                "Write one short, encouraging message as S.P.I.D.E.Y, a warm personal "
                "assistant. The user just completed a task. Acknowledge it briefly, "
                "then recommend the next task they should focus on — mention its "
                "name and priority. Be specific, motivating, and concise. "
                "Do not mention AI, prompts, or instructions. Return only the "
                "message text, no quotes.\n"
                f"Just completed: {completed_task_title}\n"
                f"Remaining tasks (highest priority first):\n{task_list}\n"
                f"Current time: {current_time}"
            )
        else:
            llm_prompt = (
                "Write one short, friendly message as S.P.I.D.E.Y, a warm personal "
                "assistant. The user has pending tasks. Gently suggest the "
                "top-priority task they should focus on — mention its name and "
                "priority level. Be motivating and brief. "
                "Do not mention AI, prompts, or instructions. Return only the "
                "message text, no quotes.\n"
                f"Pending tasks (highest priority first):\n{task_list}\n"
                f"Current time: {current_time}"
            )

    try:
        content = generate_response(llm_prompt, 64).strip()
        if not content:
            raise ValueError("Empty task recommendation.")
    except Exception as error:
        print(f"Task recommendation generation failed: {error}")
        if not tasks:
            content = (
                f"You've completed \"{completed_task_title}\" and cleared all "
                f"your tasks. Well done!"
            )
        else:
            next_task = tasks[0]
            if completed_task_title:
                content = (
                    f"Done with \"{completed_task_title}\"! "
                    f"Next up: {next_task['title']} "
                    f"({next_task['priority']} priority)."
                )
            else:
                content = (
                    f"Your top task right now is: {next_task['title']} "
                    f"({next_task['priority']} priority). Ready when you are!"
                )

    coach_id = f"task-coach-{uuid.uuid4()}"
    memory_store.add_notification(chat_id, coach_id, content)


def _schedule_reminder(reminder: dict[str, Any]) -> None:
    run_at: datetime | None = None
    try:
        run_at = datetime.fromisoformat(reminder["run_at"])
        if run_at.tzinfo is None:
            run_at = run_at.replace(tzinfo=LOCAL_TIMEZONE)
    except (TypeError, ValueError):
        run_at = dateparser.parse(
            reminder["run_at"],
            settings={"RETURN_AS_TIMEZONE_AWARE": True, "TIMEZONE": "Asia/Kolkata"},
        )
    if run_at is None:
        print(f"Skipping reminder {reminder['id']}: invalid run time.")
        return
    recurrence = (reminder.get("recurrence") or "").lower().strip()
    if recurrence:
        match = re.fullmatch(
            r"every\s+(?:(\d+)\s+)?(minute|hour|day|week)s?", recurrence
        )
        if not match:
            raise ValueError(
                "Recurrence must look like 'every day' or 'every 2 hours'."
            )
        amount, unit = int(match.group(1) or 1), match.group(2)
        reminder_scheduler.add_job(
            _run_reminder,
            IntervalTrigger(**{f"{unit}s": amount}, timezone="Asia/Kolkata"),
            args=[reminder["id"]],
            id=reminder["id"],
            replace_existing=True,
        )
        return
    now = datetime.now(run_at.tzinfo)
    if run_at <= now:
        if now - run_at > timedelta(minutes=30):
            memory_store.cancel_reminder(reminder["id"])
            return
        print(f"Delivering missed reminder {reminder['id']}.")
        run_at = now + timedelta(seconds=5)
    reminder_scheduler.add_job(
        _run_reminder,
        DateTrigger(run_date=run_at, timezone="Asia/Kolkata"),
        args=[reminder["id"]],
        id=reminder["id"],
        replace_existing=True,
    )


def _validate_recurrence(recurrence: str | None) -> None:
    if recurrence is None or not recurrence.strip():
        return
    if not re.fullmatch(
        r"every\s+(?:(\d+)\s+)?(minute|hour|day|week)s?",
        recurrence.strip().lower(),
    ):
        raise ValueError(
            "Recurrence must look like 'every day' or 'every 2 hours'."
        )


def _restore_reminders(provider: str = "local") -> None:
    """Schedule saved reminders for one chat provider."""
    for reminder in memory_store.list_reminders():
        session = memory_store.get_external_session(reminder["chat_id"])
        chat_provider = session["provider"] if session else "local"
        if chat_provider != provider:
            continue
        try:
            _schedule_reminder(reminder)
        except Exception as error:
            print(f"Skipping reminder {reminder['id']}: {error}")
    if not reminder_scheduler.running:
        reminder_scheduler.start()


DEFAULT_CHAT_ID = "default"


def _ensure_chat_exists(chat_id: str) -> None:
    if not memory_store.chat_exists(chat_id):
        now_iso = datetime.now(timezone.utc).isoformat()
        with memory_store.connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (chat_id, "Default Chat", now_iso, now_iso),
            )


def _reminder_view(r: dict[str, Any]) -> dict[str, Any]:
    """MemoryStore row -> the shape the frontend already expects (legacy keys kept)."""
    return {
        "id": r["id"],
        "title": r["text"],
        "text": r["text"],
        "remind_at": r["run_at"],
        "run_at": r["run_at"],
        "recurrence": r.get("recurrence"),
        "completed": 0 if r.get("active", 1) else 1,
        "chat_id": r.get("chat_id"),
    }


def _live_reminders() -> list[dict[str, Any]]:
    return [_reminder_view(r) for r in memory_store.list_reminders()]


def _create_reminder_unified(
    title: str,
    when: str,
    recurrence: str | None = None,
    chat_id: str | None = None,
) -> dict[str, Any]:
    _validate_recurrence(recurrence)
    parsed = _parse_reminder_time(when)
    if parsed is None:
        raise ValueError(f"I could not understand the reminder time: {when!r}.")
    target_chat = chat_id or DEFAULT_CHAT_ID
    _ensure_chat_exists(target_chat)
    row = memory_store.create_reminder(
        title, parsed.isoformat(), recurrence, target_chat
    )
    _schedule_reminder(row)
    return _reminder_view(row)


def _cancel_reminder_unified(reminder_id: str) -> bool:
    ok = memory_store.cancel_reminder(reminder_id)
    try:
        reminder_scheduler.remove_job(reminder_id)
    except Exception:  # job already fired or never existed
        pass
    return ok


def _migrate_legacy_reminders() -> int:
    from db import list_reminders as legacy_list, update_reminder_status as legacy_done
    moved = 0
    for r in legacy_list(include_completed=False):
        try:
            _create_reminder_unified(r["title"], r["remind_at"])
        except Exception as err:
            logging.getLogger("orbit007").warning("legacy reminder %s skipped: %s", r.get("id"), err)
            continue
        legacy_done(r["id"], True)  # never migrate twice
        moved += 1
    return moved


def _parse_reminder_time(value: str) -> datetime | None:
    rel_match = re.fullmatch(
        r"in\s+(\d+)\s*(seconds?|secs?|minutes?|mins?|hours?|hrs?|days?|weeks?)",
        value.strip(),
        re.IGNORECASE,
    )
    if rel_match:
        amount = int(rel_match.group(1))
        unit = rel_match.group(2).lower()
        now = datetime.now(LOCAL_TIMEZONE)
        if unit.startswith("sec"):
            return now + timedelta(seconds=amount)
        if unit.startswith("min"):
            return now + timedelta(minutes=amount)
        if unit.startswith("h"):
            return now + timedelta(hours=amount)
        if unit.startswith("d"):
            return now + timedelta(days=amount)
        if unit.startswith("w"):
            return now + timedelta(weeks=amount)

    day_part_times = {
        "morning": (6, 30),
        "afternoon": (12, 30),
        "evening": (18, 30),
        "night": (21, 0),
    }
    day_part_match = re.search(
        r"\b(morning|afternoon|evening|night)\b", value, re.IGNORECASE
    )
    settings = {
        "TIMEZONE": "Asia/Kolkata",
        "RETURN_AS_TIMEZONE_AWARE": True,
        "PREFER_DATES_FROM": "future",
    }
    if not day_part_match:
        parsed_base = dateparser.parse(value, settings=settings)
        if parsed_base is None:
            return None
        now = datetime.now(parsed_base.tzinfo)
        has_explicit_date = bool(
            re.search(
                r"\b(tomorrow|tmrw|today|tonight|next|monday|tuesday|wednesday|thursday|friday|saturday|sunday|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|\d{1,2}[/-]\d{1,2})\b",
                value,
                re.IGNORECASE,
            )
        )
        if not has_explicit_date and parsed_base <= now:
            parsed_base += timedelta(days=1)
        return parsed_base

    day_part = day_part_match.group(1).lower()
    date_text = re.sub(
        r"\b(morning|afternoon|evening|night)\b", "", value,
        count=1, flags=re.IGNORECASE,
    ).strip()
    if date_text.lower() == "next":
        date_text = "tomorrow"
    if not date_text:
        date_text = "today"
    settings_with_future = {**settings, "PREFER_DATES_FROM": "future"}
    parsed_date = dateparser.parse(date_text, settings=settings_with_future)
    if parsed_date is None:
        return None
    hour, minute = day_part_times[day_part]
    parsed = parsed_date.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if parsed <= datetime.now(parsed.tzinfo) and date_text == "today":
        parsed += timedelta(days=1)
    return parsed


_REMINDER_CREATE_PATTERN = (
    r"(?:set|create|add|make|schedule)\s+"
    r"(?:(?:a|an|another|one\s+more|new)\s+)?reminder"
)


def _reminder_result(
    text: str, when: str, recurrence: str | None, *, relative: bool = False
) -> tuple[str, str, str | None]:
    """Resolve the time part of a reminder request, or explain what is unclear."""
    parsed = _parse_reminder_time(f"in {when}" if relative else when)
    if parsed is None:
        raise ValueError(f"I could not understand the reminder time: {when}.")
    return text.strip(), parsed.isoformat(), recurrence.strip() if recurrence else None


def _parse_chat_reminder(prompt: str) -> tuple[str, str, str | None] | None:
    normalized_prompt = _normalize_command(prompt)
    create = _REMINDER_CREATE_PATTERN

    # "remind me in 10 minutes to stretch"
    match = re.fullmatch(
        r"remind me in (.+?) to (.+?)(?:,\s*(every .+))?",
        normalized_prompt,
        re.IGNORECASE,
    )
    if match:
        when, text, recurrence = match.groups()
        return _reminder_result(text, when, recurrence, relative=True)

    # "set another reminder in 10 minutes for tea"
    match = re.fullmatch(
        rf"{create} in "
        r"(\d+\s*(?:seconds?|secs?|minutes?|mins?|hours?|hrs?|days?|weeks?))\s+"
        r"(?:(?:for|to)\s+)?(.+?)(?:,\s*(every .+))?",
        normalized_prompt,
        re.IGNORECASE,
    )
    if match:
        when, text, recurrence = match.groups()
        return _reminder_result(text, when, recurrence, relative=True)

    # "set reminder 10 minutes for tea"
    match = re.fullmatch(
        rf"{create}\s+"
        r"(\d+\s*(?:seconds?|secs?|minutes?|mins?|hours?|hrs?|days?|weeks?))"
        r"\s+(?:for|to)\s+(.+?)(?:,\s*(every .+))?",
        normalized_prompt,
        re.IGNORECASE,
    )
    if match:
        when, text, recurrence = match.groups()
        return _reminder_result(text, when, recurrence, relative=True)

    # "remind me 20 minutes to stretch"
    match = re.fullmatch(
        r"remind me (.+?) to (.+?)(?:,\s*(every .+))?",
        normalized_prompt,
        re.IGNORECASE,
    )
    if match:
        when, text, recurrence = match.groups()
        return _reminder_result(text, when, recurrence)

    # "remind me to stretch in 20 minutes"
    match = re.fullmatch(
        r"remind me to (.+?) in (.+?)(?:,\s*(every .+))?",
        normalized_prompt,
        re.IGNORECASE,
    )
    if match:
        text, when, recurrence = match.groups()
        return _reminder_result(text, when, recurrence, relative=True)

    # "set a reminder in 20 minutes for tea"
    match = re.fullmatch(
        rf"{create} in (.+?) (?:for|to) (.+?)(?:,\s*(every .+))?",
        normalized_prompt,
        re.IGNORECASE,
    )
    if match:
        when, text, recurrence = match.groups()
        return _reminder_result(text, when, recurrence, relative=True)

    # "add a reminder for water in 30 minutes"
    match = re.fullmatch(
        rf"{create} (?:for|about) (.+?) in (.+?)(?:,\s*(every .+))?",
        normalized_prompt,
        re.IGNORECASE,
    )
    if match:
        text, when, recurrence = match.groups()
        return _reminder_result(text, when, recurrence, relative=True)

    # "remind me to sleep at 11pm" / "remind me about dinner at 8pm"
    match = re.fullmatch(
        r"remind me (?:to|about) (.+?) at (.+?)(?:,\s*(every .+))?",
        normalized_prompt,
        re.IGNORECASE,
    )
    if match:
        text, when, recurrence = match.groups()
        return _reminder_result(text, when, recurrence)

    # "set another reminder at 23:10 for sleeping" — time first, then topic.
    match = re.fullmatch(
        rf"{create} (?:at|on|for|by) (.+?) (?:for|to) "
        r"(.+?)(?:,\s*(every .+))?",
        normalized_prompt,
        re.IGNORECASE,
    )
    if match:
        when, text, recurrence = match.groups()
        return _reminder_result(text, when, recurrence)

    # "set a reminder for sleeping at 11pm" — topic first, then time.
    match = re.fullmatch(
        rf"{create} (?:for|about) (.+?) (?:at|by) "
        r"(.+?)(?:,\s*(every .+))?",
        normalized_prompt,
        re.IGNORECASE,
    )
    if match:
        text, when, recurrence = match.groups()
        return _reminder_result(text, when, recurrence)

    # "set a timer for 10 minutes [for pasta]" / "timer for 5 mins"
    match = re.fullmatch(
        r"(?:(?:set|start|create|add)\s+(?:a\s+)?)?timer\s+(?:for\s+)?"
        r"(\d+\s*(?:seconds?|secs?|minutes?|mins?|hours?|hrs?))"
        r"(?:\s+(?:for|to|about)\s+(.+))?",
        normalized_prompt,
        re.IGNORECASE,
    )
    if match:
        when, text = match.groups()
        label = text.strip() if text else f"Timer ({when.strip()})"
        return _reminder_result(label, when, None, relative=True)

    # "set an alarm for 7am [to wake up]" / "wake me up at 6:30am"
    match = re.fullmatch(
        r"(?:(?:set|create|add)\s+(?:an?\s+)?alarm\s+(?:for|at)\s+(.+?)"
        r"(?:\s+(?:for|to|about)\s+(.+))?"
        r"|wake\s+me(?:\s+up)?\s+(?:at|in)\s+(.+?)(?:\s+to\s+(.+))?)",
        normalized_prompt,
        re.IGNORECASE,
    )
    if match:
        g1, g2, g3, g4 = match.groups()
        when = (g1 or g3).strip()
        label = (g2 or g4 or ("Wake up" if g3 else "Alarm")).strip()
        is_rel = bool(re.match(r"^\d+\s*(?:seconds?|secs?|minutes?|mins?|hours?|hrs?)\b", when, re.IGNORECASE))
        return _reminder_result(label, when, None, relative=is_rel)

    # "ping me in 15 minutes to check the build"
    match = re.fullmatch(
        r"(?:ping|notify|alert|nudge)\s+me\s+in\s+(.+?)\s+(?:to|about|for)\s+(.+)",
        normalized_prompt,
        re.IGNORECASE,
    )
    if match:
        when, text = match.groups()
        return _reminder_result(text, when, None, relative=True)

    return None


def _normalize_command(prompt: str) -> str:
    """Strip greetings, politeness, conversational lead-ins, and trailing punctuation."""
    normalized = re.sub(r"\s+", " ", prompt.strip()).rstrip(".!?")
    normalized = re.sub(
        r"^(?:hey\s+|hi\s+|hello\s+)?(?:spidey|s\.p\.i\.d\.e\.y)[,\s]+",
        "",
        normalized,
        flags=re.IGNORECASE,
    )
    # Strip conversational filler prefixes ("Now I am...", "Okay please...", "Alright, ...")
    # while preserving short follow-up phrases like "now" or "okay now".
    if len(normalized.split()) > 2:
        normalized = re.sub(
            r"^(?:(?:now|okay|ok|alright|so|well|actually|hey)\b[,\s]*)+",
            "",
            normalized,
            flags=re.IGNORECASE,
        )
    normalized = re.sub(
        r"^(?:(?:please|kindly)\s+|(?:can|could|would)\s+you\s+)+",
        "",
        normalized,
        flags=re.IGNORECASE,
    )
    return normalized.strip()


def _reminder_clarification(prompt: str) -> str | None:
    """Ask for the missing detail instead of letting the LLM invent a reminder."""
    normalized = _normalize_command(prompt)
    create_pattern = rf"^(?:{_REMINDER_CREATE_PATTERN}|remind\s+me)\b"
    if not re.match(create_pattern, normalized, re.IGNORECASE):
        return None
    has_time = bool(
        re.search(
            r"\b(?:in\s+\d|at\s+\d|tomorrow|tonight|today|noon|midnight|"
            r"next\s+\w+|morning|afternoon|evening|night|every\s+\w+|"
            r"\d{1,2}(?::\d{2})?\s*(?:am|pm)|"
            r"\d+\s*(?:seconds?|secs?|minutes?|mins?|hours?|hrs?|days?|weeks?))\b",
            normalized,
            re.IGNORECASE,
        )
    )
    if has_time:
        return "What should I remind you about?"
    topic = re.sub(
        rf"^(?:{_REMINDER_CREATE_PATTERN}|remind\s+me)\s*",
        "",
        normalized,
        flags=re.IGNORECASE,
    ).strip()
    if not topic:
        return "What should I remind you about, and when?"
    if re.match(r"^(?:to|about)\s+", topic, re.IGNORECASE):
        return f"When should I remind you {topic}?"
    topic = re.sub(r"^for\s+", "", topic, flags=re.IGNORECASE).strip()
    return f"When should I remind you about {topic}?"


_WEB_SEARCH_PATTERN = re.compile(
    r"^(?:web\s*search|search\s+(?:the\s+)?(?:web|internet|online)"
    r"|(?:google|bing|duckduckgo)|browse\s+(?:the\s+)?(?:web|internet))"
    r"(?:\s+(?:for|about|on))?\s+(?P<query>.+)$",
    re.IGNORECASE,
)
_WEB_SEARCH_SUFFIX_PATTERN = re.compile(
    r"^(?P<query>.+?)\s+(?:on\s+(?:the\s+)?(?:web|internet|google)|"
    r"in\s+web\s*search|with\s+web\s*search|via\s+web\s*search|"
    r"search\s+(?:the\s+)?(?:web|online)|web\s*search)$",
    re.IGNORECASE,
)
_LATEST_NEWS_SEARCH_PATTERN = re.compile(
    r"^(?:what\s+(?:are|is)\s+(?:the\s+)?)?(?:latest|breaking|current|today'?s)\s+"
    r"(?:news|headlines|updates)\s+(?:in|about|on|for|from)\s+(?P<query>.+)$",
    re.IGNORECASE,
)
# "search for india sports news" and "look up the score" mean the same thing.
_PLAIN_SEARCH_PATTERN = re.compile(
    r"^(?:search|look\s+(?:it\s+|this\s+)?up)\s+(?:for\s+|about\s+|on\s+)?(?P<query>.+)$",
    re.IGNORECASE,
)
# Keep local data queries local: "search my tasks", "search reminders", ...
_LOCAL_SEARCH_TARGET = re.compile(
    r"^(?:(?:my|the|a|an|all)\s+)*(?:tasks?|to[\s-]?dos?|habits?|reminders?|"
    r"memor(?:y|ies)|notes?|chats?|conversations?|history|lists?|productivity)\b",
    re.IGNORECASE,
)


def _parse_web_search_query(prompt: str) -> str | None:
    """Return the query of a search request, else None."""
    normalized = _normalize_command(prompt)
    match = _WEB_SEARCH_PATTERN.match(normalized)
    if not match:
        match = _WEB_SEARCH_SUFFIX_PATTERN.match(normalized)
        if match:
            cleaned = re.sub(
                r"^(?:what\s+(?:are|is)\s+(?:the\s+)?|tell\s+me\s+(?:about\s+)?|"
                r"find\s+(?:me\s+)?|show\s+(?:me\s+)?)",
                "",
                match.group("query").strip(),
                flags=re.IGNORECASE,
            )
            return cleaned.strip(" \"'") if len(cleaned.strip(" \"'")) >= 2 else None
    if not match:
        match = _LATEST_NEWS_SEARCH_PATTERN.match(normalized)
        if match:
            topic = match.group("query").strip(" \"'")
            return f"latest news in {topic}" if len(topic) >= 2 else None
    if not match:
        match = _PLAIN_SEARCH_PATTERN.match(normalized)
        if not match or _LOCAL_SEARCH_TARGET.match(match.group("query")):
            return None
    query = match.group("query").strip(" \"'")
    return query if len(query) >= 2 else None


def _search_snippet(text: str, limit: int = 200) -> str:
    # Page text often arrives with scraped markdown headings and nav crumbs.
    text = re.sub(r"[#*_>`]+", " ", text or "")
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    return f"{text[:limit].rsplit(' ', 1)[0]}…"


def _web_search_reply(query: str) -> tuple[str, str]:
    """Run one live search and format the hits as chat-friendly markdown."""
    try:
        sources = web_search(query, max_results=5, search_depth="basic")
    except SearchNotConfigured as error:
        print(f"Web search unavailable: {error}")
        return (
            "I can search the web, but the search key is not set yet. Add "
            "TAVILY_API_KEY to the .env file next to app.py, restart the server, "
            "and ask me again.",
            "web-search-unconfigured",
        )
    except SearchFailed as error:
        print(f"Web search failed for {query!r}: {error}")
        return (
            "I reached for the web but the search provider didn't answer. "
            "Give it another try in a moment.",
            "web-search-failed",
        )
    if not sources:
        return (
            f"I searched for **{query}** but nothing solid came back. "
            "Want to try different wording?",
            "web-search",
        )
    lines = [f"Here's what I found on **{query}**:", ""]
    for source in sources:
        snippet = _search_snippet(source.get("content", ""))
        lines.append(
            f"{source['id']}. [{source['title']}]({source['url']}) — {source['domain']}"
        )
        if snippet:
            lines.append(f"> {snippet}")
        lines.append("")
    lines.append("Want me to dig into one of these?")
    return "\n".join(lines).strip(), "web-search"


def _parse_task_create(prompt: str) -> tuple[str, str] | None:
    normalized = _normalize_command(prompt)
    match = re.fullmatch(
        r"(?:add|create|new)\s+(?:a\s+)?(?:new\s+)?(?:task|to[\s-]?do)"
        r"(?:\s+(?:to|for|in|about))*"
        r"(?:\s+(?:called|named))?"
        r"(?:\s*[:\-]\s*|\s+)(.+?)"
        r"(?:\s+(?:with\s+)?priority\s+(low|medium|high)"
        r"|\s+(?:with\s+)?(low|medium|high)\s+priority)?",
        normalized,
        re.IGNORECASE,
    )
    if not match:
        # Shorthand prefix: "todo: buy milk" or "task: finish slides high priority"
        match = re.fullmatch(
            r"(?:task|to[\s-]?do)\s*[:\-]\s*(.+?)"
            r"(?:\s+(?:with\s+)?priority\s+(low|medium|high)"
            r"|\s+(?:with\s+)?(low|medium|high)\s+priority)?",
            normalized,
            re.IGNORECASE,
        )
    if not match:
        # Natural list phrasing: "add buy milk to my tasks" / "put finish report on my todo list"
        match = re.fullmatch(
            r"(?:add|put)\s+(.+?)\s+(?:to|on|into)\s+(?:my\s+|the\s+)?"
            r"(?:tasks?|task\s+list|to[\s-]?dos?|to[\s-]?do\s+list)"
            r"(?:\s+(?:with\s+)?priority\s+(low|medium|high)"
            r"|\s+(?:with\s+)?(low|medium|high)\s+priority)?",
            normalized,
            re.IGNORECASE,
        )
    if not match:
        return None
    title = match.group(1).strip(" :-'\"")
    if not title:
        return None
    priority = match.group(2) or match.group(3) or "medium"
    return title, priority.lower()


def _fuzzy_best_match(
    query: str, items: list[dict[str, Any]], key: str, threshold: float = 0.76
) -> dict[str, Any] | None:
    """Return a single fuzzy match when the user has a minor typo (e.g. garder vs garden)."""
    if len(query) < 3 or not items:
        return None
    scored: list[tuple[float, dict[str, Any]]] = []
    for item in items:
        candidate = re.sub(r"\s+", " ", str(item.get(key, "")).strip().lower())
        ratio = difflib.SequenceMatcher(None, query, candidate).ratio()
        if ratio >= threshold:
            scored.append((ratio, item))
    if not scored:
        return None
    scored.sort(key=lambda pair: pair[0], reverse=True)
    if len(scored) > 1 and abs(scored[0][0] - scored[1][0]) < 0.05:
        return None
    return scored[0][1]


def _find_task_reference(
    reference: str, include_completed: bool = True
) -> dict[str, Any] | None:
    """Resolve '#12', '12', or a unique (partial or fuzzy) task title."""
    cleaned = reference.strip().strip("#").strip()
    if not cleaned:
        return None
    tasks = memory_store.list_tasks(include_completed=True)
    if not include_completed:
        tasks = [task for task in tasks if not task["completed"]]
    if cleaned.isdigit():
        task_id = int(cleaned)
        return next((task for task in tasks if task["id"] == task_id), None)

    normalized = re.sub(r"\s+", " ", cleaned.lower())
    exact_matches = [
        task for task in tasks if task["title"].strip().lower() == normalized
    ]
    if len(exact_matches) == 1:
        return exact_matches[0]
    partial_matches = [
        task for task in tasks if normalized in task["title"].strip().lower()
    ]
    if len(partial_matches) == 1:
        return partial_matches[0]
    if len(partial_matches) > 1:
        raise ValueError(
            f"More than one task matches '{reference}'. Use 'task <number>'."
        )
    requested_words = set(re.findall(r"[a-z0-9]+", normalized))
    word_matches = [
        task for task in tasks
        if requested_words
        and requested_words.issubset(
            set(re.findall(r"[a-z0-9]+", task["title"].lower()))
        )
    ]
    if len(word_matches) == 1:
        return word_matches[0]
    if len(word_matches) > 1:
        raise ValueError(
            f"More than one task matches '{reference}'. Use 'task <number>'."
        )
    return _fuzzy_best_match(normalized, tasks, "title")


def _format_task_list(tasks: list[dict[str, Any]], completed: bool = False) -> str:
    if not tasks:
        if completed:
            return "You haven't completed any tasks yet."
        return "You're all caught up—there's nothing waiting on your task list."
    if completed:
        lines = "\n".join(f"- #{task['id']} **{task['title']}**" for task in tasks)
        return f"Here are your completed tasks:\n{lines}"
    if len(tasks) == 1:
        task = tasks[0]
        return (
            f"You still have **{task['title']}** (#{task['id']}, "
            f"{task['priority']} priority) on your list. Want to tackle that now?"
        )
    if len(tasks) == 2:
        first, second = tasks[0], tasks[1]
        return (
            f"You've got **{first['title']}** (#{first['id']}) and "
            f"**{second['title']}** (#{second['id']}) still to do. "
            "Which one do you want to tackle first?"
        )
    lines = "\n".join(
        f"- #{task['id']} **{task['title']}** ({task['priority']})"
        for task in tasks
    )
    return (
        f"You've got a few tasks waiting:\n{lines}\n"
        "Which one should we start with?"
    )


def _parse_task_update(prompt: str) -> tuple[str, str, str] | None:
    """Return (field, reference, value) for task priority, due date, or rename."""
    normalized = _normalize_command(prompt)
    priority_match = re.fullmatch(
        r"(?:set|change|update|make)\s+(?:(?:the|my)\s+)?(?:priority\s+of\s+)?"
        r"(?:(?:the|my)\s+)?task\s+(.+?)\s+"
        r"(?:to\s+)?(?:be\s+)?(?:priority\s+)?(low|medium|high)(?:\s+priority)?",
        normalized,
        re.IGNORECASE,
    )
    if priority_match:
        return (
            "priority",
            priority_match.group(1).strip(),
            priority_match.group(2).lower(),
        )
    due_match = re.fullmatch(
        r"(?:set|change|reschedule|update)\s+(?:(?:the|my)\s+)?"
        r"(?:due\s+date\s+of\s+)?(?:(?:the|my)\s+)?task\s+"
        r"(.+?)\s+due(?:\s+date)?\s+(.+)",
        normalized,
        re.IGNORECASE,
    )
    if due_match:
        return "due", due_match.group(1).strip(), due_match.group(2).strip()
    title_match = re.fullmatch(
        r"(?:rename|retitle|update)\s+(?:(?:the|my)\s+)?task\s+(.+?)\s+to\s+(.+)",
        normalized,
        re.IGNORECASE,
    )
    if title_match:
        return "title", title_match.group(1).strip(), title_match.group(2).strip()
    return None


def _parse_task_delete(prompt: str) -> str | None:
    match = re.fullmatch(
        r"(?:delete|remove|cancel|forget)\s+(?:(?:the|my)\s+)?task\s+(.+)",
        _normalize_command(prompt),
        re.IGNORECASE,
    )
    return match.group(1).strip() if match else None


def _parse_task_reopen(prompt: str) -> str | None:
    """Match 'reopen task 3' or 'mark task 3 as incomplete'."""
    match = re.fullmatch(
        r"(?:reopen|re-open|uncomplete|un-complete)\s+"
        r"(?:(?:the|my)\s+)?task\s+(.+)"
        r"|mark\s+(?:(?:the|my)\s+)?task\s+(.+?)\s+as\s+"
        r"(?:incomplete|open|pending|not\s+completed|not\s+done)",
        _normalize_command(prompt),
        re.IGNORECASE,
    )
    if not match:
        return None
    return (match.group(1) or match.group(2)).strip()


_ADVICE_PATTERN = re.compile(
    r"\b(?:what\s+(?:new|else|other|should|can|could|would)|"
    r"should\s+i|how\s+(?:can|do|should|could|to)|"
    r"suggest\w*|recommend\w*|advice|tips?|ideas?|improv\w+|better|"
    r"think\s+about|opinion|why|help\s+me|ways?\s+to|"
    r"good\s+habits?|bad\s+habits?)\b",
    re.IGNORECASE,
)


def _is_advice_question(normalized_prompt: str) -> bool:
    """Open-ended questions belong to the LLM, not the list handlers."""
    return bool(_ADVICE_PATTERN.search(normalized_prompt))


def _is_task_list_request(prompt: str) -> bool:
    normalized_prompt = _normalize_command(prompt).lower()
    # Habit and reminder queries belong to their own handlers.
    if re.search(r"\b(?:habits?|reminders?)\b", normalized_prompt):
        return False
    if _is_advice_question(normalized_prompt):
        return False
    if re.match(
        r"^(?:add|create|new|complete|completed|finish|finished|mark|"
        r"check off|delete|remove|rename|retitle|reopen|uncomplete|set|change|"
        r"update|make|cancel|done\s+with)\b",
        normalized_prompt,
    ):
        return False

    # Bare / short task queries: "tasks", "my tasks", "todos", "pending tasks", "tasks today"
    if re.fullmatch(
        r"(?:(?:my|the|all|today'?s|daily|current)\s+)*"
        r"(?:tasks?|to[\s-]?dos?|task\s+list|to[\s-]?do\s+list)"
        r"(?:\s+(?:today|for\s+today|status|list))?",
        normalized_prompt,
    ):
        return True

    task_reference = re.search(
        r"\b(?:tasks?|to[\s-]?dos?|things to do)\b",
        normalized_prompt,
    )
    if not task_reference:
        # Common shorthand such as “what's pending?” or “what are the
        # pending to do?” still means the user's open task list.
        return bool(
            re.search(r"\b(?:what|which|show|list|check|tell me|any)\b", normalized_prompt)
            and re.search(
                r"\b(?:pending|unfinished|remaining|left|to do|need to complete|need to do)\b",
                normalized_prompt,
            )
        )

    list_intent = re.search(
        r"\b(?:show|list|check|review|see|find|what|which|tell me|"
        r"do i have|are there|any)\b",
        normalized_prompt,
    )
    pending_intent = re.search(
        r"\b(?:pending|incomplete|unfinished|remaining|left|to do|"
        r"need to do|still need to do|need to complete|to complete|"
        r"need to finish|open|completed|done|finished)\b",
        normalized_prompt,
    )
    return bool(list_intent or pending_intent)


def _is_habit_list_request(prompt: str) -> bool:
    normalized_prompt = _normalize_command(prompt).lower()

    if not re.search(r"\bhabits?\b", normalized_prompt):
        return False
    if _is_advice_question(normalized_prompt):
        return False
    # Let explicit create/complete/update commands reach their own handlers.
    if re.match(
        r"^(?:add|create|new|start|track|complete|finish|done with|mark done|"
        r"check off|delete|remove|drop|rename|retitle|set|change|update)\b",
        normalized_prompt,
    ):
        return False

    # Bare / short habit queries: "habits", "my habits", "habits today", "daily habits", "habit status"
    if re.fullmatch(
        r"(?:(?:my|the|all|today'?s|daily|current)\s+)*"
        r"habits?(?:\s+(?:today|for\s+today|status|list|streaks?))?",
        normalized_prompt,
    ):
        return True

    list_intent = re.search(
        r"\b(?:check|show|list|review|see|tell me|find|what|which|"
        r"do i have|are there|any|status)\b",
        normalized_prompt,
    )
    incomplete_intent = re.search(
        r"\b(?:pending|incomplete|unfinished|remaining|left|not complete|"
        r"not completed|still not complete|still not completed|to complete|"
        r"to do|need to complete|need to finish|done|completed|today)\b",
        normalized_prompt,
    )
    return bool(list_intent or incomplete_intent)


def _is_habit_follow_up(prompt: str, previous_messages: list[dict[str, str]]) -> bool:
    normalized_prompt = re.sub(r"\s+", " ", prompt.strip().lower()).rstrip(".!?")
    if normalized_prompt not in {
        "now", "what now", "now what", "and now", "okay now", "ok now",
        "then what", "what next", "what should i do now",
    }:
        return False
    previous_assistant_messages = [
        item["content"].lower()
        for item in previous_messages
        if item["role"] == "assistant"
    ]
    return any(
        "habit" in content
        and any(marker in content for marker in (
            "list", "all caught up", "all set for today", "haven't set up",
            "haven't added", "set up yet",
        ))
        for content in previous_assistant_messages[-2:]
    )


def _parse_chat_habit(prompt: str) -> tuple[str, str] | None:
    normalized_prompt = _normalize_command(prompt)
    if _is_advice_question(normalized_prompt.lower()):
        return None
    match = re.fullmatch(
        r"(?:add|create|start|track|set(?:\s+up)?|make|new)\s+"
        r"(?:(?:a|an)\s+)?(?:new\s+)?(?:daily\s+)?habits?\b"
        r"(?:\s+(?:for|to|called|named|of))?\s*[:\-]?\s*(.+)",
        normalized_prompt,
        re.IGNORECASE,
    )
    if not match:
        match = re.fullmatch(
            r"habit\s*[:\-]\s*(.+)"
            r"|(?:add|put)\s+(.+?)\s+(?:to|on|into)\s+(?:my\s+|the\s+)?(?:daily\s+)?habits?"
            r"|track\s+(.+?)\s+(?:every\s*day|daily)",
            normalized_prompt,
            re.IGNORECASE,
        )
        if match:
            raw = next((g for g in match.groups() if g), "")
            habit_name = raw.strip(" :-'\"")
            return (habit_name, "Activity") if habit_name else None
        return None
    habit_name = match.group(1).strip(" :-'\"")
    return (habit_name, "Activity") if habit_name else None


def _parse_habit_uncomplete(prompt: str) -> str | None:
    """Match 'uncheck habit X', 'undo habit X', or 'mark habit X as not done'."""
    normalized = _normalize_command(prompt)
    match = re.fullmatch(
        r"(?:uncheck|un-check|undo|uncomplete|un-complete|reset)\s+"
        r"(?:(?:the|my)\s+)?habits?\s+(.+)"
        r"|mark\s+(?:(?:the|my)\s+)?(?:habit\s+)?(.+?)\s+as\s+"
        r"(?:not\s+done|not\s+completed|incomplete|unchecked|pending)",
        normalized,
        re.IGNORECASE,
    )
    if not match:
        return None
    target = (match.group(1) or match.group(2)).strip()
    return re.sub(r"^(?:my|the)\s+", "", target, flags=re.IGNORECASE)


def _parse_habit_completion(prompt: str) -> str | None:
    """Match an explicit 'complete habit <name>' command."""
    match = re.fullmatch(
        r"(?:complete|completed|finish|finished|done\s+with|mark\s+done|check\s+off)\s+"
        r"(?:(?:the|my)\s+)?habits?\s+(.+)",
        _normalize_command(prompt),
        re.IGNORECASE,
    )
    if not match:
        return None
    return re.sub(r"^(?:my|the)\s+", "", match.group(1).strip(), flags=re.IGNORECASE)


def _parse_habit_rename(prompt: str) -> tuple[str, str] | None:
    match = re.fullmatch(
        r"(?:rename|retitle|update)\s+(?:(?:the|my)\s+)?habit\s+(.+?)\s+to\s+(.+)",
        _normalize_command(prompt),
        re.IGNORECASE,
    )
    if not match:
        return None
    return match.group(1).strip(), match.group(2).strip()


def _parse_habit_delete(prompt: str) -> str | None:
    match = re.fullmatch(
        r"(?:delete|remove|stop|drop)\s+(?:(?:the|my)\s+)?habit\s+(.+)",
        _normalize_command(prompt),
        re.IGNORECASE,
    )
    return match.group(1).strip() if match else None


def _find_habit(habit_title: str) -> dict[str, Any] | None:
    normalized_title = re.sub(r"\s+", " ", habit_title.strip().lower())
    if not normalized_title:
        return None
    habits = memory_store.list_habits()
    exact_matches = [
        habit for habit in habits
        if habit["name"].strip().lower() == normalized_title
    ]
    if len(exact_matches) == 1:
        return exact_matches[0]
    matches = [
        habit for habit in habits
        if normalized_title in habit["name"].strip().lower()
        or habit["name"].strip().lower() in normalized_title
    ]
    if len(matches) > 1:
        raise ValueError(
            f"More than one habit matches '{habit_title}'. Be more specific."
        )
    if len(matches) == 1:
        return matches[0]
    return _fuzzy_best_match(normalized_title, habits, "name")


def _parse_task_completion(prompt: str) -> str | None:
    """Match an explicit 'complete task <number|title>' command."""
    match = re.fullmatch(
        r"(?:complete|completed|finish|finished|mark\s+done|check\s+off|done\s+with)\s+"
        r"(?:(?:the|my)\s+)?tasks?\s+(.+)",
        _normalize_command(prompt),
        re.IGNORECASE,
    )
    if not match:
        return None
    return re.sub(r"^(?:my|the)\s+", "", match.group(1).strip(), flags=re.IGNORECASE)


_COMPLETION_VERB_PATTERN = (
    r"(?:complete|completed|finish|finished|mark\s+done|mark\s+completed|"
    r"check\s+off|tick\s+off|done\s+with)"
)

_ACTION_WITHOUT_HANDLER = re.compile(
    r"^(?:add|create|new|set|make|mark|complete|finish|delete|remove|"
    r"rename|reschedule|cancel)\b.*\b(?:tasks?|habits?|reminders?)\b",
    re.IGNORECASE,
)


def _parse_completion_target(prompt: str) -> str | None:
    """Catch generic 'complete X' or 'now I am completed X' phrasing without an entity keyword."""
    normalized = _normalize_command(prompt)
    patterns = (
        rf"{_COMPLETION_VERB_PATTERN}\s+(.+)",
        r"mark\s+(?:the\s+|my\s+)?(.+?)\s+(?:tasks?\s+|habits?\s+)?(?:as\s+)?"
        r"(?:being\s+)?(?:completed|complete|done|finished)",
        r"(?:i\s+)?(?:am|'m|was|have|'ve|had|'d)?\s*"
        r"(?:just\s+|already\s+|finally\s+|now\s+)*"
        r"(?:completed|complete|finished|finish|done(?:\s+with)?|checked\s+off)\s+"
        r"(?:with\s+|doing\s+)?(.+)",
        r"(.+?)\s+(?:task|habit)\s+(?:is\s+)?(?:completed|complete|finished|finish|done)"
        r"(?:\s+(?:and\s+)?mark\s+it(?:\s+as)?(?:\s+completed|\s+done)?)?",
        r"(.+?)\s+is\s+(?:now\s+)?(?:completed|complete|finished|done)",
    )
    for pattern in patterns:
        match = re.fullmatch(pattern, normalized, re.IGNORECASE)
        if not match:
            continue
        target = match.group(1).strip()
        target = re.sub(
            r"^(?:(?:the|my)\s+)?(?:task|habit)\s+",
            "",
            target,
            flags=re.IGNORECASE,
        )
        target = re.sub(
            r"^(?:the|my)\s+", "", target, flags=re.IGNORECASE
        )
        target = re.sub(
            r"\s+(?:today|right\s+now|now|already)$", "", target, flags=re.IGNORECASE
        ).strip()
        target = re.sub(r"\s+mark\s+it.*$", "", target, flags=re.IGNORECASE).strip()
        if target:
            return target

    did_match = re.fullmatch(r"(?:i\s+)?did\s+(.+)", normalized, re.IGNORECASE)
    if did_match:
        words = re.findall(r"[a-z0-9]+", did_match.group(1).lower())
        deduplicated = [
            word for index, word in enumerate(words)
            if index == 0 or word != words[index - 1]
        ]
        return " ".join(deduplicated) or None
    return None


def _parse_reminder_cancel(prompt: str) -> str | None:
    match = re.fullmatch(
        r"(?:cancel|delete|remove|stop|forget)\s+(?:(?:the|my)\s+)?reminder\s+"
        r"(?:to\s+|for\s+|about\s+)?(.+)",
        _normalize_command(prompt),
        re.IGNORECASE,
    )
    return match.group(1).strip() if match else None


def _parse_reminder_reschedule(prompt: str) -> tuple[str, str] | None:
    """Return (reference, when) for 'reschedule reminder X to Y'."""
    match = re.fullmatch(
        r"(?:reschedule|move|postpone|delay|change|update)\s+"
        r"(?:(?:the|my)\s+)?reminder\s+(.+?)\s+to\s+(.+)",
        _normalize_command(prompt),
        re.IGNORECASE,
    )
    if not match:
        return None
    return match.group(1).strip(), match.group(2).strip()


def _find_reminder(reference: str) -> dict[str, Any]:
    """Resolve a unique active reminder by a substring of its text."""
    normalized = re.sub(r"\s+", " ", reference.strip().lower())
    if not normalized:
        raise ValueError("Tell me which reminder you mean.")
    matches = [
        reminder for reminder in memory_store.list_reminders()
        if normalized in reminder["text"].strip().lower()
    ]
    if not matches:
        raise ValueError(f"No active reminder matched '{reference}'.")
    if len(matches) > 1:
        options = "\n".join(f"- {reminder['text']}" for reminder in matches)
        raise ValueError(
            f"More than one reminder matches '{reference}':\n{options}"
        )
    return matches[0]


def _is_reminder_list_request(prompt: str) -> bool:
    normalized = _normalize_command(prompt).lower()
    if not re.search(r"\b(?:reminders?|alarms?|timers?)\b", normalized):
        return False
    if _is_advice_question(normalized):
        return False
    if re.match(
        r"^(?:cancel|delete|remove|stop|forget|set|create|add|remind|"
        r"reschedule|move|postpone|delay|change|update)\b",
        normalized,
    ):
        return False
    if re.fullmatch(
        r"(?:(?:my|the|all|active|upcoming|pending|current)\s+)*"
        r"(?:reminders?|alarms?|timers?)(?:\s+list)?",
        normalized,
    ):
        return True
    return bool(
        re.search(
            r"\b(?:show|list|check|what|which|tell me|do i have|any|upcoming|active|pending)\b",
            normalized,
        )
    )


def _format_reminder_list(reminders: list[dict[str, Any]]) -> str:
    if not reminders:
        return "You don't have any active reminders right now."
    lines = []
    for reminder in reminders:
        recurrence = reminder.get("recurrence")
        suffix = f", repeating {recurrence}" if recurrence else ""
        lines.append(
            f"- **{reminder['text']}** at "
            f"{_format_due_date(reminder['run_at'])}{suffix}"
        )
    return "Here's what I'm keeping track of:\n" + "\n".join(lines)


def _format_habit_list(
    all_habits: list[dict[str, Any]], pending_habits: list[dict[str, Any]]
) -> str:
    if not all_habits:
        return "I don't see any habits set up yet. What would you like to start with?"
    if not pending_habits:
        return "You're all caught up on habits for today. What else is on your mind?"
    names = [habit["name"] for habit in pending_habits]
    if len(names) == 1:
        return (
            f"You've still got **{names[0]}** on your habit list today. "
            "Want to start with that?"
        )
    if len(names) == 2:
        return (
            f"You've still got **{names[0]}** and **{names[1]}** on your habit "
            "list today. Which one feels easier to start with?"
        )
    formatted_names = "\n".join(f"- **{name}**" for name in names)
    return (
        f"You've still got a few habits on your list today:\n{formatted_names}\n"
        "Which one do you feel like starting with?"
    )


def _format_due_date(value: str) -> str:
    try:
        return datetime.fromisoformat(value).strftime("%d %b %Y, %I:%M %p")
    except (TypeError, ValueError):
        return value


def _notify_completion(chat_id: str, prefix: str, content: str) -> None:
    """Drop a completion follow-up into the conversation the web panel polls."""
    if not chat_id or not content:
        return
    memory_store.add_notification(chat_id, f"{prefix}-{uuid.uuid4()}", content)


def _learning_suggest_for_task(task_id: int) -> None:
    """Background hook: propose topic links for a freshly created task."""
    if not learning_schema.is_initialized():
        return
    try:
        learning_classifier.suggest_links_for_task(
            task_id,
            embed=memory_store.embed,
            llm=_fallback_llm_for_learning,
        )
    except Exception as error:  # pragma: no cover - defensive
        print(f"Learning link suggestion skipped: {error}")


def _fallback_llm_for_learning(title: str, topics) -> str | None:
    """Ask the local model to name the best topic; skip when not loaded."""
    if model is None or tokenizer is None:
        return None
    numbered = "\n".join(
        f"{index}. {topic['name']}" for index, topic in enumerate(topics, 1)
    )
    prompt = (
        "Pick the single learning topic from the list below that best "
        "matches this task title. Reply with only the exact topic name, "
        "or the word none if nothing fits. No explanation."
        f"\n\nTask title: {title}\n\nTopics:\n{numbered}"
    )
    try:
        reply = generate_response(prompt, 16)
    except Exception as error:  # pragma: no cover - defensive
        print(f"Learning LLM fallback skipped: {error}")
        return None
    return reply or None


def _learning_on_task_completed(task_id: int) -> None:
    """Background hook: apply the completion rule to confirmed topic links."""
    if not learning_schema.is_initialized():
        return
    try:
        learning_progress.on_task_completed_links(task_id)
    except Exception as error:  # pragma: no cover - defensive
        print(f"Learning completion rule skipped: {error}")


def _run_learning_hook(job) -> None:
    """Run a learning-module job off the request path so it can never block
    or break task creation and completion."""
    threading.Thread(target=job, daemon=True).start()


WEEKLY_LEARNING_NUDGE_JOB = "weekly-learning-nudge"


def _weekly_learning_nudge(provider: str = "local") -> None:
    """Monday-morning recap for one provider's chats, at most weekly each."""
    if not learning_schema.is_initialized():
        return
    try:
        if provider == "discord":
            chat_ids = [
                chat["id"] for chat in memory_store.list_external_chats("discord")
            ]
        else:
            chat_ids = memory_store.list_local_chat_ids()
        learning_nudges.deliver_weekly_nudge(
            chat_ids,
            memory_store.add_notification,
            recently_sent=memory_store.has_recent_weekly_nudge,
        )
    except Exception as error:  # pragma: no cover - defensive
        print(f"Weekly learning nudge skipped: {error}")


def _schedule_weekly_learning_nudge(provider: str = "local") -> None:
    if not learning_schema.is_initialized():
        return
    try:
        reminder_scheduler.add_job(
            _weekly_learning_nudge,
            CronTrigger(day_of_week="mon", hour=9, minute=0,
                        timezone="Asia/Kolkata"),
            args=[provider],
            id=WEEKLY_LEARNING_NUDGE_JOB if provider == "local" else f"{WEEKLY_LEARNING_NUDGE_JOB}-{provider}",
            replace_existing=True,
        )
    except Exception as error:  # pragma: no cover - defensive
        print(f"Weekly learning nudge not scheduled: {error}")


def init_learning_module(provider: str = "local") -> bool:
    """Idempotent. Safe to call from both the web server and the Discord bot."""
    try:
        learning_schema.init(DATABASE_PATH)
        learning_roadmaps.load_seeds()
        _schedule_weekly_learning_nudge(provider)
        return learning_schema.is_initialized()
    except Exception as error:  # pragma: no cover - defensive
        print(f"Learning module init skipped: {error}")
        return False


def _task_completion_line(task: dict[str, Any], name: str = "") -> str:
    """Encouraging reply that names the next task, if one is waiting."""
    # list_tasks already orders by priority, so the first row is the best next pick.
    remaining = memory_store.list_tasks(include_completed=False)
    next_task = remaining[0] if remaining else None
    return assistant_tone.task_completed(
        task["title"],
        len(remaining),
        next_title=next_task["title"] if next_task else "",
        next_priority=next_task["priority"] if next_task else "",
        name=name,
    )


def _habit_completion_line(habit: dict[str, Any], name: str = "") -> str:
    """Encouraging reply that names the habits still open today."""
    today = datetime.now(LOCAL_TIMEZONE).date().isoformat()
    pending = [
        item["name"]
        for item in memory_store.list_habits()
        if item["id"] != habit["id"] and today not in item["completed_dates"]
    ]
    return assistant_tone.habit_completed(
        habit["name"], habit["streak"], pending=pending, person=name
    )


def _complete_task(task: dict[str, Any], name: str = "") -> str:
    if task["completed"]:
        return (
            f"**{task['title']}** is already done — say "
            f"'reopen task {task['title']}' if you want it back on the list."
        )
    updated = memory_store.update_task(task["id"], completed=True)
    if updated is None:
        raise ValueError(f"Task {task['id']} was not found.")
    _run_learning_hook(lambda: _learning_on_task_completed(task["id"]))
    reply = _task_completion_line(updated, name=name)
    if learning_schema.is_initialized():
        try:
            with learning_schema.connect() as connection:
                row = connection.execute(
                    "SELECT t.id, t.name FROM task_topic_links AS link "
                    "JOIN topics AS t ON t.id = link.topic_id "
                    "WHERE link.task_id = ? AND link.state = 'confirmed' "
                    "LIMIT 1",
                    (task["id"],),
                ).fetchone()
            if row:
                topic_id = row["id"]
                topic_name = row["name"]
                links = learning_progress.list_confirmed_task_links(topic_id)
                total = len(links)
                completed = sum(1 for link in links if link.get("task_completed"))
                reply = f"{reply.rstrip()}\nThat counts toward {topic_name}, now {completed}/{total}."
        except Exception:
            pass
    return reply


def _complete_habit(habit: dict[str, Any], name: str = "") -> str:
    today = datetime.now(LOCAL_TIMEZONE).date().isoformat()
    if today in habit["completed_dates"]:
        return f"**{habit['name']}** is already ticked off today — I won't count it twice."
    updated = memory_store.set_habit_completion(habit["id"], today, True)
    return _habit_completion_line(updated, name=name)


def _uncomplete_habit(habit: dict[str, Any], name: str = "") -> str:
    today = datetime.now(LOCAL_TIMEZONE).date().isoformat()
    if today not in habit["completed_dates"]:
        return f"**{habit['name']}** wasn't checked off today."
    updated = memory_store.set_habit_completion(habit["id"], today, False)
    return f"Unchecked **{updated['name']}** for today (streak: {updated['streak']})."


_IDENTITY_QUESTION_PATTERNS = (
    re.compile(
        r"\bwho\s+(?:built|builed|builded|build|made|created|developed|"
        r"designed|coded|programmed)\s+(?:you|spidey|s\.p\.i\.d\.e\.y)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bwho\s+(?:is|are)\s+your\s+(?:creator|developer|maker|builder|"
        r"author|owner|designer)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bwho\s+(?:is|are)\s+behind\s+(?:you|spidey|s\.p\.i\.d\.e\.y)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:how|why)\s+(?:is|was|are|were)\s+(?:you|spidey|s\.p\.i\.d\.e\.y)\s+"
        r"(?:built|made|created|developed|designed)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:how|why)\s+(?:is|was|are|were)\s+(?:built|made|created)\s+"
        r"(?:you|spidey|s\.p\.i\.d\.e\.y)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:how|why)\s+(?:you|spidey|s\.p\.i\.d\.e\.y)\s+(?:are|were|is|was)\s+"
        r"(?:built|made|created)\b",
        re.IGNORECASE,
    ),
)

_BUILDER_NAME_PATTERN = re.compile(
    r"\b(?:built|builed|builded|made|created|developed|designed|coded|"
    r"programmed)\s+by\s+([A-Z][\w.'-]*(?:\s+[A-Z][\w.'-]*){0,2})"
)

_BUILDER_LOOSE_PATTERN = re.compile(
    r"\b(?:built|builed|builded|made|created|developed|designed|coded|"
    r"programmed)\s+by\s+([^.;\n]{1,40})",
    re.IGNORECASE,
)

_ASSISTANT_REFERENCE_PATTERN = re.compile(
    r"\b(?:you|your|spidey|s\.p\.i\.d\.e\.y|assistant)\b", re.IGNORECASE
)


def _builder_names() -> list[str]:
    """Collect creator names from saved memories such as "built by Jaya Haris"."""
    names: list[str] = []
    for memory in memory_store.list_memories():
        content = memory["content"]
        # Only memories that talk about the assistant describe its creator.
        if not _ASSISTANT_REFERENCE_PATTERN.search(content):
            continue
        match = _BUILDER_NAME_PATTERN.search(content) or _BUILDER_LOOSE_PATTERN.search(content)
        if not match:
            continue
        name = re.sub(r"\s+", " ", match.group(1)).strip(" .,!-")
        if name and name.lower() not in {existing.lower() for existing in names}:
            names.append(name)
    return names


def _identity_answer(prompt: str) -> str | None:
    """Answer creator questions straight from memory instead of the small model."""
    normalized = _normalize_command(prompt)
    if not any(pattern.search(normalized) for pattern in _IDENTITY_QUESTION_PATTERNS):
        return None
    names = _builder_names()
    if not names:
        return None
    if len(names) == 1:
        return f"I was built by {names[0]}."
    return "I was built by " + ", ".join(names[:-1]) + f" and {names[-1]}."


# Notes about the assistant ("You was built by ...") skip the first person;
# notes the user saves about themselves usually start with I, my, or me.
_USER_FACT_PATTERN = re.compile(r"^(?:i|i'm|i’m|i've|my|me)\b", re.IGNORECASE)

_PROFILE_QUESTION_PATTERNS = (
    re.compile(
        r"\btell\s+(?:me\s+)?(?:something\s+|a bit\s+|more\s+)?about\s+me\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\btell\s+(?:me\s+)?(?:something\s+|a bit\s+|more\s+)?about\s+myself\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bwhat\s+do\s+you\s+(?:know|remember|have)\s+(?:about|on)\s+me\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bdo\s+you\s+(?:know|remember)\s+me\b", re.IGNORECASE),
    re.compile(r"^(?:who|what)\s+am\s+i\s*$", re.IGNORECASE),
    re.compile(r"^describe\s+me\s*$", re.IGNORECASE),
    re.compile(r"\bsummar(?:y|ise|ize)\s+(?:me|myself)\b", re.IGNORECASE),
    re.compile(
        r"\b(?:what(?:'s| is)|show(?:\s+me)?|tell\s+me)\s+my\s+"
        r"(?:profile|bio|details|background)\b",
        re.IGNORECASE,
    ),
    re.compile(r"^about\s+me\b", re.IGNORECASE),
)

_PROFILE_QUESTION_LEAD = re.compile(
    r"^(?:what|which|who|where|when|tell|show|list|do|does|can|could|how)\b",
    re.IGNORECASE,
)

# Words in the user's question that point at one part of their profile.
_PROFILE_TOPIC_WORDS = (
    ("The basics", ("name", "call me")),
    (
        "Education",
        (
            "education", "study", "studies", "studying", "college",
            "diploma", "degree", "school", "qualification", "background",
        ),
    ),
    (
        "Skills",
        ("skill", "skills", "stack", "technologies", "technology", "tech", "tools"),
    ),
    (
        "Currently building",
        ("working on", "developing", "building right now"),
    ),
    (
        "Projects",
        ("project", "projects", "built", "building", "portfolio", "work on"),
    ),
    ("Light House", ("startup", "business", "company", "light house")),
    ("Your setup", ("laptop", "computer", "pc", "hardware", "specs", "gpu", "ram")),
    (
        "How you like to work",
        ("goal", "goals", "ambition", "prefer", "preferences", "working style"),
    ),
)

# Keywords that decide which section of the summary a saved note belongs to.
_PROFILE_BUCKETS = (
    ("The basics", ("name", "developer", "enthusiast", "india", "entrepreneur")),
    (
        "Education",
        (
            "diploma", "degree", "college", "polytechnic", "b.e", "b.tech",
            "university", "school", "graduated",
        ),
    ),
    (
        "Skills",
        (
            "skill", "react", "javascript", "node", "python", "postgres",
            "firebase", "supabase", "tailwind", "rest api", "learning",
        ),
    ),
    (
        "Currently building",
        ("developing", "in progress", "currently working", "right now"),
    ),
    (
        "Projects",
        (
            "project", "spidey", "assistant", "application", "website", "build",
            "e-commerce", "client", "smart food",
        ),
    ),
    ("Light House", ("startup", "business", "company", "light house", "clients")),
    ("Your setup", ("laptop", "asus", "ryzen", "rtx", "gpu", "ram", "ssd", "hardware")),
    (
        "How you like to work",
        ("prefer", "enjoy", "goal", "long-term", "guidance", "interested", "ui design"),
    ),
)

_PROFILE_TOPIC_INTROS = {
    "The basics": "Here's what I have:",
    "Education": "Here's your background:",
    "Skills": "Here's your stack:",
    "Currently building": "Here's what you're building right now:",
    "Projects": "Here's what you've built:",
    "Light House": "Here's where Light House stands:",
    "Your setup": "Here's your setup:",
    "How you like to work": "Here's how you like to work:",
}

_SECOND_PERSON_REPLACEMENTS = (
    (re.compile(r"\bThe user's\b"), "Your"),
    (re.compile(r"\bthe user's\b"), "your"),
    (re.compile(r"\bThe user\b"), "You"),
    (re.compile(r"\bthe user\b"), "you"),
    (re.compile(r"\bMy\b"), "Your"),
    (re.compile(r"\bmy\b"), "your"),
    (re.compile(r"\bI[’']m\b"), "you're"),
    (re.compile(r"\bI am\b"), "you're"),
    (re.compile(r"\bI[’']ve\b"), "you've"),
    (re.compile(r"\bI[’']ll\b"), "you'll"),
    (re.compile(r"\bI[’']d\b"), "you'd"),
    (re.compile(r"\bI\b"), "you"),
    (re.compile(r"\bmyself\b"), "yourself"),
    (re.compile(r"\bmine\b"), "yours"),
    (re.compile(r"\bme\b"), "you"),
)

_NAME_IN_MEMORY_PATTERN = re.compile(
    r"\b(?:My name is|The user's name is|I'm|I am) "
    r"([A-Z][\w'-]*(?:\s+[A-Z][\w'-]*){0,2})"
)


def _describes_assistant(content: str) -> bool:
    """True for notes like "You was built by Jaya Haris", which describe S.P.I.D.E.Y."""
    return bool(
        _BUILDER_LOOSE_PATTERN.search(content)
        and _ASSISTANT_REFERENCE_PATTERN.search(content)
        and not _USER_FACT_PATTERN.match(content)
    )


def _user_fact_memories() -> list[dict[str, Any]]:
    """Everything saved about the user, minus the notes about the assistant."""
    facts = [
        memory
        for memory in memory_store.list_memories()
        if not _describes_assistant(memory["content"].strip())
    ]
    facts.sort(key=lambda item: item["id"])
    return facts


def _user_name(facts: list[dict[str, Any]]) -> str:
    for memory in facts:
        match = _NAME_IN_MEMORY_PATTERN.search(memory["content"])
        if match:
            return match.group(1).strip(" .,!-")
    return ""


_THIRD_PERSON_VERBS = {
    "prefers": "prefer", "likes": "like", "loves": "love", "wants": "want",
    "needs": "need", "works": "work", "lives": "live", "studies": "study",
    "enjoys": "enjoy", "uses": "use", "has": "have", "does": "do",
    "goes": "go", "is": "are", "builds": "build", "makes": "make",
    "plans": "plan", "knows": "know", "plays": "play", "runs": "run",
    "keeps": "keep", "feels": "feel", "thinks": "think", "hopes": "hope",
}


def _second_person(text: str) -> str:
    """Rewrite a note the user saved about themselves so it addresses them."""
    result = text
    for pattern, replacement in _SECOND_PERSON_REPLACEMENTS:
        result = pattern.sub(replacement, result)
    # Notes written in the third person ("The user prefers ...") need their
    # verbs adjusted once "The user" becomes "you".
    result = re.sub(
        r"\b([Yy]ou) (\w+)\b",
        lambda match: (
            f"{match.group(1)} "
            f"{_THIRD_PERSON_VERBS.get(match.group(2).lower(), match.group(2))}"
        ),
        result,
    )
    # "I am developing" becomes "you're developing"; restore the capital.
    return re.sub(
        r"(^|[.!?]\s+)([a-z])",
        lambda match: match.group(1) + match.group(2).upper(),
        result,
    )


def _profile_label(content: str) -> str:
    """Pick the summary section a saved note belongs to."""
    lowered = content.lower()
    best_label = "Also worth knowing"
    best_score = 0
    for label, keywords in _PROFILE_BUCKETS:
        score = sum(1 for keyword in keywords if keyword in lowered)
        if score > best_score:
            best_label, best_score = label, score
    return best_label


def _profile_buckets(facts: list[dict[str, Any]]) -> dict[str, list[str]]:
    buckets: dict[str, list[str]] = {}
    for memory in facts:
        label = _profile_label(memory["content"])
        buckets.setdefault(label, []).append(_second_person(memory["content"].strip()))
    return buckets


def _profile_topic(normalized: str) -> str | None:
    """Return the profile section a direct question like "what are my skills" asks about."""
    if not re.search(r"\b(?:my|i|me|mine)\b", normalized):
        return None
    if not _PROFILE_QUESTION_LEAD.match(normalized):
        return None
    for label, words in _PROFILE_TOPIC_WORDS:
        if any(re.search(rf"\b{re.escape(word)}\b", normalized) for word in words):
            return label
    return None


def _profile_overview(facts: list[dict[str, Any]]) -> str:
    buckets = _profile_buckets(facts)
    name = _user_name(facts)
    intro = (
        f"Here's what I've got on you, {name}:"
        if name
        else "Here's what I've got on you:"
    )
    lines: list[str] = []
    for label, _ in _PROFILE_BUCKETS:
        items = buckets.pop(label, [])
        if items:
            lines.append(f"- **{label}:** {' '.join(items)}")
    leftovers = [text for texts in buckets.values() for text in texts]
    if leftovers:
        lines.append(f"- **Also worth knowing:** {' '.join(leftovers)}")
    return (
        f"{intro}\n\n"
        + "\n".join(lines)
        + "\n\nAsk me to go deeper on any of it — your projects, your skills, "
        "or what you're working on right now."
    )


def _profile_topic_answer(label: str, facts: list[dict[str, Any]]) -> str:
    items = _profile_buckets(facts).get(label, [])
    if not items:
        return "I don't have anything saved on that yet."
    intro = _PROFILE_TOPIC_INTROS.get(label, "Here's what I have:")
    return f"{intro} {' '.join(items)}"


def _profile_answer(prompt: str) -> str | None:
    """Answer questions about the user from the notes they saved about themselves."""
    normalized = _normalize_command(prompt).lower()
    if not normalized:
        return None
    overview = any(
        pattern.search(normalized) for pattern in _PROFILE_QUESTION_PATTERNS
    )
    topic = None if overview else _profile_topic(normalized)
    if not overview and topic is None:
        return None

    facts = _user_fact_memories()
    if not facts:
        return (
            "I don't have anything saved about you yet. Add a few details to "
            "your memory list — your name, what you work on, how you like to be "
            "helped — and I'll keep them in mind."
        )
    if topic is None:
        return _profile_overview(facts)
    name = _user_name(facts)
    if topic == "The basics" and name and re.search(r"\bname\b", normalized):
        return f"You're {name}."
    return _profile_topic_answer(topic, facts)


def _memories_for_prompt(prompt: str) -> list[dict[str, Any]]:
    """Semantic matches first, topped up with the most recent saved memories."""
    found = memory_store.search_memories(prompt)
    if len(found) >= MAX_RETRIEVED_MEMORIES:
        return found
    seen = {item["id"] for item in found}
    recent = [
        item for item in memory_store.list_memories() if item["id"] not in seen
    ]
    return [*found, *recent[: MAX_RETRIEVED_MEMORIES - len(found)]]


_GREETING_PATTERN = re.compile(
    r"(?:hi|hello|hey|yo|hiya|good\s+(?:morning|afternoon|evening))"
    r"[\s!.,]*(?:spidey|s\.p\.i\.d\.e\.y)?[\s!.,]*",
    re.IGNORECASE,
)

_ASSISTANT_INTRO_PATTERN = re.compile(
    r"(?:what(?:'s|s| is) your name|who are you|what are you|introduce yourself)",
    re.IGNORECASE,
)


def _greeting_answer(prompt: str) -> str | None:
    """Answer bare greetings instantly so a model load never delays hello."""
    normalized = _normalize_command(prompt)
    if not _GREETING_PATTERN.fullmatch(normalized):
        return None
    name = _user_name(_user_fact_memories())
    salutation = f"Hey {name.split()[0]}!" if name else "Hey!"
    return (
        f"{salutation} I'm S.P.I.D.E.Y. Ask me about your tasks, habits, "
        "reminders, learning progress, or anything you've saved."
    )


def _assistant_intro_answer(prompt: str) -> str | None:
    """Introduce S.P.I.D.E.Y without asking the small model to guess its name."""
    normalized = _normalize_command(prompt)
    if not _ASSISTANT_INTRO_PATTERN.fullmatch(normalized):
        return None
    return (
        "I'm S.P.I.D.E.Y, your local assistant. I manage tasks, habits, and "
        "reminders, remember what you tell me, and answer questions — all on "
        "your own machine."
    )


_PAGE_BY_PROVIDER = {
    "local-habit-manager": "habits",
    "local-task-manager": "tasks",
    "local-scheduler": "reminders",
    "local-learning": "learning",
}

_PAGE_ALIASES = {
    "tasks": ("task", "tasks", "todo", "todos", "to-do", "to-dos", "to do", "to dos"),
    "habits": ("habit", "habits"),
    "reminders": ("reminder", "reminders"),
    "dashboard": ("dashboard", "home", "overview"),
    "chat": ("chat", "assistant"),
    "research": ("research", "web research"),
    "memories": ("memory", "memories", "notes"),
    "learning": ("learning", "learning tracker", "roadmap", "roadmaps"),
    "settings": ("settings",),
}

_NAVIGATE_PATTERN = re.compile(
    r"^(?:(?:go|move|switch|navigate|jump|head)\s+(?:over\s+)?(?:to|into)"
    r"|goto|take\s+me\s+(?:to|into)|open|bring\s+up)"
    r"\s+(?:the\s+|my\s+)?(?P<target>[a-z\- ]+?)"
    r"(?:\s+(?:page|section|tab|screen|view|area))?$",
    re.IGNORECASE,
)


def _parse_navigation(prompt: str) -> str | None:
    """Return a page key for 'go to tasks' style commands, else None."""
    match = _NAVIGATE_PATTERN.match(_normalize_command(prompt))
    if not match:
        return None
    target = re.sub(r"\s+", " ", match.group("target").strip().lower())
    for page, words in _PAGE_ALIASES.items():
        if target in words:
            return page
    return None


def _page_for_provider(provider: str) -> str | None:
    if provider.startswith("local-navigate:"):
        return provider.split(":", 1)[1]
    return _PAGE_BY_PROVIDER.get(provider)


_LEARNING_PROGRESS_PATTERN = re.compile(
    r"^(?:(?:show|check|view|tell|give)\s+(?:me\s+)?)?(?:my\s+)?learning\s+"
    r"(?:progress|status|stats|statistics|summary|overview)$"
    r"|^learning\s+(?:progress|status|stats|summary)$"
    r"|^how(?:'s| is)\s+my\s+learning\s+(?:going|progress|doing)$",
    re.IGNORECASE,
)

_LEARNING_NEXT_PATTERN = re.compile(
    r"^what(?:\s+should|\s+can|\s+do|\s+could|\s+shall)?\s+i\s+(?:learn|study|tackle|practice|master)\s+next$"
    r"|^what(?:'s| is)\s+next\s+(?:to\s+|in\s+my\s+)?(?:learn|learning|study|studying)$"
    r"|^(?:what(?:'s| is)\s+)?(?:my\s+|the\s+)?next\s+(?:learning\s+)?(?:topic|lesson|module|subject)s?(?:\s+to\s+(?:learn|study))?$"
    r"|^(?:suggest|recommend)(?:\s+me)?\s+(?:a\s+|some\s+)?(?:learning\s+)?(?:topics?|something\s+to\s+(?:learn|study))(?:\s+next)?$",
    re.IGNORECASE,
)

_LEARNING_ROADMAP_PROGRESS_PATTERN = re.compile(
    r"^(?:check|show|view)\s+(?:me\s+)?(?:my\s+)?(?:the\s+)?"
    r"(?:(?:status|progress)\s+(?:of|for|in)\s+(.+)"
    r"|(.+?)\s+(?:roadmap\s+)?(?:status|progress))$"
    r"|^what(?:'s| is)\s+(?:my\s+)?(?:the\s+)?"
    r"(?:status|progress)\s+(?:of|for|in)\s+(.+)$"
    r"|^how(?:'s| is)\s+(?:my\s+)?(.+?)\s+(?:going|progress)$",
    re.IGNORECASE,
)

_LEARNING_ROADMAP_TOPICS_PATTERN = re.compile(
    r"^(?:(?:show|list|view|check|what(?:'s| is| are))\s+(?:me\s+)?)?"
    r"(?:(?:all|the|my)\s+)*topics?\s+(?:in|for|under|from|of|inside)\s+(.+)$"
    r"|^(?:show|list|view|inspect)\s+(?:me\s+)?(?:the\s+|my\s+)?(.+?)\s+(?:roadmap\s+)?topics$"
    r"|^what(?:'s| is)\s+(?:inside|in)\s+(?:the\s+|my\s+)?(.+?)\s+roadmap$",
    re.IGNORECASE,
)

_LEARNING_STATUS_LIST_PATTERN = re.compile(
    r"^what\s+(?:topics?\s+)?am\s+i\s+(?:currently\s+)?(?:learning|studying)(?:\s+right\s+now|\s+now)?$"
    r"|^(?:(?:show|list|view|check|what\s+are)\s+(?:me\s+)?)?(?:(?:my|the|all)\s+)*"
    r"(?P<filter>in[\s\-]progress|active|current|currently\s+learning|completed|finished|learned|mastered|on[\s\-]hold|paused|not\s+started|unlocked)\s+"
    r"(?:learning\s+)?topics?$"
    r"|^(?:(?:show|list|view|check|what\s+are)\s+(?:me\s+)?)?(?:(?:my|the|all)\s+)*"
    r"(?:learning\s+)?topics?\s+(?:that\s+are\s+|which\s+are\s+|i\s+have\s+|i've\s+)?"
    r"(?P<filter2>in\s+progress|currently\s+learning|completed|finished|learned|mastered|on\s+hold|paused|not\s+started)$"
    r"|^what\s+(?:topics?\s+)?have\s+i\s+(?P<filter3>learned|completed|mastered|finished)(?:\s+so\s+far)?$",
    re.IGNORECASE,
)

_LEARNING_ROADMAP_NEXT_PATTERN = re.compile(
    r"^(?:check\s+)?what\s+"
    r"(?:(?:to\s+)?should\s+i|i\s+should|can\s+i|to)\s+"
    r"(?:learn|study)(?:\s+next)?\s+(?:from|in|on|for)\s+(.+)$"
    r"|^what(?:'s| is)\s+next\s+(?:in|for|on)\s+(.+)$",
    re.IGNORECASE,
)

_LEARNING_ROADMAP_TASK_PATTERN = re.compile(
    r"^(?:suggest|recommend)(?:\s+me)?\s+(?:a\s+)?"
    r"(?:task|topic|something)\s+"
    r"(?:(?:to\s+)?(?:learn|study)(?:\s+next)?\s+(?:from|in|on)\s+(.+)"
    r"|(?:for|about)\s+(.+))$",
    re.IGNORECASE,
)

_LEARNING_START_PATTERN = re.compile(
    r"^(?:start|begin|resume|continue)\s+(?:learning|studying|working\s+on|with\s+topic|topic)\s+(.+)$"
    r"|^(?:start|begin|resume|continue)\s+(it|that|this|that\s+topic|this\s+topic|(?:the\s+)?(?:first|second|third|1st|2nd|3rd)(?:\s+one|\s+topic)?)$"
    r"|^i(?:'ve| have|'m| am)?\s+(?:just\s+|now\s+)?(?:started|begun|learning|studying|working\s+on)\s+(?:learning\s+|studying\s+|the\s+topic\s+|topic\s+)?(.+)$"
    r"|^mark\s+(.+?)\s+as\s+(?:learning|in\s+progress|studying|started)$",
    re.IGNORECASE,
)

_LEARNING_COMPLETE_PATTERN = re.compile(
    r"^(?:complete|finish|master)\s+(?:the\s+)?(?:topic|learning\s+topic|lesson|module)\s+(.+)$"
    r"|^mark\s+(?:(?:the\s+)?(?:topic|learning\s+topic)\s+)?(.+?)\s+(?:as\s+)?(?:learned|mastered|completed|done|finished)$",
    re.IGNORECASE,
)

_LEARNING_FINISHED_PATTERN = re.compile(
    r"^(?:now\s+)?i(?:'ve|'d| have| had| was| am|'m)?\s+"
    r"(?:just\s+|already\s+|finally\s+|now\s+)?"
    r"(?:comp\w*|finis\w*ed|finis\w*ht|finish|learn(?:ed|t)|master\w*|done\s+with)\s+"
    r"(?:learning\s+|studying\s+|the\s+topic\s+|topic\s+|with\s+)?(.+)$",
    re.IGNORECASE,
)

_LEARNING_HOLD_PATTERN = re.compile(
    r"^(?:put|place|set|move)\s+(?:(?:the\s+)?(?:topic|learning\s+topic)\s+)?(.+?)\s+(?:on\s+hold|to\s+on\s+hold|as\s+on\s+hold)$"
    r"|^(?:pause|postpone|hold|defer)\s+(?:learning\s+|studying\s+|(?:the\s+)?topic\s+)?(.+)$",
    re.IGNORECASE,
)

_LEARNING_REOPEN_PATTERN = re.compile(
    r"^(?:reopen|re-open|reset|redo|uncomplete|restart)\s+(?:learning\s+|studying\s+|(?:the\s+)?topic\s+)(.+)$"
    r"|^mark\s+(?:(?:the\s+)?topic\s+)?(.+?)\s+as\s+(?:not\s+started|unlearned|incomplete)$",
    re.IGNORECASE,
)

_LEARNING_ROADMAP_CREATE_PATTERN = re.compile(
    r"^(?:create|add|start|make)\s+(?:a\s+)?(?:new\s+)?"
    r"(?:(?P<cat>academic|technical|personal)\s+)?(?:learning\s+)?roadmap\s+"
    r"(?:called\s+|named\s+|for\s+|on\s+)?(?P<name>.+?)"
    r"(?:\s+in\s+(?P<cat2>academic|technical|personal)(?:\s+category)?)?$",
    re.IGNORECASE,
)

_LEARNING_ROADMAP_DELETE_PATTERN = re.compile(
    r"^(?:delete|remove|drop)\s+(?:the\s+|my\s+)?(?:learning\s+)?roadmap\s+(?:called\s+|named\s+)?(.+)$",
    re.IGNORECASE,
)

_LEARNING_TOPIC_CREATE_PATTERN = re.compile(
    r"^(?:add|create)\s+(?:a\s+)?(?:new\s+)?(?:learning\s+)?topic\s+"
    r"(?:called\s+|named\s+)?(?P<topic>.+?)\s+(?:to|in|under|into|for)\s+"
    r"(?:the\s+|my\s+)?(?P<roadmap>.+?)(?:\s+roadmap)?$",
    re.IGNORECASE,
)

_LEARNING_TOPIC_DELETE_PATTERN = re.compile(
    r"^(?:delete|remove|drop)\s+(?:the\s+|my\s+)?(?:learning\s+)?topic\s+(?:called\s+|named\s+)?(.+)$",
    re.IGNORECASE,
)

_LEARNING_TOPIC_PRIORITY_PATTERN = re.compile(
    r"^(?:set|change|update|make)\s+(?:the\s+)?(?:priority\s+of\s+)?(?:learning\s+)?topic\s+(.+?)\s+"
    r"(?:priority\s+)?(?:to\s+|as\s+)?(high|medium|low|1|2|3)(?:\s+priority)?$",
    re.IGNORECASE,
)

_LEARNING_LINKS_LIST_PATTERN = re.compile(
    r"^(?:(?:show|list|view|check|what\s+are)\s+(?:me\s+)?)?"
    r"(?:(?:my|the|all|pending|suggested)\s+)*"
    r"(?:suggested\s+learning\s+links?|learning\s+links?|task[\s\-]topic\s+links?|topic\s+links?)$",
    re.IGNORECASE,
)

_LEARNING_LINK_ACTION_PATTERN = re.compile(
    r"^(?P<action>confirm|accept|approve|reject|dismiss|decline)\s+"
    r"(?:(?P<all>all)\s+)?(?:(?:the|my|suggested)\s+)*(?:learning\s+|task[\s\-]topic\s+)?links?"
    r"(?:\s+(?:for\s+|#)(?P<target>.+))?$",
    re.IGNORECASE,
)

_LEARNING_ACTIVITY_PATTERN = re.compile(
    r"^(?:(?:show|list|view|check|what\s+is)\s+(?:me\s+)?)?"
    r"(?:(?:my|the|recent|latest)\s+)*"
    r"(?:learning\s+activity|learning\s+activities|learning\s+history|learning\s+log|study\s+history|study\s+log)$",
    re.IGNORECASE,
)

_LEARNING_NUDGES_PATTERN = re.compile(
    r"^(?:(?P<enable>enable|turn\s+on|activate|unmute|resume)"
    r"|(?P<disable>disable|turn\s+off|deactivate|mute|pause))"
    r"\s+(?:my\s+|the\s+)?learning\s+(?:nudges?|reminders?|notifications?)$"
    r"|^(?:learning\s+nudges?\s+(?P<state>on|off|status))$",
    re.IGNORECASE,
)

_LEARNING_BARE_PATTERN = re.compile(
    r"^(?:(?:show|check|view|list|what(?:'s| is| are))\s+(?:me\s+)?)?"
    r"(?:(?:my|the|all|current)\s+)*"
    r"(?:learning|learning\s+roadmaps?|roadmaps?|learning\s+tracker|study\s+plan|learning\s+goals?)$",
    re.IGNORECASE,
)

_LEARNING_PRONOUNS = {
    "it", "that", "this", "this one", "that one", "the one",
    "the topic", "that topic", "this topic", "the lesson", "that lesson",
    "the roadmap", "that roadmap", "this roadmap",
    "first", "the first", "first one", "the first one", "first topic", "the first topic", "1st", "the 1st",
    "second", "the second", "second one", "the second one", "second topic", "the second topic", "2nd", "the 2nd",
    "third", "the third", "third one", "the third one", "third topic", "the third topic", "3rd", "the 3rd",
    "both", "both of them", "them", "all", "all of them",
}


def _clean_topic_reference(reference: str) -> str:
    """Drop a trailing 'mark it done/complete…' clause from a spoken topic.

    Users often tack an instruction onto the confession, e.g. "i had completed
    HTML Fundamentals mark it done". The topic name is everything before that.
    """
    cleaned = re.sub(
        r"\s+(?:and\s+)?mark\s+\S.*$", "", reference, flags=re.IGNORECASE
    ).strip()
    cleaned = re.sub(
        r"\s+(?:in|from|under)\s+(?:my\s+|the\s+)?learning(?:\s+tracker|\s+roadmap)?$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip()
    return cleaned.strip("\"'`?!.,;:")


def _extract_learning_context(
    chat_id: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool]:
    """Return (mentioned_topics, mentioned_roadmaps, is_most_recent_learning) from recent assistant turns."""
    if not chat_id or not learning_schema.is_initialized():
        return [], [], False
    messages = memory_store.get_messages(chat_id)
    assistant_msgs = [
        m["content"] for m in reversed(messages) if m.get("role") == "assistant"
    ][:3]
    if not assistant_msgs:
        return [], [], False

    all_topics = learning_progress.list_topics()
    all_roadmaps = learning_progress.roadmap_summaries()

    def _scan_msg(text: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        lower_text = text.lower()
        found_topics: list[tuple[int, dict[str, Any]]] = []
        for topic in all_topics:
            t_name = topic["name"].strip().lower()
            pos = lower_text.find(t_name)
            if pos != -1:
                found_topics.append((pos, topic))
        found_topics.sort(key=lambda pair: pair[0])

        found_roadmaps: list[tuple[int, dict[str, Any]]] = []
        for rm in all_roadmaps:
            r_name = rm["name"].strip().lower()
            pos = lower_text.find(r_name)
            if pos != -1:
                found_roadmaps.append((pos, rm))
        found_roadmaps.sort(key=lambda pair: pair[0])
        return [t for _, t in found_topics], [r for _, r in found_roadmaps]

    most_recent_lower = assistant_msgs[0].lower()
    most_recent_topics, most_recent_roadmaps = _scan_msg(assistant_msgs[0])
    has_learning_cue = any(
        cue in most_recent_lower
        for cue in (
            "ready to learn",
            "roadmap",
            "next-topic",
            "in progress",
            "what should i learn next",
            "start learning",
            "reopen learning",
            "prerequisites",
            "unlocked",
            "learning link",
            "learning tracker",
        )
    )
    is_most_recent_learning = bool(
        has_learning_cue or most_recent_topics or most_recent_roadmaps
    ) and not bool(re.search(r"#\d+\b", assistant_msgs[0]))

    for content in assistant_msgs:
        topics, roadmaps = _scan_msg(content)
        if topics or roadmaps:
            return topics, roadmaps, is_most_recent_learning

    return [], [], False


def _resolve_topic_from_context_or_ref(
    reference: str, chat_id: str
) -> list[dict[str, Any]]:
    """Return 1+ matching topics either from pronoun/ordinal context or by name/fuzzy lookup."""
    cleaned = _clean_topic_reference(reference)
    lowered = re.sub(r"\s+", " ", cleaned.lower()).strip()
    if lowered in _LEARNING_PRONOUNS and chat_id:
        ctx_topics, ctx_roadmaps, _ = _extract_learning_context(chat_id)
        if not ctx_topics and ctx_roadmaps:
            next_in_rm = learning_progress.next_topics(
                limit=3, roadmap_id=ctx_roadmaps[0]["id"]
            )
            if next_in_rm:
                found = learning_progress.find_topic(next_in_rm[0]["name"])
                if found:
                    ctx_topics = [found]
        if ctx_topics:
            if lowered in {"both", "both of them", "them", "all", "all of them"}:
                return ctx_topics
            ordinal = (
                1
                if ("second" in lowered or "2nd" in lowered)
                else (2 if ("third" in lowered or "3rd" in lowered) else 0)
            )
            chosen = ctx_topics[ordinal] if ordinal < len(ctx_topics) else ctx_topics[0]
            fresh = learning_progress.find_topic(chosen["name"])
            return [fresh or chosen]
    topic = learning_progress.find_topic(cleaned)
    return [topic] if topic is not None else []


def _resolve_roadmap_from_context_or_ref(
    reference: str, chat_id: str
) -> dict[str, Any] | None:
    cleaned = reference.strip().strip("\"'`?!.,;:")
    lowered = re.sub(r"\s+", " ", cleaned.lower()).strip()
    if lowered in _LEARNING_PRONOUNS and chat_id:
        _, ctx_roadmaps, _ = _extract_learning_context(chat_id)
        if ctx_roadmaps:
            ordinal = (
                1
                if ("second" in lowered or "2nd" in lowered)
                else (2 if ("third" in lowered or "3rd" in lowered) else 0)
            )
            chosen = ctx_roadmaps[ordinal] if ordinal < len(ctx_roadmaps) else ctx_roadmaps[0]
            return learning_progress.find_roadmap(chosen["name"]) or chosen
    return learning_progress.find_roadmap(cleaned)


def _learning_progress_reply() -> tuple[str, str]:
    summary = learning_progress.overall_progress()
    if summary["topic_count"] == 0:
        return (
            "Your learning tracker is empty. Open the Learning page to add a "
            "roadmap, then I can coach you through it.",
            "local-learning",
        )
    lines = [
        f"- **{item['roadmap_name']}**: {item['completed']}/{item['total']} "
        f"({item['percent']}%)"
        for item in summary["per_roadmap"]
    ]
    return (
        f"You've completed **{summary['completed_topics']} of "
        f"{summary['topic_count']}** topics "
        f"({summary['completion_percent']}%) across "
        f"{summary['roadmap_count']} roadmaps.\n"
        + "\n".join(lines)
        + "\n\nSay 'what should I learn next' for a suggestion.",
        "local-learning",
    )


def _learning_roadmap_progress_reply(
    reference: str, chat_id: str = ""
) -> tuple[str, str]:
    roadmap = _resolve_roadmap_from_context_or_ref(reference, chat_id)
    if roadmap is None:
        return _learning_unknown_roadmap_reply(reference)
    reply = (
        f"**{roadmap['name']}**: you've completed "
        f"**{roadmap['completed_count']} of {roadmap['topic_count']}** topics "
        f"({roadmap['completion_percent']}%), with "
        f"{roadmap['in_progress_count']} in progress."
    )
    if roadmap["next_topic"]:
        reply += f" Next up: **{roadmap['next_topic']}**."
    else:
        reply += " There's nothing unlocked left to learn in this roadmap."
    return reply, "local-learning"


def _learning_roadmap_topics_reply(
    reference: str, chat_id: str = ""
) -> tuple[str, str]:
    roadmap = _resolve_roadmap_from_context_or_ref(reference, chat_id)
    if roadmap is None:
        return _learning_unknown_roadmap_reply(reference)
    detail = learning_progress.get_roadmap(roadmap["id"])
    topics = (detail or {}).get("topics") or []
    if not topics:
        return (
            f"**{roadmap['name']}** doesn't have any topics yet. "
            f"Say 'add topic <name> to {roadmap['name']}' to add one.",
            "local-learning",
        )
    status_badges = {
        "completed": "✅ Completed",
        "learning": "🔄 In progress",
        "on_hold": "⏸️ On hold",
        "not_started": "🔓 Ready",
    }
    lines = []
    for t in topics:
        if t["status"] == "not_started" and not t.get("available", True):
            prereqs = ", ".join(t.get("unmet_prerequisites") or [])
            badge = f"🔒 Locked (needs {prereqs})"
        else:
            badge = status_badges.get(t["status"], t["status"])
        lines.append(f"- **{t['name']}** — {badge}")
    return (
        f"Topics in **{roadmap['name']}** ({roadmap['completed_count']}/{roadmap['topic_count']} completed):\n"
        + "\n".join(lines),
        "local-learning",
    )


def _learning_status_topics_reply(raw_filter: str) -> tuple[str, str]:
    norm = re.sub(r"[\s\-]+", " ", raw_filter.strip().lower())
    if norm in {"in progress", "active", "current", "currently learning", "learning", "studying"}:
        status = "learning"
        label = "currently in progress"
        empty_msg = "You don't have any learning topics in progress right now. Say 'what should I learn next' to pick one!"
    elif norm in {"completed", "finished", "learned", "mastered"}:
        status = "completed"
        label = "completed"
        empty_msg = "You haven't marked any learning topics completed yet."
    elif norm in {"on hold", "paused"}:
        status = "on_hold"
        label = "on hold"
        empty_msg = "You don't have any learning topics on hold."
    else:
        status = "not_started"
        label = "not started"
        empty_msg = "All topics are either in progress or completed!"

    items = learning_progress.list_topics(status=status)
    if not items:
        return (empty_msg, "local-learning")
    lines = [
        f"- **{item['name']}** ({item['roadmap_name']})"
        for item in items
    ]
    return (
        f"Here are your learning topics **{label}** ({len(items)}):\n"
        + "\n".join(lines),
        "local-learning",
    )


def _learning_unknown_roadmap_reply(reference: str) -> tuple[str, str]:
    names = ", ".join(
        item["name"] for item in learning_progress.roadmap_summaries()
    )
    reply = f"I couldn't find a learning roadmap named **{reference}**."
    if names:
        reply += f" Your roadmaps are: {names}."
    return reply, "local-learning"


def _learning_next_reply(
    roadmap_reference: str | None = None,
    *,
    suggest_task: bool = False,
    chat_id: str = "",
) -> tuple[str, str]:
    roadmap = None
    if roadmap_reference:
        roadmap = _resolve_roadmap_from_context_or_ref(roadmap_reference, chat_id)
        if roadmap is None:
            return _learning_unknown_roadmap_reply(roadmap_reference)
    items = learning_progress.next_topics(
        limit=3, roadmap_id=roadmap["id"] if roadmap else None
    )
    if not items:
        scope = f" in {roadmap['name']}" if roadmap else ""
        return (
            f"Everything that's unlocked{scope} is already completed — add a "
            "new topic or roadmap when you're ready.",
            "local-learning",
        )
    if suggest_task:
        first = items[0]
        return (
            f"Try this task for **{first['roadmap_name']}**: "
            f"**Practice {first['name']}**. {first['reason']} "
            f"Say 'start learning {first['name']}' when you begin.",
            "local-learning",
        )
    lines = [
        f"- **{item['name']}** ({item['roadmap_name']}) — {item['reason']}"
        for item in items
    ]
    header = "Here's what's ready to learn"
    if roadmap:
        header += f" in **{roadmap['name']}**"
    return (
        header + ":\n"
        + "\n".join(lines)
        + "\n\nSay 'start learning <topic>' when you begin one.",
        "local-learning",
    )


def _learning_start_reply(
    reference: str, name: str = "", chat_id: str = ""
) -> tuple[str, str]:
    topics = _resolve_topic_from_context_or_ref(reference, chat_id)
    if not topics:
        raise ValueError(f"No topic matched '{reference}'.")
    topic = topics[0]
    if topic["status"] == "completed":
        return (
            f"**{topic['name']}** is already completed — say 'reopen "
            f"learning {topic['name']}' only if you really want to redo it.",
            "local-learning",
        )
    if topic["status"] == "learning":
        return (
            f"**{topic['name']}** is already in progress. Keep going!"
            + (f" {name.split()[0]}, you've got this." if name else ""),
            "local-learning",
        )
    unmet = learning_progress.unavailable_prerequisite_names(topic["id"])
    if unmet:
        return (
            f"**{topic['name']}** isn't unlocked yet — finish "
            f"{', '.join(unmet)} first.",
            "local-learning",
        )
    learning_progress.set_topic_status(topic["id"], "learning")
    return (
        f"Marked **{topic['name']}** as in progress. I'll keep it at the "
        f"top of your next-topic list.",
        "local-learning",
    )


def _learning_complete_reply(
    reference: str, name: str = "", chat_id: str = ""
) -> tuple[str, str]:
    topics = _resolve_topic_from_context_or_ref(reference, chat_id)
    if not topics:
        raise ValueError(f"No topic matched '{reference}'.")
    if len(topics) > 1:
        names = []
        for t in topics:
            learning_progress.set_topic_status(t["id"], "completed")
            names.append(f"**{t['name']}**")
        return (
            f"Nice work — marked {', '.join(names)} as completed. "
            "Say 'what should I learn next' to keep rolling.",
            "local-learning",
        )
    topic = topics[0]
    if topic["status"] == "completed":
        return (
            f"**{topic['name']}** was already marked completed.",
            "local-learning",
        )
    learning_progress.set_topic_status(topic["id"], "completed")
    return (
        f"Nice work — **{topic['name']}** is marked completed"
        + (f", {name.split()[0]}" if name else "")
        + ". Say 'what should I learn next' to keep rolling.",
        "local-learning",
    )


def _learning_hold_reply(
    reference: str, chat_id: str = ""
) -> tuple[str, str]:
    topics = _resolve_topic_from_context_or_ref(reference, chat_id)
    if not topics:
        raise ValueError(f"No topic matched '{reference}'.")
    topic = topics[0]
    learning_progress.set_topic_status(topic["id"], "on_hold")
    return (
        f"Put **{topic['name']}** on hold. Say 'start learning {topic['name']}' "
        "whenever you're ready to pick it back up.",
        "local-learning",
    )


def _learning_reopen_reply(
    reference: str, chat_id: str = ""
) -> tuple[str, str]:
    topics = _resolve_topic_from_context_or_ref(reference, chat_id)
    if not topics:
        raise ValueError(f"No topic matched '{reference}'.")
    topic = topics[0]
    learning_progress.set_topic_status(topic["id"], "not_started")
    return (
        f"Reopened **{topic['name']}** and reset its status to not started.",
        "local-learning",
    )


def _learning_command(
    prompt: str, name: str = "", chat_id: str = ""
) -> tuple[str, str] | None:
    """Handle learning tracker commands; requires the schema to be ready."""
    if not learning_schema.is_initialized():
        return None
    normalized = _normalize_command(prompt)
    if _LEARNING_PROGRESS_PATTERN.fullmatch(normalized) or _LEARNING_BARE_PATTERN.fullmatch(normalized):
        return _learning_progress_reply()
    if _LEARNING_NEXT_PATTERN.fullmatch(normalized):
        return _learning_next_reply(chat_id=chat_id)

    # Learning activity log
    if _LEARNING_ACTIVITY_PATTERN.fullmatch(normalized):
        activities = learning_progress.list_activities(limit=8)
        if not activities:
            return ("No learning activity recorded yet. Start a topic to log your progress!", "local-learning")
        lines = [
            f"- **{a['topic_name']}** ({a['roadmap_name']}): {a['kind']}"
            + (f" via task *{a['task_title']}*" if a.get("task_title") else "")
            for a in activities
        ]
        return ("Recent learning activity:\n" + "\n".join(lines), "local-learning")

    # Learning nudges toggle / status
    nudges_match = _LEARNING_NUDGES_PATTERN.fullmatch(normalized)
    if nudges_match:
        state_word = (nudges_match.group("state") or "").lower()
        if state_word == "status":
            enabled = learning_progress.get_nudges_enabled()
            return (
                f"Learning nudges are currently **{'enabled' if enabled else 'disabled'}**.",
                "local-learning",
            )
        enable = bool(nudges_match.group("enable")) or state_word == "on"
        learning_progress.set_nudges_enabled(enable)
        return (
            f"Learning nudges are now **{'enabled' if enable else 'disabled'}**.",
            "local-learning",
        )

    # Suggested task-topic links list & confirmation
    if _LEARNING_LINKS_LIST_PATTERN.fullmatch(normalized):
        links = learning_progress.list_suggested_links()
        if not links:
            return ("You don't have any pending suggested task-topic links right now.", "local-learning")
        lines = [
            f"- Link #{l['id']}: Task **{l['task_title']}** → Topic **{l['topic_name']}** ({l['roadmap_name']})"
            for l in links[:10]
        ]
        return (
            "Suggested task-topic learning links:\n"
            + "\n".join(lines)
            + "\n\nSay 'confirm learning link' or 'reject learning link' to manage them.",
            "local-learning",
        )

    link_act = _LEARNING_LINK_ACTION_PATTERN.fullmatch(normalized)
    if link_act:
        action = link_act.group("action").lower()
        new_state = "confirmed" if action in {"confirm", "accept", "approve"} else "rejected"
        links = learning_progress.list_suggested_links()
        if not links:
            return ("There are no pending suggested learning links right now.", "local-learning")
        target_str = (link_act.group("target") or "").strip()
        if link_act.group("all"):
            for l in links:
                learning_progress.update_task_link(l["id"], new_state)
            return (
                f"{new_state.capitalize()} all {len(links)} suggested learning links.",
                "local-learning",
            )
        chosen = links[0]
        if target_str:
            if target_str.isdigit():
                lid = int(target_str)
                chosen = next((l for l in links if l["id"] == lid), chosen)
            else:
                t_low = target_str.lower()
                chosen = next(
                    (
                        l for l in links
                        if t_low in l["topic_name"].lower() or t_low in l["task_title"].lower()
                    ),
                    chosen,
                )
        learning_progress.update_task_link(chosen["id"], new_state)
        return (
            f"{new_state.capitalize()} link between task **{chosen['task_title']}** and topic **{chosen['topic_name']}**.",
            "local-learning",
        )

    # Create / delete roadmap
    rm_create = _LEARNING_ROADMAP_CREATE_PATTERN.fullmatch(normalized)
    if rm_create:
        rm_name = rm_create.group("name").strip(" \"'")
        cat = (rm_create.group("cat") or rm_create.group("cat2") or "technical").lower()
        created = learning_progress.create_roadmap(rm_name, category=cat)
        return (
            f"Created **{created['category']}** learning roadmap **{created['name']}**. "
            f"Say 'add topic <name> to {created['name']}' to add topics.",
            "local-learning",
        )

    rm_delete = _LEARNING_ROADMAP_DELETE_PATTERN.fullmatch(normalized)
    if rm_delete:
        ref = rm_delete.group(1).strip()
        roadmap = _resolve_roadmap_from_context_or_ref(ref, chat_id)
        if roadmap is None:
            return _learning_unknown_roadmap_reply(ref)
        learning_progress.delete_roadmap(roadmap["id"])
        return (f"Deleted learning roadmap **{roadmap['name']}**.", "local-learning")

    # Create / delete / reprioritize topic
    t_create = _LEARNING_TOPIC_CREATE_PATTERN.fullmatch(normalized)
    if t_create:
        t_name = t_create.group("topic").strip(" \"'")
        rm_ref = t_create.group("roadmap").strip(" \"'")
        roadmap = _resolve_roadmap_from_context_or_ref(rm_ref, chat_id)
        if roadmap is None:
            return _learning_unknown_roadmap_reply(rm_ref)
        created = learning_progress.create_topic(roadmap["id"], t_name)
        return (
            f"Added topic **{created['name']}** to **{roadmap['name']}**.",
            "local-learning",
        )

    t_delete = _LEARNING_TOPIC_DELETE_PATTERN.fullmatch(normalized)
    if t_delete:
        ref = t_delete.group(1).strip()
        topics = _resolve_topic_from_context_or_ref(ref, chat_id)
        if not topics:
            raise ValueError(f"No topic matched '{ref}'.")
        topic = topics[0]
        learning_progress.delete_topic(topic["id"])
        return (f"Removed topic **{topic['name']}**.", "local-learning")

    t_prio = _LEARNING_TOPIC_PRIORITY_PATTERN.fullmatch(normalized)
    if t_prio:
        ref = t_prio.group(1).strip()
        prio_raw = t_prio.group(2).lower()
        prio_map = {"low": 1, "1": 1, "medium": 2, "2": 2, "high": 3, "3": 3}
        prio_label = {1: "low", 2: "medium", 3: "high"}
        prio_val = prio_map[prio_raw]
        topics = _resolve_topic_from_context_or_ref(ref, chat_id)
        if not topics:
            raise ValueError(f"No topic matched '{ref}'.")
        updated = learning_progress.update_topic(topics[0]["id"], priority=prio_val)
        return (
            f"Updated **{updated['name']}** priority to **{prio_label[prio_val]}**.",
            "local-learning",
        )

    # Filter topics by status ("what am I currently learning", "completed topics", "topics on hold")
    status_match = _LEARNING_STATUS_LIST_PATTERN.fullmatch(normalized)
    if status_match:
        raw_filter = (
            status_match.group("filter")
            or status_match.group("filter2")
            or status_match.group("filter3")
            or "learning"
        )
        return _learning_status_topics_reply(raw_filter)

    # List topics in a specific roadmap
    rm_topics_match = _LEARNING_ROADMAP_TOPICS_PATTERN.fullmatch(normalized)
    if rm_topics_match:
        reference = next(group for group in rm_topics_match.groups() if group)
        return _learning_roadmap_topics_reply(reference.strip(), chat_id=chat_id)

    match = _LEARNING_ROADMAP_PROGRESS_PATTERN.fullmatch(normalized)
    if match:
        reference = next(group for group in match.groups() if group)
        return _learning_roadmap_progress_reply(reference.strip(), chat_id=chat_id)
    match = _LEARNING_ROADMAP_TASK_PATTERN.fullmatch(normalized)
    if match:
        reference = next(group for group in match.groups() if group)
        return _learning_next_reply(reference.strip(), suggest_task=True, chat_id=chat_id)
    match = _LEARNING_ROADMAP_NEXT_PATTERN.fullmatch(normalized)
    if match:
        reference = next(group for group in match.groups() if group)
        return _learning_next_reply(reference.strip(), chat_id=chat_id)

    # Pause / On hold
    hold_match = _LEARNING_HOLD_PATTERN.fullmatch(normalized)
    if hold_match:
        reference = next(group for group in hold_match.groups() if group)
        return _learning_hold_reply(reference.strip(), chat_id=chat_id)

    # Reopen / Reset
    reopen_match = _LEARNING_REOPEN_PATTERN.fullmatch(normalized)
    if reopen_match:
        reference = next(group for group in reopen_match.groups() if group)
        return _learning_reopen_reply(reference.strip(), chat_id=chat_id)

    match = _LEARNING_START_PATTERN.fullmatch(normalized)
    if match:
        reference = next(group for group in match.groups() if group)
        return _learning_start_reply(reference.strip(), name=name, chat_id=chat_id)
    match = _LEARNING_COMPLETE_PATTERN.fullmatch(normalized)
    if match:
        reference = next(group for group in match.groups() if group)
        return _learning_complete_reply(reference.strip(), name=name, chat_id=chat_id)
    match = _LEARNING_FINISHED_PATTERN.fullmatch(normalized)
    if match:
        return _learning_complete_reply(match.group(1).strip(), name=name, chat_id=chat_id)
    return None


_MEMORY_CREATE_PATTERN = re.compile(
    r"^(?:(?:add|create|save|write|take)\s+(?:a\s+)?(?:new\s+)?(?:note|memory)"
    r"(?:\s+(?:that|about|for))?(?:\s*[:\-]\s*|\s+)"
    r"|remember\s+(?:that\s+|this\s*[:\-]?\s*)"
    r"|note\s+down(?:\s*[:\-]\s*|\s+))(.+)$",
    re.IGNORECASE,
)

_MEMORY_LIST_PATTERN = re.compile(
    r"^(?:(?:show|list|check|view|read|what\s+are)\s+(?:me\s+)?)?"
    r"(?:(?:my|the|all|saved)\s+)*(?:notes?|memories|saved\s+notes?|saved\s+memories)$",
    re.IGNORECASE,
)

_MEMORY_DELETE_PATTERN = re.compile(
    r"^(?:delete|remove|forget)\s+(?:(?:the|my)\s+)?(?:note|memory)\s+(?:#?)(.+)$",
    re.IGNORECASE,
)


def _memory_command(prompt: str) -> tuple[str, str] | None:
    """Deterministic CRUD handler for notes and memories."""
    normalized = _normalize_command(prompt)
    create_match = _MEMORY_CREATE_PATTERN.fullmatch(normalized)
    if create_match:
        content = create_match.group(1).strip(" \"'")
        if not content:
            return ("What would you like me to save in your notes?", "local-memory")
        try:
            saved = memory_store.add_memory(content)
            return (f"Saved to your notes: **{saved['content']}**", "local-memory")
        except ValueError as err:
            if "already exists" in str(err).lower():
                return (f"I already have **{content}** in your notes.", "local-memory")
            raise

    delete_match = _MEMORY_DELETE_PATTERN.fullmatch(normalized)
    if delete_match:
        ref = delete_match.group(1).strip(" \"'")
        memories = memory_store.list_memories()
        target_mem = None
        if ref.isdigit():
            mem_id = int(ref)
            target_mem = next((m for m in memories if m["id"] == mem_id), None)
        if target_mem is None:
            lowered = ref.lower()
            matches = [m for m in memories if lowered in m["content"].lower()]
            if len(matches) == 1:
                target_mem = matches[0]
            elif len(matches) > 1:
                raise ValueError(f"More than one note matches '{ref}'. Be more specific.")
        if target_mem is None:
            raise ValueError(f"No saved note matched '{ref}'.")
        memory_store.delete_memory(target_mem["id"])
        return (f"Removed note: **{target_mem['content']}**.", "local-memory")

    if _MEMORY_LIST_PATTERN.fullmatch(normalized):
        memories = memory_store.list_memories()
        if not memories:
            return ("You don't have any saved notes yet. Say 'add note: <text>' to save one.", "local-memory")
        lines = [f"- #{m['id']} {m['content']}" for m in memories[:15]]
        return ("Here are your saved notes:\n" + "\n".join(lines), "local-memory")

    return None


_PRONOUN_TARGETS = {
    "it", "that", "this", "this one", "that one", "the one",
    "the task", "the habit", "that task", "that habit", "this task", "this habit",
    "the topic", "that topic", "this topic",
    "first", "the first", "first one", "the first one", "1st", "the 1st",
    "second", "the second", "second one", "the second one", "2nd", "the 2nd",
    "third", "the third", "third one", "the third one", "3rd", "the 3rd",
    "both", "both of them", "them", "all", "all of them", "everything",
    "all tasks", "all my tasks", "all habits", "all my habits",
}


def _extract_context_entities(
    chat_id: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (mentioned_tasks, mentioned_habits) from the recent assistant messages in chat_id."""
    messages = memory_store.get_messages(chat_id) if chat_id else []
    assistant_msgs = [
        m["content"] for m in reversed(messages) if m.get("role") == "assistant"
    ][:3]
    all_habits = memory_store.list_habits()
    all_tasks = memory_store.list_tasks(include_completed=True)

    for content in assistant_msgs:
        lower_content = content.lower()
        task_ids = [int(tid) for tid in re.findall(r"#(\d+)\b", content)]
        if task_ids and (
            "task" in lower_content
            or "list" in lower_content
            or "still to do" in lower_content
            or "priority" in lower_content
            or "next up" in lower_content
        ):
            matched_tasks = [
                t for tid in task_ids for t in all_tasks if t["id"] == tid
            ]
            if matched_tasks:
                return matched_tasks, []

        mentioned_habits: list[tuple[int, dict[str, Any]]] = []
        for habit in all_habits:
            h_name = habit["name"].strip().lower()
            pos = lower_content.find(h_name)
            if pos != -1:
                mentioned_habits.append((pos, habit))
        if mentioned_habits and (
            "habit" in lower_content
            or "streak" in lower_content
            or "tackle" in lower_content
            or "list today" in lower_content
            or "checked off" in lower_content
        ):
            mentioned_habits.sort(key=lambda pair: pair[0])
            return [], [h for _, h in mentioned_habits]

        mentioned_tasks: list[tuple[int, dict[str, Any]]] = []
        for task in all_tasks:
            t_title = task["title"].strip().lower()
            pos = lower_content.find(t_title)
            if pos != -1:
                mentioned_tasks.append((pos, task))
        if mentioned_tasks:
            mentioned_tasks.sort(key=lambda pair: pair[0])
            return [t for _, t in mentioned_tasks], []

    return [], []


def _resolve_contextual_completion(
    target: str, chat_id: str, name: str = ""
) -> tuple[str, str] | None:
    """Resolve pronouns ('it', 'that', 'the first one', 'both', 'all') using recent chat history."""
    cleaned = re.sub(r"\s+", " ", target.strip().lower())
    if cleaned not in _PRONOUN_TARGETS:
        return None

    # Check if the most recent assistant message was about learning topics
    ctx_topics, _, is_recent_learning = _extract_learning_context(chat_id)
    if (is_recent_learning or "topic" in cleaned) and ctx_topics and "task" not in cleaned and "habit" not in cleaned:
        return _learning_complete_reply(cleaned, name=name, chat_id=chat_id)

    all_habits = memory_store.list_habits()
    pending_tasks = memory_store.list_tasks(include_completed=False)
    today = datetime.now(LOCAL_TIMEZONE).date().isoformat()
    pending_habits = [h for h in all_habits if today not in h["completed_dates"]]

    # Handle explicit batch targets ("all tasks", "all habits")
    if cleaned in {"all tasks", "all my tasks"}:
        if not pending_tasks:
            return ("All your tasks are already completed!", "local-task-manager")
        titles = [t["title"] for t in pending_tasks]
        for t in pending_tasks:
            _complete_task(t, name=name)
        return (
            f"Marked all {len(titles)} pending tasks completed: "
            + ", ".join(f"**{title}**" for title in titles) + ".",
            "local-task-manager",
        )
    if cleaned in {"all habits", "all my habits"}:
        if not pending_habits:
            return ("All your habits are already checked off for today!", "local-habit-manager")
        names = [h["name"] for h in pending_habits]
        for h in pending_habits:
            _complete_habit(h, name=name)
        return (
            f"Checked off all {len(names)} habits for today: "
            + ", ".join(f"**{n}**" for n in names) + ".",
            "local-habit-manager",
        )

    ctx_tasks, ctx_habits = _extract_context_entities(chat_id)
    is_batch = cleaned in {"both", "both of them", "them", "all", "all of them", "everything"}

    if is_batch:
        if ctx_tasks:
            open_ctx = [t for t in ctx_tasks if not t["completed"]] or ctx_tasks
            for t in open_ctx:
                _complete_task(t, name=name)
            return (
                "Completed: " + ", ".join(f"**{t['title']}**" for t in open_ctx) + ".",
                "local-task-manager",
            )
        if ctx_habits:
            for h in ctx_habits:
                _complete_habit(h, name=name)
            return (
                "Checked off: " + ", ".join(f"**{h['name']}**" for h in ctx_habits) + ".",
                "local-habit-manager",
            )
        if ctx_topics:
            return _learning_complete_reply(cleaned, name=name, chat_id=chat_id)

    ordinal = 0
    if "second" in cleaned or "2nd" in cleaned:
        ordinal = 1
    elif "third" in cleaned or "3rd" in cleaned:
        ordinal = 2

    if ctx_tasks and "habit" not in cleaned:
        chosen = ctx_tasks[ordinal] if ordinal < len(ctx_tasks) else ctx_tasks[0]
        return (_complete_task(chosen, name=name), "local-task-manager")

    if ctx_habits and "task" not in cleaned:
        chosen = ctx_habits[ordinal] if ordinal < len(ctx_habits) else ctx_habits[0]
        return (_complete_habit(chosen, name=name), "local-habit-manager")

    if ctx_topics and "task" not in cleaned and "habit" not in cleaned:
        return _learning_complete_reply(cleaned, name=name, chat_id=chat_id)

    # Fallback: if there is globally only 1 pending task, complete it
    if len(pending_tasks) == 1 and ("habit" not in cleaned) and ("topic" not in cleaned):
        return (_complete_task(pending_tasks[0], name=name), "local-task-manager")

    if len(pending_habits) == 1 and ("task" not in cleaned) and ("topic" not in cleaned):
        return (_complete_habit(pending_habits[0], name=name), "local-habit-manager")

    return None


def _resolve_contextual_mutation(
    prompt: str, chat_id: str, name: str = ""
) -> tuple[str, str] | None:
    """Handle contextual 'delete it', 'make it high priority', 'undo it', or 'reopen it'."""
    normalized = _normalize_command(prompt).lower()
    ctx_topics, _, is_recent_learning = _extract_learning_context(chat_id)

    # 1. Contextual Delete: "delete it", "remove that", "delete the first one"
    del_match = re.fullmatch(
        r"(?:delete|remove|drop|cancel)\s+"
        r"(it|that|this|that\s+one|this\s+one|the\s+task|the\s+habit|the\s+topic|that\s+topic|"
        r"(?:the\s+)?(?:first|second|third|1st|2nd|3rd)(?:\s+one)?)",
        normalized,
        re.IGNORECASE,
    )
    if del_match:
        target_ref = del_match.group(1)
        ordinal = 1 if ("second" in target_ref or "2nd" in target_ref) else (2 if ("third" in target_ref or "3rd" in target_ref) else 0)
        if (is_recent_learning or "topic" in target_ref) and ctx_topics and "task" not in target_ref and "habit" not in target_ref:
            chosen_t = ctx_topics[ordinal] if ordinal < len(ctx_topics) else ctx_topics[0]
            learning_progress.delete_topic(chosen_t["id"])
            return (f"Removed topic **{chosen_t['name']}**.", "local-learning")
        ctx_tasks, ctx_habits = _extract_context_entities(chat_id)
        if ctx_tasks and "habit" not in target_ref:
            chosen = ctx_tasks[ordinal] if ordinal < len(ctx_tasks) else ctx_tasks[0]
            memory_store.delete_task(chosen["id"])
            remaining = len(memory_store.list_tasks(include_completed=False))
            return (assistant_tone.task_deleted(chosen["title"], remaining), "local-task-manager")
        if ctx_habits and "task" not in target_ref:
            chosen = ctx_habits[ordinal] if ordinal < len(ctx_habits) else ctx_habits[0]
            memory_store.delete_habit(chosen["id"])
            return (assistant_tone.habit_deleted(chosen["name"]), "local-habit-manager")

    # 2. Contextual Priority Update: "make it high priority", "set its priority to low"
    prio_match = re.fullmatch(
        r"(?:make|set|change|update)\s+"
        r"(?:it|that|this|its\s+priority|the\s+priority|the\s+task|the\s+topic)\s+"
        r"(?:to\s+)?(?:be\s+)?(?:priority\s+)?(low|medium|high)(?:\s+priority)?",
        normalized,
        re.IGNORECASE,
    )
    if prio_match:
        new_prio = prio_match.group(1).lower()
        if (is_recent_learning or "topic" in normalized) and ctx_topics and "task" not in normalized:
            prio_val = {"low": 1, "medium": 2, "high": 3}[new_prio]
            updated_t = learning_progress.update_topic(ctx_topics[0]["id"], priority=prio_val)
            return (f"Updated **{updated_t['name']}** priority to **{new_prio}**.", "local-learning")
        ctx_tasks, _ = _extract_context_entities(chat_id)
        if ctx_tasks:
            chosen = ctx_tasks[0]
            updated = memory_store.update_task(chosen["id"], priority=new_prio)
            return (assistant_tone.task_priority(updated["title"], updated["priority"]), "local-task-manager")

    # 3. Contextual Undo / Reopen / Uncheck: "undo it", "uncheck it", "reopen it", "mark it not done"
    undo_match = re.fullmatch(
        r"(?:undo|uncheck|un-check|reopen|re-open|uncomplete|reset)\s+"
        r"(it|that|this|the\s+task|the\s+habit|the\s+topic|that\s+topic)"
        r"|mark\s+(it|that|this)\s+(?:as\s+)?(?:not\s+done|not\s+completed|not\s+started|incomplete|open|pending)",
        normalized,
        re.IGNORECASE,
    )
    if undo_match:
        if (is_recent_learning or "topic" in normalized) and ctx_topics and "task" not in normalized and "habit" not in normalized:
            return _learning_reopen_reply("it", chat_id=chat_id)
        ctx_tasks, ctx_habits = _extract_context_entities(chat_id)
        if ctx_tasks:
            chosen = ctx_tasks[0]
            updated = memory_store.update_task(chosen["id"], completed=False)
            return (assistant_tone.task_reopened(updated["title"], name=name), "local-task-manager")
        if ctx_habits:
            chosen = ctx_habits[0]
            return (_uncomplete_habit(chosen, name=name), "local-habit-manager")

    return None


_BRIEFING_PATTERN = re.compile(
    r"^(?:(?:show|give|tell|check|what(?:'s| is))\s+(?:me\s+)?)?"
    r"(?:(?:my|the|today'?s|daily|morning|evening|current)\s+)*"
    r"(?:briefing|daily\s+summary|summary|status|overview|day\s+summary|"
    r"dashboard\s+status|main\s+page\s+status|plate|on\s+my\s+plate)"
    r"|^how\s+am\s+i\s+doing(?:\s+today)?$"
    r"|^what(?:'s| is)\s+on\s+my\s+plate(?:\s+today)?$",
    re.IGNORECASE,
)

_XP_STATUS_PATTERN = re.compile(
    r"^(?:(?:show|check|what(?:'s| is| are))\s+(?:me\s+)?)?"
    r"(?:(?:my|the|current)\s+)*"
    r"(?:xp|level|gamification|achievements?|streaks?|global\s+streak|score|points)$",
    re.IGNORECASE,
)


def _xp_status_command(prompt: str) -> tuple[str, str] | None:
    normalized = _normalize_command(prompt)
    if not _XP_STATUS_PATTERN.fullmatch(normalized):
        return None
    stats = memory_store.get_gamification()
    xp = stats.get("xp", 0)
    level = (xp // 100) + 1
    streak = stats.get("global_streak", 0)
    habits = memory_store.list_habits()
    top_habit = max(habits, key=lambda h: h.get("streak", 0)) if habits else None
    lines = [
        f"⚡ **Level {level}** ({xp} XP) — 🔥 **{streak}-day global streak**"
    ]
    if top_habit and top_habit.get("streak", 0) > 0:
        lines.append(f"- Top habit streak: **{top_habit['name']}** ({top_habit['streak']} days)")
    achievements = stats.get("achievements") or []
    if achievements:
        unlocked = [a.get("title") or a.get("name") or str(a) for a in achievements if isinstance(a, dict) and a.get("unlocked", True)]
        if unlocked:
            lines.append(f"- Achievements: {', '.join(unlocked[:5])}")
    return ("\n".join(lines), "local-gamification")


def _daily_briefing_command(prompt: str) -> tuple[str, str] | None:
    normalized = _normalize_command(prompt)
    if not _BRIEFING_PATTERN.fullmatch(normalized):
        return None
    today = datetime.now(LOCAL_TIMEZONE).date().isoformat()
    open_tasks = memory_store.list_tasks(include_completed=False)
    all_habits = memory_store.list_habits()
    pending_habits = [h for h in all_habits if today not in h["completed_dates"]]
    done_habits = [h for h in all_habits if today in h["completed_dates"]]
    reminders = memory_store.list_reminders()
    stats = memory_store.get_gamification()
    xp = stats.get("xp", 0)
    level = (xp // 100) + 1

    sections = [f"**Daily Briefing** (Level {level} • {xp} XP)"]
    if open_tasks:
        preview = ", ".join(f"#{t['id']} **{t['title']}** ({t['priority']})" for t in open_tasks[:3])
        sections.append(f"- **Tasks ({len(open_tasks)} pending):** {preview}")
    else:
        sections.append("- **Tasks:** All caught up! ✅")

    if all_habits:
        if pending_habits:
            h_preview = ", ".join(f"**{h['name']}**" for h in pending_habits)
            sections.append(
                f"- **Habits ({len(done_habits)}/{len(all_habits)} done today):** Still open: {h_preview}"
            )
        else:
            sections.append(f"- **Habits ({len(all_habits)}/{len(all_habits)}):** All completed today! ✅")
    else:
        sections.append("- **Habits:** None tracked yet.")

    if reminders:
        r_preview = ", ".join(f"**{r['text']}** ({_format_due_date(r['run_at'])})" for r in reminders[:2])
        sections.append(f"- **Reminders ({len(reminders)} active):** {r_preview}")

    if learning_schema.is_initialized():
        try:
            next_items = learning_progress.next_topics(limit=1)
            if next_items:
                sections.append(
                    f"- **Next Learning Topic:** **{next_items[0]['name']}** ({next_items[0]['roadmap_name']})"
                )
        except Exception:
            pass

    return ("\n".join(sections), "local-briefing")


def _handle_local_command(prompt: str, chat_id: str) -> tuple[str, str] | None:
    """Answer tasks, habits, reminders, notes, greetings, and identity locally."""
    page = _parse_navigation(prompt)
    if page:
        return (f"Opening {page}.", f"local-navigate:{page}")

    briefing = _daily_briefing_command(prompt)
    if briefing is not None:
        return briefing

    xp_status = _xp_status_command(prompt)
    if xp_status is not None:
        return xp_status

    profile = _profile_answer(prompt)
    if profile is not None:
        return (profile, "local-profile")

    greeting = _greeting_answer(prompt)
    if greeting is not None:
        return (greeting, "local-greeting")

    intro = _assistant_intro_answer(prompt)
    if intro is not None:
        return (intro, "local-identity")

    identity = _identity_answer(prompt)
    if identity is not None:
        return (identity, "local-identity")

    memory_cmd = _memory_command(prompt)
    if memory_cmd is not None:
        return memory_cmd

    # Entity commands address the user by name when a name was saved.
    name = _user_name(_user_fact_memories())

    ctx_mutation = _resolve_contextual_mutation(prompt, chat_id, name=name)
    if ctx_mutation is not None:
        return ctx_mutation

    reminder_request = _parse_chat_reminder(prompt)
    if reminder_request:
        reminder_text, run_at, recurrence = reminder_request
        _validate_recurrence(recurrence)
        reminder = memory_store.create_reminder(
            reminder_text, run_at, recurrence, chat_id
        )
        _schedule_reminder(reminder)
        return (
            assistant_tone.reminder_scheduled(
                reminder["text"],
                reminder["run_at"],
                recurrence=reminder.get("recurrence"),
                name=name,
            ),
            "local-scheduler",
        )

    clarification = _reminder_clarification(prompt)
    if clarification is not None:
        return (clarification, "local-scheduler")

    reminder_reschedule = _parse_reminder_reschedule(prompt)
    if reminder_reschedule:
        reference, when = reminder_reschedule
        reminder = _find_reminder(reference)
        run_at = _parse_reminder_time(when)
        if run_at is None:
            raise ValueError(f"I could not understand the reminder time: {when}.")
        try:
            reminder_scheduler.remove_job(reminder["id"])
        except Exception:
            pass
        memory_store.cancel_reminder(reminder["id"])
        updated = memory_store.create_reminder(
            reminder["text"],
            run_at.isoformat(),
            reminder["recurrence"],
            reminder["chat_id"] or chat_id,
        )
        _schedule_reminder(updated)
        return (
            assistant_tone.reminder_moved(
                updated["text"], updated["run_at"], name=name
            ),
            "local-scheduler",
        )

    reminder_reference = _parse_reminder_cancel(prompt)
    if reminder_reference is not None:
        reminder = _find_reminder(reminder_reference)
        try:
            reminder_scheduler.remove_job(reminder["id"])
        except Exception:
            pass
        memory_store.cancel_reminder(reminder["id"])
        return (
            assistant_tone.reminder_cancelled(reminder["text"], name=name),
            "local-scheduler",
        )

    if _is_reminder_list_request(prompt):
        return (
            _format_reminder_list(memory_store.list_reminders()),
            "local-scheduler",
        )

    task_request = _parse_task_create(prompt)
    if task_request:
        title, priority = task_request
        task = memory_store.create_task(title, priority)
        _run_learning_hook(lambda: _learning_suggest_for_task(task["id"]))
        open_count = len(memory_store.list_tasks(include_completed=False))
        return (
            assistant_tone.task_created(
                task["title"], task["priority"], open_count, name=name
            ),
            "local-task-manager",
        )

    if _is_task_list_request(prompt):
        normalized_prompt = _normalize_command(prompt).lower()
        wants_completed = bool(
            re.search(r"\b(?:completed|done|finished|closed)\b", normalized_prompt)
        ) and not re.search(
            r"\b(?:pending|open|remaining|left|incomplete|unfinished|need to complete|to complete|need to finish)\b"
            r"|\bnot\s+(?:done|completed|finished)\b",
            normalized_prompt,
        )
        tasks = [
            task for task in memory_store.list_tasks(include_completed=True)
            if bool(task["completed"]) == wants_completed
        ]
        return (
            _format_task_list(tasks, completed=wants_completed),
            "local-task-manager",
        )

    task_update = _parse_task_update(prompt)
    if task_update:
        field, reference, value = task_update
        task = _find_task_reference(reference)
        if task is None:
            raise ValueError(f"No task matched '{reference}'.")
        if field == "priority":
            updated = memory_store.update_task(task["id"], priority=value)
            return (
                assistant_tone.task_priority(updated["title"], updated["priority"]),
                "local-task-manager",
            )
        if field == "title":
            updated = memory_store.update_task(task["id"], title=value)
            return (
                assistant_tone.task_renamed(updated["title"]),
                "local-task-manager",
            )
        try:
            parsed_due = _parse_reminder_time(value)
        except (TypeError, ValueError):
            parsed_due = None
        if parsed_due is None:
            raise ValueError(f"I could not understand the due date: {value}.")
        updated = memory_store.update_task(
            task["id"], due_at=parsed_due.isoformat()
        )
        return (
            assistant_tone.task_due(updated["title"], updated["due_at"]),
            "local-task-manager",
        )

    task_delete = _parse_task_delete(prompt)
    if task_delete is not None:
        task = _find_task_reference(task_delete)
        if task is None:
            raise ValueError(f"No task matched '{task_delete}'.")
        memory_store.delete_task(task["id"])
        remaining = len(memory_store.list_tasks(include_completed=False))
        return (
            assistant_tone.task_deleted(task["title"], remaining),
            "local-task-manager",
        )

    task_reopen = _parse_task_reopen(prompt)
    if task_reopen is not None:
        task = _find_task_reference(task_reopen)
        if task is None:
            raise ValueError(f"No task matched '{task_reopen}'.")
        if not task["completed"]:
            return (
                f"**{task['title']}** was still open — nothing to reopen.",
                "local-task-manager",
            )
        updated = memory_store.update_task(task["id"], completed=False)
        return (
            assistant_tone.task_reopened(updated["title"], name=name),
            "local-task-manager",
        )

    habit_request = _parse_chat_habit(prompt)
    if habit_request:
        habit_name, icon = habit_request
        habit = memory_store.create_habit(habit_name, icon)
        total_habits = len(memory_store.list_habits())
        return (
            assistant_tone.habit_created(habit["name"], total_habits),
            "local-habit-manager",
        )

    if _is_habit_list_request(prompt) and re.search(
        r"\b(?:have|having|all|my habits)\b", _normalize_command(prompt).lower()
    ) and not re.search(
        r"\b(?:pending|left|remaining|incomplete)\b", prompt.lower()
    ):
        habits = memory_store.list_habits()
        if not habits:
            return ("You don't have any habits set up yet.", "local-habit-manager")
        today = datetime.now(LOCAL_TIMEZONE).date().isoformat()
        rows = "\n".join(
            f"- **{h['name']}** — streak {h['streak']}"
            f"{' ✅' if today in h['completed_dates'] else ''}"
            for h in habits
        )
        return (f"Here are your habits:\n{rows}", "local-habit-manager")

    if _is_habit_list_request(prompt):
        today = datetime.now(LOCAL_TIMEZONE).date().isoformat()
        all_habits = memory_store.list_habits()
        pending_habits = [
            habit for habit in all_habits if today not in habit["completed_dates"]
        ]
        return (
            _format_habit_list(all_habits, pending_habits),
            "local-habit-manager",
        )

    if _is_habit_follow_up(prompt, memory_store.get_messages(chat_id)):
        today = datetime.now(LOCAL_TIMEZONE).date().isoformat()
        habits = memory_store.list_habits()
        pending_habits = [
            habit for habit in habits if today not in habit["completed_dates"]
        ]
        if not habits:
            response = (
                "I don't see any habits set up yet. "
                "What would you like your first one to be?"
            )
        elif not pending_habits:
            response = "You're all set for today. What else would you like to tackle?"
        else:
            response = (
                f"Let's take them one at a time. **{pending_habits[0]['name']}** "
                "feels like a good place to start."
            )
        return (response, "local-habit-manager")

    habit_rename = _parse_habit_rename(prompt)
    if habit_rename:
        reference, new_name = habit_rename
        habit = _find_habit(reference)
        if habit is None:
            raise ValueError(f"No habit matched '{reference}'.")
        updated = memory_store.update_habit(habit["id"], name=new_name)
        return (assistant_tone.habit_renamed(updated["name"]), "local-habit-manager")

    habit_delete = _parse_habit_delete(prompt)
    if habit_delete is not None:
        habit = _find_habit(habit_delete)
        if habit is None:
            raise ValueError(f"No habit matched '{habit_delete}'.")
        memory_store.delete_habit(habit["id"])
        return (assistant_tone.habit_deleted(habit["name"]), "local-habit-manager")

    habit_uncomplete = _parse_habit_uncomplete(prompt)
    if habit_uncomplete is not None:
        habit = _find_habit(habit_uncomplete)
        if habit is None:
            raise ValueError(f"No habit matched '{habit_uncomplete}'.")
        return (_uncomplete_habit(habit, name=name), "local-habit-manager")

    habit_completion = _parse_habit_completion(prompt)
    if habit_completion is not None:
        contextual = _resolve_contextual_completion(habit_completion, chat_id, name=name)
        if contextual is not None:
            return contextual
        habit = _find_habit(habit_completion)
        if habit is None:
            raise ValueError(f"No habit matched '{habit_completion}'.")
        return (_complete_habit(habit, name=name), "local-habit-manager")

    task_completion = _parse_task_completion(prompt)
    if task_completion is not None:
        contextual = _resolve_contextual_completion(task_completion, chat_id, name=name)
        if contextual is not None:
            return contextual
        task = _find_task_reference(task_completion)
        if task is None:
            raise ValueError(f"No task matched '{task_completion}'.")
        return (_complete_task(task, name=name), "local-task-manager")

    completion_target = _parse_completion_target(prompt)
    if completion_target is not None:
        contextual = _resolve_contextual_completion(completion_target, chat_id, name=name)
        if contextual is not None:
            return contextual
        pending_task = _find_task_reference(
            completion_target, include_completed=False
        )
        habit = _find_habit(completion_target)
        if pending_task and habit:
            raise ValueError(
                f"I found both a task and a habit called '{completion_target}'. "
                f"Say 'complete task {completion_target}' or "
                f"'complete habit {completion_target}'."
            )
        if pending_task:
            return (_complete_task(pending_task, name=name), "local-task-manager")
        if habit:
            return (_complete_habit(habit, name=name), "local-habit-manager")
        completed_task = _find_task_reference(completion_target)
        if completed_task is not None:
            return (_complete_task(completed_task, name=name), "local-task-manager")
        # No task or habit matched — let learning commands have a look.

    learning_command = _learning_command(prompt, name=name, chat_id=chat_id)
    if learning_command is not None:
        return learning_command

    if completion_target is not None and learning_schema.is_initialized():
        matched_topic = learning_progress.find_topic(
            _clean_topic_reference(completion_target)
        )
        if matched_topic is not None:
            return _learning_complete_reply(
                completion_target, name=name, chat_id=chat_id
            )

    web_query = _parse_web_search_query(prompt)
    if web_query:
        return _web_search_reply(web_query)

    normalized = _normalize_command(prompt)
    if _ACTION_WITHOUT_HANDLER.search(normalized) and not _is_advice_question(
        normalized.lower()
    ):
        return (
            "I couldn't parse that as a command, so nothing was changed. "
            "Try 'add habit exercise' or 'complete task explore AI basics'.",
            "local-help",
        )
    return None


def select_generation_context(
    context: list[dict[str, str]], prompt: str
) -> list[dict[str, str]]:
    words = prompt.split()
    if len(words) <= 4 and len(context) > 1:
        # Short replies such as "now", "yes", or "that one" depend on the
        # assistant's last answer as much as the user's earlier messages.
        return context[-4:]
    return context


def resolve_short_follow_up(
    prompt: str, context: list[dict[str, str]]
) -> str:
    normalized = prompt.strip().lower()
    if "cricket" in normalized and "captain" in normalized:
        return (
            "Name one real cricket captain. Answer with only one captain's "
            "name and one short sentence."
        )
    if not re.fullmatch(r"(name|give) (one|any one|an example)", normalized):
        return prompt
    previous_user_messages = [
        message["content"].lower()
        for message in context[:-1]
        if message["role"] == "user"
    ]
    topic = " ".join(previous_user_messages[-2:])
    if "cricket" in topic or "captain" in topic:
        return (
            "Name one real cricket captain. Answer with only one captain's "
            "name and one short sentence."
        )
    return prompt


def _productivity_context(prompt: str) -> str:
    """Current habits/tasks/learning, injected when the question is about them or the dashboard."""
    if not re.search(
        r"\b(?:habits?|tasks?|to[\s-]?dos?|routine|schedule|productiv\w*|"
        r"improve\w*|discipline|goals?|learning|studying|study|topics?|course\w*|"
        r"skills?|practice|main\s+page|dashboard|page|app|nothing|there|pending|"
        r"completed|done|list)\b",
        prompt,
        re.IGNORECASE,
    ):
        return ""
    today = datetime.now(LOCAL_TIMEZONE).date().isoformat()
    lines = []
    habits = memory_store.list_habits()
    if habits:
        lines.append("Current habits:")
        for habit in habits:
            done = "done today" if today in habit["completed_dates"] else "not done today"
            lines.append(f"- {habit['name']} (streak {habit['streak']}, {done})")
    else:
        lines.append("Current habits: none yet.")
    tasks = memory_store.list_tasks(include_completed=False)
    if tasks:
        lines.append("Pending tasks:")
        lines.extend(f"- #{t['id']} {t['title']} ({t['priority']})" for t in tasks[:8])
    else:
        lines.append("Pending tasks: none.")
    lines.extend(_learning_digest())
    return "\n".join(lines)


def _learning_digest() -> list[str]:
    """Short learning snapshot for the LLM, or nothing when not initialized."""
    if not learning_schema.is_initialized():
        return []
    try:
        overview = learning_progress.overall_progress()
    except Exception:  # pragma: no cover - defensive
        return []
    if overview["topic_count"] == 0:
        return []
    digest = [
        "Learning progress: "
        f"{overview['completed_topics']} of {overview['topic_count']} "
        f"topics completed ({overview['completion_percent']}%), "
        f"{overview['in_progress_topics']} in progress."
    ]
    upcoming = learning_progress.next_topics(limit=3)
    if upcoming:
        digest.append(
            "Next learning topics: "
            + ", ".join(
                f"{topic['name']} ({topic['roadmap_name']})" for topic in upcoming
            )
            + "."
        )
    return digest


def _trim_incomplete(text: str) -> str:
    """Cut a reply back to its last finished sentence or line."""
    ends = list(re.finditer(r"[.!?](?=\s|$)|\n", text))
    if ends and ends[-1].end() >= len(text) * 0.4:
        text = text[: ends[-1].end()]
    if text.count("**") % 2:
        text = text.rsplit("**", 1)[0]
    return text.rstrip()


def generate_response(
    prompt: str,
    max_new_tokens: int,
    messages: list[dict[str, str]] | None = None,
    memories: list[dict[str, Any]] | None = None,
    on_token: Any | None = None,
) -> str:
    global model, tokenizer
    ensure_model_loaded()
    if model is None or tokenizer is None:
        from runtime.models import REGISTRY
        if REGISTRY.is_ready("llm"):
            model, tokenizer = REGISTRY.get("llm")
        else:
            raise RuntimeError("The language model is currently unavailable.")
    if not prompt.strip():
        raise ValueError("Prompt cannot be empty.")
    if len(prompt) > MAX_PROMPT_CHARS:
        raise ValueError(
            f"Prompt is limited to {MAX_PROMPT_CHARS} characters, "
            f"but received {len(prompt)}."
        )

    with generation_lock:
        context = messages or []
        if not context or context[-1]["content"] != prompt:
            context = [*context, {"role": "user", "content": prompt}]
        memory_text = "\n".join(f"- {item['content']}" for item in (memories or []))
        system_prompt = DEFAULT_SYSTEM_PROMPT
        if memory_text:
            system_prompt += (
                "\nNotes saved from earlier conversations. Lines written in the "
                "first person are facts the user shared about themselves; treat "
                "them as background about the user and address them as 'you'. "
                "Lines about you describe you. Use them whenever they are "
                f"relevant and never contradict them:\n{memory_text}"
            )
        words = int(max_new_tokens * 0.6)
        system_prompt += (
            f"\nKeep this reply under about {words} words and finish your "
            "last sentence. Answer in plain conversational sentences: no "
            "headings, no emoji, no bullet lists unless the user asks for a list."
        )
        extra = _productivity_context(prompt)
        if extra:
            system_prompt += (
                "\nThe user's live habit and task data (use it to answer; "
                "for advice questions, give 3-5 specific suggestions and do "
                "not just list pending items):\n" + extra
            )
        if os.environ.get("ORBIT_LLM_TOOLS", "0") == "1":
            system_prompt += (
                "\nYou can create or delete reminders using tool calls. "
                "Output: <tool_call>{\"action\": \"create_reminder\", \"title\": \"...\", \"when\": \"...\"}</tool_call>"
            )
        model_context = select_generation_context(context, prompt)
        model_messages = [{"role": "system", "content": system_prompt}, *model_context]
        process_started = time.perf_counter()
        inputs = tokenizer.apply_chat_template(
            model_messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
            enable_thinking=False,
        )
        input_device = next(
            (
                torch.device(device)
                for device in getattr(model, "hf_device_map", {}).values()
                if str(device).startswith("cuda")
            ),
            torch.device("cuda:0" if cuda_available else "cpu"),
        )
        inputs = inputs.to(input_device)
        input_len = inputs["input_ids"].shape[-1]

        if cuda_available:
            torch.cuda.synchronize()
        process_seconds = time.perf_counter() - process_started
        generation_started = time.perf_counter()
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.05,
            )
        if cuda_available:
            torch.cuda.synchronize()
        generation_seconds = time.perf_counter() - generation_started
        generated_tokens = outputs.shape[-1] - input_len
        response = tokenizer.decode(
            outputs[0][input_len:], skip_special_tokens=True
        ).strip()
        if generated_tokens >= max_new_tokens:
            response = _trim_incomplete(response)
        print(
            f"[input {process_seconds:.2f}s | generation {generation_seconds:.2f}s | "
            f"{generated_tokens} tokens | "
            f"{generated_tokens / generation_seconds:.2f} tokens/s]"
        )
        if cuda_available:
            print(
                "[GPU memory: "
                f"{torch.cuda.memory_allocated(0) / 1024**3:.2f} GiB allocated, "
                f"{torch.cuda.memory_reserved(0) / 1024**3:.2f} GiB reserved]"
            )
        return response


@asynccontextmanager
async def lifespan(_: FastAPI):
    global MAIN_LOOP
    MAIN_LOOP = asyncio.get_running_loop()
    _cleanup_tts()
    init_db()
    from runtime.guard import acquire_model_process_lock, release_model_process_lock
    acquire_model_process_lock()
    try:
        from runtime.loaders import load_all_models_eager
        await asyncio.to_thread(
            load_all_models_eager,
            MODEL_DIR,
            GPU_MEMORY_LIMIT,
            CPU_MEMORY_LIMIT,
            generation_lock,
        )
        await asyncio.to_thread(ensure_model_loaded)
        _restore_reminders(provider="local")
        try:
            migrated_count = _migrate_legacy_reminders()
            if migrated_count:
                logging.getLogger("orbit007").info("Migrated %d legacy reminders to MemoryStore", migrated_count)
        except Exception as e:
            logging.getLogger("orbit007").warning("Legacy reminder migration error: %s", e)
        init_learning_module("local")
        purge_voice_tmp()

        discord_token = os.environ.get("AI_DISCORD_TOKEN", "").strip()
        if discord_token:
            try:
                from discord_bot import start_discord_bot_in_process
                await start_discord_bot_in_process(discord_token)
            except Exception as err:
                print(f"Warning: Failed to start in-process Discord bot: {err}")

        yield
    finally:
        discord_token = os.environ.get("AI_DISCORD_TOKEN", "").strip()
        if discord_token:
            try:
                from discord_bot import stop_discord_bot_in_process
                await stop_discord_bot_in_process()
            except Exception:
                pass
        if reminder_scheduler.running:
            reminder_scheduler.shutdown(wait=False)
        release_model_process_lock()


cors_origins_env = os.environ.get(
    "ORBIT_CORS_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000",
)
ORBIT_CORS_ORIGINS = [orig.strip() for orig in cors_origins_env.split(",") if orig.strip()]

API_TOKEN = os.environ.get("ORBIT_API_TOKEN", "").strip()
_OPEN_PATHS = ("/health", "/api/health", "/tts/")

app = FastAPI(title="ORBIT007 Backend", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ORBIT_CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def token_guard(request: Request, call_next):
    if (
        API_TOKEN
        and request.method != "OPTIONS"
        and not request.url.path.startswith(_OPEN_PATHS)
    ):
        supplied = request.headers.get("x-orbit-token", "")
        if not secrets.compare_digest(supplied, API_TOKEN):
            return JSONResponse({"detail": "Unauthorized"}, status_code=401)
    return await call_next(request)


async def ws_authorized(ws: WebSocket) -> bool:
    if not API_TOKEN:
        return True
    ok = secrets.compare_digest(ws.query_params.get("token", ""), API_TOKEN)
    if not ok:
        await ws.close(code=1008)
    return ok
app.mount("/tts", StaticFiles(directory=str(TTS_DIR)), name="tts")


@app.exception_handler(Exception)
async def handle_unexpected_error(request: Request, error: Exception):
    print(f"Unhandled API error: {error}")
    return JSONResponse(
        status_code=500,
        content={"error": "The backend could not complete that request."},
    )


@app.exception_handler(HTTPException)
async def handle_http_error(request: Request, error: HTTPException):
    detail = error.detail
    return JSONResponse(
        status_code=error.status_code,
        content={
            "detail": detail if isinstance(detail, str) else str(detail),
            "error": detail if isinstance(detail, str) else str(detail),
        },
    )


# Web research lives in its own package; this is the only wiring app.py needs.
app.include_router(research_router)
# Adaptive learning tracker: tables, seeds, and its API surface.
app.include_router(learning_router)

# Voice subsystem
async def _voice_chat_turn(prompt: str, chat_id: str | None = None):
    payload = {"prompt": prompt}
    if chat_id:
        payload["chat_id"] = chat_id
    return await generate(payload)

set_voice_chat_handler(_voice_chat_turn)
app.include_router(voice_router)
app.include_router(voice_ws_router)


@app.get("/api/health")
@app.get("/health")
async def health():
    from runtime.models import REGISTRY
    return {
        "status": "ok",
        "db_connected": DB_PATH.exists(),
        "db_path": str(DB_PATH),
        "api_version": "spidey-local-2026-11",
        "model": "Qwen3-4B",
        "cuda": cuda_available,
        "load_seconds": round(load_seconds, 2) if load_seconds is not None else 0.0,
        "database": str(DB_PATH),
        "semantic_memory": os.environ.get("AI_ENABLE_SEMANTIC_MEMORY", "1") != "0",
        "scheduler_running": reminder_scheduler.running if "reminder_scheduler" in globals() else False,
        "scheduled_jobs": len(reminder_scheduler.get_jobs()) if "reminder_scheduler" in globals() and reminder_scheduler.running else 0,
        "active_ws_clients": len(ws_manager.active) if "ws_manager" in globals() else 0,
        "models": REGISTRY.get_status_summary() if hasattr(REGISTRY, "get_status_summary") else {},
    }


TOKENS_DEFAULT = 64
TOKENS_BRIEF = 100
TOKENS_MEDIUM = 200
TOKENS_LONG = 300


def _token_budget(prompt: str) -> int:
    """Pick a reply length from what the user is asking for."""
    text = prompt.lower()
    if re.search(
        r"\b(?:brief|briefly|in short|short answer|one line|one sentence|"
        r"quick|tl;?dr|yes or no|in a word)\b",
        text,
    ):
        return TOKENS_BRIEF
    if re.search(
        r"\b(?:longer|long answer|more detail|in detail|detailed|elaborate|"
        r"expand|step by step|full guide|deep dive|essay|code|script|"
        r"function|implement|program|debug|write (?:a|an|the))\b",
        text,
    ):
        return TOKENS_LONG
    if re.search(
        r"\b(?:explain|how (?:does|do|to|can)|compare|difference|why|"
        r"suggest\w*|recommend\w*|ideas?|tips?|advice|improve\w*|"
        r"pros and cons|list \d+|\d+ (?:habits|ideas|tips|ways))\b",
        text,
    ):
        return TOKENS_MEDIUM
    return TOKENS_DEFAULT


@app.post("/generate")
@app.post("/api/generate")
async def generate(payload: dict[str, Any] = Body(...)):
    await asyncio.to_thread(ensure_model_loaded)
    if not isinstance(payload, dict):
        raise HTTPException(400, "Request body must be a JSON object.")

    prompt = payload.get("prompt")
    max_new_tokens = payload.get("max_new_tokens")
    if not isinstance(prompt, str):
        raise HTTPException(400, "'prompt' must be a string.")
    if not prompt.strip():
        raise HTTPException(400, "Prompt cannot be empty.")
    if len(prompt) > MAX_PROMPT_CHARS:
        raise HTTPException(400, f"Prompt is limited to {MAX_PROMPT_CHARS} characters.")
    if max_new_tokens is None:
        max_new_tokens = _token_budget(prompt)
    if (
        isinstance(max_new_tokens, bool)
        or not isinstance(max_new_tokens, int)
        or not 1 <= max_new_tokens <= MAX_NEW_TOKENS
    ):
        raise HTTPException(
            400,
            detail=(
                f"'max_new_tokens' must be an integer between 1 and "
                f"{MAX_NEW_TOKENS}."
            )
        )

    chat_id = payload.get("chat_id")
    if chat_id is not None and (
        not isinstance(chat_id, str) or not memory_store.chat_exists(chat_id)
    ):
        raise HTTPException(404, "'chat_id' does not identify an existing chat.")

    if chat_id is None:
        chat = memory_store.create_chat()
        chat_id = chat["id"]

    try:
        command = _handle_local_command(prompt, chat_id)
        print(f"[route] {'local:' + command[1] if command else 'llm'} <- {prompt!r}")
        if command is not None:
            response, provider = command
            memory_store.add_message(chat_id, "user", prompt.strip())
            memory_store.add_message(chat_id, "assistant", response)
            return {
                "response": response,
                "chat_id": chat_id,
                "messages": memory_store.get_messages(chat_id),
                "provider": provider.split(":", 1)[0],
                "navigate": _page_for_provider(provider),
            }
        memory_store.add_message(chat_id, "user", prompt.strip())
        context = memory_store.build_context(chat_id, tokenizer, prompt)
        model_prompt = resolve_short_follow_up(prompt, context)
        if model_prompt != prompt and context:
            context[-1] = {"role": "user", "content": model_prompt}
        memories = _memories_for_prompt(prompt)
        response = await asyncio.to_thread(
            generate_response,
            model_prompt,
            max_new_tokens,
            context,
            memories,
        )
        memory_store.add_message(chat_id, "assistant", response)
        return {
            "response": response,
            "chat_id": chat_id,
            "messages": memory_store.get_messages(chat_id),
            "provider": "local-qwen",
            "navigate": None,
        }
    except ValueError as error:
        raise HTTPException(400, str(error)) from error


@app.get("/api/chats")
def list_chats():
    return {"chats": memory_store.list_chats()}


@app.post("/api/chats")
def create_chat():
    return JSONResponse(status_code=201, content=memory_store.create_chat())


@app.get("/api/history/{chat_id}")
def history(chat_id: str):
    if not memory_store.chat_exists(chat_id):
        raise HTTPException(404, "Conversation not found.")
    return {"messages": memory_store.get_messages(chat_id)}


@app.delete("/api/chats/{chat_id}")
def delete_chat(chat_id: str):
    if not memory_store.delete_chat(chat_id):
        raise HTTPException(404, "Conversation not found.")
    return {"status": "deleted"}


@app.delete("/api/chats")
def clear_chats():
    memory_store.clear_chats()
    return {"status": "deleted"}


@app.get("/api/memories")
def list_memories():
    return {"memories": memory_store.list_memories()}


@app.get("/api/tasks")
def list_tasks(include_completed: int = Query(0)):
    return {"tasks": memory_store.list_tasks(include_completed == 1)}


@app.post("/api/tasks")
def create_task(payload: dict[str, Any] = Body(...)):
    if not isinstance(payload, dict) or not isinstance(payload.get("title"), str):
        raise HTTPException(400, "'title' must be a string.")
    try:
        task = memory_store.create_task(
            payload["title"],
            payload.get("priority", "medium"),
            payload.get("due_at"),
            payload.get("description", ""),
        )
        _run_learning_hook(lambda: _learning_suggest_for_task(task["id"]))
        return JSONResponse(status_code=201, content=task)
    except ValueError as error:
        raise HTTPException(400, str(error)) from error


@app.patch("/api/tasks/{task_id}")
def update_task(task_id: int, payload: dict[str, Any] = Body(...)):
    if not isinstance(payload, dict):
        raise HTTPException(400, "Request body must be a JSON object.")
    chat_id = payload.get("chat_id")
    was_completed = any(
        item["id"] == task_id and item["completed"]
        for item in memory_store.list_tasks(include_completed=True)
    )
    try:
        task = memory_store.update_task(
            task_id,
            payload.get("title"),
            payload.get("priority"),
            payload.get("due_at"),
            payload.get("completed"),
            payload.get("description"),
            clear_due_at=(
                "due_at" in payload and payload.get("due_at") in (None, "")
            ),
        )
    except ValueError as error:
        raise HTTPException(400, str(error)) from error
    if task is None:
        raise HTTPException(404, "Task not found.")
    # Checking a task off in the UI deserves the same follow-up as telling me in chat.
    if payload.get("completed") and not was_completed:
        _notify_completion(
            chat_id,
            "task-coach",
            _task_completion_line(task, name=_user_name(_user_fact_memories())),
        )
        _run_learning_hook(lambda: _learning_on_task_completed(task_id))
    return task


@app.delete("/api/tasks/{task_id}")
def delete_task(task_id: int):
    if not memory_store.delete_task(task_id):
        raise HTTPException(404, "Task not found.")
    return {"status": "deleted"}


@app.get("/api/habits")
def list_habits():
    return {"habits": memory_store.list_habits()}


@app.post("/api/habits")
def create_habit(payload: dict[str, Any] = Body(...)):
    if not isinstance(payload, dict) or not isinstance(payload.get("name"), str):
        raise HTTPException(400, "'name' must be a string.")
    try:
        return JSONResponse(
            status_code=201,
            content=memory_store.create_habit(
                payload["name"],
                payload.get("icon", "Activity"),
                payload.get("daily_goal", 1),
            ),
        )
    except ValueError as error:
        raise HTTPException(400, str(error)) from error


@app.delete("/api/habits/{habit_id}")
def delete_habit(habit_id: int):
    if not memory_store.delete_habit(habit_id):
        raise HTTPException(404, "Habit not found.")
    return {"status": "deleted"}


@app.patch("/api/habits/{habit_id}")
def update_habit(habit_id: int, payload: dict[str, Any] = Body(...)):
    if not isinstance(payload, dict):
        raise HTTPException(400, "Request body must be a JSON object.")
    try:
        habit = memory_store.update_habit(
            habit_id,
            payload.get("name"),
            payload.get("icon"),
            payload.get("daily_goal"),
        )
    except ValueError as error:
        raise HTTPException(400, str(error)) from error
    if habit is None:
        raise HTTPException(404, "Habit not found.")
    return habit


@app.put("/api/habits/{habit_id}/completions/{completion_date}")
def set_habit_completion(
    habit_id: int, completion_date: str, payload: dict[str, Any] = Body(...)
):
    if not isinstance(payload, dict) or not isinstance(payload.get("completed"), bool):
        raise HTTPException(400, "'completed' must be a boolean.")
    was_done = any(
        item["id"] == habit_id and completion_date in item["completed_dates"]
        for item in memory_store.list_habits()
    )
    try:
        habit = memory_store.set_habit_completion(
            habit_id, completion_date, payload["completed"]
        )
    except ValueError as error:
        raise HTTPException(400, str(error)) from error
    if habit is None:
        raise HTTPException(404, "Habit not found.")
    # Same nudge as the chat path when a habit is ticked off in the UI.
    if payload["completed"] and not was_done:
        _notify_completion(
            payload.get("chat_id"),
            "habit-coach",
            _habit_completion_line(habit, name=_user_name(_user_fact_memories())),
        )
    return habit


@app.get("/api/gamification")
def get_gamification():
    return memory_store.get_gamification()


@app.post("/api/tasks/nudge")
def nudge_task(payload: dict[str, Any] = Body(...)):
    """Proactively suggest the top pending task if the user hasn't been nudged recently."""
    if not isinstance(payload, dict) or not isinstance(payload.get("chat_id"), str):
        raise HTTPException(400, "'chat_id' is required.")
    chat_id = payload["chat_id"]
    if not memory_store.chat_exists(chat_id):
        raise HTTPException(404, "Chat not found.")
    tasks = memory_store.list_tasks(include_completed=False)
    if not tasks:
        return {"nudge": False, "reason": "no_pending_tasks"}
    if memory_store.has_recent_coach_notification(chat_id, minutes=30):
        return {"nudge": False, "reason": "already_nudged_recently"}
    threading.Thread(
        target=_recommend_next_task,
        args=(chat_id,),
        daemon=True,
    ).start()
    return {"nudge": True, "task_count": len(tasks)}


class ChatPayload(BaseModel):
    message: str
    session_id: Optional[str] = "default"

class ReminderCreatePayload(BaseModel):
    title: Optional[str] = None
    remind_at: Optional[str] = None
    text: Optional[str] = None
    run_at: Optional[str] = None
    recurrence: Optional[str] = None
    chat_id: Optional[str] = None

class ReminderUpdatePayload(BaseModel):
    completed: bool = True

# legacy, unused
def parse_and_execute_reminder_intent(user_text: str) -> Optional[str]:
    """Fallback intent parser if the LLM does not emit structured tool calls."""
    match = re.search(
        r"remind me to (.+?) (?:at|on|in) (.+)", user_text, re.IGNORECASE
    )
    if match:
        title = match.group(1).strip()
        remind_at = match.group(2).strip()
        created = create_reminder(title=title, remind_at=remind_at)
        return f"Reminder set: '{created['title']}' for {created['remind_at']}."
    return None

def run_existing_llm(user_message: str, history: list[dict[str, Any]], active_reminders: list[dict[str, Any]]) -> str:
    """Run existing LLM inference + assistant tone logic without blocking FastAPI."""
    try:
        ensure_model_loaded()
        now_str = datetime.now(LOCAL_TIMEZONE).strftime("%Y-%m-%d %I:%M %p %Z")
        reminders_text = "\n".join(
            f"- #{r.get('id', '')} {r.get('title', '')} (due {r.get('remind_at', '')})"
            for r in active_reminders
        ) if active_reminders else "No active reminders."

        reminders_context = f"Current local time: {now_str}\nLive Database Reminders:\n{reminders_text}"
        memories = _memories_for_prompt(user_message) or []
        memories.append({"content": reminders_context})

        context_messages = [
            {"role": m.get("role", "user"), "content": m.get("content", "")}
            for m in history
            if m.get("role") and m.get("content")
        ]

        tokens = _token_budget(user_message)
        reply = generate_response(
            prompt=user_message,
            max_new_tokens=tokens,
            messages=context_messages,
            memories=memories,
        )
        return reply.strip()
    except Exception as exc:
        logging.getLogger("orbit007").warning("LLM inference failed: %s", exc)
        if active_reminders and any(k in user_message.lower() for k in ["remind", "task", "list", "what"]):
            items = ", ".join(f"'{r['title']}' for {r['remind_at']}" for r in active_reminders)
            return f"You currently have these active reminders: {items}."
        return f"Acknowledged: '{user_message}'. How else can I assist with your reminders and schedule?"

def run_existing_tts(text: str, output_path: str) -> None:
    """Synthesize speech into WAV audio file."""
    try:
        from voice.tts import get_tts_engine, audio_to_wav_bytes
        engine = get_tts_engine()
        engine.load()
        if engine.available:
            audio_array = engine.synthesize(text)
            if audio_array is not None and len(audio_array) > 0:
                wav_bytes = audio_to_wav_bytes(audio_array)
                with open(output_path, "wb") as f:
                    f.write(wav_bytes)
                return
    except Exception as err:
        logging.getLogger("orbit007").warning("OmniVoice TTS warning: %s", err)

    try:
        import wave, struct, math
        sr = 16000
        duration = 0.4
        with wave.open(output_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            frames = bytearray()
            for i in range(int(sr * duration)):
                val = int(2500 * math.sin(2 * math.pi * 130.81 * (i / sr)))
                frames.extend(struct.pack("<h", val))
            wf.writeframes(frames)
    except Exception as wav_err:
        logging.getLogger("orbit007").warning("Fallback WAV warning: %s", wav_err)

def transcribe_pcm_bytes(pcm_bytes: bytes) -> str:
    """Transcribe 16 kHz mono 16-bit PCM bytes."""
    try:
        from voice.stt import get_stt
        stt = get_stt()
        return stt.transcribe(pcm_bytes)
    except Exception as err:
        logging.getLogger("orbit007").warning("STT failed: %s", err)
        return ""

class ConnectionManager:
    def __init__(self) -> None:
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket) -> None:
        if ws in self.active:
            self.active.remove(ws)

    async def broadcast(self, payload: dict) -> None:
        for ws in list(self.active):
            try:
                await ws.send_json(payload)
            except Exception:
                self.disconnect(ws)


ws_manager = ConnectionManager()
MAIN_LOOP: asyncio.AbstractEventLoop | None = None
_tts_lock: asyncio.Lock | None = None


async def generate_tts_url(text: str) -> str | None:
    global _tts_lock
    if not text or not text.strip():
        return None
    _tts_lock = _tts_lock or asyncio.Lock()
    name = f"reply_{uuid.uuid4().hex[:12]}.wav"
    path = TTS_DIR / name
    try:
        async with _tts_lock:  # TTS engine is not guaranteed thread-safe
            await asyncio.to_thread(run_existing_tts, text, str(path))
        return f"/tts/{name}" if path.exists() else None
    except Exception:
        logging.getLogger("orbit007").exception("TTS failed")
        return None


def _cleanup_tts(max_age_s: int = 600) -> None:
    cutoff = time.time() - max_age_s
    for f in TTS_DIR.glob("reply_*.wav"):
        try:
            if f.stat().st_mtime < cutoff:
                f.unlink()
        except OSError:
            pass


def _handle_llm_tool_calls(
    llm_reply: str, user_message: str, chat_id: str | None = None
) -> str:
    """Parse and execute at most one whitelisted tool call if ORBIT_LLM_TOOLS=1 and intent words are present."""
    if os.environ.get("ORBIT_LLM_TOOLS", "0") != "1":
        return llm_reply

    intent_words = ("remind", "reminder", "alarm", "alert me")
    user_lower = user_message.lower()
    if not any(word in user_lower for word in intent_words):
        return re.sub(r"<tool_call>.*?</tool_call>", "", llm_reply, flags=re.DOTALL).strip()

    match = re.search(r"<tool_call>(.*?)</tool_call>", llm_reply, re.DOTALL)
    if not match:
        return llm_reply

    raw_json = match.group(1).strip()
    cleaned_reply = re.sub(r"<tool_call>.*?</tool_call>", "", llm_reply, flags=re.DOTALL).strip()

    try:
        call_data = json.loads(raw_json)
        action = call_data.get("action")
        if action == "create_reminder":
            title = call_data.get("title") or call_data.get("text")
            when = call_data.get("when") or call_data.get("remind_at") or call_data.get("run_at")
            recurrence = call_data.get("recurrence")
            if title and when:
                _create_reminder_unified(
                    str(title).strip(), str(when).strip(), recurrence=recurrence, chat_id=chat_id
                )
        elif action == "delete_reminder":
            rem_id = call_data.get("reminder_id") or call_data.get("id")
            if rem_id:
                _cancel_reminder_unified(str(rem_id).strip())
    except Exception as exc:
        logging.getLogger("orbit007").warning("Error executing LLM tool call: %s", exc)

    return cleaned_reply


async def process_agent_turn(
    user_message: str, session_id: str = "default", chat_id: str | None = None
) -> dict:
    user_message = user_message.strip()
    target_chat = chat_id or session_id or DEFAULT_CHAT_ID
    _ensure_chat_exists(target_chat)
    save_message("user", user_message, session_id=session_id)

    # 1. Order: _handle_local_command first (rules-first pipeline)
    reply_text: str | None = None
    try:
        cmd_result = await asyncio.to_thread(_handle_local_command, user_message, target_chat)
        if cmd_result:
            reply_text = cmd_result[0]
    except Exception:
        logging.getLogger("orbit007").exception("Local command handling failed for %r", user_message)

    # 2. If no local command match, route to LLM
    active_reminders = _live_reminders()
    if reply_text is None:
        history = get_recent_messages(session_id=session_id, limit=10)
        reply_text = await asyncio.to_thread(
            run_existing_llm, user_message, history, active_reminders
        )
        reply_text = _handle_llm_tool_calls(reply_text, user_message, chat_id=target_chat)

    save_message("assistant", reply_text, session_id=session_id)

    # 3. Audio generation: per-reply unique audio file
    audio_url = await generate_tts_url(reply_text)
    _cleanup_tts()

    return {
        "type": "agent_response",
        "text": reply_text,
        "audio_url": audio_url,
        "reminders": _live_reminders(),
        "history": get_recent_messages(session_id=session_id, limit=20),
    }

@app.get("/api/reminders")
async def api_get_reminders(include_completed: bool = False):
    return {"reminders": _live_reminders()}

@app.post("/api/reminders")
async def api_create_reminder(payload: ReminderCreatePayload):
    title = (payload.title or payload.text or "").strip()
    remind_at = (payload.remind_at or payload.run_at or "").strip()
    if not title:
        raise HTTPException(status_code=400, detail="Reminder title is required")
    if not remind_at:
        raise HTTPException(status_code=400, detail="Reminder time is required")
    try:
        reminder = _create_reminder_unified(
            title, remind_at, recurrence=payload.recurrence, chat_id=payload.chat_id
        )
        return {"reminder": reminder, "reminders": _live_reminders()}
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err))

@app.patch("/api/reminders/{reminder_id}")
async def api_update_reminder(reminder_id: str, payload: ReminderUpdatePayload):
    ok = _cancel_reminder_unified(reminder_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Reminder not found")
    return {"status": "success", "reminders": _live_reminders()}

@app.delete("/api/reminders/{reminder_id}")
async def api_delete_reminder(reminder_id: str):
    ok = _cancel_reminder_unified(reminder_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Reminder not found")
    return {"deleted": True, "reminders": _live_reminders()}

@app.get("/api/history")
async def api_get_history(session_id: str = "default"):
    return {"messages": get_recent_messages(session_id=session_id, limit=50)}

@app.post("/api/chat")
async def api_chat(payload: ChatPayload):
    if not payload.message or not payload.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    return await process_agent_turn(payload.message, payload.session_id or "default")

@app.websocket("/ws")
async def websocket_agent_endpoint(websocket: WebSocket):
    if not await ws_authorized(websocket):
        return
    await ws_manager.connect(websocket)
    try:
        await websocket.send_json({
            "type": "connected",
            "reminders": _live_reminders(),
            "history": get_recent_messages(limit=20),
        })
        while True:
            packet = await websocket.receive()
            if "bytes" in packet and packet["bytes"]:
                pcm_bytes = packet["bytes"]
                if len(pcm_bytes) > 2 * 1024 * 1024:
                    await websocket.send_json({"type": "error", "error": "Binary payload exceeds 2 MB limit"})
                    continue
                transcript = await asyncio.to_thread(transcribe_pcm_bytes, pcm_bytes)
                if transcript and transcript.strip():
                    await websocket.send_json({
                        "type": "transcript",
                        "text": transcript,
                    })
                    await websocket.send_json({"type": "status", "state": "thinking"})
                    result = await process_agent_turn(transcript)
                    await websocket.send_json(result)

            elif "text" in packet and packet["text"]:
                try:
                    payload = json.loads(packet["text"])
                except json.JSONDecodeError:
                    payload = {"message": packet["text"]}

                msg_type = payload.get("type")
                if msg_type == "ping":
                    await websocket.send_json({"type": "pong"})
                    continue

                user_msg = payload.get("message") or payload.get("text") or ""
                session_id = payload.get("session_id", "default")
                chat_id = payload.get("chat_id")

                if user_msg.strip():
                    await websocket.send_json({"type": "status", "state": "thinking"})
                    result = await process_agent_turn(user_msg, session_id=session_id, chat_id=chat_id)
                    await websocket.send_json(result)
    except WebSocketDisconnect:
        logging.getLogger("orbit007").info("WebSocket client disconnected")
    except Exception as exc:
        logging.getLogger("orbit007").warning("WebSocket error: %s", exc)
        try:
            await websocket.send_json({"type": "error", "error": str(exc)})
        except Exception:
            pass
    finally:
        ws_manager.disconnect(websocket)


@app.get("/api/notifications/{chat_id}")
def list_notifications(chat_id: str, after_id: int = Query(0)):
    if not memory_store.chat_exists(chat_id):
        raise HTTPException(404, "Conversation not found.")
    return {
        "notifications": memory_store.list_notifications(chat_id, after_id)
    }


# Compatibility aliases for older Vite proxy configurations.
@app.get("/notifications/{chat_id}")
def compat_list_notifications(chat_id: str, after_id: int = Query(0)):
    return list_notifications(chat_id, after_id)


@app.post("/api/memories")
def create_memory(payload: dict[str, Any] = Body(...)):
    if not isinstance(payload, dict) or not isinstance(payload.get("content"), str):
        raise HTTPException(400, "'content' must be a string.")
    try:
        return JSONResponse(status_code=201, content=memory_store.add_memory(payload["content"]))
    except ValueError as error:
        raise HTTPException(400, str(error)) from error


@app.put("/api/memories/{memory_id}")
def update_memory(memory_id: int, payload: dict[str, Any] = Body(...)):
    if not isinstance(payload, dict) or not isinstance(payload.get("content"), str):
        raise HTTPException(400, "'content' must be a string.")
    try:
        memory = memory_store.update_memory(memory_id, payload["content"])
    except ValueError as error:
        raise HTTPException(400, str(error)) from error
    if memory is None:
        raise HTTPException(404, "Memory not found.")
    return memory


@app.delete("/api/memories/{memory_id}")
def delete_memory(memory_id: int):
    if not memory_store.delete_memory(memory_id):
        raise HTTPException(404, "Memory not found.")
    return {"status": "deleted"}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local Qwen AI assistant")
    parser.add_argument(
        "--server",
        action="store_true",
        help="Start the HTTP server instead of the CLI.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Server bind host.")
    parser.add_argument("--port", type=int, default=8000, help="Server port.")
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Reload the server when Python source files change (development only).",
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        help="Prompt to run once. Omit it to start interactive mode.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help=f"Maximum response tokens (1-{MAX_NEW_TOKENS}).",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = parse_arguments()
    if args.server:
        if args.host == "0.0.0.0" and not os.environ.get("ORBIT_API_TOKEN", "").strip():
            print(
                "[SECURITY WARNING] Server is binding to 0.0.0.0 with no ORBIT_API_TOKEN set! "
                "Any machine on your network can access this assistant."
            )
        from runtime.guard import SPIDEY_MODEL_GUARD_PORT, get_port_owner_pid
        owner_pid = get_port_owner_pid(args.port) or get_port_owner_pid(SPIDEY_MODEL_GUARD_PORT)
        if owner_pid is not None and owner_pid != os.getpid():
            print(f"\n[FATAL] S.P.I.D.E.Y server or models are already running in process [PID {owner_pid}].")
            print(f"To stop the previous instance, run: taskkill /PID {owner_pid} /F (or Stop-Process -Id {owner_pid})\n")
            return

        uvicorn.run(
            "app:app" if args.reload else app,
            host=args.host,
            port=args.port,
            log_level="info",
            reload=args.reload,
        )
        return

    if (
        args.max_new_tokens is not None
        and not 1 <= args.max_new_tokens <= MAX_NEW_TOKENS
    ):
        raise ValueError(
            f"--max-new-tokens must be between 1 and {MAX_NEW_TOKENS}."
        )

    if args.prompt is not None:
        tokens = args.max_new_tokens
        if tokens is None:
            tokens = _token_budget(args.prompt)
        print(generate_response(args.prompt, tokens))
        return

    print("Local AI assistant. Type 'exit' or 'quit' to stop.")
    while True:
        try:
            prompt = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if prompt.lower() in {"exit", "quit"}:
            return
        try:
            tokens = args.max_new_tokens
            if tokens is None:
                tokens = _token_budget(prompt)
            print(f"AI: {generate_response(prompt, tokens)}")
        except ValueError as error:
            print(f"Error: {error}")


if __name__ == "__main__":
    main()
