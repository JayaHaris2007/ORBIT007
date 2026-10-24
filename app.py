import os
import argparse
import asyncio
import re
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import dateparser
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.date import DateTrigger
from apscheduler.triggers.interval import IntervalTrigger

CPU_THREADS = 4
CPU_INTEROP_THREADS = 1
GPU_MEMORY_LIMIT = "5GiB"
CPU_MEMORY_LIMIT = "6GiB"
MAX_PROMPT_CHARS = 2000
MAX_NEW_TOKENS = 128
DATABASE_PATH = Path(__file__).resolve().parent / "assistant.db"

from transformers import AutoModelForCausalLM, AutoTokenizer
import uvicorn
from fastapi import Body, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from memory import MemoryStore

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
    "You are a helpful assistant. Answer the latest user message directly and "
    "concisely. Use earlier messages only to resolve references such as 'it' or "
    "'one'. Do not repeat an earlier assistant answer unless the user asks for "
    "it again. For factual questions, give a direct factual answer; do not "
    "refuse just because the question uses words like 'best'. Use only the "
    "supplied conversation and saved memories. Never invent personal facts. "
    "Always answer in clear English. Never answer in Chinese, Hindi, or any "
    "other language unless the user explicitly asks for that language."
)


def _run_reminder(reminder_id: str) -> None:
    reminder = memory_store.get_reminder(reminder_id)
    if not reminder or not reminder["active"] or not reminder["chat_id"]:
        return
    current_time = datetime.now().astimezone().strftime("%d %B %Y, %I:%M %p %Z")
    prompt = (
        "Write one short, natural reminder notification as JARVIS. "
        "Sound like a warm human personal assistant speaking directly to the user. "
        "Mention the exact task and that it is time to do it. "
        "Use the current date and time only for context. Do not mention AI, "
        "prompts, scheduling, or these instructions. Return only the notification "
        "text, with no heading or quotation marks.\n"
        f"Reminder task: {reminder['text']}\n"
        f"Current date and time: {current_time}"
    )
    try:
        content = generate_response(prompt, 48).strip()
        if not content:
            raise ValueError("The reminder notification response was empty.")
    except Exception as error:
        print(
            f"Reminder notification generation failed for {reminder_id}: {error}"
        )
        content = f"It's time to {reminder['text']}."
    memory_store.add_notification(reminder["chat_id"], reminder_id, content)
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
                "Write one short, celebratory message as JARVIS, a warm personal "
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
                "Write one short, encouraging message as JARVIS, a warm personal "
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
                "Write one short, friendly message as JARVIS, a warm personal "
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
    elif run_at > datetime.now(run_at.tzinfo):
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


def _restore_reminders() -> None:
    for reminder in memory_store.list_reminders():
        _schedule_reminder(reminder)
    reminder_scheduler.start()


def _parse_reminder_time(value: str) -> datetime | None:
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
        return dateparser.parse(value, settings=settings)

    day_part = day_part_match.group(1).lower()
    date_text = re.sub(
        r"\b(morning|afternoon|evening|night)\b", "", value,
        count=1, flags=re.IGNORECASE,
    ).strip()
    if date_text.lower() == "next":
        date_text = "tomorrow"
    if not date_text:
        date_text = "today"
    parsed_date = dateparser.parse(date_text, settings=settings)
    if parsed_date is None:
        return None
    hour, minute = day_part_times[day_part]
    parsed = parsed_date.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if parsed <= datetime.now(parsed.tzinfo) and date_text == "today":
        parsed += timedelta(days=1)
    return parsed


def _parse_chat_reminder(prompt: str) -> tuple[str, str, str | None] | None:
    normalized_prompt = re.sub(r"\s+", " ", prompt.strip()).rstrip(".!?")
    relative_match = re.fullmatch(
        r"remind me in (.+?) to (.+?)(?:,\s*(every .+))?",
        normalized_prompt,
        re.IGNORECASE,
    )
    if relative_match:
        when, text, recurrence = relative_match.groups()
        parsed = _parse_reminder_time(
            f"in {when}",
        )
        if parsed is None:
            raise ValueError(f"I could not understand the reminder time: {when}.")
        return text.strip(), parsed.isoformat(), recurrence.strip() if recurrence else None

    relative_match = re.fullmatch(
        r"set (?:a )?reminder in "
        r"(\d+\s+(?:second|minute|hour|day|week)s?)\s+"
        r"(?:(?:for|to)\s+)?(.+?)(?:,\s*(every .+))?",
        normalized_prompt,
        re.IGNORECASE,
    )
    if relative_match:
        when, text, recurrence = relative_match.groups()
        parsed = _parse_reminder_time(f"in {when}")
        if parsed is None:
            raise ValueError(f"I could not understand the reminder time: {when}.")
        return text.strip(), parsed.isoformat(), recurrence.strip() if recurrence else None

    relative_match = re.fullmatch(
        r"(?:hey\s+jarvis,?\s+)?remind me (.+?) to (.+?)(?:,\s*(every .+))?",
        normalized_prompt,
        re.IGNORECASE,
    )
    if relative_match:
        when, text, recurrence = relative_match.groups()
        parsed = _parse_reminder_time(when)
        if parsed is None:
            raise ValueError(f"I could not understand the reminder time: {when}.")
        return text.strip(), parsed.isoformat(), recurrence.strip() if recurrence else None

    relative_match = re.fullmatch(
        r"set (?:a )?reminder in (.+?) (?:for|to) (.+?)(?:,\s*(every .+))?",
        normalized_prompt,
        re.IGNORECASE,
    )
    if relative_match:
        when, text, recurrence = relative_match.groups()
        parsed = _parse_reminder_time(
            f"in {when}",
        )
        if parsed is None:
            raise ValueError(f"I could not understand the reminder time: {when}.")
        return text.strip(), parsed.isoformat(), recurrence.strip() if recurrence else None

    relative_match = re.fullmatch(
        r"remind me to (.+?) in (.+?)(?:,\s*(every .+))?",
        normalized_prompt,
        re.IGNORECASE,
    )
    if relative_match:
        text, when, recurrence = relative_match.groups()
        parsed = _parse_reminder_time(
            f"in {when}",
        )
        if parsed is None:
            raise ValueError(f"I could not understand the reminder time: {when}.")
        return text.strip(), parsed.isoformat(), recurrence.strip() if recurrence else None

    match = re.fullmatch(
        r"remind me to (.+?) at (.+?)(?:,\s*(every .+))?",
        normalized_prompt,
        re.IGNORECASE,
    )
    if match:
        text, when, recurrence = match.groups()
    else:
        match = re.fullmatch(
            r"set (?:a )?reminder (?:for|on|at) (.+?) for (.+?)(?:,\s*(every .+))?",
            normalized_prompt,
            re.IGNORECASE,
        )
        if not match:
            return None
        when, text, recurrence = match.groups()
    parsed = _parse_reminder_time(when)
    if parsed is None:
        raise ValueError(f"I could not understand the reminder time: {when}.")
    return text.strip(), parsed.isoformat(), recurrence.strip() if recurrence else None


def _parse_chat_task(prompt: str) -> tuple[str, str] | None:
    match = re.fullmatch(
        r"\s*(?:add|create)\s+(?:a\s+)?task(?:\s+to)?\s+(.+?)(?:\s+priority\s+(low|medium|high))?\s*",
        prompt,
        re.IGNORECASE,
    )
    if not match:
        return None
    title, priority = match.groups()
    return title.strip(), priority or "medium"


def _is_task_list_request(prompt: str) -> bool:
    normalized_prompt = re.sub(r"\s+", " ", prompt.strip()).rstrip(".!?")
    return bool(re.fullmatch(
        r"(?:ok(?:ay)?[,\s]+|hey\s+jarvis[,\s]+)?(?:"
        r"(?:show|list)(?:\s+(?:me|my|all|the|pending))*\s+tasks"
        r"|(?:what are|what're)(?:\s+(?:all|my|the|pending))*\s+tasks"
        r"|(?:do i have|are there)(?:\s+(?:any|the|pending))*\s+tasks"
        r")",
        normalized_prompt,
        re.IGNORECASE,
    ))


def _parse_task_completion(prompt: str) -> tuple[int | None, str | None]:
    match = re.fullmatch(
        r"\s*(?:complete|completed|finish|finished|mark done)\s+"
        r"(?:the\s+)?task\s+(\d+)\s*",
        prompt,
        re.IGNORECASE,
    )
    if match:
        return int(match.group(1)), None

    normalized_prompt = re.sub(r"\s+", " ", prompt.strip()).rstrip(".!?")
    title_patterns = (
        r"(?:please\s+)?(?:complete|finish)\s+(?:the\s+)?(?:task\s+)?(.+)",
        r"(?:please\s+)?mark\s+(?:the\s+)?(?:task\s+)?(.+?)"
        r"\s+as\s+(?:being\s+)?completed",
        r"(?:i\s+)?(?:have\s+)?(?:completed|finished)\s+(.+)",
        r"(.+?)\s+task\s+(?:completed|complete|finished|finish)"
        r"(?:\s+(?:and\s+)?mark\s+it(?:\s+as)?(?:\s+completed)?)?",
    )
    for pattern in title_patterns:
        match = re.fullmatch(pattern, normalized_prompt, re.IGNORECASE)
        if match:
            title = match.group(1).strip()
            title = re.sub(r"^(?:the|my)\s+", "", title, flags=re.IGNORECASE)
            title = re.sub(r"\s+mark\s+it$", "", title, flags=re.IGNORECASE)
            if title:
                return None, title

    match = re.fullmatch(
        r"\s*(?:i\s+have\s+)?did\s+(.+?)\s*",
        normalized_prompt,
        re.IGNORECASE,
    )
    if match:
        words = re.findall(r"[a-z0-9]+", match.group(1).lower())
        deduplicated_words = [
            word for index, word in enumerate(words)
            if index == 0 or word != words[index - 1]
        ]
        title = " ".join(deduplicated_words)
        return None, title
    return None, None


def _find_pending_task(task_title: str) -> dict[str, Any] | None:
    normalized_title = re.sub(r"\s+", " ", task_title.strip().lower())
    requested_words = set(re.findall(r"[a-z0-9]+", normalized_title))
    tasks = memory_store.list_tasks()
    exact_matches = [
        task for task in tasks
        if task["title"].strip().lower() == normalized_title
    ]
    if len(exact_matches) == 1:
        return exact_matches[0]
    partial_matches = [
        task for task in tasks
        if normalized_title in task["title"].strip().lower()
    ]
    if len(partial_matches) == 1:
        return partial_matches[0]
    if len(partial_matches) > 1:
        raise ValueError(
            f"More than one pending task matches '{task_title}'. "
            "Use 'complete task <number>'."
        )
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
            f"More than one pending task matches '{task_title}'. "
            "Use 'complete task <number>'."
        )
    return None


def select_generation_context(
    context: list[dict[str, str]], prompt: str
) -> list[dict[str, str]]:
    words = prompt.split()
    if len(words) <= 4 and len(context) > 1:
        user_context = [message for message in context if message["role"] == "user"]
        return user_context[-4:]
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


def generate_response(
    prompt: str,
    max_new_tokens: int,
    messages: list[dict[str, str]] | None = None,
    memories: list[dict[str, Any]] | None = None,
) -> str:
    ensure_model_loaded()
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
            system_prompt += f"\nRelevant saved memories:\n{memory_text}"
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
                repetition_penalty=1.15,
                no_repeat_ngram_size=3,
            )
        if cuda_available:
            torch.cuda.synchronize()
        generation_seconds = time.perf_counter() - generation_started
        generated_tokens = outputs.shape[-1] - input_len
        response = tokenizer.decode(
            outputs[0][input_len:], skip_special_tokens=True
        ).strip()
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
    await asyncio.to_thread(ensure_model_loaded)
    _restore_reminders()
    try:
        yield
    finally:
        if reminder_scheduler.running:
            reminder_scheduler.shutdown(wait=False)


app = FastAPI(title="JARVIS Local Assistant", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


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
        content={"error": detail if isinstance(detail, str) else str(detail)},
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "api_version": "jarvis-local-2026-11",
        "model": "Qwen3-1.7B",
        "cuda": cuda_available,
        "load_seconds": round(load_seconds, 2),
        "database": str(DATABASE_PATH),
        "semantic_memory": os.environ.get("AI_ENABLE_SEMANTIC_MEMORY", "1") != "0",
    }


@app.get("/api/health")
def api_health():
    return health()


@app.post("/generate")
@app.post("/api/generate")
async def generate(payload: dict[str, Any] = Body(...)):
    await asyncio.to_thread(ensure_model_loaded)
    if not isinstance(payload, dict):
        raise HTTPException(400, "Request body must be a JSON object.")

    prompt = payload.get("prompt")
    max_new_tokens = payload.get("max_new_tokens", MAX_NEW_TOKENS)
    if not isinstance(prompt, str):
        raise HTTPException(400, "'prompt' must be a string.")
    if not prompt.strip():
        raise HTTPException(400, "Prompt cannot be empty.")
    if len(prompt) > MAX_PROMPT_CHARS:
        raise HTTPException(400, f"Prompt is limited to {MAX_PROMPT_CHARS} characters.")
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
        task_request = _parse_chat_task(prompt)
        if task_request:
            title, priority = task_request
            task = memory_store.create_task(title, priority)
            response = f"Added {priority}-priority task: {task['title']}"
            memory_store.add_message(chat_id, "user", prompt.strip())
            memory_store.add_message(chat_id, "assistant", response)
            return {
                "response": response,
                "chat_id": chat_id,
                "messages": memory_store.get_messages(chat_id),
                "provider": "local-task-manager",
            }
        if _is_task_list_request(prompt):
            tasks = memory_store.list_tasks()
            response = (
                "You have no pending tasks."
                if not tasks
                else "\n".join(
                    f"{task['id']}. [{task['priority']}] {task['title']}"
                    for task in tasks
                )
            )
            memory_store.add_message(chat_id, "user", prompt.strip())
            memory_store.add_message(chat_id, "assistant", response)
            return {
                "response": response,
                "chat_id": chat_id,
                "messages": memory_store.get_messages(chat_id),
                "provider": "local-task-manager",
            }
        task_id, task_title = _parse_task_completion(prompt)
        if task_title is not None:
            task = _find_pending_task(task_title)
            if task is None:
                raise ValueError(f"No pending task matched '{task_title}'.")
            task_id = task["id"]
        if task_id is not None:
            task = memory_store.update_task(task_id, completed=True)
            if task is None:
                raise ValueError(f"Task {task_id} was not found.")
            response = f"Completed task: {task['title']}"
            memory_store.add_message(chat_id, "user", prompt.strip())
            memory_store.add_message(chat_id, "assistant", response)
            threading.Thread(
                target=_recommend_next_task,
                args=(chat_id, task["title"]),
                daemon=True,
            ).start()
            return {
                "response": response,
                "chat_id": chat_id,
                "messages": memory_store.get_messages(chat_id),
                "provider": "local-task-manager",
            }
        reminder_request = _parse_chat_reminder(prompt)
        if reminder_request:
            reminder_text, run_at, recurrence = reminder_request
            _validate_recurrence(recurrence)
            reminder = memory_store.create_reminder(
                reminder_text, run_at, recurrence, chat_id
            )
            _schedule_reminder(reminder)
            response = (
                f"Scheduled reminder for {datetime.fromisoformat(run_at).strftime('%d %b %Y, %I:%M %p')}: "
                f"{reminder_text}"
            )
            memory_store.add_message(chat_id, "user", prompt.strip())
            memory_store.add_message(chat_id, "assistant", response)
            return {
                "response": response,
                "chat_id": chat_id,
                "messages": memory_store.get_messages(chat_id),
                "provider": "local-scheduler",
            }
        memory_store.add_message(chat_id, "user", prompt.strip())
        context = memory_store.build_context(chat_id, tokenizer, prompt)
        model_prompt = resolve_short_follow_up(prompt, context)
        if model_prompt != prompt and context:
            context[-1] = {"role": "user", "content": model_prompt}
        memories = memory_store.search_memories(prompt)
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
        return JSONResponse(status_code=201, content=memory_store.create_task(
            payload["title"],
            payload.get("priority", "medium"),
            payload.get("due_at"),
        ))
    except ValueError as error:
        raise HTTPException(400, str(error)) from error


@app.patch("/api/tasks/{task_id}")
def update_task(task_id: int, payload: dict[str, Any] = Body(...)):
    if not isinstance(payload, dict):
        raise HTTPException(400, "Request body must be a JSON object.")
    chat_id = payload.get("chat_id")
    try:
        task = memory_store.update_task(
            task_id,
            payload.get("title"),
            payload.get("priority"),
            payload.get("due_at"),
            payload.get("completed"),
        )
    except ValueError as error:
        raise HTTPException(400, str(error)) from error
    if task is None:
        raise HTTPException(404, "Task not found.")
    if payload.get("completed") and chat_id:
        threading.Thread(
            target=_recommend_next_task,
            args=(chat_id, task["title"]),
            daemon=True,
        ).start()
    return task


@app.delete("/api/tasks/{task_id}")
def delete_task(task_id: int):
    if not memory_store.delete_task(task_id):
        raise HTTPException(404, "Task not found.")
    return {"status": "deleted"}


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


@app.get("/api/reminders")
def list_reminders(include_cancelled: int = Query(0)):
    return {
        "reminders": memory_store.list_reminders(not include_cancelled),
        "scheduler": {
            "running": reminder_scheduler.running,
            "mode": "local-process",
        },
    }


@app.post("/api/reminders")
def create_reminder(payload: dict[str, Any] = Body(...)):
    if (
        not isinstance(payload, dict)
        or not isinstance(payload.get("text"), str)
        or not isinstance(payload.get("run_at"), str)
    ):
        raise HTTPException(400, "'text' and 'run_at' must be strings.")
    try:
        _validate_recurrence(payload.get("recurrence"))
        reminder = memory_store.create_reminder(
            payload["text"],
            payload["run_at"],
            payload.get("recurrence"),
            payload.get("chat_id"),
        )
        _schedule_reminder(reminder)
        return JSONResponse(status_code=201, content=reminder)
    except ValueError as error:
        raise HTTPException(400, str(error)) from error


@app.delete("/api/reminders/{reminder_id}")
def cancel_reminder(reminder_id: str):
    try:
        reminder_scheduler.remove_job(reminder_id)
    except Exception:
        pass
    if not memory_store.cancel_reminder(reminder_id):
        raise HTTPException(404, "Reminder not found.")
    return {"status": "cancelled"}


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
        default=MAX_NEW_TOKENS,
        help=f"Maximum response tokens (1-{MAX_NEW_TOKENS}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    if args.server:
        uvicorn.run(
            "app:app" if args.reload else app,
            host=args.host,
            port=args.port,
            log_level="info",
            reload=args.reload,
        )
        return

    if not 1 <= args.max_new_tokens <= MAX_NEW_TOKENS:
        raise ValueError(
            f"--max-new-tokens must be between 1 and {MAX_NEW_TOKENS}."
        )

    if args.prompt is not None:
        print(generate_response(args.prompt, args.max_new_tokens))
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
            print(f"AI: {generate_response(prompt, args.max_new_tokens)}")
        except ValueError as error:
            print(f"Error: {error}")


if __name__ == "__main__":
    main()