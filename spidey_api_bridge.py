#!/usr/bin/env python3
"""Lightweight JSON bridge for Vite dev server when full GPU server isn't running.

Uses the real MemoryStore, learning engine, and deterministic NLP handler from app.py
against assistant.db so the React frontend works end-to-end in cloud preview environments.
"""

from __future__ import annotations

import json
import sys
import types
import typing
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlparse

if not hasattr(typing, "NotRequired"):
    typing.NotRequired = typing.Optional


class _StubModule(types.ModuleType):
    def __init__(self, name: str):
        super().__init__(name)
        self.__path__ = []

    def __getattr__(self, attr: str):
        if attr.startswith("__"):
            raise AttributeError(attr)
        return _StubCallable()


class _StubCallable:
    def __init__(self, *a, **k):
        pass

    def __call__(self, *a, **k):
        return self

    def __getattr__(self, attr: str):
        return self

    def __iter__(self):
        return iter([])


def _install_stubs_if_needed() -> None:
    optional_mods = [
        "torch",
        "transformers",
        "accelerate",
        "bitsandbytes",
        "duckduckgo_search",
        "ddgs",
        "bs4",
        "httpx",
        "fastapi",
        "fastapi.middleware",
        "fastapi.middleware.cors",
        "fastapi.responses",
        "fastapi.staticfiles",
        "uvicorn",
        "sentence_transformers",
        "numpy",
        "scipy",
        "soundfile",
        "speech_recognition",
        "discord",
        "discord.ext",
        "discord.ext.commands",
        "dotenv",
        "pydantic",
        "rapidfuzz",
        "dateparser",
        "apscheduler",
        "apscheduler.schedulers",
        "apscheduler.schedulers.background",
        "apscheduler.triggers",
        "apscheduler.triggers.cron",
        "apscheduler.triggers.date",
        "apscheduler.triggers.interval",
    ]
    for mod_name in optional_mods:
        try:
            __import__(mod_name)
        except Exception:
            sys.modules[mod_name] = _StubModule(mod_name)

    if isinstance(sys.modules.get("pydantic"), _StubModule):
        class _BaseModelStub:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)
        sys.modules["pydantic"].BaseModel = _BaseModelStub
        sys.modules["pydantic"].Field = lambda *a, **k: None

    if isinstance(sys.modules.get("fastapi"), _StubModule):
        class _FastAPIStub:
            def __init__(self, *a, **k): pass
            def add_middleware(self, *a, **k): pass
            def include_router(self, *a, **k): pass
            def mount(self, *a, **k): pass
            def get(self, *a, **k): return lambda f: f
            def post(self, *a, **k): return lambda f: f
            def put(self, *a, **k): return lambda f: f
            def patch(self, *a, **k): return lambda f: f
            def delete(self, *a, **k): return lambda f: f
            def websocket(self, *a, **k): return lambda f: f
            def exception_handler(self, *a, **k): return lambda f: f

        sys.modules["fastapi"].FastAPI = _FastAPIStub
        sys.modules["fastapi"].APIRouter = _FastAPIStub
        sys.modules["fastapi"].HTTPException = Exception


_install_stubs_if_needed()

import app  # noqa: E402
from learning import progress as learning_progress  # noqa: E402
from learning import roadmaps as learning_roadmaps  # noqa: E402
from learning import schema as learning_schema  # noqa: E402

store = app.memory_store
try:
    learning_schema.init(app.DATABASE_PATH)
    learning_roadmaps.load_seeds()
except Exception:
    pass


def _ensure_seed_data() -> None:
    if not store.list_habits() and not store.list_tasks(include_completed=True):
        store.create_habit("Watering Garden", "Activity")
        store.create_habit("Sleeping", "Activity")
        store.create_task("explore AI Basics", priority="medium")


_ensure_seed_data()


def handle_request(method: str, raw_url: str, body: dict | None) -> tuple[int, dict]:
    parsed = urlparse(raw_url)
    path = parsed.path.rstrip("/") or "/"
    query = parse_qs(parsed.query)
    payload = body or {}

    if path == "/api/health":
        return 200, {
            "status": "ok",
            "model_loaded": False,
            "mode": "local-deterministic-bridge",
            "gamification": store.get_gamification(),
        }

    # Chats
    if path == "/api/chats" and method == "GET":
        chats = store.list_chats()
        if not chats:
            created = store.create_chat("New conversation")
            chats = [created]
        return 200, {"chats": chats}

    if path == "/api/chats" and method == "POST":
        title = str(payload.get("title") or "New conversation")
        return 201, store.create_chat(title)

    if path.startswith("/api/chats/"):
        chat_id = path.split("/")[3]
        if method == "GET":
            chat = store.get_chat(chat_id)
            if not chat:
                chat = store.create_chat("New conversation")
            return 200, {
                "chat": chat,
                "messages": store.get_messages(chat["id"]),
                "latest_notification_id": store.latest_notification_id(chat["id"]),
            }
        if method == "DELETE":
            store.delete_chat(chat_id)
            return 200, {"status": "deleted"}

    if path in ("/api/chat", "/chat") and method == "POST":
        prompt = str(payload.get("prompt") or payload.get("message") or "").strip()
        chat_id = payload.get("chat_id")
        if not chat_id or not store.chat_exists(chat_id):
            chat = (store.list_chats() or [store.create_chat("New conversation")])[0]
            chat_id = chat["id"]
        try:
            cmd = app._handle_local_command(prompt, chat_id)
        except ValueError as err:
            cmd = (str(err), "local-validation")
        if cmd is not None:
            reply, provider = cmd
        else:
            brief = app._daily_briefing_command("briefing")
            summary_line = brief[0] if brief else ""
            reply = (
                f"I processed **{prompt}** using local deterministic context.\n\n"
                f"{summary_line}\n\n"
                "Try natural commands like `habits`, `tasks`, `mark it completed`, "
                "`Now I am completed sleeping`, `what's on my plate`, or `add note: ...`."
            )
            provider = "local-assistant"
        store.add_message(chat_id, "user", prompt)
        store.add_message(chat_id, "assistant", reply)
        page = app._page_for_provider(provider)
        return 200, {
            "response": reply,
            "reply": reply,
            "chat_id": chat_id,
            "provider": provider,
            "navigate": page,
            "messages": store.get_messages(chat_id),
        }

    # Notifications
    if path.startswith("/api/notifications/") or path.startswith("/notifications/"):
        parts = path.strip("/").split("/")
        chat_id = parts[-1]
        after_id = int((query.get("after_id") or ["0"])[0])
        if not store.chat_exists(chat_id):
            return 200, {"notifications": []}
        return 200, {"notifications": store.list_notifications(chat_id, after_id)}

    # Tasks
    if path == "/api/tasks" and method == "GET":
        return 200, {"tasks": store.list_tasks(include_completed=True)}

    if path == "/api/tasks" and method == "POST":
        task = store.create_task(
            title=str(payload.get("title") or "").strip(),
            priority=str(payload.get("priority") or "medium"),
            due_at=payload.get("due_at"),
            recurrence=payload.get("recurrence"),
        )
        return 201, task

    if path.startswith("/api/tasks/"):
        parts = path.split("/")
        task_id = int(parts[3])
        if len(parts) > 4 and parts[4] == "toggle" and method == "POST":
            existing = next((t for t in store.list_tasks(include_completed=True) if t["id"] == task_id), None)
            if not existing:
                return 404, {"error": "Task not found."}
            updated = store.update_task(task_id, completed=not bool(existing["completed"]))
            return 200, {"task": updated, "gamification": store.get_gamification()}
        if method in ("PUT", "PATCH"):
            updated = store.update_task(
                task_id,
                title=payload.get("title"),
                completed=payload.get("completed"),
                priority=payload.get("priority"),
                due_at=payload.get("due_at", ...),
                recurrence=payload.get("recurrence", ...),
            )
            return 200, {"task": updated, "gamification": store.get_gamification()}
        if method == "DELETE":
            store.delete_task(task_id)
            return 200, {"status": "deleted", "gamification": store.get_gamification()}

    # Habits
    if path == "/api/habits" and method == "GET":
        return 200, {
            "habits": store.list_habits(),
            "gamification": store.get_gamification(),
        }

    if path == "/api/habits" and method == "POST":
        habit = store.create_habit(
            name=str(payload.get("name") or "").strip(),
            icon=str(payload.get("icon") or "Activity"),
        )
        return 201, habit

    if path.startswith("/api/habits/"):
        parts = path.split("/")
        habit_id = int(parts[3])
        if len(parts) > 4 and parts[4] == "completions" and method in ("POST", "PUT"):
            date_str = str(payload.get("date") or datetime.now(timezone.utc).date().isoformat())
            completed = bool(payload.get("completed", True))
            updated = store.set_habit_completion(habit_id, date_str, completed)
            return 200, {"habit": updated, "gamification": store.get_gamification()}
        if method in ("PUT", "PATCH"):
            updated = store.update_habit(
                habit_id,
                name=payload.get("name"),
                icon=payload.get("icon"),
            )
            return 200, updated
        if method == "DELETE":
            store.delete_habit(habit_id)
            return 200, {"status": "deleted", "gamification": store.get_gamification()}

    # Gamification
    if path == "/api/gamification" and method == "GET":
        return 200, store.get_gamification()

    # Reminders
    if path == "/api/reminders" and method == "GET":
        return 200, {
            "reminders": store.list_reminders(active_only=True),
            "scheduler": {"running": True, "mode": "local-process"},
        }

    if path == "/api/reminders" and method == "POST":
        reminder = store.create_reminder(
            str(payload.get("text") or "").strip(),
            str(payload.get("run_at") or "").strip(),
            payload.get("recurrence"),
            payload.get("chat_id"),
        )
        return 201, reminder

    if path.startswith("/api/reminders/") and method == "DELETE":
        rid = path.split("/")[3]
        store.cancel_reminder(rid)
        return 200, {"status": "cancelled"}

    # Memories / Notes
    if path == "/api/memories" and method == "GET":
        return 200, {"memories": store.list_memories()}

    if path == "/api/memories" and method == "POST":
        return 201, store.add_memory(str(payload.get("content") or ""))

    if path.startswith("/api/memories/"):
        mid = int(path.split("/")[3])
        if method == "PUT":
            return 200, store.update_memory(mid, str(payload.get("content") or ""))
        if method == "DELETE":
            store.delete_memory(mid)
            return 200, {"status": "deleted"}

    # Learning endpoints
    if path == "/api/learning/roadmaps" and method == "GET":
        return 200, {"roadmaps": learning_progress.roadmap_summaries()}

    if path == "/api/learning/roadmaps" and method == "POST":
        created = learning_progress.create_roadmap(
            name=str(payload.get("name") or "").strip(),
            category=str(payload.get("category") or "technical"),
            description=str(payload.get("description") or ""),
        )
        return 201, created

    if path.startswith("/api/learning/roadmaps/"):
        rid = int(path.split("/")[4])
        if method == "GET":
            detail = learning_progress.get_roadmap(rid)
            return (200, detail) if detail else (404, {"error": "Roadmap not found."})
        if method == "PATCH":
            return 200, learning_progress.update_roadmap(rid, **payload)
        if method == "DELETE":
            learning_progress.delete_roadmap(rid)
            return 200, {"status": "deleted"}

    if path == "/api/learning/progress" and method == "GET":
        return 200, learning_progress.overall_progress()

    if path == "/api/learning/next" and method == "GET":
        return 200, {"next": learning_progress.next_topics(limit=5)}

    if path == "/api/learning/links" and method == "GET":
        state = (query.get("state") or ["suggested"])[0]
        return 200, {"links": learning_progress.list_task_links(state=state)}

    if path == "/api/learning/settings":
        if method == "GET":
            return 200, {"nudges_enabled": learning_progress.get_nudges_enabled()}
        if method == "PATCH":
            enabled = bool(payload.get("nudges_enabled", True))
            learning_progress.set_nudges_enabled(enabled)
            return 200, {"nudges_enabled": enabled}

    if path.startswith("/api/learning/topics"):
        parts = path.split("/")
        if len(parts) == 4 and method == "POST":
            return 201, learning_progress.create_topic(
                roadmap_id=int(payload.get("roadmap_id") or payload.get("roadmapId")),
                name=str(payload.get("name") or "").strip(),
                description=str(payload.get("description") or ""),
                priority=int(payload.get("priority") or 2),
            )
        if len(parts) >= 5:
            tid = int(parts[4])
            if len(parts) == 6 and parts[5] == "links":
                return 200, {"links": learning_progress.list_confirmed_task_links(tid)}
            if method == "PATCH":
                if "status" in payload:
                    return 200, learning_progress.set_topic_status(tid, payload["status"])
                return 200, learning_progress.update_topic(tid, **payload)
            if method == "DELETE":
                learning_progress.delete_topic(tid)
                return 200, {"status": "deleted"}

    # Voice settings fallback
    if path == "/api/voice/settings":
        if method == "GET":
            return 200, {"speak_replies": False, "voice": "default", "available": False}
        return 200, payload

    if path == "/api/voice/stop":
        return 200, {"status": "stopped"}

    return 200, {"status": "ok"}


if __name__ == "__main__":
    req = json.loads(sys.stdin.read() or "{}")
    try:
        status, out = handle_request(
            req.get("method", "GET"),
            req.get("url", "/api/health"),
            req.get("body"),
        )
    except Exception as exc:
        status, out = 400, {"error": str(exc)}
    sys.stdout.write(json.dumps({"status": status, "body": out}))
