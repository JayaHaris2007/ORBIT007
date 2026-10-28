"""Local SQLite storage and bounded memory retrieval for the assistant."""

from __future__ import annotations

import json
import os
import re
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


MAX_CONTEXT_TOKENS = 1800
MAX_MEMORY_TEXT = 500
MAX_RETRIEVED_MEMORIES = 5
HABIT_COMPLETION_XP = 10
LOCAL_TIMEZONE = ZoneInfo("Asia/Kolkata")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class MemoryStore:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self._embedder = None
        self._embedding_warning_printed = False
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL REFERENCES conversations(id)
                        ON DELETE CASCADE,
                    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(conversation_id, role, content, created_at)
                );
                CREATE INDEX IF NOT EXISTS messages_conversation_idx
                    ON messages(conversation_id, id);
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT NOT NULL UNIQUE,
                    embedding TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS external_sessions (
                    provider TEXT NOT NULL,
                    external_user_id TEXT NOT NULL,
                    conversation_id TEXT NOT NULL REFERENCES conversations(id)
                        ON DELETE CASCADE,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(provider, external_user_id)
                );
                CREATE TABLE IF NOT EXISTS tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL DEFAULT '',
                    priority TEXT NOT NULL DEFAULT 'medium'
                        CHECK(priority IN ('low', 'medium', 'high')),
                    due_at TEXT,
                    completed INTEGER NOT NULL DEFAULT 0,
                    completed_at TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS tasks_status_idx
                    ON tasks(completed, priority, due_at);
                CREATE TABLE IF NOT EXISTS reminders (
                    id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    run_at TEXT NOT NULL,
                    recurrence TEXT,
                    chat_id TEXT REFERENCES conversations(id) ON DELETE SET NULL,
                    active INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS reminders_active_idx
                    ON reminders(active, run_at);
                CREATE TABLE IF NOT EXISTS habits (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    icon TEXT NOT NULL DEFAULT 'Activity',
                    daily_goal INTEGER NOT NULL DEFAULT 1 CHECK(daily_goal > 0),
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS habit_completions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    habit_id INTEGER NOT NULL REFERENCES habits(id) ON DELETE CASCADE,
                    completion_date TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(habit_id, completion_date)
                );
                CREATE INDEX IF NOT EXISTS habit_completions_date_idx
                    ON habit_completions(completion_date, habit_id);
                CREATE TABLE IF NOT EXISTS gamification (
                    id INTEGER PRIMARY KEY CHECK(id = 1),
                    xp INTEGER NOT NULL DEFAULT 0 CHECK(xp >= 0),
                    valid_streak_dates TEXT NOT NULL DEFAULT '[]',
                    achievements TEXT NOT NULL DEFAULT '[]',
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS notifications (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id TEXT NOT NULL REFERENCES conversations(id)
                        ON DELETE CASCADE,
                    reminder_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(reminder_id, created_at)
                );
                DELETE FROM messages
                WHERE role = 'assistant'
                  AND EXISTS (
                      SELECT 1
                      FROM notifications
                      WHERE notifications.chat_id = messages.conversation_id
                        AND notifications.content = messages.content
                  );
                """
            )
            reminder_columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(reminders)")
            }
            if "chat_id" not in reminder_columns:
                connection.execute(
                    "ALTER TABLE reminders ADD COLUMN chat_id TEXT "
                    "REFERENCES conversations(id) ON DELETE SET NULL"
                )
            task_columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(tasks)")
            }
            if "completed_at" not in task_columns:
                connection.execute(
                    "ALTER TABLE tasks ADD COLUMN completed_at TEXT"
                )
                connection.execute(
                    "UPDATE tasks SET completed_at = updated_at "
                    "WHERE completed = 1"
                )
            if "description" not in task_columns:
                connection.execute(
                    "ALTER TABLE tasks ADD COLUMN description TEXT NOT NULL DEFAULT ''"
                )
            connection.execute(
                "INSERT OR IGNORE INTO gamification "
                "(id, updated_at) VALUES (1, ?)",
                (_now(),),
            )

    def create_chat(self, title: str = "New conversation") -> dict[str, Any]:
        chat_id = str(uuid.uuid4())
        now = _now()
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO conversations VALUES (?, ?, ?, ?)",
                (chat_id, title[:120], now, now),
            )
        return {"id": chat_id, "title": title[:120], "created_at": now, "updated_at": now}

    def list_chats(self) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT id, title, created_at, updated_at FROM conversations "
                "ORDER BY updated_at DESC"
            ).fetchall()
        return [dict(row) for row in rows]

    def chat_exists(self, chat_id: str) -> bool:
        with self.connect() as connection:
            return connection.execute(
                "SELECT 1 FROM conversations WHERE id = ?", (chat_id,)
            ).fetchone() is not None

    def get_messages(self, chat_id: str) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT id, role, content, created_at FROM messages "
                "WHERE conversation_id = ? ORDER BY id",
                (chat_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def add_message(self, chat_id: str, role: str, content: str) -> dict[str, Any]:
        if role not in {"user", "assistant"}:
            raise ValueError("Invalid message role.")
        if not content.strip():
            raise ValueError("Message cannot be empty.")
        now = _now()
        with self.connect() as connection:
            cursor = connection.execute(
                "INSERT INTO messages(conversation_id, role, content, created_at) "
                "VALUES (?, ?, ?, ?)",
                (chat_id, role, content, now),
            )
            if role == "user":
                title = content.strip().replace("\n", " ")[:80]
                connection.execute(
                    "UPDATE conversations SET title = CASE WHEN title = "
                    "'New conversation' THEN ? ELSE title END, updated_at = ? "
                    "WHERE id = ?",
                    (title, now, chat_id),
                )
            else:
                connection.execute(
                    "UPDATE conversations SET updated_at = ? WHERE id = ?",
                    (now, chat_id),
                )
        return {"id": cursor.lastrowid, "role": role, "content": content, "created_at": now}

    def delete_chat(self, chat_id: str) -> bool:
        with self.connect() as connection:
            result = connection.execute(
                "DELETE FROM conversations WHERE id = ?", (chat_id,)
            )
        return result.rowcount > 0

    def clear_chats(self) -> None:
        with self.connect() as connection:
            connection.execute("DELETE FROM conversations")

    def get_or_create_external_chat(
        self, provider: str, external_user_id: str
    ) -> dict[str, Any]:
        if not provider.strip() or not external_user_id.strip():
            raise ValueError("External session identifiers cannot be empty.")
        with self.connect() as connection:
            row = connection.execute(
                "SELECT c.id, c.title, c.created_at, c.updated_at "
                "FROM external_sessions AS s "
                "JOIN conversations AS c ON c.id = s.conversation_id "
                "WHERE s.provider = ? AND s.external_user_id = ?",
                (provider, external_user_id),
            ).fetchone()
            if row is not None:
                return dict(row)

        chat = self.create_chat(title=f"{provider} conversation")
        now = _now()
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO external_sessions "
                "(provider, external_user_id, conversation_id, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (provider, external_user_id, chat["id"], now, now),
            )
        return chat

    def get_external_session(self, chat_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT provider, external_user_id FROM external_sessions "
                "WHERE conversation_id = ?",
                (chat_id,),
            ).fetchone()
        return dict(row) if row else None

    def list_external_chats(self, provider: str) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT c.id, s.external_user_id "
                "FROM external_sessions AS s "
                "JOIN conversations AS c ON c.id = s.conversation_id "
                "WHERE s.provider = ?",
                (provider,),
            ).fetchall()
        return [dict(row) for row in rows]

    def list_local_chat_ids(self) -> list[str]:
        """Conversation ids belonging to local web sessions."""
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT c.id FROM conversations AS c "
                "LEFT JOIN external_sessions AS s ON s.conversation_id = c.id "
                "WHERE s.provider IS NULL"
            ).fetchall()
        return [row["id"] for row in rows]

    def list_tasks(self, include_completed: bool = False) -> list[dict[str, Any]]:
        query = "SELECT * FROM tasks"
        if not include_completed:
            query += " WHERE completed = 0"
        query += (
            " ORDER BY completed, CASE priority WHEN 'high' THEN 1 "
            "WHEN 'medium' THEN 2 ELSE 3 END, due_at, id"
        )
        with self.connect() as connection:
            rows = connection.execute(query).fetchall()
        return [dict(row) for row in rows]

    def create_task(
        self,
        title: str,
        priority: str = "medium",
        due_at: str | None = None,
        description: str = "",
    ) -> dict[str, Any]:
        title = title.strip()
        description = description.strip()
        if not title or len(title) > 500:
            raise ValueError("Task title must contain 1-500 characters.")
        if len(description) > 2000:
            raise ValueError("Task description must be 2000 characters or fewer.")
        if priority not in {"low", "medium", "high"}:
            raise ValueError("Priority must be low, medium, or high.")
        now = _now()
        with self.connect() as connection:
            cursor = connection.execute(
                "INSERT INTO tasks(title, description, priority, due_at, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (title, description, priority, due_at, now, now),
            )
            row = connection.execute(
                "SELECT * FROM tasks WHERE id = ?", (cursor.lastrowid,)
            ).fetchone()
        return dict(row)

    def update_task(
        self,
        task_id: int,
        title: str | None = None,
        priority: str | None = None,
        due_at: str | None = None,
        completed: bool | None = None,
        description: str | None = None,
        clear_due_at: bool = False,
    ) -> dict[str, Any] | None:
        with self.connect() as connection:
            existing = connection.execute(
                "SELECT * FROM tasks WHERE id = ?", (task_id,)
            ).fetchone()
            if existing is None:
                return None
            if clear_due_at:
                due_value = None
            elif due_at is not None:
                due_value = due_at
            else:
                due_value = existing["due_at"]
            values = (
                title.strip() if title is not None else existing["title"],
                description.strip() if description is not None else existing["description"],
                priority if priority is not None else existing["priority"],
                due_value,
                int(completed) if completed is not None else existing["completed"],
                (
                    datetime.now(timezone.utc).isoformat()
                    if completed is True
                    else None
                    if completed is False
                    else existing["completed_at"]
                ),
                _now(),
                task_id,
            )
            if not values[0] or len(values[0]) > 500:
                raise ValueError("Task title must contain 1-500 characters.")
            if len(values[1]) > 2000:
                raise ValueError("Task description must be 2000 characters or fewer.")
            if values[2] not in {"low", "medium", "high"}:
                raise ValueError("Priority must be low, medium, or high.")
            connection.execute(
                "UPDATE tasks SET title = ?, description = ?, priority = ?, due_at = ?, "
                "completed = ?, completed_at = ?, updated_at = ? WHERE id = ?",
                values,
            )
            row = connection.execute(
                "SELECT * FROM tasks WHERE id = ?", (task_id,)
            ).fetchone()
        return dict(row)

    def delete_task(self, task_id: int) -> bool:
        with self.connect() as connection:
            result = connection.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
        return result.rowcount > 0

    @staticmethod
    def _calculate_streak(completed_dates: list[str]) -> int:
        completed = set(completed_dates)
        today = datetime.now(LOCAL_TIMEZONE).date()
        if today.isoformat() not in completed:
            today -= timedelta(days=1)
            if today.isoformat() not in completed:
                return 0
        streak = 0
        while today.isoformat() in completed:
            streak += 1
            today -= timedelta(days=1)
        return streak

    def _get_gamification(self, connection: sqlite3.Connection) -> dict[str, Any]:
        row = connection.execute(
            "SELECT xp, valid_streak_dates, achievements FROM gamification WHERE id = 1"
        ).fetchone()
        if row is None:
            now = _now()
            connection.execute(
                "INSERT INTO gamification(id, updated_at) VALUES (1, ?)", (now,)
            )
            return {"xp": 0, "valid_streak_dates": [], "achievements": []}
        return {
            "xp": row["xp"],
            "valid_streak_dates": json.loads(row["valid_streak_dates"]),
            "achievements": json.loads(row["achievements"]),
        }

    def _refresh_gamification(self, connection: sqlite3.Connection, xp_delta: int = 0) -> None:
        data = self._get_gamification(connection)
        habits = connection.execute("SELECT id FROM habits").fetchall()
        valid_dates: list[str] = []
        if habits:
            rows = connection.execute(
                "SELECT completion_date, COUNT(*) AS completed FROM habit_completions "
                "GROUP BY completion_date"
            ).fetchall()
            threshold = len(habits) * 0.7
            valid_dates = [row["completion_date"] for row in rows if row["completed"] >= threshold]
        achievements = set(data["achievements"])
        if habits:
            achievements.add("first-habit")
        if self._calculate_streak(valid_dates) >= 7:
            achievements.add("seven-day-streak")
        xp = max(0, data["xp"] + xp_delta)
        if xp >= 100:
            achievements.add("hundred-xp")
        connection.execute(
            "UPDATE gamification SET xp = ?, valid_streak_dates = ?, achievements = ?, updated_at = ? "
            "WHERE id = 1",
            (xp, json.dumps(sorted(valid_dates)), json.dumps(sorted(achievements)), _now()),
        )

    def list_habits(self) -> list[dict[str, Any]]:
        with self.connect() as connection:
            habits = connection.execute("SELECT * FROM habits ORDER BY id").fetchall()
            result = []
            for habit in habits:
                dates = [
                    row["completion_date"]
                    for row in connection.execute(
                        "SELECT completion_date FROM habit_completions "
                        "WHERE habit_id = ? ORDER BY completion_date",
                        (habit["id"],),
                    )
                ]
                item = dict(habit)
                item["completed_dates"] = dates
                item["streak"] = self._calculate_streak(dates)
                result.append(item)
        return result

    def create_habit(self, name: str, icon: str = "Activity", daily_goal: int = 1) -> dict[str, Any]:
        name = name.strip()
        if not name or len(name) > 200:
            raise ValueError("Habit name must contain 1-200 characters.")
        if not isinstance(daily_goal, int) or isinstance(daily_goal, bool) or daily_goal < 1:
            raise ValueError("daily_goal must be a positive integer.")
        now = _now()
        with self.connect() as connection:
            cursor = connection.execute(
                "INSERT INTO habits(name, icon, daily_goal, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                (name, icon.strip() or "Activity", daily_goal, now, now),
            )
            self._refresh_gamification(connection)
        return next(habit for habit in self.list_habits() if habit["id"] == cursor.lastrowid)

    def delete_habit(self, habit_id: int) -> bool:
        with self.connect() as connection:
            result = connection.execute("DELETE FROM habits WHERE id = ?", (habit_id,))
            if result.rowcount:
                self._refresh_gamification(connection)
        return result.rowcount > 0

    def update_habit(
        self,
        habit_id: int,
        name: str | None = None,
        icon: str | None = None,
        daily_goal: int | None = None,
    ) -> dict[str, Any] | None:
        with self.connect() as connection:
            existing = connection.execute(
                "SELECT * FROM habits WHERE id = ?", (habit_id,)
            ).fetchone()
            if existing is None:
                return None
            next_name = name.strip() if name is not None else existing["name"]
            next_icon = icon.strip() if icon is not None else existing["icon"]
            next_goal = daily_goal if daily_goal is not None else existing["daily_goal"]
            if not next_name or len(next_name) > 200:
                raise ValueError("Habit name must contain 1-200 characters.")
            if not next_icon:
                raise ValueError("Habit icon cannot be empty.")
            if (
                not isinstance(next_goal, int)
                or isinstance(next_goal, bool)
                or next_goal < 1
            ):
                raise ValueError("daily_goal must be a positive integer.")
            connection.execute(
                "UPDATE habits SET name = ?, icon = ?, daily_goal = ?, updated_at = ? "
                "WHERE id = ?",
                (next_name, next_icon, next_goal, _now(), habit_id),
            )
        return next(habit for habit in self.list_habits() if habit["id"] == habit_id)

    def set_habit_completion(self, habit_id: int, completion_date: str, completed: bool) -> dict[str, Any] | None:
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", completion_date):
            raise ValueError("completion_date must use YYYY-MM-DD format.")
        with self.connect() as connection:
            if connection.execute("SELECT 1 FROM habits WHERE id = ?", (habit_id,)).fetchone() is None:
                return None
            existing = connection.execute(
                "SELECT 1 FROM habit_completions WHERE habit_id = ? AND completion_date = ?",
                (habit_id, completion_date),
            ).fetchone()
            xp_delta = 0
            if completed and existing is None:
                connection.execute(
                    "INSERT INTO habit_completions(habit_id, completion_date, created_at) VALUES (?, ?, ?)",
                    (habit_id, completion_date, _now()),
                )
                xp_delta = HABIT_COMPLETION_XP
            elif not completed and existing is not None:
                connection.execute(
                    "DELETE FROM habit_completions WHERE habit_id = ? AND completion_date = ?",
                    (habit_id, completion_date),
                )
                xp_delta = -HABIT_COMPLETION_XP
            connection.execute("UPDATE habits SET updated_at = ? WHERE id = ?", (_now(), habit_id))
            self._refresh_gamification(connection, xp_delta)
        return next(habit for habit in self.list_habits() if habit["id"] == habit_id)

    def get_gamification(self) -> dict[str, Any]:
        with self.connect() as connection:
            self._refresh_gamification(connection)
            data = self._get_gamification(connection)
        return {
            "xp": data["xp"],
            "global_streak": self._calculate_streak(data["valid_streak_dates"]),
            "valid_streak_dates": data["valid_streak_dates"],
            "achievements": data["achievements"],
        }

    def list_reminders(self, active_only: bool = True) -> list[dict[str, Any]]:
        query = "SELECT * FROM reminders"
        if active_only:
            query += " WHERE active = 1"
        query += " ORDER BY run_at"
        with self.connect() as connection:
            rows = connection.execute(query).fetchall()
        return [dict(row) for row in rows]

    def create_reminder(
        self,
        text: str,
        run_at: str,
        recurrence: str | None = None,
        chat_id: str | None = None,
    ) -> dict[str, Any]:
        text = text.strip()
        if not text or len(text) > 500:
            raise ValueError("Reminder text must contain 1-500 characters.")
        if not run_at.strip():
            raise ValueError("Reminder time cannot be empty.")
        reminder_id = str(uuid.uuid4())
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO reminders(id, text, run_at, recurrence, chat_id, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (reminder_id, text, run_at.strip(), recurrence, chat_id, _now()),
            )
            row = connection.execute(
                "SELECT * FROM reminders WHERE id = ?", (reminder_id,)
            ).fetchone()
        return dict(row)

    def get_reminder(self, reminder_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM reminders WHERE id = ?", (reminder_id,)
            ).fetchone()
        return dict(row) if row else None

    def add_notification(
        self, chat_id: str, reminder_id: str, content: str
    ) -> dict[str, Any]:
        created_at = _now()
        with self.connect() as connection:
            cursor = connection.execute(
                "INSERT OR IGNORE INTO notifications "
                "(chat_id, reminder_id, content, created_at) VALUES (?, ?, ?, ?)",
                (chat_id, reminder_id, content, created_at),
            )
            row = connection.execute(
                "SELECT * FROM notifications WHERE id = ?", (cursor.lastrowid,)
            ).fetchone()
        return dict(row)

    def list_notifications(self, chat_id: str, after_id: int = 0) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM notifications WHERE chat_id = ? AND id > ? "
                "ORDER BY id",
                (chat_id, after_id),
            ).fetchall()
        return [dict(row) for row in rows]

    def latest_notification_id(self, chat_id: str) -> int:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT COALESCE(MAX(id), 0) AS latest FROM notifications "
                "WHERE chat_id = ?",
                (chat_id,),
            ).fetchone()
        return int(row["latest"])

    def has_recent_coach_notification(
        self, chat_id: str, minutes: int = 30
    ) -> bool:
        """Return True if a task-coach notification was sent recently."""
        cutoff = (
            datetime.now(timezone.utc) - timedelta(minutes=minutes)
        ).isoformat()
        with self.connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM notifications WHERE chat_id = ? "
                "AND reminder_id LIKE 'task-coach-%' AND created_at > ? LIMIT 1",
                (chat_id, cutoff),
            ).fetchone()
        return row is not None

    def has_recent_weekly_nudge(
        self, chat_id: str, now: datetime | None = None, days: int = 6
    ) -> bool:
        """Return True if a weekly learning nudge landed in the last ``days``."""
        now = now or datetime.now(timezone.utc)
        cutoff = (now - timedelta(days=days)).isoformat()
        with self.connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM notifications WHERE chat_id = ? "
                "AND reminder_id LIKE 'weekly-learning-nudge-%' "
                "AND created_at > ? LIMIT 1",
                (chat_id, cutoff),
            ).fetchone()
        return row is not None

    def cancel_reminder(self, reminder_id: str) -> bool:
        with self.connect() as connection:
            result = connection.execute(
                "UPDATE reminders SET active = 0 WHERE id = ?", (reminder_id,)
            )
        return result.rowcount > 0

    def list_memories(self) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT id, content, created_at, updated_at FROM memories "
                "ORDER BY updated_at DESC"
            ).fetchall()
        return [dict(row) for row in rows]

    def add_memory(self, content: str, embedding: list[float] | None = None) -> dict[str, Any]:
        content = content.strip()
        if not content or len(content) > MAX_MEMORY_TEXT:
            raise ValueError(f"Memory must contain 1-{MAX_MEMORY_TEXT} characters.")
        now = _now()
        if embedding is None:
            embedding = self.embed(content)
        serialized = json.dumps(embedding) if embedding is not None else None
        with self.connect() as connection:
            try:
                cursor = connection.execute(
                    "INSERT INTO memories(content, embedding, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?)",
                    (content, serialized, now, now),
                )
            except sqlite3.IntegrityError as error:
                raise ValueError("That memory already exists.") from error
        return {"id": cursor.lastrowid, "content": content, "created_at": now, "updated_at": now}

    def update_memory(self, memory_id: int, content: str) -> dict[str, Any] | None:
        content = content.strip()
        if not content or len(content) > MAX_MEMORY_TEXT:
            raise ValueError(f"Memory must contain 1-{MAX_MEMORY_TEXT} characters.")
        now = _now()
        embedding = self.embed(content)
        with self.connect() as connection:
            result = connection.execute(
                "UPDATE memories SET content = ?, embedding = ?, updated_at = ? "
                "WHERE id = ?",
                (content, json.dumps(embedding) if embedding is not None else None, now, memory_id),
            )
            if not result.rowcount:
                return None
            row = connection.execute(
                "SELECT id, content, created_at, updated_at FROM memories WHERE id = ?",
                (memory_id,),
            ).fetchone()
        return dict(row)

    def delete_memory(self, memory_id: int) -> bool:
        with self.connect() as connection:
            result = connection.execute(
                "DELETE FROM memories WHERE id = ?", (memory_id,)
            )
        return result.rowcount > 0

    def search_memories(self, query: str, limit: int = MAX_RETRIEVED_MEMORIES) -> list[dict[str, Any]]:
        query_embedding = self.embed(query)
        if query_embedding is not None:
            with self.connect() as connection:
                rows = connection.execute(
                    "SELECT id, content, embedding, created_at, updated_at "
                    "FROM memories WHERE embedding IS NOT NULL"
                ).fetchall()
            ranked = []
            query_norm = sum(value * value for value in query_embedding) ** 0.5
            for row in rows:
                values = json.loads(row["embedding"])
                denominator = query_norm * (sum(value * value for value in values) ** 0.5)
                score = sum(a * b for a, b in zip(query_embedding, values)) / denominator if denominator else 0
                ranked.append((score, dict(row)))
            ranked.sort(key=lambda item: item[0], reverse=True)
            return [item[1] for item in ranked if item[0] >= 0.35][:limit]

        terms = set(re.findall(r"[a-z0-9]+", query.lower()))
        if not terms:
            return []
        scored: list[tuple[int, dict[str, Any]]] = []
        for memory in self.list_memories():
            memory_terms = set(re.findall(r"[a-z0-9]+", memory["content"].lower()))
            score = len(terms & memory_terms)
            if score:
                scored.append((score, memory))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [memory for _, memory in scored[:limit]]

    def embed(self, text: str) -> list[float] | None:
        if os.environ.get("AI_ENABLE_SEMANTIC_MEMORY", "1") == "0":
            return None
        if self._embedder is not None:
            return self._embedder.encode(text, normalize_embeddings=True).tolist()
        try:
            from runtime.models import REGISTRY
            if REGISTRY.is_ready("embedder"):
                self._embedder = REGISTRY.get("embedder")
                return self._embedder.encode(text, normalize_embeddings=True).tolist()
        except Exception:
            pass
        if not self._embedding_warning_printed:
            print("Semantic memory model unavailable; using keyword retrieval.")
            self._embedding_warning_printed = True
        return None

    def build_context(
        self,
        chat_id: str,
        tokenizer: Any,
        query: str,
        max_tokens: int = MAX_CONTEXT_TOKENS,
    ) -> list[dict[str, str]]:
        messages = self.get_messages(chat_id)
        selected: list[dict[str, str]] = []
        for message in reversed(messages):
            candidate = {"role": message["role"], "content": message["content"]}
            trial = [candidate, *selected]
            try:
                if tokenizer is None:
                    raise TypeError("Tokenizer not loaded")
                encoded = tokenizer.apply_chat_template(
                    trial, tokenize=True, add_generation_prompt=True
                )
                if isinstance(encoded, dict) or hasattr(encoded, "input_ids"):
                    ids = encoded["input_ids"]
                    token_count = len(ids[0]) if ids and isinstance(ids[0], list) else len(ids)
                else:
                    token_count = len(encoded)
            except Exception:
                token_count = sum(len(item["content"]) // 4 + 8 for item in trial)
            if token_count > max_tokens:
                break
            selected = trial
        return selected

