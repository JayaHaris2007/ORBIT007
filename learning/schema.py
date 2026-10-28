"""Learning tables and SQLite connection handling for the learning module.

Follows the conventions of memory.py: CREATE TABLE IF NOT EXISTS, foreign
keys with cascading deletes, a row factory returning dict-like rows, and a
connection manager that commits on success and rolls back on failure.

Nothing in this package owns assistant.db exclusively: ``init()`` must be
called once by app.py (with the same database path MemoryStore uses) before
any query runs. Tests call ``init()`` with a temporary database instead.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Literal

from pydantic import BaseModel, Field, StrictBool

ROADMAP_CATEGORIES = ("academic", "technical", "personal")
TOPIC_STATUSES = ("not_started", "learning", "completed", "on_hold")
LINK_STATES = ("suggested", "confirmed", "rejected")
LINK_SOURCES = ("rule", "embedding", "llm", "manual")
ACTIVITY_KINDS = ("planned", "started", "completed", "postponed")

Category = Literal["academic", "technical", "personal"]
TopicStatus = Literal["not_started", "learning", "completed", "on_hold"]
LinkState = Literal["suggested", "confirmed", "rejected"]


class RoadmapIn(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    category: Category = "technical"
    description: str = Field(default="", max_length=2000)


class RoadmapPatch(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=120)
    category: Category | None = None
    description: str | None = Field(default=None, max_length=2000)


class TopicIn(BaseModel):
    roadmap_id: int
    name: str = Field(min_length=1, max_length=120)
    aliases: list[str] = Field(default_factory=list, max_length=20)
    description: str = Field(default="", max_length=2000)
    position: int | None = None
    priority: int = Field(default=1, ge=1, le=3)
    status: TopicStatus = "not_started"
    requires_topic_ids: list[int] = Field(default_factory=list, max_length=20)


class TopicPatch(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=120)
    aliases: list[str] | None = Field(default=None, max_length=20)
    description: str | None = Field(default=None, max_length=2000)
    position: int | None = None
    priority: int | None = Field(default=None, ge=1, le=3)
    status: TopicStatus | None = None


class LinkPatch(BaseModel):
    state: LinkState
    topic_id: int | None = None


class SettingsPatch(BaseModel):
    nudges_enabled: StrictBool

_database_path: Path | None = None


def is_initialized() -> bool:
    """Whether init() has pointed the module at a database yet."""
    return _database_path is not None


def init(database_path: Path) -> None:
    """Point the learning module at a database and create its tables."""
    global _database_path
    _database_path = Path(database_path)
    _database_path.parent.mkdir(parents=True, exist_ok=True)
    with connect() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS roadmaps (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                category TEXT NOT NULL CHECK(category IN ('academic', 'technical', 'personal')),
                description TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS topics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                roadmap_id INTEGER NOT NULL REFERENCES roadmaps(id)
                    ON DELETE CASCADE,
                name TEXT NOT NULL,
                aliases TEXT NOT NULL DEFAULT '[]',
                description TEXT NOT NULL DEFAULT '',
                position INTEGER NOT NULL DEFAULT 0,
                priority INTEGER NOT NULL DEFAULT 1,
                status TEXT NOT NULL DEFAULT 'not_started'
                    CHECK(status IN ('not_started', 'learning', 'completed', 'on_hold')),
                embedding BLOB,
                updated_at TEXT NOT NULL,
                UNIQUE(roadmap_id, name)
            );
            CREATE INDEX IF NOT EXISTS topics_roadmap_idx
                ON topics(roadmap_id, position);
            CREATE TABLE IF NOT EXISTS topic_prerequisites (
                topic_id INTEGER NOT NULL REFERENCES topics(id)
                    ON DELETE CASCADE,
                requires_topic_id INTEGER NOT NULL REFERENCES topics(id)
                    ON DELETE CASCADE,
                PRIMARY KEY(topic_id, requires_topic_id),
                CHECK(topic_id != requires_topic_id)
            );
            CREATE TABLE IF NOT EXISTS task_topic_links (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id INTEGER NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
                topic_id INTEGER NOT NULL REFERENCES topics(id) ON DELETE CASCADE,
                confidence REAL NOT NULL DEFAULT 0,
                state TEXT NOT NULL DEFAULT 'suggested'
                    CHECK(state IN ('suggested', 'confirmed', 'rejected')),
                source TEXT NOT NULL DEFAULT 'rule'
                    CHECK(source IN ('rule', 'embedding', 'llm', 'manual')),
                created_at TEXT NOT NULL,
                UNIQUE(task_id, topic_id)
            );
            CREATE INDEX IF NOT EXISTS task_topic_links_state_idx
                ON task_topic_links(state, id);
            CREATE TABLE IF NOT EXISTS learning_activities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic_id INTEGER NOT NULL REFERENCES topics(id)
                    ON DELETE CASCADE,
                task_id INTEGER REFERENCES tasks(id) ON DELETE SET NULL,
                kind TEXT NOT NULL CHECK(kind IN ('planned', 'started', 'completed', 'postponed')),
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS learning_activities_topic_idx
                ON learning_activities(topic_id, id);
            CREATE TABLE IF NOT EXISTS learning_settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            """
        )


def database_path() -> Path:
    if _database_path is None:
        raise RuntimeError("learning.schema.init() was not called yet.")
    return _database_path


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    connection = sqlite3.connect(database_path())
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


def row_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return dict(row) if row is not None else None
