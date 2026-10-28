"""Roadmap storage: idempotent seed loading plus topic/roadmap queries.

The seed loader is the only writer allowed to touch rows it did not create:
it inserts missing roadmaps, topics, and prerequisites and never UPDATEs an
existing row, so user edits survive re-running the loader.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import schema

SEEDS_PATH = Path(__file__).resolve().parent / "seeds" / "roadmaps.json"
SEEDED_ROADMAPS_KEY = "seeded_roadmaps"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_seeds(seeds_path: Path | None = None) -> dict[str, int]:
    """Insert the bundled starter roadmaps; idempotent, never overwrites edits.

    Roadmap names that were seeded before are recorded in learning_settings
    and skipped on later runs, so any rename, delete, or edit the user makes
    is never undone or duplicated by re-seeding. Returns counts of rows
    created on this run.
    """
    seeds_path = seeds_path or SEEDS_PATH
    with open(seeds_path, encoding="utf-8") as handle:
        seed_roadmaps = json.load(handle)

    counts = {"roadmaps": 0, "topics": 0, "prerequisites": 0}
    with schema.connect() as connection:
        seeded_names = _seeded_roadmap_names(connection)
        for seed in seed_roadmaps:
            if seed["name"] in seeded_names:
                continue
            roadmap_id = _ensure_roadmap(connection, seed, counts)
            _ensure_topics(connection, roadmap_id, seed.get("topics", []), counts)
            _ensure_prerequisites(
                connection, roadmap_id, seed.get("topics", []), counts
            )
            seeded_names.add(seed["name"])
            connection.execute(
                "INSERT INTO learning_settings (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (SEEDED_ROADMAPS_KEY, json.dumps(sorted(seeded_names))),
            )
    return counts


def _seeded_roadmap_names(connection) -> set[str]:
    row = connection.execute(
        "SELECT value FROM learning_settings WHERE key = ?",
        (SEEDED_ROADMAPS_KEY,),
    ).fetchone()
    if row is None:
        return set()
    try:
        return set(json.loads(row["value"]))
    except (TypeError, ValueError):
        return set()


def _ensure_roadmap(
    connection, seed: dict[str, Any], counts: dict[str, int]
) -> int:
    existing = connection.execute(
        "SELECT id FROM roadmaps WHERE name = ?", (seed["name"],)
    ).fetchone()
    if existing is not None:
        return existing["id"]
    cursor = connection.execute(
        "INSERT INTO roadmaps (name, category, description, created_at) "
        "VALUES (?, ?, ?, ?)",
        (
            seed["name"],
            seed.get("category", "technical"),
            seed.get("description", ""),
            _now(),
        ),
    )
    counts["roadmaps"] += 1
    return int(cursor.lastrowid)


def _ensure_topics(
    connection, roadmap_id: int, seed_topics: list[dict[str, Any]], counts: dict[str, int]
) -> None:
    for position, seed_topic in enumerate(seed_topics):
        existing = connection.execute(
            "SELECT id FROM topics WHERE roadmap_id = ? AND name = ?",
            (roadmap_id, seed_topic["name"]),
        ).fetchone()
        if existing is not None:
            continue
        connection.execute(
            "INSERT INTO topics (roadmap_id, name, aliases, description, "
            "position, priority, status, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                roadmap_id,
                seed_topic["name"],
                json.dumps(seed_topic.get("aliases", [])),
                seed_topic.get("description", ""),
                position,
                int(seed_topic.get("priority", 1)),
                "not_started",
                _now(),
            ),
        )
        counts["topics"] += 1


def _ensure_prerequisites(
    connection, roadmap_id: int, seed_topics: list[dict[str, Any]], counts: dict[str, int]
) -> None:
    for seed_topic in seed_topics:
        topic = connection.execute(
            "SELECT id FROM topics WHERE roadmap_id = ? AND name = ?",
            (roadmap_id, seed_topic["name"]),
        ).fetchone()
        if topic is None:
            continue
        for required_name in seed_topic.get("requires", []):
            required = connection.execute(
                "SELECT id FROM topics WHERE roadmap_id = ? AND name = ?",
                (roadmap_id, required_name),
            ).fetchone()
            if required is None:
                raise ValueError(
                    f"Seed error in '{seed_topic['name']}': unknown prerequisite "
                    f"'{required_name}'."
                )
            cursor = connection.execute(
                "INSERT OR IGNORE INTO topic_prerequisites (topic_id, "
                "requires_topic_id) VALUES (?, ?)",
                (topic["id"], required["id"]),
            )
            counts["prerequisites"] += cursor.rowcount
