"""Progress rules and activity logging for the learning module.

Plain SQL and Python only — no LLM, no graph library:

- Completing a linked task logs an activity and moves its topic to
  "learning". It never marks the topic completed: a finished task is not
  mastery; only the user confirms completion.
- A topic is available when every prerequisite is completed.
- Next topics are available, not completed, ordered by priority, then
  position, then least recently touched.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any

from rapidfuzz import fuzz

from . import schema

NUDGES_KEY = "nudges_enabled"
IDLE_DAYS_FOR_REASON = 14
HIGH_PRIORITY = 3
NEXT_TOPIC_LIMIT = 5
ACTIVITY_LIMIT = 50


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log_activity(
    topic_id: int, task_id: int | None, kind: str
) -> int | None:
    """Record one learning event, deduplicating (topic, task, kind)."""
    if kind not in schema.ACTIVITY_KINDS:
        raise ValueError(f"Unknown activity kind: {kind}")
    with schema.connect() as connection:
        duplicate = connection.execute(
            "SELECT id FROM learning_activities WHERE topic_id = ? "
            "AND task_id IS ? AND kind = ? LIMIT 1",
            (topic_id, task_id, kind),
        ).fetchone()
        if duplicate is not None:
            return None
        cursor = connection.execute(
            "INSERT INTO learning_activities (topic_id, task_id, kind, "
            "created_at) VALUES (?, ?, ?, ?)",
            (topic_id, task_id, kind, _now()),
        )
        return int(cursor.lastrowid)


def touch_topic(topic_id: int) -> None:
    with schema.connect() as connection:
        connection.execute(
            "UPDATE topics SET updated_at = ? WHERE id = ?", (_now(), topic_id)
        )


def on_linked_task_completed(topic_id: int, task_id: int) -> None:
    """Completion rule: log the event and move the topic to 'learning'."""
    log_activity(topic_id, task_id, "completed")
    with schema.connect() as connection:
        topic = connection.execute(
            "SELECT status FROM topics WHERE id = ?", (topic_id,)
        ).fetchone()
        if topic is None:
            return
        if topic["status"] in ("not_started", "on_hold"):
            connection.execute(
                "UPDATE topics SET status = 'learning', updated_at = ? "
                "WHERE id = ?",
                (_now(), topic_id),
            )
        else:
            connection.execute(
                "UPDATE topics SET updated_at = ? WHERE id = ?",
                (_now(), topic_id),
            )


def on_task_completed_links(task_id: int) -> list[int]:
    """Apply the completion rule to every confirmed link of a task."""
    with schema.connect() as connection:
        rows = connection.execute(
            "SELECT topic_id FROM task_topic_links "
            "WHERE task_id = ? AND state = 'confirmed'",
            (task_id,),
        ).fetchall()
    topic_ids = [row["topic_id"] for row in rows]
    for topic_id in topic_ids:
        on_linked_task_completed(topic_id, task_id)
    return topic_ids


def set_topic_status(topic_id: int, status: str) -> bool:
    """Update a topic status and log the matching activity."""
    if status not in schema.TOPIC_STATUSES:
        raise ValueError(f"Unknown topic status: {status}")
    with schema.connect() as connection:
        topic = connection.execute(
            "SELECT status FROM topics WHERE id = ?", (topic_id,)
        ).fetchone()
        if topic is None:
            return False
        connection.execute(
            "UPDATE topics SET status = ?, updated_at = ? WHERE id = ?",
            (status, _now(), topic_id),
        )
    previous = topic["status"]
    if status == previous:
        return True
    if status == "learning":
        log_activity(topic_id, None, "started")
    elif status == "completed":
        log_activity(topic_id, None, "completed")
    elif status == "on_hold" and previous == "learning":
        log_activity(topic_id, None, "postponed")
    return True


def find_topic(reference: str) -> dict[str, Any] | None:
    """Best topic for a spoken reference: exact name/alias, else fuzzy."""
    reference = reference.strip().lower()
    if not reference:
        return None
    with schema.connect() as connection:
        topics = connection.execute(
            "SELECT t.*, r.name AS roadmap_name "
            "FROM topics AS t JOIN roadmaps AS r ON r.id = t.roadmap_id"
        ).fetchall()
    exact = None
    for topic in topics:
        names = [topic["name"], *_safe_aliases(topic["aliases"])]
        if any(name.lower() == reference for name in names):
            exact = dict(topic)
            break
    if exact is not None:
        return exact
    best = None
    best_score = 0
    for topic in topics:
        names = [topic["name"], *_safe_aliases(topic["aliases"])]
        score = max(fuzz.partial_ratio(reference, name.lower()) for name in names)
        if score > best_score:
            best_score = score
            best = dict(topic)
    return best if best_score >= 80 else None


def find_roadmap(reference: str) -> dict[str, Any] | None:
    """Best roadmap summary for a spoken name, including progress counts."""
    reference = _normalize_reference(reference)
    if not reference:
        return None
    summaries = roadmap_summaries()
    for summary in summaries:
        if _normalize_reference(summary["name"]) == reference:
            return summary
    best = None
    best_score = 0
    for summary in summaries:
        score = fuzz.partial_ratio(
            reference, _normalize_reference(summary["name"])
        )
        if score > best_score:
            best_score = score
            best = summary
    return best if best_score >= 85 else None


def _normalize_reference(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", value.lower().replace("&", " and "))
    return re.sub(r"\s+", " ", normalized).strip()


def _safe_aliases(raw: str | None) -> list[str]:
    try:
        aliases = json.loads(raw or "[]")
    except (TypeError, ValueError):
        return []
    return [alias for alias in aliases if isinstance(alias, str)]


def unavailable_prerequisite_names(topic_id: int) -> list[str]:
    """Names of prerequisites that are not completed yet."""
    with schema.connect() as connection:
        rows = connection.execute(
            "SELECT req.name FROM topic_prerequisites AS prereq "
            "JOIN topics AS req ON req.id = prereq.requires_topic_id "
            "WHERE prereq.topic_id = ? AND req.status != 'completed' "
            "ORDER BY req.position",
            (topic_id,),
        ).fetchall()
    return [row["name"] for row in rows]


def topic_available(topic_id: int) -> bool:
    return not unavailable_prerequisite_names(topic_id)


def roadmap_summaries() -> list[dict[str, Any]]:
    """Every roadmap with topic counts, percent complete, and a next hint."""
    with schema.connect() as connection:
        roadmaps = connection.execute(
            "SELECT * FROM roadmaps ORDER BY id"
        ).fetchall()
        summaries = []
        for roadmap in roadmaps:
            counts = connection.execute(
                "SELECT "
                "SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) AS done, "
                "SUM(CASE WHEN status = 'learning' THEN 1 ELSE 0 END) AS active, "
                "COUNT(*) AS total "
                "FROM topics WHERE roadmap_id = ?",
                (roadmap["id"],),
            ).fetchone()
            total = counts["total"] or 0
            done = counts["done"] or 0
            next_hint = next_topic_for_roadmap(connection, roadmap["id"])
            summaries.append(
                {
                    "id": roadmap["id"],
                    "name": roadmap["name"],
                    "category": roadmap["category"],
                    "description": roadmap["description"],
                    "created_at": roadmap["created_at"],
                    "topic_count": total,
                    "completed_count": done,
                    "in_progress_count": counts["active"] or 0,
                    "completion_percent": round(done * 100 / total) if total else 0,
                    "next_topic": next_hint,
                }
            )
    return summaries


def next_topic_for_roadmap(connection, roadmap_id: int) -> str | None:
    rows = _available_candidates(connection, roadmap_id, limit=1)
    return rows[0]["name"] if rows else None


def next_topics(
    limit: int = NEXT_TOPIC_LIMIT, roadmap_id: int | None = None
) -> list[dict[str, Any]]:
    """Recommended next topics with a short reason each."""
    with schema.connect() as connection:
        return _available_candidates(connection, roadmap_id, limit=limit)


def _available_candidates(connection, roadmap_id: int | None, limit: int):
    where = "WHERE req.id IS NULL AND t.status != 'completed'"
    params: list[Any] = []
    if roadmap_id is not None:
        where += " AND t.roadmap_id = ?"
        params.append(roadmap_id)
    rows = connection.execute(
        "SELECT t.id, t.name, t.status, t.priority, t.updated_at, "
        "r.id AS roadmap_id, r.name AS roadmap_name, "
        "COALESCE(MAX(a.created_at), t.updated_at) AS last_touched "
        "FROM topics AS t "
        "JOIN roadmaps AS r ON r.id = t.roadmap_id "
        "LEFT JOIN topic_prerequisites AS prereq ON prereq.topic_id = t.id "
        "LEFT JOIN topics AS req ON req.id = prereq.requires_topic_id "
        "AND req.status != 'completed' "
        "LEFT JOIN learning_activities AS a ON a.topic_id = t.id "
        f"{where} "
        "GROUP BY t.id "
        "ORDER BY t.priority DESC, t.position, last_touched ASC "
        "LIMIT ?",
        (*params, limit),
    ).fetchall()
    now = datetime.now(timezone.utc)
    results = []
    for row in rows:
        try:
            last = datetime.fromisoformat(row["last_touched"])
            idle_days = max((now - last).days, 0)
        except (TypeError, ValueError):
            idle_days = 0
        results.append(
            {
                "topic_id": row["id"],
                "name": row["name"],
                "roadmap_id": row["roadmap_id"],
                "roadmap_name": row["roadmap_name"],
                "status": row["status"],
                "priority": row["priority"],
                "idle_days": idle_days,
                "reason": _next_reason(row, idle_days),
            }
        )
    return results


def _next_reason(row, idle_days: int) -> str:
    if idle_days >= IDLE_DAYS_FOR_REASON:
        return f"Idle for {idle_days} days — pick it back up."
    if row["priority"] >= HIGH_PRIORITY:
        return "High priority."
    return "Prerequisites complete — ready to start."


def overall_progress() -> dict[str, Any]:
    with schema.connect() as connection:
        totals = connection.execute(
            "SELECT COUNT(*) AS total, "
            "SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) AS done, "
            "SUM(CASE WHEN status = 'learning' THEN 1 ELSE 0 END) AS active "
            "FROM topics"
        ).fetchone()
        total = totals["total"] or 0
        done = totals["done"] or 0
        roadmap_count = connection.execute(
            "SELECT COUNT(*) AS n FROM roadmaps"
        ).fetchone()["n"]
        per_roadmap = [
            {
                "roadmap_id": item["id"],
                "roadmap_name": item["name"],
                "completed": item["completed_count"],
                "total": item["topic_count"],
                "percent": item["completion_percent"],
            }
            for item in roadmap_summaries()
        ]
    return {
        "roadmap_count": roadmap_count,
        "topic_count": total,
        "completed_topics": done,
        "in_progress_topics": totals["active"] or 0,
        "completion_percent": round(done * 100 / total) if total else 0,
        "per_roadmap": per_roadmap,
    }


def list_activities(limit: int = ACTIVITY_LIMIT) -> list[dict[str, Any]]:
    with schema.connect() as connection:
        rows = connection.execute(
            "SELECT a.id, a.kind, a.created_at, a.topic_id, a.task_id, "
            "t.name AS topic_name, r.name AS roadmap_name, "
            "task.title AS task_title "
            "FROM learning_activities AS a "
            "JOIN topics AS t ON t.id = a.topic_id "
            "JOIN roadmaps AS r ON r.id = t.roadmap_id "
            "LEFT JOIN tasks AS task ON task.id = a.task_id "
            "ORDER BY a.id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def list_suggested_links() -> list[dict[str, Any]]:
    with schema.connect() as connection:
        rows = connection.execute(
            "SELECT link.id, link.task_id, link.topic_id, link.confidence, "
            "link.source, link.created_at, task.title AS task_title, "
            "t.name AS topic_name, r.name AS roadmap_name, "
            "r.id AS roadmap_id "
            "FROM task_topic_links AS link "
            "JOIN tasks AS task ON task.id = link.task_id "
            "JOIN topics AS t ON t.id = link.topic_id "
            "JOIN roadmaps AS r ON r.id = t.roadmap_id "
            "WHERE link.state = 'suggested' "
            "ORDER BY link.id DESC"
        ).fetchall()
    return [dict(row) for row in rows]


def list_confirmed_task_links(topic_id: int) -> list[dict[str, Any]]:
    with schema.connect() as connection:
        rows = connection.execute(
            "SELECT link.id, link.task_id, link.confidence, link.source, "
            "link.state, task.title AS task_title, "
            "task.completed AS task_completed "
            "FROM task_topic_links AS link "
            "JOIN tasks AS task ON task.id = link.task_id "
            "WHERE link.topic_id = ? AND link.state = 'confirmed' "
            "ORDER BY link.id DESC",
            (topic_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def get_setting(key: str, default: str | None = None) -> str | None:
    with schema.connect() as connection:
        row = connection.execute(
            "SELECT value FROM learning_settings WHERE key = ?", (key,)
        ).fetchone()
    return row["value"] if row is not None else default


def set_setting(key: str, value: str) -> None:
    with schema.connect() as connection:
        connection.execute(
            "INSERT INTO learning_settings (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )


def nudges_enabled() -> bool:
    return get_setting(NUDGES_KEY, "1") != "0"
