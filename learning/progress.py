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
from difflib import SequenceMatcher
from typing import Any

try:
    from rapidfuzz import fuzz as _rf_fuzz
    if not isinstance(_rf_fuzz.partial_ratio("a", "a"), (int, float)):
        raise ImportError
    fuzz = _rf_fuzz
except Exception:
    class _FallbackFuzz:
        @staticmethod
        def ratio(s1: str, s2: str) -> float:
            if not s1 and not s2:
                return 100.0
            if not s1 or not s2:
                return 0.0
            return SequenceMatcher(None, s1, s2).ratio() * 100.0

        @staticmethod
        def partial_ratio(s1: str, s2: str) -> float:
            if not s1 or not s2:
                return 0.0
            shorter, longer = (s1, s2) if len(s1) <= len(s2) else (s2, s1)
            if shorter in longer:
                return 100.0
            m = SequenceMatcher(None, shorter, longer)
            blocks = m.get_matching_blocks()
            scores = [SequenceMatcher(None, shorter, longer).ratio() * 100.0]
            k = len(shorter)
            for b in blocks:
                if b.size == 0:
                    continue
                start = max(0, b.b - b.a)
                window = longer[start : start + k]
                scores.append(SequenceMatcher(None, shorter, window).ratio() * 100.0)
            return max(scores)

    fuzz = _FallbackFuzz()  # type: ignore[assignment]

from . import schema

NUDGES_KEY = "nudges_enabled"
IDLE_DAYS_FOR_REASON = 14
HIGH_PRIORITY = 3
NEXT_TOPIC_LIMIT = 5
ACTIVITY_LIMIT = 50

_REF_NOISE_PREFIX = re.compile(
    r"^(?:the\s+|my\s+|a\s+|an\s+)*(?:learning\s+)?(?:topic|roadmap|track|course|module|lesson|subject|skill)\s+(?:called\s+|named\s+|on\s+|for\s+|about\s+)?",
    re.IGNORECASE,
)
_REF_NOISE_SUFFIX = re.compile(
    r"\s+(?:topic|roadmap|track|course|module|lesson)$",
    re.IGNORECASE,
)


def _clean_lookup_ref(reference: str) -> str:
    cleaned = reference.strip().strip("\"'`?!.,;:")
    cleaned = _REF_NOISE_PREFIX.sub("", cleaned).strip()
    cleaned = _REF_NOISE_SUFFIX.sub("", cleaned).strip()
    return re.sub(r"\s+", " ", cleaned).lower()


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


def find_topic(
    reference: str, roadmap_id: int | None = None
) -> dict[str, Any] | None:
    """Best topic for a spoken reference: exact name/alias, else fuzzy."""
    raw_ref = reference.strip().lower()
    cleaned_ref = _clean_lookup_ref(reference)
    needle = cleaned_ref or raw_ref
    if not needle:
        return None
    with schema.connect() as connection:
        if roadmap_id is None:
            topics = connection.execute(
                "SELECT t.*, r.name AS roadmap_name "
                "FROM topics AS t JOIN roadmaps AS r ON r.id = t.roadmap_id"
            ).fetchall()
        else:
            topics = connection.execute(
                "SELECT t.*, r.name AS roadmap_name "
                "FROM topics AS t JOIN roadmaps AS r ON r.id = t.roadmap_id "
                "WHERE t.roadmap_id = ?",
                (roadmap_id,),
            ).fetchall()
    for candidate_needle in (raw_ref, cleaned_ref):
        if not candidate_needle:
            continue
        for topic in topics:
            names = [topic["name"], *_safe_aliases(topic["aliases"])]
            if any(name.lower() == candidate_needle for name in names):
                return dict(topic)
    best = None
    best_score = 0.0
    for topic in topics:
        names = [topic["name"], *_safe_aliases(topic["aliases"])]
        score = max(
            (
                max(
                    fuzz.partial_ratio(needle, name.lower()),
                    fuzz.ratio(needle, name.lower()),
                )
                for name in names
                if name
            ),
            default=0.0,
        )
        if score > best_score:
            best_score = score
            best = dict(topic)
    return best if best_score >= 80 else None


def find_roadmap(reference: str) -> dict[str, Any] | None:
    """Best roadmap summary for a spoken name, including progress counts."""
    raw_norm = _normalize_reference(reference)
    clean_norm = _normalize_reference(_clean_lookup_ref(reference))
    needle = clean_norm or raw_norm
    if not needle:
        return None
    summaries = roadmap_summaries()
    for candidate in (raw_norm, clean_norm):
        if not candidate:
            continue
        for summary in summaries:
            if _normalize_reference(summary["name"]) == candidate:
                return summary
    best = None
    best_score = 0.0
    for summary in summaries:
        norm_name = _normalize_reference(summary["name"])
        score = max(
            fuzz.partial_ratio(needle, norm_name),
            fuzz.ratio(needle, norm_name),
        )
        if score > best_score:
            best_score = score
            best = summary
    return best if best_score >= 85 else None


def _normalize_reference(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", value.lower().replace("&", " and "))
    return re.sub(r"\s+", " ", normalized).strip()


def _safe_aliases(raw: Any) -> list[str]:
    if isinstance(raw, list):
        return [str(alias) for alias in raw]
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


def list_task_links(state: str = "suggested") -> list[dict[str, Any]]:
    if state == "suggested":
        return list_suggested_links()
    with schema.connect() as connection:
        rows = connection.execute(
            "SELECT link.id, link.task_id, link.topic_id, link.confidence, "
            "link.source, link.state, link.created_at, task.title AS task_title, "
            "t.name AS topic_name, r.name AS roadmap_name, "
            "r.id AS roadmap_id "
            "FROM task_topic_links AS link "
            "JOIN tasks AS task ON task.id = link.task_id "
            "JOIN topics AS t ON t.id = link.topic_id "
            "JOIN roadmaps AS r ON r.id = t.roadmap_id "
            "WHERE link.state = ? "
            "ORDER BY link.id DESC",
            (state,),
        ).fetchall()
    return [dict(row) for row in rows]


def update_task_link(
    link_id: int, state: str, topic_id: int | None = None
) -> dict[str, Any]:
    if state not in schema.LINK_STATES:
        raise ValueError(f"'state' must be one of {schema.LINK_STATES}.")
    with schema.connect() as connection:
        link = connection.execute(
            "SELECT * FROM task_topic_links WHERE id = ?", (link_id,)
        ).fetchone()
        if link is None:
            raise ValueError("Link not found.")
        resolved_topic_id = topic_id or link["topic_id"]
        topic = connection.execute(
            "SELECT id, name FROM topics WHERE id = ?", (resolved_topic_id,)
        ).fetchone()
        if topic is None:
            raise ValueError("Topic not found.")
        source = "manual" if topic_id else link["source"]
        connection.execute(
            "UPDATE task_topic_links SET state = ?, topic_id = ?, source = ? "
            "WHERE id = ?",
            (state, resolved_topic_id, source, link_id),
        )
    if state == "confirmed":
        log_activity(resolved_topic_id, link["task_id"], "planned")
    with schema.connect() as connection:
        row = connection.execute(
            "SELECT * FROM task_topic_links WHERE id = ?", (link_id,)
        ).fetchone()
    return dict(row)


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


def _topic_payload(topic: dict[str, Any]) -> dict[str, Any]:
    payload = dict(topic)
    payload["aliases"] = _safe_aliases(payload.get("aliases"))
    payload.pop("embedding", None)
    return payload


def get_roadmap(roadmap_id: int) -> dict[str, Any] | None:
    with schema.connect() as connection:
        row = connection.execute(
            "SELECT * FROM roadmaps WHERE id = ?", (roadmap_id,)
        ).fetchone()
        if row is None:
            return None
        roadmap = dict(row)
        topic_rows = connection.execute(
            "SELECT * FROM topics WHERE roadmap_id = ? ORDER BY position, id",
            (roadmap_id,),
        ).fetchall()
    topics = []
    for t_row in topic_rows:
        topic = _topic_payload(dict(t_row))
        unmet = unavailable_prerequisite_names(topic["id"])
        topic["available"] = not unmet
        topic["unmet_prerequisites"] = unmet
        topics.append(topic)
    roadmap["topics"] = topics
    return roadmap


def create_roadmap(
    name: str, category: str = "technical", description: str = ""
) -> dict[str, Any]:
    clean_name = name.strip()
    if not clean_name:
        raise ValueError("Roadmap name cannot be empty.")
    cat = category.strip().lower() if category else "technical"
    if cat not in schema.ROADMAP_CATEGORIES:
        cat = "technical"
    with schema.connect() as connection:
        existing = connection.execute(
            "SELECT id FROM roadmaps WHERE lower(name) = lower(?)",
            (clean_name,),
        ).fetchone()
        if existing is not None:
            raise ValueError("A roadmap with that name already exists.")
        cursor = connection.execute(
            "INSERT INTO roadmaps (name, category, description, created_at) "
            "VALUES (?, ?, ?, ?)",
            (clean_name, cat, description.strip(), _now()),
        )
        row = connection.execute(
            "SELECT * FROM roadmaps WHERE id = ?", (cursor.lastrowid,)
        ).fetchone()
    return dict(row)


def update_roadmap(roadmap_id: int, **updates: Any) -> dict[str, Any]:
    with schema.connect() as connection:
        existing = connection.execute(
            "SELECT * FROM roadmaps WHERE id = ?", (roadmap_id,)
        ).fetchone()
        if existing is None:
            raise ValueError("Roadmap not found.")
        fields: list[str] = []
        values: list[Any] = []
        for field in ("name", "category", "description"):
            if field in updates and updates[field] is not None:
                fields.append(f"{field} = ?")
                values.append(updates[field])
        if fields:
            values.append(roadmap_id)
            connection.execute(
                f"UPDATE roadmaps SET {', '.join(fields)} WHERE id = ?",
                values,
            )
        row = connection.execute(
            "SELECT * FROM roadmaps WHERE id = ?", (roadmap_id,)
        ).fetchone()
    return dict(row)


def delete_roadmap(roadmap_id: int) -> None:
    with schema.connect() as connection:
        row = connection.execute(
            "SELECT id FROM roadmaps WHERE id = ?", (roadmap_id,)
        ).fetchone()
        if row is None:
            raise ValueError("Roadmap not found.")
        connection.execute("DELETE FROM roadmaps WHERE id = ?", (roadmap_id,))


def create_topic(
    roadmap_id: int,
    name: str,
    aliases: list[str] | None = None,
    description: str = "",
    position: int | None = None,
    priority: int = 1,
    status: str = "not_started",
    requires_topic_ids: list[int] | None = None,
) -> dict[str, Any]:
    clean_name = name.strip()
    if not clean_name:
        raise ValueError("Topic name cannot be empty.")
    prio = max(1, min(3, int(priority)))
    stat = status if status in schema.TOPIC_STATUSES else "not_started"
    req_ids = list(requires_topic_ids or [])
    with schema.connect() as connection:
        rm = connection.execute(
            "SELECT id FROM roadmaps WHERE id = ?", (roadmap_id,)
        ).fetchone()
        if rm is None:
            raise ValueError("Roadmap not found.")
        clash = connection.execute(
            "SELECT id FROM topics WHERE roadmap_id = ? AND lower(name) = lower(?)",
            (roadmap_id, clean_name),
        ).fetchone()
        if clash is not None:
            raise ValueError("A topic with that name already exists.")
        if position is None:
            last = connection.execute(
                "SELECT COALESCE(MAX(position), -1) + 1 AS next FROM topics WHERE roadmap_id = ?",
                (roadmap_id,),
            ).fetchone()
            pos = int(last["next"])
        else:
            pos = int(position)
        cursor = connection.execute(
            "INSERT INTO topics (roadmap_id, name, aliases, description, position, priority, status, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                roadmap_id,
                clean_name,
                json.dumps(aliases or []),
                description.strip(),
                pos,
                prio,
                stat,
                _now(),
            ),
        )
        topic_id = int(cursor.lastrowid)
        for req_id in req_ids:
            connection.execute(
                "INSERT OR IGNORE INTO topic_prerequisites (topic_id, requires_topic_id) VALUES (?, ?)",
                (topic_id, req_id),
            )
        row = connection.execute(
            "SELECT * FROM topics WHERE id = ?", (topic_id,)
        ).fetchone()
    if stat == "learning":
        log_activity(topic_id, None, "started")
    elif stat == "completed":
        log_activity(topic_id, None, "completed")
    return _topic_payload(dict(row))


def update_topic(topic_id: int, **updates: Any) -> dict[str, Any]:
    with schema.connect() as connection:
        existing = connection.execute(
            "SELECT * FROM topics WHERE id = ?", (topic_id,)
        ).fetchone()
        if existing is None:
            raise ValueError("Topic not found.")
        topic = dict(existing)
        if "aliases" in updates and isinstance(updates["aliases"], list):
            updates["aliases"] = json.dumps(updates["aliases"])
        fields: list[str] = []
        values: list[Any] = []
        for field in ("name", "aliases", "description", "position", "priority", "status"):
            if field in updates and updates[field] is not None:
                fields.append(f"{field} = ?")
                values.append(updates[field])
        if fields:
            values.extend([_now(), topic_id])
            connection.execute(
                f"UPDATE topics SET {', '.join(fields)}, updated_at = ? WHERE id = ?",
                values,
            )
        row = connection.execute(
            "SELECT * FROM topics WHERE id = ?", (topic_id,)
        ).fetchone()
    new_status = updates.get("status")
    if new_status and new_status != topic["status"]:
        if new_status == "learning":
            log_activity(topic_id, None, "started")
        elif new_status == "completed":
            log_activity(topic_id, None, "completed")
        elif new_status == "on_hold" and topic["status"] == "learning":
            log_activity(topic_id, None, "postponed")
    return _topic_payload(dict(row))


def delete_topic(topic_id: int) -> None:
    with schema.connect() as connection:
        row = connection.execute(
            "SELECT id FROM topics WHERE id = ?", (topic_id,)
        ).fetchone()
        if row is None:
            raise ValueError("Topic not found.")
        connection.execute("DELETE FROM topics WHERE id = ?", (topic_id,))


def list_topics(
    roadmap_id: int | None = None, status: str | None = None
) -> list[dict[str, Any]]:
    query = (
        "SELECT t.*, r.name AS roadmap_name, r.category AS roadmap_category "
        "FROM topics AS t JOIN roadmaps AS r ON r.id = t.roadmap_id"
    )
    clauses: list[str] = []
    params: list[Any] = []
    if roadmap_id is not None:
        clauses.append("t.roadmap_id = ?")
        params.append(roadmap_id)
    if status is not None:
        clauses.append("t.status = ?")
        params.append(status)
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY t.roadmap_id, t.position, t.id"
    with schema.connect() as connection:
        rows = connection.execute(query, params).fetchall()
    result: list[dict[str, Any]] = []
    for row in rows:
        item = _topic_payload(dict(row))
        unmet = unavailable_prerequisite_names(item["id"])
        item["available"] = not unmet
        item["unmet_prerequisites"] = unmet
        result.append(item)
    return result


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


def get_nudges_enabled() -> bool:
    return nudges_enabled()


def set_nudges_enabled(enabled: bool) -> bool:
    set_setting(NUDGES_KEY, "1" if enabled else "0")
    return bool(enabled)
