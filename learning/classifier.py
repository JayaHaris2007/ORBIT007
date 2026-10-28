"""Task-to-topic classifier for the learning module.

Pipeline (cheap to expensive):

1. **Rules** — normalized name/alias matching against the task title:
   exact substring containment first, then rapidfuzz partial similarity.
   Free, deterministic, and precise for "Study Java basics".
2. **Embeddings** — if the rules find nothing, embed the task title with
   the shared ``memory_store.embed`` sentence-transformer and compare it
   against lazily computed topic embeddings (cached on the topic row).
3. **LLM fallback** — if both stages miss, ask the local Qwen model to
   pick the best topic from a numbered list. The raw reply is parsed
   back to a topic id here; the caller (app.py) owns the model.

Every function is defensive: the task hooks call this from background
threads, so any failure must degrade to "no suggestions", never raise.
"""

from __future__ import annotations

import json
import re
import sqlite3
from datetime import datetime, timezone
from typing import Any, Callable

import numpy

from rapidfuzz import fuzz

from . import schema

RULE_FUZZ_THRESHOLD = 85
RULE_TOKEN_THRESHOLD = 80
RULE_CONTAINMENT_LENGTH_GAP = 2
RULE_FUZZ_CONFIDENCE = 0.85
EMBED_SIM_THRESHOLD = 0.45
EMBED_CONFIDENCE_FLOOR = 0.5
EMBED_CONFIDENCE_CAP = 0.95
LLM_FALLBACK_CONFIDENCE = 0.8
TOPIC_MATCH_LIMIT = 3

TOKEN_STOPWORDS = {
    "a", "an", "and", "as", "at", "be", "by", "do", "for", "if",
    "in", "is", "it", "my", "of", "on", "or", "so", "the", "to",
    "up", "we",
}

Embedder = Callable[[str], list[float] | None]
#: Callable receiving the task title and the open topic rows, returning the
#: raw model reply ("Dynamic Programming", "none", ...) or None on failure.
LlmMatcher = Callable[[str, list[sqlite3.Row]], str | None]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _task_title(connection: sqlite3.Connection, task_id: int) -> str | None:
    row = connection.execute(
        "SELECT title FROM tasks WHERE id = ?", (task_id,)
    ).fetchone()
    return row["title"] if row is not None else None


def _open_topics(connection: sqlite3.Connection) -> list[sqlite3.Row]:
    """Topics that can still receive links, across every roadmap."""
    return connection.execute(
        "SELECT * FROM topics WHERE status != 'completed'"
    ).fetchall()


def _topic_aliases(topic: sqlite3.Row) -> list[str]:
    try:
        aliases = json.loads(topic["aliases"] or "[]")
    except (TypeError, ValueError):
        return []
    return [alias for alias in aliases if isinstance(alias, str)]


def _significant_tokens(text: str) -> list[str]:
    return [
        token for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) >= 2 and token not in TOKEN_STOPWORDS
    ]


def _token_hits(name_token: str, title_token: str) -> bool:
    """One name token matches one title token (exact, fuzzy, or near plural)."""
    if name_token == title_token:
        return True
    if fuzz.ratio(name_token, title_token) >= RULE_TOKEN_THRESHOLD:
        return True
    # Substring containment only for near-equal lengths, so "java" does not
    # match "javascript" but "tree" still matches "trees".
    if abs(len(name_token) - len(title_token)) <= RULE_CONTAINMENT_LENGTH_GAP and (
        name_token in title_token or title_token in name_token
    ):
        return True
    return False


def _token_matches(title_tokens: list[str], name: str) -> bool:
    """Every significant token of the name must hit a title token."""
    name_tokens = _significant_tokens(name)
    if not name_tokens:
        return False
    for token in name_tokens:
        if any(_token_hits(token, title_token) for title_token in title_tokens):
            continue
        return False
    return True


def _rule_matches(title: str, topics: list[sqlite3.Row]) -> list[tuple[int, float, str]]:
    """Topic ids the rules match, with confidence and source each."""
    title_lower = title.lower()
    title_tokens = _significant_tokens(title_lower)
    matches: list[tuple[int, float, str]] = []
    for topic in topics:
        names = [topic["name"], *_topic_aliases(topic)]
        for name in names:
            name_lower = name.lower()
            if not name_lower:
                continue
            if name_lower in title_lower or title_lower in name_lower:
                matches.append((topic["id"], 1.0, "rule"))
                break
            # Fuzzy spelling tolerance, but every significant word of the
            # topic must be present in the title to stay precise.
            ratio = fuzz.partial_ratio(title_lower, name_lower)
            if (
                ratio >= RULE_FUZZ_THRESHOLD
                and _token_matches(title_tokens, name_lower)
            ):
                matches.append((topic["id"], RULE_FUZZ_CONFIDENCE, "rule"))
                break
    return matches


def _topic_text(topic: sqlite3.Row) -> str:
    parts = [topic["name"], topic["description"] or "", *_topic_aliases(topic)]
    return " ".join(part for part in parts if part)


def _topic_embedding(
    connection: sqlite3.Connection,
    topic: sqlite3.Row,
    embed: Embedder,
) -> list[float] | None:
    if topic["embedding"] is not None:
        try:
            return numpy.frombuffer(topic["embedding"], dtype=numpy.float32).tolist()
        except ValueError:
            pass
    vector = embed(_topic_text(topic))
    if vector is None:
        return None
    blob = numpy.asarray(vector, dtype=numpy.float32).tobytes()
    connection.execute(
        "UPDATE topics SET embedding = ? WHERE id = ?", (blob, topic["id"])
    )
    return vector


def _cosine(a: list[float], b: list[float]) -> float:
    first = numpy.asarray(a, dtype=numpy.float32)
    second = numpy.asarray(b, dtype=numpy.float32)
    denominator = numpy.linalg.norm(first) * numpy.linalg.norm(second)
    if denominator == 0:
        return 0.0
    return float(numpy.dot(first, second) / denominator)


def _embedding_matches(
    title: str,
    topics: list[sqlite3.Row],
    embed: Embedder,
    connection: sqlite3.Connection,
) -> list[tuple[int, float, str]]:
    title_vector = embed(title)
    if title_vector is None:
        return []
    scored: list[tuple[int, float, str]] = []
    for topic in topics:
        vector = _topic_embedding(connection, topic, embed)
        if vector is None:
            continue
        similarity = _cosine(title_vector, vector)
        if similarity >= EMBED_SIM_THRESHOLD:
            confidence = min(
                EMBED_CONFIDENCE_CAP,
                EMBED_CONFIDENCE_FLOOR + similarity,
            )
            scored.append((topic["id"], confidence, "embedding"))
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:TOPIC_MATCH_LIMIT]


def _parse_llm_reply(reply: str, topics: list[sqlite3.Row]) -> int | None:
    """Map the model's reply back to one topic id, longest name first."""
    cleaned = re.sub(r"[^a-z0-9 ]", " ", reply.lower())
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned or "none" in cleaned.split():
        return None
    for topic in sorted(topics, key=lambda item: len(item["name"]), reverse=True):
        name = topic["name"].lower()
        if name in cleaned or cleaned in name:
            return topic["id"]
    return None


def _llm_matches(
    title: str,
    topics: list[sqlite3.Row],
    llm: LlmMatcher,
) -> list[tuple[int, float, str]]:
    reply = llm(title, topics)
    if not reply or not reply.strip():
        return []
    topic_id = _parse_llm_reply(reply, topics)
    if topic_id is None:
        return []
    return [(topic_id, LLM_FALLBACK_CONFIDENCE, "llm")]


def _insert_suggested_link(
    connection: sqlite3.Connection,
    task_id: int,
    topic_id: int,
    confidence: float,
    source: str,
) -> int | None:
    """Insert a suggested link unless this (task, topic) pair is taken."""
    existing = connection.execute(
        "SELECT id FROM task_topic_links WHERE task_id = ? AND topic_id = ?",
        (task_id, topic_id),
    ).fetchone()
    if existing is not None:
        return None
    cursor = connection.execute(
        "INSERT INTO task_topic_links (task_id, topic_id, confidence, state, "
        "source, created_at) VALUES (?, ?, ?, 'suggested', ?, ?)",
        (task_id, topic_id, confidence, source, _now()),
    )
    return int(cursor.lastrowid)


def suggest_links_for_task(
    task_id: int,
    embed: Embedder | None = None,
    llm: LlmMatcher | None = None,
) -> list[int]:
    """Run the classifier for one task; return the created link ids."""
    with schema.connect() as connection:
        title = _task_title(connection, task_id)
        if not title:
            return []
        topics = _open_topics(connection)
        if not topics:
            return []
        matches = _rule_matches(title, topics)
        if not matches and embed is not None:
            try:
                matches = _embedding_matches(title, topics, embed, connection)
            except Exception:  # pragma: no cover - embedder is best-effort
                matches = []
        if not matches and llm is not None:
            try:
                matches = _llm_matches(title, topics, llm)
            except Exception:  # pragma: no cover - model is best-effort
                matches = []
        created = []
        for topic_id, confidence, source in matches:
            link_id = _insert_suggested_link(
                connection, task_id, topic_id, confidence, source
            )
            if link_id is not None:
                created.append(link_id)
        return created


def suggest_links_for_all_open_tasks(
    embed: Embedder | None = None, llm: LlmMatcher | None = None
) -> int:
    """Suggest links for every open task; returns how many links were made."""
    with schema.connect() as connection:
        rows = connection.execute(
            "SELECT id FROM tasks WHERE completed = 0"
        ).fetchall()
    created = 0
    for row in rows:
        created += len(suggest_links_for_task(row["id"], embed=embed, llm=llm))
    return created
