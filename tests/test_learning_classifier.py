"""Phase 6 tests: the task-to-topic classifier and the completion rule."""

import re
import sqlite3

import pytest

from learning import classifier, progress, roadmaps, schema
from memory import MemoryStore


@pytest.fixture
def store(tmp_path):
    db_file = tmp_path / "test_learning_classifier.db"
    MemoryStore(db_file)
    schema.init(db_file)
    roadmaps.load_seeds()
    yield db_file


def _make_task(db_file, title, completed=0):
    with sqlite3.connect(db_file) as connection:
        cursor = connection.execute(
            "INSERT INTO tasks (title, priority, completed, created_at, "
            "updated_at) VALUES (?, 'medium', ?, 'now', 'now')",
            (title, completed),
        )
        return cursor.lastrowid


def _connect(db_file):
    connection = sqlite3.connect(db_file)
    connection.row_factory = sqlite3.Row
    return connection


def _topic_id(db_file, name):
    with _connect(db_file) as connection:
        return connection.execute(
            "SELECT id FROM topics WHERE name = ? LIMIT 1", (name,)
        ).fetchone()["id"]


def _links(db_file):
    with _connect(db_file) as connection:
        rows = connection.execute(
            "SELECT * FROM task_topic_links ORDER BY id"
        ).fetchall()
    return [dict(row) for row in rows]


def test_rule_exact_substring_suggests_link(store):
    task_id = _make_task(store, "Study Java basics for the interview")
    created = classifier.suggest_links_for_task(task_id)
    assert len(created) == 1
    links = _links(store)
    assert links[0]["topic_id"] == _topic_id(store, "Java Basics")
    assert links[0]["state"] == "suggested"
    assert links[0]["source"] == "rule"
    assert links[0]["confidence"] == 1.0


def test_rule_fuzzy_match_with_typo(store):
    task_id = _make_task(store, "Review java bascis notes")
    created = classifier.suggest_links_for_task(task_id)
    assert len(created) == 1
    assert _links(store)[0]["source"] == "rule"
    assert _links(store)[0]["confidence"] == classifier.RULE_FUZZ_CONFIDENCE


def test_no_match_creates_no_links(store):
    task_id = _make_task(store, "Buy groceries for the week")
    assert classifier.suggest_links_for_task(task_id) == []
    assert _links(store) == []


def test_suggestions_are_deduplicated(store):
    task_id = _make_task(store, "Study Java basics")
    first = classifier.suggest_links_for_task(task_id)
    second = classifier.suggest_links_for_task(task_id)
    assert len(first) == 1
    assert second == []
    assert len(_links(store)) == 1


def test_embedding_stage_when_rules_miss(store):
    def fake_embed(text):
        lowered = text.lower()
        # Topic embeddings are built from "name description aliases…", so
        # the topic name is the prefix; the task title is free text. Use a
        # word-boundary check so "containers" (Generics description) is not
        # confused with the title's "container".
        if lowered.startswith("collections framework") or re.search(
            r"\bcontainer\b", lowered
        ):
            return [1.0, 0.0, 0.0]
        return [0.0, 1.0, 0.0]

    task_id = _make_task(store, "Container stuff")
    created = classifier.suggest_links_for_task(task_id, embed=fake_embed)
    assert len(created) == 1
    link = _links(store)[0]
    assert link["topic_id"] == _topic_id(store, "Collections Framework")
    assert link["source"] == "embedding"
    assert 0.5 < link["confidence"] <= classifier.EMBED_CONFIDENCE_CAP

    # The topic embedding was cached on the row for the next run.
    with _connect(store) as connection:
        row = connection.execute(
            "SELECT embedding FROM topics WHERE id = ?", (link["topic_id"],)
        ).fetchone()
    assert row["embedding"] is not None


def test_completed_topics_are_skipped(store):
    with _connect(store) as connection:
        connection.execute(
            "UPDATE topics SET status = 'completed' WHERE name = 'Java Basics'"
        )
    task_id = _make_task(store, "Study Java basics again")
    assert classifier.suggest_links_for_task(task_id) == []


def test_suggest_all_open_tasks(store):
    _make_task(store, "Study Java basics")
    _make_task(store, "Buy groceries")
    _make_task(store, "Practice DBMS SQL", completed=1)
    created = classifier.suggest_links_for_all_open_tasks()
    assert created >= 1
    links = _links(store)
    assert len(links) == created
    assert all(link["state"] == "suggested" for link in links)


def test_task_completion_rule_applies_to_confirmed_links(store):
    task_id = _make_task(store, "Study Java basics")
    topic_id = _topic_id(store, "Java Basics")
    with _connect(store) as connection:
        connection.execute(
            "INSERT INTO task_topic_links (task_id, topic_id, confidence, "
            "state, source, created_at) VALUES (?, ?, 0.9, 'confirmed', "
            "'rule', 'now')",
            (task_id, topic_id),
        )
    assert progress.on_task_completed_links(task_id) == [topic_id]
    with _connect(store) as connection:
        topic = connection.execute(
            "SELECT status FROM topics WHERE id = ?", (topic_id,)
        ).fetchone()
        activity = connection.execute(
            "SELECT kind FROM learning_activities WHERE topic_id = ? "
            "AND task_id = ?",
            (topic_id, task_id),
        ).fetchone()
    assert topic["status"] == "learning"
    assert activity["kind"] == "completed"


def test_task_completion_rule_ignores_suggested_links(store):
    task_id = _make_task(store, "Study Java basics")
    topic_id = _topic_id(store, "Java Basics")
    with _connect(store) as connection:
        connection.execute(
            "INSERT INTO task_topic_links (task_id, topic_id, confidence, "
            "state, source, created_at) VALUES (?, ?, 0.7, 'suggested', "
            "'rule', 'now')",
            (task_id, topic_id),
        )
    assert progress.on_task_completed_links(task_id) == []
    with _connect(store) as connection:
        topic = connection.execute(
            "SELECT status FROM topics WHERE id = ?", (topic_id,)
        ).fetchone()
    assert topic["status"] == "not_started"


def _fake_llm(reply):
    def match(title, topics):
        return reply
    return match


def test_llm_fallback_creates_suggested_link(store):
    task_id = _make_task(store, "Prepare for tomorrow's quiz")
    created = classifier.suggest_links_for_task(
        task_id, llm=_fake_llm("Dynamic Programming")
    )
    assert len(created) == 1
    links = _links(store)
    assert links[0]["topic_id"] == _topic_id(store, "Dynamic Programming")
    assert links[0]["source"] == "llm"
    assert links[0]["confidence"] == classifier.LLM_FALLBACK_CONFIDENCE
    assert links[0]["state"] == "suggested"


def test_llm_fallback_rejects_none(store):
    task_id = _make_task(store, "Paint a watercolor landscape")
    assert classifier.suggest_links_for_task(
        task_id, llm=_fake_llm("none")
    ) == []
    assert _links(store) == []


def test_llm_fallback_ignores_unknown_names(store):
    task_id = _make_task(store, "Paint a watercolor landscape")
    assert classifier.suggest_links_for_task(
        task_id, llm=_fake_llm("Rocket Surgery")
    ) == []
    assert _links(store) == []


def test_llm_fallback_tolerates_numbered_prefix(store):
    task_id = _make_task(store, "Prepare for tomorrow's quiz")
    created = classifier.suggest_links_for_task(
        task_id, llm=_fake_llm("17. Dynamic Programming\n")
    )
    assert len(created) == 1
    assert _links(store)[0]["topic_id"] == _topic_id(
        store, "Dynamic Programming"
    )


def test_llm_fallback_not_used_when_rules_match(store):
    task_id = _make_task(store, "Study Java basics")
    called = []

    def tracking_llm(title, topics):
        called.append(title)
        return "none"

    created = classifier.suggest_links_for_task(task_id, llm=tracking_llm)
    assert len(created) == 1
    assert _links(store)[0]["source"] == "rule"
    assert called == []  # The rule stage answered; the model was never asked.


def test_llm_fallback_catches_crashes(store):
    task_id = _make_task(store, "Prepare for tomorrow's quiz")

    def crashing_llm(title, topics):
        raise RuntimeError("model exploded")

    assert classifier.suggest_links_for_task(task_id, llm=crashing_llm) == []
    assert _links(store) == []
