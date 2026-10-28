"""Phase 8 tests: chat commands that drive the learning tracker."""

import sqlite3

import pytest

import app
from learning import progress, roadmaps, schema
from memory import MemoryStore


@pytest.fixture
def learning_db(tmp_path, monkeypatch):
    """Point the learning module at a seeded temp database for intent tests."""
    db_file = tmp_path / "test_learning_intents.db"
    MemoryStore(db_file)  # Creates tasks and the other core tables first.
    schema.init(db_file)
    roadmaps.load_seeds()
    yield db_file
    # Reset so other app tests keep the "not initialized" default.
    monkeypatch.setattr(schema, "_database_path", None)


def _status(db_file, name):
    with sqlite3.connect(db_file) as connection:
        connection.row_factory = sqlite3.Row
        return connection.execute(
            "SELECT status FROM topics WHERE name = ? LIMIT 1", (name,)
        ).fetchone()["status"]


def test_learning_progress_reply(learning_db):
    reply, provider = app._handle_local_command("show my learning progress", "c1")
    assert provider == "local-learning"
    assert "50" in reply  # Seeded topic count.
    assert "Java & DSA" in reply


def test_learning_how_going_phrasing(learning_db):
    reply, provider = app._handle_local_command("how is my learning going", "c1")
    assert provider == "local-learning"
    assert "completed" in reply


def test_learning_next_reply(learning_db):
    reply, provider = app._handle_local_command("what should I learn next", "c1")
    assert provider == "local-learning"
    assert "ready to learn" in reply
    assert "Java Basics" in reply


def test_start_learning_topic(learning_db):
    reply, provider = app._handle_local_command("start learning Java Basics", "c1")
    assert provider == "local-learning"
    assert "in progress" in reply
    assert _status(learning_db, "Java Basics") == "learning"


def test_start_locked_topic_names_prerequisites(learning_db):
    reply, provider = app._handle_local_command("start learning Trees", "c1")
    assert provider == "local-learning"
    assert "Recursion" in reply
    assert _status(learning_db, "Trees") == "not_started"


def test_start_unknown_topic_raises(learning_db):
    with pytest.raises(ValueError):
        app._handle_local_command("start learning Rocket Surgery", "c1")


def test_complete_topic(learning_db):
    reply, provider = app._handle_local_command("complete topic Recursion", "c1")
    assert provider == "local-learning"
    assert "completed" in reply
    assert _status(learning_db, "Recursion") == "completed"

    # The prerequisite unlock flows into the candidate pool.
    names = {item["name"] for item in progress.next_topics(limit=20)}
    assert "Trees" in names


def test_i_finished_topic_phrasing(learning_db):
    app._handle_local_command("I finished Java Basics", "c1")
    assert _status(learning_db, "Java Basics") == "completed"


def test_mark_as_learned_phrasing(learning_db):
    app._handle_local_command("mark SQL Queries as learned", "c1")
    assert _status(learning_db, "SQL Queries") == "completed"


def test_learning_provider_navigates_to_learning_page(learning_db):
    page = app._page_for_provider("local-learning")
    assert page == "learning"


def test_roadmap_status_uses_saved_progress(learning_db):
    java_basics = progress.find_topic("Java Basics")
    progress.set_topic_status(java_basics["id"], "completed")

    reply, provider = app._handle_local_command(
        "check my status of Java & DSA", "c1"
    )

    assert provider == "local-learning"
    assert "Java & DSA" in reply
    assert "1 of 18" in reply
    assert "6%" in reply
    assert "Next up" in reply


def test_roadmap_progress_accepts_and_alias(learning_db):
    reply, provider = app._handle_local_command(
        "show my java and dsa progress", "c1"
    )
    assert provider == "local-learning"
    assert "Java & DSA" in reply
    assert "0 of 18" in reply


def test_roadmap_next_reply_only_uses_that_roadmap(learning_db):
    reply, provider = app._handle_local_command(
        "check what to should i learn from Java & DSA", "c1"
    )

    assert provider == "local-learning"
    assert "Java Basics" in reply
    assert "Java & DSA" in reply
    assert "Web Development" not in reply
    assert "DBMS" not in reply


def test_roadmap_task_suggestion_uses_next_available_topic(learning_db):
    reply, provider = app._handle_local_command(
        "suggest task to learn from web development", "c1"
    )

    assert provider == "local-learning"
    assert "Try this task" in reply
    assert "Web Development" in reply
    assert "Practice HTML Fundamentals" in reply
    assert "start learning HTML Fundamentals" in reply


def test_unknown_roadmap_gets_a_helpful_local_reply(learning_db):
    reply, provider = app._handle_local_command(
        "check my status of Rocket Science", "c1"
    )
    assert provider == "local-learning"
    assert "couldn't find" in reply
    assert "Java & DSA" in reply


@pytest.mark.parametrize(
    "phrasing, topic",
    [
        ("i have just compeleted Java Basics", "Java Basics"),
        ("I have complete Java Basics", "Java Basics"),
        ("i had completed Java Basics mark it done", "Java Basics"),
        ("i had completed Java Basics mark it dobe", "Java Basics"),
        ("i just completed Java Basics", "Java Basics"),
        ("I'm done with Java Basics", "Java Basics"),
        ("i already completed Java Basics", "Java Basics"),
        ("I have complete HTML Fundamentals", "HTML Fundamentals"),
    ],
)
def test_loose_completion_phrasings_mark_topic_done(learning_db, phrasing, topic):
    reply, provider = app._handle_local_command(phrasing, "c1")
    assert provider == "local-learning", phrasing
    assert _status(learning_db, topic) == "completed", phrasing
