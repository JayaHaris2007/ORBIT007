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


def test_bare_learning_and_roadmap_topics_listing(learning_db):
    reply, provider = app._handle_local_command("learning", "c1")
    assert provider == "local-learning"
    assert "Java & DSA" in reply

    reply_topics, provider = app._handle_local_command("show topics in Java & DSA", "c1")
    assert provider == "local-learning"
    assert "Java Basics" in reply_topics
    assert "Object-Oriented Programming" in reply_topics
    assert "Locked" in reply_topics


def test_hold_reopen_and_status_filters(learning_db):
    app._handle_local_command("start learning Java Basics", "c1")
    assert _status(learning_db, "Java Basics") == "learning"

    in_prog, provider = app._handle_local_command("what am I currently learning", "c1")
    assert provider == "local-learning"
    assert "Java Basics" in in_prog

    hold_reply, provider = app._handle_local_command("put Java Basics on hold", "c1")
    assert provider == "local-learning"
    assert _status(learning_db, "Java Basics") == "on_hold"

    paused, _ = app._handle_local_command("topics on hold", "c1")
    assert "Java Basics" in paused

    app._handle_local_command("complete topic Java Basics", "c1")
    assert _status(learning_db, "Java Basics") == "completed"

    done_list, _ = app._handle_local_command("what topics have I completed", "c1")
    assert "Java Basics" in done_list

    reopen_reply, provider = app._handle_local_command("reopen learning Java Basics", "c1")
    assert provider == "local-learning"
    assert _status(learning_db, "Java Basics") == "not_started"


def test_contextual_pronouns_and_ordinals_for_learning(learning_db):
    chat_id = app.memory_store.create_chat()["id"]
    app.memory_store.add_message(chat_id, "user", "what should I learn next in Java & DSA")
    next_reply, _ = app._handle_local_command("what should I learn next in Java & DSA", chat_id)
    app.memory_store.add_message(chat_id, "assistant", next_reply)

    # "start it" should resolve to Java Basics from the assistant message
    start_reply, provider = app._handle_local_command("start it", chat_id)
    assert provider == "local-learning"
    assert _status(learning_db, "Java Basics") == "learning"
    app.memory_store.add_message(chat_id, "assistant", start_reply)

    # "mark it completed" should complete Java Basics (not fall through to tasks)
    comp_reply, provider = app._handle_local_command("mark it completed", chat_id)
    assert provider == "local-learning"
    assert _status(learning_db, "Java Basics") == "completed"
    app.memory_store.add_message(chat_id, "assistant", comp_reply)

    # "reopen it" should reopen Java Basics
    reopen_reply, provider = app._handle_local_command("reopen it", chat_id)
    assert provider == "local-learning"
    assert _status(learning_db, "Java Basics") == "not_started"


def test_roadmap_and_topic_crud_via_chat(learning_db):
    reply, provider = app._handle_local_command(
        "create technical roadmap Machine Learning", "c1"
    )
    assert provider == "local-learning"
    assert "Machine Learning" in reply

    t_reply, provider = app._handle_local_command(
        "add topic Linear Regression to Machine Learning", "c1"
    )
    assert provider == "local-learning"
    assert "Linear Regression" in t_reply
    assert _status(learning_db, "Linear Regression") == "not_started"

    prio_reply, provider = app._handle_local_command(
        "set topic Linear Regression priority to high", "c1"
    )
    assert provider == "local-learning"
    assert "high" in prio_reply

    del_t_reply, provider = app._handle_local_command(
        "delete topic Linear Regression", "c1"
    )
    assert provider == "local-learning"
    assert "Removed topic **Linear Regression**" in del_t_reply

    del_rm_reply, provider = app._handle_local_command(
        "delete roadmap Machine Learning", "c1"
    )
    assert provider == "local-learning"
    assert "Deleted learning roadmap **Machine Learning**" in del_rm_reply


def test_learning_activity_and_nudges_commands(learning_db):
    app._handle_local_command("start learning Java Basics", "c1")
    act_reply, provider = app._handle_local_command("show learning activity", "c1")
    assert provider == "local-learning"
    assert "Java Basics" in act_reply

    off_reply, provider = app._handle_local_command("disable learning nudges", "c1")
    assert provider == "local-learning"
    assert "disabled" in off_reply
    assert not progress.get_nudges_enabled()

    on_reply, provider = app._handle_local_command("enable learning nudges", "c1")
    assert provider == "local-learning"
    assert "enabled" in on_reply
    assert progress.get_nudges_enabled()

