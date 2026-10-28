"""Phase 1 tests: learning schema, idempotent seeding, and cascading deletes."""

import sqlite3

import pytest

from learning import roadmaps, schema
from memory import MemoryStore


@pytest.fixture
def learning_db(tmp_path):
    """A temporary database with both the core and the learning tables."""
    db_file = tmp_path / "test_learning.db"
    MemoryStore(db_file)  # Creates the tasks table the FK references.
    schema.init(db_file)
    return db_file


def _counts(db_file):
    with sqlite3.connect(db_file) as connection:
        connection.row_factory = sqlite3.Row
        tables = (
            "roadmaps",
            "topics",
            "topic_prerequisites",
            "task_topic_links",
            "learning_activities",
            "learning_settings",
        )
        return {
            table: connection.execute(
                f"SELECT COUNT(*) AS n FROM {table}"
            ).fetchone()["n"]
            for table in tables
        }


def test_seed_loading_is_idempotent(learning_db):
    first = roadmaps.load_seeds()
    assert first == {"roadmaps": 3, "topics": 50, "prerequisites": 50}
    assert _counts(learning_db) == {
        "roadmaps": 3,
        "topics": 50,
        "topic_prerequisites": 50,
        "task_topic_links": 0,
        "learning_activities": 0,
        "learning_settings": 1,
    }

    second = roadmaps.load_seeds()
    assert second == {"roadmaps": 0, "topics": 0, "prerequisites": 0}
    assert _counts(learning_db)["topics"] == 50


def test_seeds_never_overwrite_user_edits(learning_db):
    roadmaps.load_seeds()
    with schema.connect() as connection:
        connection.execute(
            "UPDATE topics SET name = 'Renamed by me' WHERE name = 'Java Basics'"
        )
        connection.execute(
            "UPDATE roadmaps SET name = 'My DSA Roadmap' WHERE name = 'Java & DSA'"
        )

    roadmaps.load_seeds()

    with schema.connect() as connection:
        renamed_topic = connection.execute(
            "SELECT id FROM topics WHERE name = 'Renamed by me'"
        ).fetchone()
        original_topic = connection.execute(
            "SELECT id FROM topics WHERE name = 'Java Basics'"
        ).fetchone()
        renamed_roadmap = connection.execute(
            "SELECT id FROM roadmaps WHERE name = 'My DSA Roadmap'"
        ).fetchone()
        original_roadmap = connection.execute(
            "SELECT id FROM roadmaps WHERE name = 'Java & DSA'"
        ).fetchone()

    assert renamed_topic is not None
    assert original_topic is None  # Not re-added after the rename.
    assert renamed_roadmap is not None
    assert original_roadmap is None
    assert _counts(learning_db)["topics"] == 50
    assert _counts(learning_db)["roadmaps"] == 3


def test_task_topic_links_are_unique(learning_db):
    roadmaps.load_seeds()
    with schema.connect() as connection:
        connection.execute(
            "INSERT INTO tasks (title, priority, created_at, updated_at) "
            "VALUES ('Study Java', 'medium', 'now', 'now')"
        )
        task_id = connection.execute(
            "SELECT id FROM tasks WHERE title = 'Study Java'"
        ).fetchone()["id"]
        topic_id = connection.execute(
            "SELECT id FROM topics WHERE name = 'Java Basics' LIMIT 1"
        ).fetchone()["id"]

    with schema.connect() as connection:
        connection.execute(
            "INSERT INTO task_topic_links (task_id, topic_id, confidence, "
            "state, source, created_at) VALUES (?, ?, 0.9, 'suggested', "
            "'rule', 'now')",
            (task_id, topic_id),
        )

    with pytest.raises(sqlite3.IntegrityError):
        with schema.connect() as connection:
            connection.execute(
                "INSERT INTO task_topic_links (task_id, topic_id, confidence, "
                "state, source, created_at) VALUES (?, ?, 0.8, 'suggested', "
                "'rule', 'now')",
                (task_id, topic_id),
            )
    assert _counts(learning_db)["task_topic_links"] == 1


def test_cascade_deletes_clean_up_learning_rows(learning_db):
    roadmaps.load_seeds()
    with schema.connect() as connection:
        connection.execute(
            "INSERT INTO tasks (title, priority, created_at, updated_at) "
            "VALUES ('Study Java', 'medium', 'now', 'now')"
        )
        task_id = connection.execute(
            "SELECT id FROM tasks WHERE title = 'Study Java'"
        ).fetchone()["id"]
        topic_id = connection.execute(
            "SELECT id FROM topics WHERE name = 'Java Basics' LIMIT 1"
        ).fetchone()["id"]
        connection.execute(
            "INSERT INTO task_topic_links (task_id, topic_id, confidence, "
            "state, source, created_at) VALUES (?, ?, 0.9, 'confirmed', "
            "'rule', 'now')",
            (task_id, topic_id),
        )
        connection.execute(
            "INSERT INTO learning_activities (topic_id, task_id, kind, "
            "created_at) VALUES (?, ?, 'planned', 'now')",
            (topic_id, task_id),
        )
        roadmap_id = connection.execute(
            "SELECT roadmap_id FROM topics WHERE id = ?", (topic_id,)
        ).fetchone()["roadmap_id"]

    with schema.connect() as connection:
        connection.execute("DELETE FROM roadmaps WHERE id = ?", (roadmap_id,))

    with schema.connect() as connection:
        link_count = connection.execute(
            "SELECT COUNT(*) AS n FROM task_topic_links"
        ).fetchone()["n"]
        activity_count = connection.execute(
            "SELECT COUNT(*) AS n FROM learning_activities"
        ).fetchone()["n"]
        topic_count = connection.execute(
            "SELECT COUNT(*) AS n FROM topics WHERE roadmap_id = ?", (roadmap_id,)
        ).fetchone()["n"]
        task_count = connection.execute(
            "SELECT COUNT(*) AS n FROM tasks WHERE id = ?", (task_id,)
        ).fetchone()["n"]

    assert topic_count == 0
    assert link_count == 0
    assert activity_count == 0
    assert task_count == 1  # The task itself survives; only links cascade.


def test_deleting_a_task_cascades_to_links(learning_db):
    roadmaps.load_seeds()
    with schema.connect() as connection:
        connection.execute(
            "INSERT INTO tasks (title, priority, created_at, updated_at) "
            "VALUES ('Study Java', 'medium', 'now', 'now')"
        )
        task_id = connection.execute(
            "SELECT id FROM tasks WHERE title = 'Study Java'"
        ).fetchone()["id"]
        topic_id = connection.execute(
            "SELECT id FROM topics WHERE name = 'Java Basics' LIMIT 1"
        ).fetchone()["id"]
        connection.execute(
            "INSERT INTO task_topic_links (task_id, topic_id, confidence, "
            "state, source, created_at) VALUES (?, ?, 0.9, 'suggested', "
            "'rule', 'now')",
            (task_id, topic_id),
        )

    with schema.connect() as connection:
        connection.execute("DELETE FROM tasks WHERE id = ?", (task_id,))

    assert _counts(learning_db)["task_topic_links"] == 0
