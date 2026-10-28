"""Phase 9 tests: weekly learning nudges — message building and delivery."""

import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

import app
from learning import nudges, progress, roadmaps, schema
from memory import MemoryStore


@pytest.fixture
def learning_db(tmp_path, monkeypatch):
    """Point the learning module at a seeded temp database for nudge tests."""
    db_file = tmp_path / "test_learning_nudges.db"
    MemoryStore(db_file)  # Creates conversations, tasks, notifications.
    schema.init(db_file)
    roadmaps.load_seeds()
    yield db_file
    # Reset so other app tests keep the "not initialized" default.
    monkeypatch.setattr(schema, "_database_path", None)


def _set_status(db_file, name, status, *, days_ago=0):
    updated = (
        datetime.now(timezone.utc) - timedelta(days=days_ago)
    ).isoformat()
    with sqlite3.connect(db_file) as connection:
        connection.execute(
            "UPDATE topics SET status = ?, updated_at = ? WHERE name = ?",
            (status, updated, name),
        )


def test_nudges_enabled_by_default(learning_db):
    assert progress.nudges_enabled() is True


def test_disabled_nudges_build_nothing(learning_db):
    progress.set_setting(progress.NUDGES_KEY, "0")
    assert nudges.build_weekly_nudge() is None


def test_empty_database_builds_nothing(tmp_path, monkeypatch):
    db_file = tmp_path / "empty.db"
    MemoryStore(db_file)
    schema.init(db_file)
    assert nudges.build_weekly_nudge() is None
    monkeypatch.setattr(schema, "_database_path", None)


def test_fresh_progress_mentions_next_topic(learning_db):
    message = nudges.build_weekly_nudge()
    assert message is not None
    assert "0 of 50" in message
    assert "Java Basics" in message


def test_idle_topic_gets_pick_back_up(learning_db):
    _set_status(learning_db, "Recursion", "completed")
    _set_status(learning_db, "Trees", "learning", days_ago=20)
    message = nudges.build_weekly_nudge()
    assert message is not None
    assert "Trees" in message
    assert "pick it back up" in message
    assert "20 days" in message


def test_full_completion_celebrates_without_next(learning_db):
    with sqlite3.connect(learning_db) as connection:
        connection.execute("UPDATE topics SET status = 'completed'")
    message = nudges.build_weekly_nudge()
    assert message is not None
    assert "100% complete" in message
    assert "Up next" not in message


def _has_recent(store, chat_id, now):
    return store.has_recent_weekly_nudge(chat_id, now)


def test_deliver_sends_one_nudge_per_local_chat(learning_db, temp_db):
    chat_id = temp_db.create_chat(title="local conversation")["id"]
    sent = nudges.deliver_weekly_nudge(
        [chat_id], temp_db.add_notification,
        recently_sent=lambda chat, now: _has_recent(temp_db, chat, now),
    )
    assert sent == 1
    notifications = temp_db.list_notifications(chat_id)
    assert len(notifications) == 1
    assert notifications[0]["reminder_id"].startswith(
        nudges.NUDGE_NOTIFICATION_KEY
    )


def test_deliver_does_not_repeat_within_the_week(learning_db, temp_db):
    chat_id = temp_db.create_chat(title="local conversation")["id"]
    recent = lambda chat, now: _has_recent(temp_db, chat, now)
    assert nudges.deliver_weekly_nudge(
        [chat_id], temp_db.add_notification, recently_sent=recent
    ) == 1
    assert nudges.deliver_weekly_nudge(
        [chat_id], temp_db.add_notification, recently_sent=recent
    ) == 0


def test_app_schedules_weekly_nudge(learning_db):
    app._schedule_weekly_learning_nudge()
    app.reminder_scheduler.add_job.assert_called()
    (_, trigger), kwargs = app.reminder_scheduler.add_job.call_args
    assert kwargs["id"] == app.WEEKLY_LEARNING_NUDGE_JOB
    assert trigger.timezone.zone == "Asia/Kolkata"


def test_weekly_job_skips_when_learning_not_initialized(monkeypatch):
    monkeypatch.setattr(schema, "_database_path", None)
    app.reminder_scheduler.add_job.reset_mock()
    app._schedule_weekly_learning_nudge()
    app.reminder_scheduler.add_job.assert_not_called()


def test_learning_digest_appears_for_study_questions(learning_db):
    context = app._productivity_context("What should I study next?")
    assert "Learning progress" in context
    assert "0 of 50" in context
    assert "Java Basics" in context


def test_learning_digest_skipped_for_unrelated_questions(learning_db):
    context = app._productivity_context("Tell me about space travel")
    assert context == ""


def test_learning_digest_empty_when_not_initialized(monkeypatch):
    monkeypatch.setattr(schema, "_database_path", None)
    context = app._productivity_context("What should I study next?")
    assert "Learning progress" not in context
