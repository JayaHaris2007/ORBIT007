"""Phase 2 tests: the /api/learning contract, progress rules, availability."""

import sqlite3

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from learning import roadmaps, schema
from learning.router import router
from memory import MemoryStore


@pytest.fixture
def client(tmp_path):
    db_file = tmp_path / "test_learning_api.db"
    MemoryStore(db_file)
    schema.init(db_file)
    roadmaps.load_seeds()
    app = FastAPI()
    app.include_router(router)
    yield TestClient(app), db_file


def _task_id(db_file, title="Study Java"):
    with sqlite3.connect(db_file) as connection:
        connection.execute(
            "INSERT INTO tasks (title, priority, created_at, updated_at) "
            "VALUES (?, 'medium', 'now', 'now')",
            (title,),
        )
        return connection.execute(
            "SELECT id FROM tasks WHERE title = ?", (title,)
        ).fetchone()[0]


def _topic_id(db_file, name="Java Basics"):
    with sqlite3.connect(db_file) as connection:
        return connection.execute(
            "SELECT id FROM topics WHERE name = ? LIMIT 1", (name,)
        ).fetchone()[0]


def test_list_roadmaps_has_progress(client):
    http, _ = client
    response = http.get("/api/learning/roadmaps")
    assert response.status_code == 200
    roadmaps_list = response.json()["roadmaps"]
    assert len(roadmaps_list) == 3
    for item in roadmaps_list:
        assert item["topic_count"] > 0
        assert item["completion_percent"] == 0
        assert item["in_progress_count"] == 0
        assert item["next_topic"]  # Every starter roadmap has a next hint.


def test_roadmap_crud(client):
    http, _ = client
    created = http.post(
        "/api/learning/roadmaps",
        json={"name": "Game Development", "category": "personal"},
    ).json()
    roadmap_id = created["id"]

    duplicate = http.post(
        "/api/learning/roadmaps",
        json={"name": "Game Development", "category": "personal"},
    )
    assert duplicate.status_code == 400

    renamed = http.patch(
        f"/api/learning/roadmaps/{roadmap_id}", json={"name": "Game Dev Pro"}
    )
    assert renamed.status_code == 200
    assert renamed.json()["name"] == "Game Dev Pro"

    clash = http.patch(
        f"/api/learning/roadmaps/{roadmap_id}", json={"name": "DBMS"}
    )
    assert clash.status_code == 400

    missing = http.patch("/api/learning/roadmaps/9999", json={"name": "X"})
    assert missing.status_code == 404

    deleted = http.delete(f"/api/learning/roadmaps/{roadmap_id}")
    assert deleted.status_code == 200
    assert deleted.json() == {"status": "deleted"}
    assert http.get(f"/api/learning/roadmaps/{roadmap_id}").status_code == 404
    assert http.delete(f"/api/learning/roadmaps/{roadmap_id}").status_code == 404


def test_roadmap_detail_availability_and_unmet_names(client):
    http, _ = client
    detail = http.get("/api/learning/roadmaps/1").json()
    by_name = {topic["name"]: topic for topic in detail["topics"]}
    assert by_name["Java Basics"]["available"] is True
    assert by_name["Java Basics"]["unmet_prerequisites"] == []
    assert by_name["Trees"]["available"] is False
    assert by_name["Trees"]["unmet_prerequisites"] == ["Recursion"]

    # Completing prerequisites unlocks the dependent topic.
    recursion_id = by_name["Recursion"]["id"]
    http.patch(f"/api/learning/topics/{recursion_id}", json={"status": "completed"})
    detail = http.get("/api/learning/roadmaps/1").json()
    by_name = {topic["name"]: topic for topic in detail["topics"]}
    assert by_name["Trees"]["available"] is True
    assert by_name["Trees"]["unmet_prerequisites"] == []


def test_topic_crud_and_status_activity(client):
    http, _ = client
    created = http.post(
        "/api/learning/topics",
        json={"roadmap_id": 1, "name": "System Design", "priority": 2},
    )
    assert created.status_code == 200
    topic_id = created.json()["id"]

    duplicate = http.post(
        "/api/learning/topics", json={"roadmap_id": 1, "name": "System Design"}
    )
    assert duplicate.status_code == 400

    invalid_status = http.patch(
        f"/api/learning/topics/{topic_id}", json={"status": "not-a-status"}
    )
    assert invalid_status.status_code == 422

    started = http.patch(
        f"/api/learning/topics/{topic_id}", json={"status": "learning"}
    )
    assert started.status_code == 200
    assert started.json()["status"] == "learning"

    activity = http.get("/api/learning/activity").json()["activities"]
    assert any(
        item["topic_id"] == topic_id and item["kind"] == "started"
        for item in activity
    )

    http.delete(f"/api/learning/topics/{topic_id}")
    assert http.get(f"/api/learning/topics/{topic_id}/links").status_code == 404


def test_topic_links_list_only_confirmed(client):
    http, db_file = client
    task_id = _task_id(db_file)
    topic_id = _topic_id(db_file)
    other_topic_id = _topic_id(db_file, "Collections Framework")
    with sqlite3.connect(db_file) as connection:
        connection.execute(
            "INSERT INTO task_topic_links (task_id, topic_id, confidence, "
            "state, source, created_at) VALUES (?, ?, 0.9, 'confirmed', "
            "'rule', 'now')",
            (task_id, topic_id),
        )
        connection.execute(
            "INSERT INTO task_topic_links (task_id, topic_id, confidence, "
            "state, source, created_at) VALUES (?, ?, 0.5, 'suggested', "
            "'embedding', 'now')",
            (task_id, other_topic_id),
        )
    links = http.get(f"/api/learning/topics/{topic_id}/links").json()["links"]
    assert len(links) == 1
    assert links[0]["state"] == "confirmed"
    assert links[0]["task_title"] == "Study Java"


def test_progress_numbers(client):
    http, _ = client
    topics = http.get("/api/learning/roadmaps/1").json()["topics"]
    for topic in topics[:2]:
        http.patch(
            f"/api/learning/topics/{topic['id']}", json={"status": "completed"}
        )
    progress = http.get("/api/learning/progress").json()
    assert progress["topic_count"] == 50
    assert progress["completed_topics"] == 2
    assert progress["completion_percent"] == 4
    java = next(
        item for item in progress["per_roadmap"] if item["roadmap_name"] == "Java & DSA"
    )
    assert java["completed"] == 2
    assert java["total"] == 18
    assert java["percent"] == 11


def test_next_topics_availability_and_reasons(client):
    http, _ = client
    first = http.get("/api/learning/next").json()["next"]
    assert 1 <= len(first) <= 5
    for item in first:
        assert item["reason"]
        assert item["roadmap_name"]
    java_basics = next(item for item in first if item["name"] == "Java Basics")

    http.patch(
        f"/api/learning/topics/{java_basics['topic_id']}",
        json={"status": "completed"},
    )
    second = http.get("/api/learning/next").json()["next"]
    names = {item["name"] for item in second}
    assert "Java Basics" not in names
    assert "Object-Oriented Programming" in names  # Prerequisite just completed.


def test_link_confirm_change_and_reject(client):
    http, db_file = client
    task_id = _task_id(db_file)
    topic_id = _topic_id(db_file, "Java Basics")
    other_topic_id = _topic_id(db_file, "Collections Framework")
    with sqlite3.connect(db_file) as connection:
        connection.execute(
            "INSERT INTO task_topic_links (task_id, topic_id, confidence, "
            "state, source, created_at) VALUES (?, ?, 0.7, 'suggested', "
            "'rule', 'now')",
            (task_id, topic_id),
        )
    suggested = http.get("/api/learning/links?state=suggested").json()["links"]
    assert len(suggested) == 1
    link_id = suggested[0]["id"]

    confirmed = http.patch(
        f"/api/learning/links/{link_id}", json={"state": "confirmed"}
    )
    assert confirmed.status_code == 200
    assert confirmed.json()["state"] == "confirmed"

    activity = http.get("/api/learning/activity").json()["activities"]
    assert any(
        item["topic_id"] == topic_id and item["kind"] == "planned"
        for item in activity
    )

    changed = http.patch(
        f"/api/learning/links/{link_id}",
        json={"state": "confirmed", "topic_id": other_topic_id},
    )
    assert changed.json()["topic_id"] == other_topic_id
    assert changed.json()["source"] == "manual"

    rejected = http.patch(
        f"/api/learning/links/{link_id}", json={"state": "rejected"}
    )
    assert rejected.json()["state"] == "rejected"
    assert http.get("/api/learning/links?state=suggested").json()["links"] == []

    bad_state = http.get("/api/learning/links?state=nope")
    assert bad_state.status_code == 400

    missing = http.patch("/api/learning/links/9999", json={"state": "confirmed"})
    assert missing.status_code == 404


def test_activity_limit_newest_first(client):
    http, _ = client
    topics = http.get("/api/learning/roadmaps/1").json()["topics"]
    for topic in topics[:3]:
        http.patch(
            f"/api/learning/topics/{topic['id']}", json={"status": "learning"}
        )
    activity = http.get("/api/learning/activity?limit=2").json()["activities"]
    assert len(activity) == 2
    assert activity[0]["id"] > activity[1]["id"]


def test_settings_toggle_round_trip(client):
    http, _ = client
    assert http.get("/api/learning/settings").json() == {"nudges_enabled": True}
    off = http.patch("/api/learning/settings", json={"nudges_enabled": False})
    assert off.json() == {"nudges_enabled": False}
    assert http.get("/api/learning/settings").json() == {"nudges_enabled": False}
    invalid = http.patch("/api/learning/settings", json={"nudges_enabled": "yes"})
    assert invalid.status_code == 422


def _fresh_client(db_file):
    """A brand-new TestClient over the same database file.

    This models a browser reload: nothing lives in process memory between the
    two clients, so any value that comes back was genuinely persisted.
    """
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_edits_persist_across_a_fresh_connection(client):
    """Guard the reported bug: frontend edits reset to their initial state."""
    http, db_file = client
    topic_id = _topic_id(db_file, "Java Basics")

    # Edit made through one connection...
    started = http.patch(
        f"/api/learning/topics/{topic_id}", json={"status": "learning"}
    )
    assert started.status_code == 200
    assert started.json()["status"] == "learning"

    created = http.post(
        "/api/learning/roadmaps",
        json={"name": "Operating Systems", "category": "academic"},
    ).json()

    # ...is visible through a completely separate connection (a reload).
    reloaded = _fresh_client(db_file)
    topics = reloaded.get("/api/learning/roadmaps/1").json()["topics"]
    by_name = {topic["name"]: topic for topic in topics}
    assert by_name["Java Basics"]["status"] == "learning"

    names = {r["name"] for r in reloaded.get("/api/learning/roadmaps").json()["roadmaps"]}
    assert "Operating Systems" in names
    assert reloaded.get(f"/api/learning/roadmaps/{created['id']}").status_code == 200
