import asyncio
import os
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import app
import db
from memory import LOCAL_TIMEZONE, MemoryStore


def test_1_parse_reminder_time_and_recurrence():
    """1. _parse_reminder_time variations, recurrence validation, and garbage input."""
    now = datetime.now(LOCAL_TIMEZONE)

    # 4pm should be a future datetime
    parsed_4pm = app._parse_reminder_time("4pm")
    assert parsed_4pm is not None
    assert parsed_4pm.hour == 16
    assert parsed_4pm.minute == 0
    assert parsed_4pm > now

    # in 15 minutes
    parsed_15m = app._parse_reminder_time("in 15 minutes")
    assert parsed_15m is not None
    assert (parsed_15m - now).total_seconds() >= 14 * 60

    # tomorrow 5pm
    parsed_tmrw = app._parse_reminder_time("tomorrow 5pm")
    assert parsed_tmrw is not None
    assert parsed_tmrw.hour == 17
    assert parsed_tmrw.minute == 0
    assert parsed_tmrw.date() > now.date()

    # 5:30 PM
    parsed_530 = app._parse_reminder_time("5:30 PM")
    assert parsed_530 is not None
    assert parsed_530.hour == 17
    assert parsed_530.minute == 30
    assert parsed_530 > now

    # Recurrence validation
    app._validate_recurrence("every day")
    app._validate_recurrence("every 2 hours")
    app._validate_recurrence("every 15 minutes")
    app._validate_recurrence(None)
    app._validate_recurrence("")

    with pytest.raises(ValueError, match="Recurrence must look like"):
        app._validate_recurrence("every millennium")

    # Garbage input must return None, never silently become 'now + 5 minutes'
    garbage = app._parse_reminder_time("completely random unparseable gibberish text 999")
    assert garbage is None


def test_2_api_reminders_crud(monkeypatch, temp_db):
    """2. POST creates a job; GET returns title/remind_at/completed; DELETE removes."""
    client = TestClient(app.app)

    # Unknown ID -> 404
    res_del_unknown = client.delete("/api/reminders/unknown-uuid-12345")
    assert res_del_unknown.status_code == 404

    # Bad time -> 400 with readable message
    res_bad = client.post(
        "/api/reminders",
        json={"title": "Test", "remind_at": "unparseable garbage xyz"},
    )
    assert res_bad.status_code == 400
    detail = (res_bad.json().get("detail") or res_bad.json().get("error") or "").lower()
    assert "could not understand" in detail

    # Create reminder
    res_post = client.post(
        "/api/reminders",
        json={"title": "Team sync", "remind_at": "tomorrow 11am"},
    )
    assert res_post.status_code == 200
    data = res_post.json()
    assert "reminder" in data
    rem = data["reminder"]
    assert rem["title"] == "Team sync"
    assert rem["completed"] == 0
    rem_id = rem["id"]

    # Verify scheduler was called
    app.reminder_scheduler.add_job.assert_called()

    # GET reminders
    res_get = client.get("/api/reminders")
    assert res_get.status_code == 200
    rems = res_get.json()["reminders"]
    matching = [r for r in rems if r["id"] == rem_id]
    assert len(matching) == 1
    assert matching[0]["title"] == "Team sync"
    assert matching[0]["completed"] == 0
    assert "remind_at" in matching[0]

    # DELETE reminder
    res_del = client.delete(f"/api/reminders/{rem_id}")
    assert res_del.status_code == 200
    remaining = res_del.json()["reminders"]
    assert not any(r["id"] == rem_id for r in remaining)


def test_3_process_agent_turn_single_source_of_truth(monkeypatch, temp_db, tmp_path):
    """3. process_agent_turn creates reminder in MemoryStore and nothing in legacy db."""
    test_legacy_db = tmp_path / "legacy_test.db"
    monkeypatch.setattr(db, "DB_PATH", test_legacy_db)
    db.init_db()

    async def _run_turn():
        return await app.process_agent_turn("remind me to call John at 4pm", session_id="test-session")

    res = asyncio.run(_run_turn())
    assert res is not None

    # Verify MemoryStore has the reminder
    mem_reminders = temp_db.list_reminders()
    matching = [r for r in mem_reminders if "call John" in r["text"]]
    assert len(matching) == 1
    assert matching[0]["active"] == 1

    # Verify legacy DB has 0 reminders
    legacy_reminders = db.list_reminders(include_completed=True)
    assert len(legacy_reminders) == 0


def test_4_migrate_legacy_reminders(monkeypatch, temp_db, tmp_path):
    """4. _migrate_legacy_reminders moves valid rows once; unparseable rows skipped."""
    test_legacy_db = tmp_path / "legacy_migration_test.db"
    monkeypatch.setattr(db, "DB_PATH", test_legacy_db)
    db.init_db()

    # Insert one valid row and one invalid row in legacy db
    r1 = db.create_reminder("Call doctor", "tomorrow 3pm")
    r2 = db.create_reminder("Broken reminder", "non-existent gibberish time")

    moved = app._migrate_legacy_reminders()
    assert moved == 1

    # MemoryStore should have "Call doctor"
    active_mems = [r for r in temp_db.list_reminders() if "Call doctor" in r["text"]]
    assert len(active_mems) == 1

    # Second migration run should be idempotent (0 moved)
    moved_second = app._migrate_legacy_reminders()
    assert moved_second == 0


def test_5_run_reminder_persists_and_broadcasts(monkeypatch, temp_db):
    """5. _run_reminder persists notification and broadcasts reminder_alert."""
    chat = temp_db.create_chat("Test Room")
    chat_id = chat["id"]

    row = temp_db.create_reminder("Stretch legs", "2026-11-01T10:00:00", chat_id=chat_id)
    rem_id = row["id"]

    # Case A: Firing with zero WS clients
    app.ws_manager.active.clear()
    app._run_reminder(rem_id)

    # Notification must be persisted in MemoryStore
    notifs = temp_db.list_notifications(chat_id)
    assert len(notifs) >= 1
    assert any("Stretch legs" in n["content"] for n in notifs)

    # Case B: Broadcasting with one WS client
    fake_ws = AsyncMock()
    app.ws_manager.active.append(fake_ws)

    async def _test_broadcast():
        app.MAIN_LOOP = asyncio.get_running_loop()
        monkeypatch.setattr(app, "generate_tts_url", AsyncMock(return_value="/tts/mock_alert.wav"))
        app._push_reminder_alert(row, "Stretch legs now!")
        await asyncio.sleep(0.05)

    asyncio.run(_test_broadcast())

    fake_ws.send_json.assert_called_once()
    payload = fake_ws.send_json.call_args[0][0]
    assert payload["type"] == "reminder_alert"
    assert payload["reminder_id"] == rem_id
    assert payload["audio_url"] == "/tts/mock_alert.wav"

    app.ws_manager.active.clear()


def test_6_tts_unique_audio_and_cleanup(tmp_path, monkeypatch):
    """6. Two concurrent generate_tts_url calls return different URLs; _cleanup_tts deletes old files."""
    test_tts_dir = tmp_path / "TTS"
    test_tts_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(app, "TTS_DIR", test_tts_dir)

    def fake_tts(text, path_str):
        Path(path_str).write_bytes(b"RIFFdummywav")

    monkeypatch.setattr(app, "run_existing_tts", fake_tts)

    async def _test_tts():
        return await asyncio.gather(
            app.generate_tts_url("First response"),
            app.generate_tts_url("Second response"),
        )

    url1, url2 = asyncio.run(_test_tts())

    assert url1 is not None and url2 is not None
    assert url1 != url2
    assert url1.startswith("/tts/reply_")
    assert url2.startswith("/tts/reply_")

    # Test _cleanup_tts
    old_file = test_tts_dir / "reply_old12345678.wav"
    old_file.write_bytes(b"old")
    old_mtime = time.time() - 1000
    os.utime(old_file, (old_mtime, old_mtime))

    new_file = test_tts_dir / "reply_new12345678.wav"
    new_file.write_bytes(b"new")

    app._cleanup_tts(max_age_s=600)

    assert not old_file.exists(), "Old TTS file was not cleaned up"
    assert new_file.exists(), "New TTS file should be preserved"


def test_7_token_guard_and_websocket_auth(monkeypatch):
    """7. Token guard blocks unauthorized requests; /health & OPTIONS open; WS requires ?token=."""
    monkeypatch.setattr(app, "API_TOKEN", "secure-token-12345")
    client = TestClient(app.app)

    # Protected route without token -> 401
    res_no_token = client.get("/api/reminders")
    assert res_no_token.status_code == 401

    # Protected route with wrong token -> 401
    res_wrong = client.get("/api/reminders", headers={"x-orbit-token": "wrong"})
    assert res_wrong.status_code == 401

    # Protected route with valid token -> 200
    res_ok = client.get("/api/reminders", headers={"x-orbit-token": "secure-token-12345"})
    assert res_ok.status_code == 200

    # Public route /health stays open without token
    res_health = client.get("/api/health")
    assert res_health.status_code == 200

    # OPTIONS preflight request with Origin stays open (not 401)
    res_opt = client.options(
        "/api/reminders",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert res_opt.status_code == 200

    # WebSocket without token is closed with code 1008
    with pytest.raises(WebSocketDisconnect) as exc_info:
        with client.websocket_connect("/ws"):
            pass
    assert exc_info.value.code == 1008

    # WebSocket with valid token connects
    with client.websocket_connect("/ws?token=secure-token-12345") as ws:
        msg = ws.receive_json()
        assert msg["type"] == "connected"
