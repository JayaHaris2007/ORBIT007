import asyncio
import json
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import app
import db
from memory import LOCAL_TIMEZONE, MemoryStore


def test_ws_set_chat_and_fallback(monkeypatch, temp_db):
    """Verify /ws set_chat changes active chat, unknown chat falls back to default, and binary/message use it."""
    # Create two chats in MemoryStore
    chat_a = temp_db.create_chat("Chat A")
    chat_a_id = chat_a["id"]

    sent_messages = []
    fake_ws = AsyncMock()

    async def _fake_send_json(payload):
        sent_messages.append(payload)

    fake_ws.send_json = AsyncMock(side_effect=_fake_send_json)

    # Sequence of packets:
    # 1. set_chat to chat_a_id
    # 2. set_chat to an unknown id (must fall back to 'default')
    # 3. set_chat back to chat_a_id
    # 4. text message creating a reminder
    # 5. disconnect
    packets = [
        {"text": json.dumps({"type": "set_chat", "chat_id": chat_a_id})},
        {"text": json.dumps({"type": "set_chat", "chat_id": "nonexistent-uuid-999"})},
        {"text": json.dumps({"type": "set_chat", "chat_id": chat_a_id})},
        {"text": json.dumps({"message": "remind me to hydrate in 10 minutes"})},
        WebSocketDisconnect(),
    ]

    async def _fake_receive():
        if packets:
            item = packets.pop(0)
            if isinstance(item, Exception):
                raise item
            return item
        raise WebSocketDisconnect()

    fake_ws.receive = AsyncMock(side_effect=_fake_receive)
    monkeypatch.setattr(app, "ws_authorized", AsyncMock(return_value=True))

    async def _run_test():
        await app.websocket_agent_endpoint(fake_ws)

    asyncio.run(_run_test())

    # 1. Initial connected greeting should have been sent
    connected_msg = next((m for m in sent_messages if m.get("type") == "connected"), None)
    assert connected_msg is not None
    assert connected_msg.get("chat_id") == app.DEFAULT_CHAT_ID

    # 2. Check set_chat responses
    chat_set_msgs = [m for m in sent_messages if m.get("type") == "chat_set"]
    assert len(chat_set_msgs) >= 3
    assert chat_set_msgs[0]["chat_id"] == chat_a_id
    assert chat_set_msgs[1]["chat_id"] == app.DEFAULT_CHAT_ID
    assert chat_set_msgs[2]["chat_id"] == chat_a_id

    # 3. The reminder created during turn should belong to chat_a_id
    rems = temp_db.list_reminders()
    hydrate_rem = next((r for r in rems if "hydrate" in r["text"]), None)
    assert hydrate_rem is not None
    assert hydrate_rem["chat_id"] == chat_a_id

    # 4. Unknown chat id should never have been created in conversations
    assert not temp_db.chat_exists("nonexistent-uuid-999")


def test_reminder_alert_includes_chat_id(monkeypatch, temp_db):
    """Verify _push_reminder_alert includes chat_id in broadcast payload."""
    chat = temp_db.create_chat("Focus Chat")
    chat_id = chat["id"]

    row = temp_db.create_reminder("Stand up and stretch", "2026-11-01T15:00:00", chat_id=chat_id)
    rem_id = row["id"]

    fake_ws = AsyncMock()
    app.ws_manager.active.clear()
    app.ws_manager.active.append(fake_ws)

    async def _test_alert():
        app.MAIN_LOOP = asyncio.get_running_loop()
        monkeypatch.setattr(app, "generate_tts_url", AsyncMock(return_value="/tts/test_alert.wav"))
        app._push_reminder_alert(row, "Time to stretch!")
        await asyncio.sleep(0.05)

    asyncio.run(_test_alert())

    fake_ws.send_json.assert_called_once()
    payload = fake_ws.send_json.call_args[0][0]
    assert payload["type"] == "reminder_alert"
    assert payload["reminder_id"] == rem_id
    assert payload["chat_id"] == chat_id
    assert payload["text"] == "Time to stretch!"

    app.ws_manager.active.clear()


def test_patch_reminder_complete_and_undo(monkeypatch, temp_db):
    """Verify PATCH completed: true cancels, completed: false reactivates or validates time."""
    client = TestClient(app.app)
    now = datetime.now(LOCAL_TIMEZONE)

    # 1. 404 for unknown reminder
    res_404 = client.patch("/api/reminders/missing-id-999", json={"completed": True})
    assert res_404.status_code == 404

    # 2. Create a future one-shot reminder
    future_iso = (now + timedelta(hours=3)).isoformat()
    rem_row = temp_db.create_reminder("Read documentation", future_iso)
    rem_id = rem_row["id"]
    assert rem_row["active"] == 1

    # 3. PATCH completed: true -> cancels reminder (active = 0, completed = 1)
    res_done = client.patch(f"/api/reminders/{rem_id}", json={"completed": True})
    assert res_done.status_code == 200
    updated_done = temp_db.get_reminder(rem_id)
    assert updated_done["active"] == 0

    # 4. PATCH completed: false on future reminder -> reactivates successfully
    res_undo_future = client.patch(f"/api/reminders/{rem_id}", json={"completed": False})
    assert res_undo_future.status_code == 200
    data_undo = res_undo_future.json()
    assert data_undo["reminder"]["completed"] == 0
    assert temp_db.get_reminder(rem_id)["active"] == 1
    app.reminder_scheduler.add_job.assert_called()

    # 5. Create a past one-shot reminder (active=0)
    past_iso = (now - timedelta(hours=2)).isoformat()
    past_row = temp_db.create_reminder("Morning yoga", past_iso)
    past_id = past_row["id"]
    temp_db.cancel_reminder(past_id)
    assert temp_db.get_reminder(past_id)["active"] == 0

    # 6. Undo past reminder without new time -> 400 "That time has passed — send a new remind_at"
    res_undo_past_fail = client.patch(f"/api/reminders/{past_id}", json={"completed": False})
    assert res_undo_past_fail.status_code == 400
    detail = res_undo_past_fail.json().get("detail", "")
    assert "That time has passed — send a new remind_at" in detail

    # 7. Undo past reminder with invalid new time -> 400 with parse error
    res_undo_bad_time = client.patch(
        f"/api/reminders/{past_id}",
        json={"completed": False, "remind_at": "unparseable gibberish 999"},
    )
    assert res_undo_bad_time.status_code == 400
    assert "could not understand" in res_undo_bad_time.json().get("detail", "").lower()

    # 8. Undo past reminder with valid new time -> 200 reactivated
    res_undo_new_time = client.patch(
        f"/api/reminders/{past_id}",
        json={"completed": False, "remind_at": "tomorrow 8am"},
    )
    assert res_undo_new_time.status_code == 200
    reactivated = temp_db.get_reminder(past_id)
    assert reactivated["active"] == 1
    assert "08:00" in reactivated["run_at"]

    # 9. Recurring past reminder reactivates without needing a new time
    rec_past_row = temp_db.create_reminder("Daily standup", past_iso, recurrence="every day")
    rec_id = rec_past_row["id"]
    temp_db.cancel_reminder(rec_id)

    res_undo_rec = client.patch(f"/api/reminders/{rec_id}", json={"completed": False})
    assert res_undo_rec.status_code == 200
    rec_updated = temp_db.get_reminder(rec_id)
    assert rec_updated["active"] == 1


def test_get_reminders_include_completed(monkeypatch, temp_db):
    """Verify GET /api/reminders returns active only by default, and both with active first when include_completed=true."""
    client = TestClient(app.app)
    now = datetime.now(LOCAL_TIMEZONE)

    r_active = temp_db.create_reminder("Active task", (now + timedelta(hours=1)).isoformat())
    r_done = temp_db.create_reminder("Done task", (now + timedelta(hours=2)).isoformat())
    temp_db.cancel_reminder(r_done["id"])

    # Default GET -> active only
    res_default = client.get("/api/reminders")
    assert res_default.status_code == 200
    rems_default = res_default.json()["reminders"]
    ids_default = [r["id"] for r in rems_default]
    assert r_active["id"] in ids_default
    assert r_done["id"] not in ids_default

    # GET with include_completed=true -> both, ordered with active first
    res_all = client.get("/api/reminders?include_completed=true")
    assert res_all.status_code == 200
    rems_all = res_all.json()["reminders"]
    ids_all = [r["id"] for r in rems_all]
    assert r_active["id"] in ids_all
    assert r_done["id"] in ids_all

    # Active items must come before completed items
    first_active_idx = next(i for i, r in enumerate(rems_all) if r["id"] == r_active["id"])
    first_done_idx = next(i for i, r in enumerate(rems_all) if r["id"] == r_done["id"])
    assert first_active_idx < first_done_idx
    assert rems_all[first_done_idx]["completed"] == 1
    assert rems_all[first_active_idx]["completed"] == 0

