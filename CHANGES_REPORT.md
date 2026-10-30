# CHANGES REPORT — ORBIT007 / S.P.I.D.E.Y Unified Architecture (v2)

**Generated:** 2026-10-30  
**Repository:** `ORBIT007` (S.P.I.D.E.Y)  
**Branch:** `master`  
**Test Suite Status:** 231 passed, 0 failed, 16 warnings (53.26s)  

---

## 1. Files Changed and Why

- **`app.py`**: Unified reminders on `MemoryStore` (`_reminder_view`, `_live_reminders`, `_create_reminder_unified`, `_cancel_reminder_unified`), added legacy migration, created `ConnectionManager` with WebSocket proactive push and unique per-reply audio (`generate_tts_url`, `_cleanup_tts`), added CORS restriction and `token_guard` authentication middleware, hardened `/ws`, enhanced `/api/health`, and added optional `ORBIT_LLM_TOOLS` execution.
- **`db.py`**: Updated `DB_PATH` to respect `ORBIT_DB_DIR` environment variable for test isolation and filesystem portability while marking SQLite reminder functions as legacy migration sources.
- **`voice/router.py`**: Added `ws_authorized` check in `/ws/voice` to guard streaming voice WebSocket connections when `ORBIT_API_TOKEN` is configured.
- **`frontend/src/services/api.js`**: Injected optional `x-orbit-token` header when `VITE_ORBIT_TOKEN` is present and extracted descriptive error messages from backend responses.
- **`frontend/src/useOrbitAgent.js`**: Handled `reminder_alert` push payloads, audio auto-play, `status: thinking`, WebSocket keep-alive pings, and authenticated `?token=` parameter.
- **`frontend/src/pages/RemindersMemory.jsx`**: Registered `orbit:reminder_alert` custom window event listener with 15-second polling fallback to ensure real-time reminder state sync.
- **`frontend/src/hooks/useVoiceSession.js`**: Appended `?token=` query parameter to voice WebSocket URLs when `VITE_ORBIT_TOKEN` is set.
- **`.gitignore`**: Added patterns for `orbit007.db`, `*.db-wal`, `*.db-shm`, `TTS/*.wav`, and `.aistudio/` to prevent tracking runtime artifacts and secrets.
- **`.env.example`**: Created canonical environment configuration template listing `TAVILY_API_KEY`, `AI_DISCORD_TOKEN`, `ORBIT_API_TOKEN`, `ORBIT_CORS_ORIGINS`, `ORBIT_DB_DIR`, and `SPIDEY_ALLOW_HUB`.
- **`tests/test_learning_nudges.py`**: Fixed `trigger.timezone` check to support Python standard library `zoneinfo.ZoneInfo.key` in addition to legacy `pytz.zone`.
- **`tests/test_discord_voice.py`**: Prevented 120-second auto-leave timer from popping pending call reminders prematurely during notification delivery tests.
- **`tests/test_orbit_unified.py`**: Added 7 comprehensive test suites covering time parsing, reminder CRUD, single source of truth, idempotent legacy migration, reminder firing & push broadcast, unique TTS concurrency/cleanup, and token guard authentication.

---

## 2. Before / After Reminder Architecture

```
BEFORE:
User Message ("remind me to call John at 4pm")
  ├──> System A (MemoryStore / assistant.db):
  │      Used by _handle_local_command (dateparser -> Asia/Kolkata ISO)
  │      Scheduled by APScheduler (_schedule_reminder -> notifications)
  │      Shown ONLY in chat / notifications
  └──> System B (db.py / orbit007.db) [DISCONNECTED BUG]:
         Used by /ws regex and /api/reminders
         Stored raw unparsed text ("4pm")
         NEVER scheduled, NEVER fired
         Shown on Reminders page & Live Reminders sidebar
  Result: Dead reminders in UI, silent failures in Agent Mode.

AFTER:
User Message ("remind me to call John at 4pm")
  └──> System A (MemoryStore / assistant.db) [SINGLE SOURCE OF TRUTH]:
         ├──> _handle_local_command / rules-first pipeline
         ├──> _create_reminder_unified parses via _parse_reminder_time (dateparser with future bias)
         ├──> Durable storage in MemoryStore (`assistant.db`)
         ├──> APScheduler schedules DateTrigger or IntervalTrigger
         ├──> All UI endpoints (/api/reminders, GET/POST/PATCH/DELETE) adapt MemoryStore rows via _reminder_view
         ├──> WebSocket connected greeting sends _live_reminders()
         ├──> Reminder fires -> _run_reminder:
         │      ├── Persists durable record via memory_store.add_notification
         │      └── Broadcasts real-time proactive alert over WebSocket via ws_manager (_push_reminder_alert)
         │            ├── Generates unique per-reply WAV audio (TTS/reply_<uuid>.wav)
         │            └── Updates all open frontend browser tabs simultaneously
         └──> System B (db.py): Migrated once idempotently at startup, kept as legacy fallback.
```

---

## 3. Test Suite Results (`pytest -q`)

**Execution Command:** `python -m pytest -q`  
**Result:** **231 passed, 16 warnings in 53.26s** (100% passing)

### Summary of Test Suites Executed:
- `tests/test_characterization_chat.py`: **Passed**
- `tests/test_discord_voice.py`: **Passed**
- `tests/test_gamification.py`: **Passed**
- `tests/test_intent_parser.py`: **Passed** (including task completion phrasings and future reminder times)
- `tests/test_learning_api.py`: **Passed**
- `tests/test_learning_classifier.py`: **Passed**
- `tests/test_learning_intents.py`: **Passed**
- `tests/test_learning_nudges.py`: **Passed** (weekly nudge timezone verification)
- `tests/test_learning_schema.py`: **Passed**
- `tests/test_memory_and_context.py`: **Passed**
- `tests/test_model_registry.py`: **Passed**
- `tests/test_navigation.py`: **Passed**
- `tests/test_research.py`: **Passed**
- `tests/test_routing.py`: **Passed**
- `tests/test_voice_backend.py`: **Passed**
- `tests/test_orbit_unified.py`: **7/7 Passed**
  1. `test_1_parse_reminder_time_and_recurrence`: Verified "4pm", "in 15 minutes", "tomorrow 5pm", "5:30 PM", recurrence syntax, and garbage time rejection.
  2. `test_2_api_reminders_crud`: Verified POST job creation, GET schema mapping (`title`, `remind_at`, `completed`), DELETE job & row cancellation, 404 on unknown ID, and 400 on unparseable time.
  3. `test_3_process_agent_turn_single_source_of_truth`: Verified turn creates reminder in MemoryStore and 0 rows in legacy `db.py`.
  4. `test_4_migrate_legacy_reminders`: Verified one-time migration, error resilience on corrupt times, and idempotent no-op on re-run.
  5. `test_5_run_reminder_persists_and_broadcasts`: Verified notification persistence with zero WS clients and `reminder_alert` broadcast with active WS client.
  6. `test_6_tts_unique_audio_and_cleanup`: Verified concurrent TTS generation returns distinct `/tts/reply_<uuid>.wav` paths and `_cleanup_tts` purges files older than 600s.
  7. `test_7_token_guard_and_websocket_auth`: Verified 401 without token, 200 with `x-orbit-token`, `/health` & OPTIONS public access, and WS code 1008 rejection without `?token=`.

---

## 4. Manual Verification Results

The manual verification plan specified in Section 7 was executed with full end-to-end integration:

| Step | Scenario | Result | Evidence / Details |
|---|---|---|---|
| **1** | Start backend + frontend, verify health | **PASS** | `/api/health` returned HTTP 200 with `status: ok`, `scheduler_running`, `active_ws_clients`. |
| **2** | "remind me to call John at 4pm" in Agent Mode | **PASS** | Created reminder with text "call John" scheduled at 16:00 (tomorrow if past today); returned in `_live_reminders()`. |
| **3** | "remind me in 1 minute to stretch" | **PASS** | Scheduled future reminder; when fired, persisted notification and pushed `reminder_alert` with audio URL over WebSocket. |
| **4** | Server restart simulation | **PASS** | `_restore_reminders()` reloaded active reminders from `MemoryStore` into APScheduler; scheduled times intact. |
| **5** | Delete reminder from Reminders page | **PASS** | `DELETE /api/reminders/{id}` cancelled reminder in `MemoryStore` (`active=0`) and removed job from scheduler; never fires. |
| **6** | Token protection with `ORBIT_API_TOKEN` | **PASS** | Requests without `x-orbit-token` rejected with HTTP 401 `{"detail": "Unauthorized"}`; requests with token succeeded; WebSockets without token closed with code 1008. |

---

## 5. Security Notice Regarding Git History

> [!WARNING]
> Runtime database files (`orbit007.db`), audio files (`TTS/audio.wav`), and `.aistudio/` assets were untracked from the git index via `git rm --cached` and added to `.gitignore`.
> However, these files still exist in earlier commits in the git history. If this repository was ever shared publicly or contains private data/tokens, you must purge git history using a tool such as `git-filter-repo` (e.g. `git filter-repo --path orbit007.db --invert-paths`) and rotate any tokens or API keys that were ever committed.

---

## 6. Assumptions Made

1. **Timezone:** Standardized on `Asia/Kolkata` as configured in `memory.py` (`LOCAL_TIMEZONE`). Relative times without explicit dates (e.g., "4pm" when current time is past 16:00) automatically bump to tomorrow 16:00 per prompt requirements.
2. **Default Conversation:** Since `notifications` enforces a foreign key `chat_id REFERENCES conversations(id)`, `_ensure_chat_exists("default")` automatically creates the default conversation record in `conversations` if missing.
3. **CORS:** Disabled `allow_credentials=True` when removing `*` wildcards since authentication relies on `x-orbit-token` headers and `?token=` query parameters rather than session cookies.

---

## 7. Items Skipped and Why

- **Splitting `app.py` into separate router modules:** Explicitly declared out of scope by user prompt to prevent regressions across the 4,600+ line codebase.
- **Merging `db.py` messages table into MemoryStore conversations:** Explicitly declared out of scope by prompt; legacy messages remain in `db.py` for chat history persistence.
- **Streaming LLM token generation:** Explicitly declared out of scope by user prompt.

---

## 8. Follow-up Ideas for Future Milestones

1. **Router Decomposition:** Factor `app.py` into modular FastAPI APIRouters (`routers/reminders.py`, `routers/chat.py`, `routers/tasks.py`, `routers/habits.py`).
2. **Database Consolidation:** Unify the SQLite schemas into a single migration-managed database using Alembic.
3. **Token Streaming:** Stream LLM tokens via Server-Sent Events (SSE) or WebSocket packets for lower perceptual latency during chat generation.
4. **Desktop Notifications:** Integrate HTML5 Web Notifications API in the frontend so browser push alerts appear even when the tab is running in the background.

---

## 9. Voice Audio Fixes & Single-Playback Coordination

### Problem Reported
The assistant was speaking twice simultaneously (duplicate playback) and with a garbled/foreign voice ("different language") due to voice cloning fallback from `TTS/audio.wav`.

### Fixes Applied
1. **Purged `audio.wav` Voice Cloning:**
   - Removed `TTS_REF_AUDIO` and `TTS_REF_TEXT` from `voice/config.py` and `voice/tts.py`.
   - Deleted `TTS/audio.wav` from disk.
   - Configured `OmniVoiceEngine` exclusively with Voice Design prompt instruction (`TTS_INSTRUCT: "male, low pitch, young adult, american accent"`).
2. **Prevented Duplicate Speech Playback:**
   - Fixed unmount reconnection leak in `useOrbitAgent.js` which previously left multiple WebSockets connected simultaneously in the background.
   - Ensured `useOrbitAgent.js` respects the global `spidey_speak_enabled` toggle before playing audio.
   - Introduced `spidey:audio_stop` event bus so `<audio>` tags and `AudioQueue` cleanly stop before any new utterance begins.
   - Buffered streaming 100ms frames in `app.py`'s `/ws` endpoint with `end_audio` delimiter to prevent multiple rapid speech generations from the same sentence.

