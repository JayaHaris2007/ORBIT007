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



---

## 10. Active Chat Scoping for Agent Mode & Full PATCH Semantics (Milestone v2.1)

**Generated:** 2026-10-31  
**Test Suite Status:** 236 passed, 0 failed, 16 warnings (63.13s)

### Problem Addressed
1. **Agent Mode Disconnected from Active Chat:** Previously, `useOrbitAgent.js` never transmitted a chat ID, so every turn and reminder fell into `"default"`. The sidebar polls notifications by its active chat ID, causing missed reminder alerts across conversations.
2. **Lossy PATCH Reminders Semantics:** `PATCH /api/reminders/{id}` unconditionally cancelled reminders regardless of payload, ignoring `completed: false` (undo/reactivate), and `GET /api/reminders?include_completed=true` was not implemented.

### Fixes Implemented

#### Fix 1: Active Chat Scoping for Agent Mode
- **Backend (`app.py`):**
  - Added per-connection `current_chat_id` tracking in `/ws` (defaults to `DEFAULT_CHAT_ID = "default"`).
  - Accepted `{"type": "set_chat", "chat_id": "..."}` and validated against `memory_store.chat_exists`; unknown IDs safely fall back to `"default"` without creating phantom conversations.
  - Accepted optional `chat_id` on text messages, validating with fallback.
  - Used `current_chat_id` for binary audio frames and `end_audio` transcripts.
  - Passed `chat_id=current_chat_id` and `session_id=current_chat_id` to `process_agent_turn` so LLM history and commands are scoped per chat.
  - Added `"chat_id": reminder.get("chat_id")` to `reminder_alert` payload in `_push_reminder_alert`.
- **Frontend (`useOrbitAgent.js`, `AssistantSidebar.jsx`, `OrbitAgentView.jsx`):**
  - Accepted `activeChatIdProp` in `useOrbitAgent(activeChatIdProp)` with fallback to `spidey_active_chat_id` and `jarvis_active_chat_id`.
  - Sent `set_chat` immediately after WebSocket opens and whenever the active chat changes.
  - Re-fetched `/api/history?session_id=...` upon switching chats.
  - Filtered in-chat `⏰ Reminder: ...` display on `reminder_alert` so it only appends when `data.chat_id === activeChatId` (while still playing audio and updating reminder badges).
  - Wired `activeChatId={chatId}` from `AssistantSidebar` into `OrbitAgentView`.
  - Handled chat deletion and data reset events cleanly to fall back to `"default"`.

#### Fix 2: Full PATCH /api/reminders/{id} Semantics
- **Backend (`memory.py`, `app.py`):**
  - Added `set_reminder_active(reminder_id, active, run_at=None) -> dict | None` in `MemoryStore`.
  - Updated `ReminderUpdatePayload` to support `completed: bool = True`, `remind_at: str | None`, `run_at: str | None`.
  - Supported `PATCH completed: true` to deactivate reminder and remove scheduler job.
  - Supported `PATCH completed: false` to reactivate reminder:
    - If `remind_at`/`run_at` is provided, parses with `_parse_reminder_time` and updates `run_at`.
    - If no new time is provided and the stored time has passed (and reminder is not recurring), returns HTTP 400: `"That time has passed — send a new remind_at"`.
    - If recurring or future, reactivates and re-schedules without requiring a new time.
    - Returns HTTP 404 for unknown reminder IDs.
  - Implemented `GET /api/reminders?include_completed=true`: returns all reminders sorted with active items first.
- **Frontend (`RemindersMemory.jsx`, `useOrbitAgent.js`):**
  - Added "Show completed" checkbox toggle in `RemindersMemory.jsx` calling `/api/reminders?include_completed=true`.
  - Added "Undo" button for completed items sending `PATCH { completed: false }` and displaying backend 400 error messages if the time has passed.
  - Exported `completeReminder(id, completed, remind_at)` in `useOrbitAgent.js` (and aliased `toggleReminder`).

### Test Results
- Created `tests/test_orbit_reminder_chat.py`:
  - `test_ws_set_chat_and_fallback`: Verified `set_chat` switches active chat, unknown chat safely falls back to `"default"` without creating conversations, and reminders are assigned to the target chat.
  - `test_reminder_alert_includes_chat_id`: Verified broadcast alert contains `chat_id`.
  - `test_patch_reminder_complete_and_undo`: Verified 404 on unknown ID, 200 on complete, 200 on future undo, 400 on past one-shot undo without new time, 400 on unparseable time, 200 on past undo with valid new time, and 200 on recurring reminder undo.
  - `test_get_reminders_include_completed`: Verified default active-only filter, `include_completed=true` return shape, and active-first ordering.
- Full pytest suite: **236 passed, 16 warnings in 63.13s** (100% passing).

---

## 4. Optional Ollama LLM Backend Implementation (Branch: `ollama-backend`)

### Summary of Changes
Added Ollama as an optional LLM backend while keeping the existing PyTorch / HuggingFace Transformers pipeline as the default backend. The system automatically falls back to Transformers if an unsupported backend name is specified.

### Files Modified & Added
- **`app.py`**:
  - Added standard library imports `urllib.request` and `urllib.error`.
  - Added configuration constants read once at startup from environment variables:
    - `LLM_BACKEND` (default: `"transformers"`, accepted: `"transformers"`, `"ollama"`)
    - `OLLAMA_URL` (default: `"http://127.0.0.1:11434"`)
    - `OLLAMA_MODEL` (default: `"qwen3.5:4b"`)
    - `OLLAMA_NUM_CTX` (default: `4096`)
    - `OLLAMA_KEEP_ALIVE` (default: `"5m"`)
    - `OLLAMA_TIMEOUT` (default: `120`)
  - Added startup logger reporting the active LLM backend and configuration.
  - Added helper `_ollama_check()`:
    - Queries `GET {OLLAMA_URL}/api/tags` with a 5-second timeout.
    - Matches model name both with and without `:latest` suffix.
    - Caches verification status in `_ollama_ready`.
    - Raises descriptive `RuntimeError` if Ollama is unreachable ("Ollama is not running at <url>. Start it with `ollama serve`.") or if the model is missing ("Model <name> not found. Run `ollama pull <name>`.").
  - Updated `ensure_model_loaded()`:
    - When `LLM_BACKEND == "ollama"`, delegates to `_ollama_check()` and skips Transformers model loading and CUDA allocation.
    - When `LLM_BACKEND == "transformers"`, existing model loading remains completely untouched.
  - Added helper `_generate_ollama(messages, max_new_tokens)`:
    - POSTs payload to `{OLLAMA_URL}/api/chat` with top-level `think: false`, `stream: false`, `keep_alive: OLLAMA_KEEP_ALIVE`, and options (`temperature: 0`, `repeat_penalty: 1.05`, `num_predict: max_new_tokens`, `num_ctx: OLLAMA_NUM_CTX`).
    - Strips `<think>...</think>` tags using `re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()`.
    - Trims incomplete sentences with `_trim_incomplete(text)` when `done_reason == "length"`.
    - Logs token generation speed (`eval_count` and tokens/sec).
    - Catches network/timeout/JSON errors and raises `RuntimeError` to trigger existing app fallback replies.
  - Updated `generate_response(prompt, max_new_tokens, ...)`:
    - Preserves all prompt formatting and `model_messages` preparation.
    - Calls `_generate_ollama` inside `generation_lock` when `LLM_BACKEND == "ollama"`.
    - Bypasses Transformers tokenizer/model checks when backend is Ollama.
  - Updated `/api/health`:
    - Added `"llm_backend": LLM_BACKEND`.
- **`.env.example`**:
  - Documented commented-out variables for `LLM_BACKEND`, `OLLAMA_URL`, `OLLAMA_MODEL`, `OLLAMA_NUM_CTX`, `OLLAMA_KEEP_ALIVE`, and `OLLAMA_TIMEOUT`.
- **`tests/test_llm_backend.py`**:
  - Comprehensive unit test suite with mocked `urllib.request.urlopen` (no real Ollama or GPU required):
    - `test_default_backend_is_transformers`: verifies default setting and that `_generate_ollama` is never invoked.
    - `test_ollama_request_body_structure`: verifies top-level `think: false`, `stream: false`, and options parameters.
    - `test_ollama_think_tags_stripped`: verifies `<think>...</think>` blocks are removed.
    - `test_ollama_done_reason_length_trims_incomplete`: verifies sentence trimming when truncated.
    - `test_ollama_connection_error_raises_runtime_error`: verifies user-friendly error on connection failure.
    - `test_ollama_missing_model_raises_pull_message`: verifies pull instruction on missing model.
    - `test_ollama_model_matches_with_or_without_latest`: verifies model tag matching flexibility.
    - `test_unknown_backend_fallback`: verifies unrecognized backend falls back to `"transformers"`.
    - `test_health_includes_llm_backend`: verifies `llm_backend` presence in `/api/health`.

### Automatic Ollama Detection & Single-Load Optimization
- **`runtime/loaders.py`**:
  - Added `check_ollama_running(url, model_name)` which queries `{url}/api/tags` with a 2-second timeout.
  - In `load_all_models_eager`:
    - Checks whether Ollama is running with the requested model before loading the local model.
    - If Ollama is active: skips `load_qwen_eager` entirely, registers `"llm"` as `"ollama"`, and activates the Ollama backend. This saves ~46s of startup latency and ~2.5 GB of GPU VRAM while preventing double-loading of Qwen models.
    - If Ollama is not active: automatically falls back to loading "our model" (`load_qwen_eager` from `Models/LLM`) using Transformers.
- **`app.py`**:
  - Updated `ensure_model_loaded()` and `generate_response()` to handle `"ollama"` string object in `REGISTRY` without tuple unpacking errors.

### Test Results
- Unit test suite `tests/test_llm_backend.py`: **11 passed in 3.39s** (includes tests for skipping local Qwen when Ollama is running, and falling back to local Qwen when Ollama is offline).
- Full test suite: **247 passed, 16 warnings in 58.55s** (100% passing).
- **Streaming**: Not included; matches existing non-streaming chat contract for `/ws` and REST endpoints.
- **Dependencies**: Zero external dependencies introduced; implemented using Python standard library `urllib.request`.
- **Backward Compatibility**: 100% backward compatible; PyTorch and HuggingFace Transformers remain default with identical signatures and behavior.

