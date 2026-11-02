# S.P.I.D.E.Y Agent Tool Calling Architecture & Catalog

This document details the native function/tool calling architecture introduced in ORBIT007. S.P.I.D.E.Y enables the local LLM (`qwen3.5:4b` running on Ollama) to autonomously manage the application—creating and modifying tasks, habits, reminders, memories, and navigation—while keeping deterministic rules fast and enforcing strict human-in-the-loop safety for destructive actions.

---

## 1. High-Level Architecture

The assistant executes incoming turns following a **Rules First, Tools Second** principle:

```
User Message
     │
     ▼
┌──────────────────────────────────────────────┐
│ 1. Turn-Start Pending Confirmation Check     │
│    (Did the user say "yes" or "no" to        │
│     a pending destructive action?)           │
└──────────────────────┬───────────────────────┘
                       │ (if no pending confirmation)
                       ▼
┌──────────────────────────────────────────────┐
│ 2. Deterministic Short-Circuit Rules         │
│    (_handle_local_command: greeting, direct  │
│     slash-style commands, XP queries)        │
└──────────────────────┬───────────────────────┘
                       │ (if not claimed by rules)
                       ▼
┌──────────────────────────────────────────────┐
│ 3. Native Ollama Tool Loop (run_agent_loop)  │
│    - Multi-turn execution (up to 4 rounds)   │
│    - Validates parameters & types            │
│    - Entity resolution via rapidfuzz         │
│    - Emits UI actions (navigate, refresh)    │
└──────────────────────────────────────────────┘
```

### Key Architectural Tenets
1. **Zero LLM Overhead for Deterministic Rules:** Exact matches (e.g., greetings, quick timers) continue to bypass the LLM entirely with zero latency.
2. **Reuse, Don't Duplicate:** Tool executors never run direct SQL queries. They call the unified helpers and `MemoryStore` methods shared with REST endpoints.
3. **Decoupled via `AgentDeps`:** Dependencies from `app.py` are injected via a typed dataclass (`AgentDeps`), completely preventing circular imports.

---

## 2. Complete Tool Catalog (18 Native Tools)

All tools are declared in `agent/tools.py` using concise descriptions (<40 words) and strict JSON schemas.

### A. Reminders

#### `create_reminder`
- **Description:** Create a new reminder with title and scheduled time.
- **Destructive:** No
- **Parameters:**
  - `title` (`string`, required): What to remember (action or note).
  - `when` (`string`, required): Natural language time (e.g., 'in 10 minutes', 'tomorrow at 9am', 'at 15:30').
  - `recurrence` (`string`, optional): Optional recurrence rule ('daily', 'weekly', 'weekdays').
- **Example Prompts:**
  - *"remind me in 30 minutes to drink water"*
  - *"set a daily reminder at 8am to take vitamins"*

#### `list_reminders`
- **Description:** List pending scheduled reminders.
- **Destructive:** No
- **Parameters:** None
- **Example Prompts:**
  - *"what reminders do I have?"*
  - *"show me my upcoming alarms"*

#### `update_reminder`
- **Description:** Update an existing reminder's due time or title.
- **Destructive:** No
- **Parameters:**
  - `reminder` (`string`, required): Reminder ID or matching words from title.
  - `when` (`string`, optional): New time string.
  - `title` (`string`, optional): New reminder title.
- **Example Prompts:**
  - *"push my gym reminder to 6pm"*
  - *"rename my call mom reminder to call mom on birthday"*

#### `delete_reminder`
- **Description:** Delete/cancel an existing reminder. Requires user confirmation.
- **Destructive:** **Yes**
- **Parameters:**
  - `reminder` (`string`, required): Reminder ID or matching words from title.
- **Example Prompts:**
  - *"delete my reminder to call dentist"*

---

### B. Tasks & Todos

#### `create_task`
- **Description:** Create a new todo task with priority and optional due date.
- **Destructive:** No
- **Parameters:**
  - `title` (`string`, required): Task description.
  - `priority` (`string`, optional, enum: `["low", "medium", "high"]`): Priority level. Default: `medium`.
  - `due` (`string`, optional): Due date or time string.
- **Example Prompts:**
  - *"create a high priority task to review pull request 42"*
  - *"add a new task buy groceries due tomorrow"*

#### `list_tasks`
- **Description:** List todo tasks filtered by status.
- **Destructive:** No
- **Parameters:**
  - `status` (`string`, optional, enum: `["pending", "completed", "all"]`): Status filter. Default: `pending`.
- **Example Prompts:**
  - *"what pending tasks do I need to work on?"*
  - *"show me all my completed tasks"*

#### `update_task`
- **Description:** Update an existing task's title, priority, or completion status.
- **Destructive:** No
- **Parameters:**
  - `task` (`string`, required): Task ID or matching words from title.
  - `title` (`string`, optional): Updated title.
  - `priority` (`string`, optional, enum: `["low", "medium", "high"]`): Updated priority.
  - `completed` (`boolean`, optional): True if task is completed, false otherwise.
- **Example Prompts:**
  - *"mark the task write documentation as completed"*
  - *"change the priority of fix login bug to high"*

#### `delete_task`
- **Description:** Permanently delete a task. Requires user confirmation.
- **Destructive:** **Yes**
- **Parameters:**
  - `task` (`string`, required): Task ID or matching words from title.
- **Example Prompts:**
  - *"delete my task buy groceries"*

---

### C. Habits

#### `create_habit`
- **Description:** Create a new daily or weekly habit to track.
- **Destructive:** No
- **Parameters:**
  - `name` (`string`, required): Habit name.
  - `daily_goal` (`integer`, optional): Target repetitions per day (default 1).
- **Example Prompts:**
  - *"create a new habit morning meditation"*
  - *"track a new habit drink 2L water"*

#### `list_habits`
- **Description:** List all tracked habits with streaks and today's status.
- **Destructive:** No
- **Parameters:** None
- **Example Prompts:**
  - *"show my current habits and streaks"*
  - *"what habits am I tracking right now?"*

#### `complete_habit`
- **Description:** Mark a habit completed or uncompleted for a date.
- **Destructive:** No
- **Parameters:**
  - `habit` (`string`, required): Habit ID or name.
  - `date` (`string`, optional): Date in YYYY-MM-DD format (defaults to today).
  - `completed` (`boolean`, optional): True to mark complete, false to unmark. Default: `true`.
- **Example Prompts:**
  - *"mark my reading habit done for today"*
  - *"I completed my workout habit"*

#### `delete_habit`
- **Description:** Permanently delete a habit and its tracking history. Requires user confirmation.
- **Destructive:** **Yes**
- **Parameters:**
  - `habit` (`string`, required): Habit ID or name.
- **Example Prompts:**
  - *"remove the habit morning stretch"*

---

### D. Memory & Knowledge

#### `save_memory`
- **Description:** Save a persistent fact, preference, or note about the user.
- **Destructive:** No
- **Parameters:**
  - `content` (`string`, required): Fact or preference to remember.
- **Example Prompts:**
  - *"remember that my favorite programming language is Rust"*
  - *"save a note that I am allergic to peanuts"*

#### `search_memory`
- **Description:** Search stored user memories and notes by keyword or query.
- **Destructive:** No
- **Parameters:**
  - `query` (`string`, required): Search keywords or query.
- **Example Prompts:**
  - *"what is my favorite programming language?"*
  - *"tell me about myself"*

#### `delete_memory`
- **Description:** Delete a specific saved memory note. Requires user confirmation.
- **Destructive:** **Yes**
- **Parameters:**
  - `memory` (`string`, required): Memory ID or text to delete.
- **Example Prompts:**
  - *"delete the memory that I like dark roast coffee"*

---

### E. Insights, Web & Navigation

#### `get_overview`
- **Description:** Get a consolidated status report of tasks, habits, reminders, and learning progress.
- **Destructive:** No
- **Parameters:** None
- **Example Prompts:**
  - *"give me a full dashboard overview of my tasks and habits"*
  - *"how am I doing with my habits, tasks and XP progress?"*

#### `web_search`
- **Description:** Search the web for current external information, news, or technical documentation.
- **Destructive:** No
- **Parameters:**
  - `query` (`string`, required): Web search query.
- **Example Prompts:**
  - *"search the web for latest features in Python 3.14"*
  - *"find news about the James Webb Space Telescope discoveries"*

#### `navigate`
- **Description:** Switch the desktop app screen to a specified view. Only active on web/desktop UI.
- **Destructive:** No
- **Parameters:**
  - `page` (`string`, required, enum: `["dashboard", "habits", "todos", "reminders", "learning", "analytics", "settings"]`): Destination screen name.
- **Example Prompts:**
  - *"take me to analytics"*
  - *"open my habits screen"*

---

## 3. Human-in-the-Loop Confirmation Lifecycle

To protect user data, destructive tools **never execute autonomously**:

1. **Interception:** When the model calls `delete_task`, `delete_habit`, `delete_reminder`, or `delete_memory`, the executor pauses and registers a `PendingAction` in `ConfirmationManager`.
2. **User Notification:** The tool returns `{"status": "needs_confirmation", "prompt": "..."}`. The model asks the user for confirmation (e.g., *"Are you sure you want to delete the task 'buy groceries'?"*).
3. **Turn-Start Gate:** On the next user message:
   - **Affirmative (`yes`, `confirm`, `proceed`, etc.):** The pending action executes immediately, the state clears, and the assistant confirms completion.
   - **Negative / Change of Subject (`no`, `cancel`, `nevermind`, or new command):** The pending action is cancelled and cleared.
   - **Timeout (120 seconds):** Actions automatically expire after 2 minutes.

---

## 4. Security & Safety Guards

- **Prompt Injection Defense:** Content returned by `web_search` is treated as untrusted data. If `web_search` was called earlier in the turn, destructive tools are strictly blocked.
- **Channel Authorization:** Desktop UI navigation (`navigate`) is only dispatched for web clients. On Discord, navigation calls are suppressed with an explanation that the feature is desktop-only.
- **Hallucination Prevention:** The system prompt explicitly instructs the assistant to consult `search_memory` for user personal facts and truthfully disclose when no record is found.
- **Fuzzy Matching Thresholds:** Rapidfuzz requires a score $\ge 60$ for entity resolution, avoiding accidental mutations on ambiguous inputs.

---

## 5. UI Actions Contract

The frontend listens for real-time actions dispatched by the agent:
- `orbit:navigate`: Dispatched when the model invokes `navigate(page)`. Triggers React Router to change views without page reloads.
- `orbit:refresh`: Dispatched whenever tasks, habits, or reminders are mutated. Triggers `refresh()` in `AppDataContext.jsx` to update UI counters and charts immediately.

---

## 6. Configuration & Developer Extension

### Environment Variables
- `ORBIT_TOOLS=1`: Enable native function calling (default `1`). Setting to `0` restores pure chat mode.
- `ORBIT_TOOL_MODE=native`: Direct function calling via Ollama's API.

### Adding a New Tool
To add a new tool to S.P.I.D.E.Y:
1. Define the executor function in `agent/tools.py` taking `(args: dict, ctx: ToolContext) -> dict`.
2. Create a `ToolDefinition` with clean descriptions and a JSON schema.
3. Add the definition to `TOOLS` list in `agent/tools.py`.
4. Add unit test coverage in `tests/test_agent_tools.py`.
