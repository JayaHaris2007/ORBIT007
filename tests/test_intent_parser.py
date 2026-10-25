"""Characterization tests for the deterministic intent parser in app.py.

Records the exact behavior and responses of S.P.I.D.E.Y's local intent handling today,
including task actions, habit actions, reminder actions, greetings, near-misses,
and queries that fall through to the LLM.
"""

import pytest
import app


@pytest.fixture
def chat_id(temp_db):
    chat = temp_db.create_chat(title="Test Chat")
    return chat["id"]


# -------------------------------------------------------------------------
# 1. TASK CREATION PHRASINGS
# -------------------------------------------------------------------------

@pytest.mark.parametrize(
    "prompt,expected_title,expected_priority",
    [
        ("add task buy milk", "buy milk", "medium"),
        ("create a new task Finish the Python project priority high", "Finish the Python project", "high"),
        ("new todo study for exams with low priority", "study for exams", "low"),
        ("add a task called review pull requests", "review pull requests", "medium"),
        ("create task write documentation with medium priority", "write documentation", "medium"),
        ("Hey Spidey, please add task prepare slide deck", "prepare slide deck", "medium"),
        ("new task: refactor database queries with priority high", "refactor database queries", "high"),
        ("add to-do clean desk", "clean desk", "medium"),
    ],
)
def test_task_create_intents(chat_id, temp_db, prompt, expected_title, expected_priority):
    result = app._handle_local_command(prompt, chat_id)
    assert result is not None, f"Expected deterministic match for: {prompt}"
    reply, provider = result
    assert provider == "local-task-manager"
    assert expected_title in reply

    tasks = temp_db.list_tasks(include_completed=True)
    created = next((t for t in tasks if t["title"] == expected_title), None)
    assert created is not None
    assert created["priority"] == expected_priority
    assert created["completed"] == 0


# -------------------------------------------------------------------------
# 2. TASK LISTING PHRASINGS
# -------------------------------------------------------------------------

@pytest.mark.parametrize(
    "prompt",
    [
        "show my pending tasks",
        "list my tasks",
        "what's pending today?",
        "what are the pending to do?",
        "do I have any open tasks",
        "check tasks",
    ],
)
def test_task_list_pending_intents(chat_id, temp_db, prompt):
    temp_db.create_task("Sample Pending Task", priority="high")
    result = app._handle_local_command(prompt, chat_id)
    assert result is not None, f"Expected deterministic match for: {prompt}"
    reply, provider = result
    assert provider == "local-task-manager"
    assert "Sample Pending Task" in reply


def test_task_list_completed_intents(chat_id, temp_db):
    task = temp_db.create_task("Sample Done Task", priority="low")
    temp_db.update_task(task["id"], completed=True)

    result = app._handle_local_command("list completed tasks", chat_id)
    assert result is not None
    reply, provider = result
    assert provider == "local-task-manager"
    assert "Sample Done Task" in reply


# -------------------------------------------------------------------------
# 3. TASK UPDATE, RENAME, & REOPEN PHRASINGS
# -------------------------------------------------------------------------

def test_task_update_priority_intent(chat_id, temp_db):
    task = temp_db.create_task("Fix UI bug", priority="low")
    result = app._handle_local_command(f"set priority of task {task['id']} to high", chat_id)
    assert result is not None
    reply, provider = result
    assert provider == "local-task-manager"
    assert "high" in reply.lower()
    updated = temp_db.list_tasks(include_completed=True)[0]
    assert updated["priority"] == "high"


def test_task_rename_intent(chat_id, temp_db):
    task = temp_db.create_task("Draft paper", priority="medium")
    result = app._handle_local_command(f"rename task {task['id']} to Finalize paper", chat_id)
    assert result is not None
    reply, provider = result
    assert provider == "local-task-manager"
    assert "Finalize paper" in reply
    updated = temp_db.list_tasks(include_completed=True)[0]
    assert updated["title"] == "Finalize paper"


def test_task_reopen_intent(chat_id, temp_db):
    task = temp_db.create_task("Clean garage", priority="low")
    temp_db.update_task(task["id"], completed=True)

    result = app._handle_local_command(f"reopen task {task['id']}", chat_id)
    assert result is not None
    reply, provider = result
    assert provider == "local-task-manager"
    updated = temp_db.list_tasks(include_completed=True)[0]
    assert updated["completed"] == 0

    result2 = app._handle_local_command(f"mark task {task['id']} as incomplete", chat_id)
    assert result2 is not None
    assert "still open" in result2[0].lower() or "reopen" in result2[0].lower()


# -------------------------------------------------------------------------
# 4. TASK COMPLETION & DELETION PHRASINGS
# -------------------------------------------------------------------------

@pytest.mark.parametrize(
    "phrase_template",
    [
        "complete task {id}",
        "finish task {id}",
        "mark task {id} as completed",
        "done with task {id}",
        "check off task {id}",
    ],
)
def test_task_completion_phrasings(chat_id, temp_db, phrase_template):
    task = temp_db.create_task("Exercise routine", priority="medium")
    prompt = phrase_template.format(id=task["id"])
    result = app._handle_local_command(prompt, chat_id)
    assert result is not None
    reply, provider = result
    assert provider == "local-task-manager"
    updated = temp_db.list_tasks(include_completed=True)[0]
    assert updated["completed"] == 1


@pytest.mark.parametrize(
    "phrase_template",
    [
        "delete task {id}",
        "remove task {id}",
        "cancel task {id}",
        "forget task {id}",
    ],
)
def test_task_delete_phrasings(chat_id, temp_db, phrase_template):
    task = temp_db.create_task("Temporary task", priority="low")
    prompt = phrase_template.format(id=task["id"])
    result = app._handle_local_command(prompt, chat_id)
    assert result is not None
    reply, provider = result
    assert provider == "local-task-manager"
    assert len(temp_db.list_tasks(include_completed=True)) == 0


# -------------------------------------------------------------------------
# 5. HABIT CREATION, LISTING, LOGGING, & MUTATION PHRASINGS
# -------------------------------------------------------------------------

@pytest.mark.parametrize(
    "prompt,expected_name",
    [
        ("create habit Read books", "Read books"),
        ("track new habit Morning Exercise", "Morning Exercise"),
        ("hey spidey add a daily habit Drink 2L water", "Drink 2L water"),
        ("start habit Evening Meditation", "Evening Meditation"),
    ],
)
def test_habit_create_intents(chat_id, temp_db, prompt, expected_name):
    result = app._handle_local_command(prompt, chat_id)
    assert result is not None
    reply, provider = result
    assert provider == "local-habit-manager"
    assert expected_name in reply
    habits = temp_db.list_habits()
    assert any(h["name"] == expected_name for h in habits)


@pytest.mark.parametrize(
    "prompt",
    [
        "show my habits",
        "check my habits to complete",
        "what habits do I need to complete today?",
        "check habits",
    ],
)
def test_habit_list_intents(chat_id, temp_db, prompt):
    temp_db.create_habit("Read 15 mins")
    result = app._handle_local_command(prompt, chat_id)
    assert result is not None
    reply, provider = result
    assert provider == "local-habit-manager"
    assert "Read 15 mins" in reply


@pytest.mark.parametrize(
    "phrase_template",
    [
        "complete habit {name}",
        "done with habit {name}",
        "finish habit {name}",
        "check off habit {name}",
    ],
)
def test_habit_completion_phrasings(chat_id, temp_db, phrase_template):
    habit = temp_db.create_habit("Walking")
    prompt = phrase_template.format(name=habit["name"])
    result = app._handle_local_command(prompt, chat_id)
    assert result is not None
    reply, provider = result
    assert provider == "local-habit-manager"
    assert "Walking" in reply
    updated = temp_db.list_habits()[0]
    assert len(updated["completed_dates"]) == 1


def test_habit_rename_and_delete_intents(chat_id, temp_db):
    habit = temp_db.create_habit("Jogging")
    renamed = app._handle_local_command("rename habit Jogging to Running 5K", chat_id)
    assert renamed is not None
    assert renamed[1] == "local-habit-manager"
    assert "Running 5K" in renamed[0]

    deleted = app._handle_local_command("delete habit Running 5K", chat_id)
    assert deleted is not None
    assert deleted[1] == "local-habit-manager"
    assert len(temp_db.list_habits()) == 0


def test_generic_completion_target(chat_id, temp_db):
    task = temp_db.create_task("Write unit test suite", priority="high")
    result = app._handle_local_command(f"i did {task['title']}", chat_id)
    assert result is not None
    assert result[1] == "local-task-manager"


# -------------------------------------------------------------------------
# 6. REMINDER PHRASINGS (RELATIVE, DAY-PART, RECURRING)
# -------------------------------------------------------------------------

@pytest.mark.parametrize(
    "prompt,expected_text",
    [
        ("remind me in 10 minutes to take a break", "take a break"),
        ("set a reminder in 1 minute for stretch", "stretch"),
        ("remind me tomorrow morning to water my plants", "water my plants"),
        ("remind me to study Java at 7 PM", "study Java"),
        ("remind me to review goals at 8 AM, every day", "review goals"),
        ("set a reminder for sleeping at 11pm", "sleeping"),
        ("remind me to stretch at 6 PM, every 2 hours", "stretch"),
        ("set another reminder in 15 minutes for tea", "tea"),
    ],
)
def test_reminder_create_intents(chat_id, temp_db, prompt, expected_text):
    result = app._handle_local_command(prompt, chat_id)
    assert result is not None, f"Expected deterministic match for: {prompt}"
    reply, provider = result
    assert provider == "local-scheduler"
    assert expected_text.lower() in reply.lower()
    reminders = temp_db.list_reminders()
    assert any(expected_text in r["text"] for r in reminders)


def test_reminder_list_and_cancel_intents(chat_id, temp_db):
    temp_db.create_reminder("Check baking cake", run_at="2026-12-01T10:00:00+05:30")
    listing = app._handle_local_command("show reminders", chat_id)
    assert listing is not None
    assert listing[1] == "local-scheduler"
    assert "Check baking cake" in listing[0]

    cancelled = app._handle_local_command("cancel reminder Check baking cake", chat_id)
    assert cancelled is not None
    assert cancelled[1] == "local-scheduler"
    assert len(temp_db.list_reminders(active_only=True)) == 0


# -------------------------------------------------------------------------
# 7. GREETINGS, INTROS, & NEAR-MISS CLARIFICATIONS
# -------------------------------------------------------------------------

@pytest.mark.parametrize(
    "prompt,expected_provider",
    [
        ("hello spidey", "local-greeting"),
        ("hey", "local-greeting"),
        ("good morning", "local-greeting"),
        ("who are you", "local-identity"),
        ("what is your name", "local-identity"),
    ],
)
def test_greetings_and_identity(chat_id, prompt, expected_provider):
    result = app._handle_local_command(prompt, chat_id)
    assert result is not None
    reply, provider = result
    assert provider == expected_provider


@pytest.mark.parametrize(
    "prompt,expected_in_clarification",
    [
        ("remind me", "what should i remind you about"),
        ("remind me tomorrow", "what should i remind you about"),
        ("remind me to sleep", "when should i remind you"),
    ],
)
def test_reminder_near_miss_clarification(chat_id, prompt, expected_in_clarification):
    result = app._handle_local_command(prompt, chat_id)
    assert result is not None
    reply, provider = result
    assert provider == "local-scheduler"
    assert expected_in_clarification in reply.lower()


# -------------------------------------------------------------------------
# 8. AMBIGUOUS / CONVERSATIONAL PHRASES (FALLBACK TO QWEN)
# -------------------------------------------------------------------------

@pytest.mark.parametrize(
    "prompt",
    [
        "What is the capital of Australia?",
        "Write a short poem about neural networks",
        "Can you explain quantum computing simply?",
        "I am thinking about whether I should buy a new mechanical keyboard",
        "How do I reverse a singly linked list in C++?",
        "Tell me a joke about programming",
        "What's the weather like in Tokyo right now?",
    ],
)
def test_conversational_queries_fall_through(chat_id, prompt):
    # These must NOT be trapped by the deterministic parser; they should return None so Qwen answers them.
    result = app._handle_local_command(prompt, chat_id)
    assert result is None, f"Query '{prompt}' was mistakenly handled by deterministic provider {result}"
