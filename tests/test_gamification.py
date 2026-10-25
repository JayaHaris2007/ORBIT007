"""Characterization tests for gamification, XP, 70% threshold, and streak logic in memory.py."""

from datetime import datetime, timedelta
import pytest
from memory import LOCAL_TIMEZONE, MemoryStore


def test_initial_gamification_state(temp_db):
    data = temp_db.get_gamification()
    assert data["xp"] == 0
    assert data["global_streak"] == 0
    assert data["valid_streak_dates"] == []
    assert data["achievements"] == []


def test_first_habit_achievement(temp_db):
    temp_db.create_habit("Read daily")
    data = temp_db.get_gamification()
    assert "first-habit" in data["achievements"]


def test_xp_addition_and_removal(temp_db):
    habit = temp_db.create_habit("Exercise")
    today = datetime.now(LOCAL_TIMEZONE).date().isoformat()

    # Complete habit -> +10 XP
    temp_db.set_habit_completion(habit["id"], today, True)
    assert temp_db.get_gamification()["xp"] == 10

    # Uncomplete habit -> -10 XP
    temp_db.set_habit_completion(habit["id"], today, False)
    assert temp_db.get_gamification()["xp"] == 0

    # Uncompleting when 0 cannot produce negative XP
    temp_db.set_habit_completion(habit["id"], today, False)
    assert temp_db.get_gamification()["xp"] == 0


def test_seventy_percent_threshold_single_habit(temp_db):
    habit = temp_db.create_habit("Meditate")
    today = datetime.now(LOCAL_TIMEZONE).date().isoformat()

    # 1 of 1 habit completed = 100% >= 70% -> date is valid
    temp_db.set_habit_completion(habit["id"], today, True)
    data = temp_db.get_gamification()
    assert today in data["valid_streak_dates"]


def test_seventy_percent_threshold_two_habits(temp_db):
    h1 = temp_db.create_habit("Habit 1")
    h2 = temp_db.create_habit("Habit 2")
    today = datetime.now(LOCAL_TIMEZONE).date().isoformat()

    # 1 of 2 habits completed = 50% < 70% -> date is NOT valid for streak
    temp_db.set_habit_completion(h1["id"], today, True)
    data = temp_db.get_gamification()
    assert today not in data["valid_streak_dates"]

    # 2 of 2 habits completed = 100% >= 70% -> date IS valid for streak
    temp_db.set_habit_completion(h2["id"], today, True)
    data2 = temp_db.get_gamification()
    assert today in data2["valid_streak_dates"]


def test_seventy_percent_threshold_multi_habits(temp_db):
    # 4 habits: threshold is 4 * 0.7 = 2.8.
    # 2 completed (50%) < 2.8 -> invalid. 3 completed (75%) >= 2.8 -> valid.
    habits = [temp_db.create_habit(f"H{i}") for i in range(4)]
    today = datetime.now(LOCAL_TIMEZONE).date().isoformat()

    temp_db.set_habit_completion(habits[0]["id"], today, True)
    temp_db.set_habit_completion(habits[1]["id"], today, True)
    assert today not in temp_db.get_gamification()["valid_streak_dates"]

    temp_db.set_habit_completion(habits[2]["id"], today, True)
    assert today in temp_db.get_gamification()["valid_streak_dates"]


def test_streak_calculation_active_and_grace_period(temp_db):
    habit = temp_db.create_habit("Journaling")
    today = datetime.now(LOCAL_TIMEZONE).date()
    yesterday = today - timedelta(days=1)
    day_before = today - timedelta(days=2)

    # Yesterday and day before completed; today not yet completed (grace period preserves streak = 2)
    temp_db.set_habit_completion(habit["id"], yesterday.isoformat(), True)
    temp_db.set_habit_completion(habit["id"], day_before.isoformat(), True)
    data = temp_db.get_gamification()
    assert data["global_streak"] == 2

    # Now complete today -> streak becomes 3
    temp_db.set_habit_completion(habit["id"], today.isoformat(), True)
    assert temp_db.get_gamification()["global_streak"] == 3


def test_streak_calculation_broken(temp_db):
    habit = temp_db.create_habit("Running")
    today = datetime.now(LOCAL_TIMEZONE).date()
    three_days_ago = today - timedelta(days=3)

    # 3 days ago was completed, but yesterday and today were not -> streak = 0
    temp_db.set_habit_completion(habit["id"], three_days_ago.isoformat(), True)
    assert temp_db.get_gamification()["global_streak"] == 0


def test_achievements_unlock_conditions(temp_db):
    habit = temp_db.create_habit("Read Code")
    today = datetime.now(LOCAL_TIMEZONE).date()

    # Complete 7 consecutive days
    for i in range(7):
        date_str = (today - timedelta(days=i)).isoformat()
        temp_db.set_habit_completion(habit["id"], date_str, True)

    data = temp_db.get_gamification()
    assert "seven-day-streak" in data["achievements"]
    assert data["global_streak"] >= 7

    # 7 completions * 10 XP = 70 XP (not 100 yet)
    assert "hundred-xp" not in data["achievements"]

    # Add 3 more completions on older days to hit 100 XP
    for i in range(7, 10):
        date_str = (today - timedelta(days=i)).isoformat()
        temp_db.set_habit_completion(habit["id"], date_str, True)

    data100 = temp_db.get_gamification()
    assert data100["xp"] >= 100
    assert "hundred-xp" in data100["achievements"]
