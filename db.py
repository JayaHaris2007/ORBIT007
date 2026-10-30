import sqlite3
from pathlib import Path
from contextlib import contextmanager
from typing import List, Dict, Any, Optional

BASE_DIR = Path(__file__).resolve().parent
import os
DB_DIR = Path(os.environ.get("ORBIT_DB_DIR", BASE_DIR))
DB_PATH = DB_DIR / "orbit007.db" 

@contextmanager
def get_db_connection():
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False, timeout=15.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

def init_db() -> None:
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS reminders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                remind_at TEXT NOT NULL,
                completed INTEGER DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT DEFAULT 'default',
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

def create_reminder(title: str, remind_at: str) -> Dict[str, Any]:
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO reminders (title, remind_at, completed) VALUES (?, ?, 0)",
            (title.strip(), remind_at.strip()),
        )
        row_id = cur.lastrowid
        cur.execute("SELECT * FROM reminders WHERE id = ?", (row_id,))
        row = cur.fetchone()
        res = dict(row) if row else {"id": row_id, "title": title, "remind_at": remind_at, "completed": 0}
        res["text"] = res.get("title", "")
        res["run_at"] = res.get("remind_at", "")
        return res

def list_reminders(include_completed: bool = False) -> List[Dict[str, Any]]:
    with get_db_connection() as conn:
        cur = conn.cursor()
        if include_completed:
            cur.execute("SELECT * FROM reminders ORDER BY completed ASC, remind_at ASC")
        else:
            cur.execute("SELECT * FROM reminders WHERE completed = 0 ORDER BY remind_at ASC")
        rows = []
        for r in cur.fetchall():
            d = dict(r)
            d["text"] = d.get("title", "")
            d["run_at"] = d.get("remind_at", "")
            rows.append(d)
        return rows

def update_reminder_status(reminder_id: int, completed: bool = True) -> Optional[Dict[str, Any]]:
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            "UPDATE reminders SET completed = ? WHERE id = ?",
            (1 if completed else 0, reminder_id),
        )
        cur.execute("SELECT * FROM reminders WHERE id = ?", (reminder_id,))
        row = cur.fetchone()
        if not row:
            return None
        res = dict(row)
        res["text"] = res.get("title", "")
        res["run_at"] = res.get("remind_at", "")
        return res

def delete_reminder(reminder_id: int) -> bool:
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM reminders WHERE id = ?", (reminder_id,))
        return cur.rowcount > 0

def save_message(role: str, content: str, session_id: str = "default") -> None:
    with get_db_connection() as conn:
        conn.execute(
            "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
            (session_id, role, content),
        )

def get_recent_messages(session_id: str = "default", limit: int = 20) -> List[Dict[str, Any]]:
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT role, content, created_at FROM messages WHERE session_id = ? ORDER BY id DESC LIMIT ?",
            (session_id, limit),
        )
        rows = [dict(r) for r in cur.fetchall()]
        rows.reverse()
        return rows

