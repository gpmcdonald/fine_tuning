from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Optional

# Repo root = .../fine_tuning
REPO_ROOT = Path(__file__).resolve().parents[3]

# Default DB location (safe, local, ignored by git if you want)
DEFAULT_DB_PATH = REPO_ROOT / "api" / "data" / "symoneural.sqlite3"

def _db_path() -> Path:
    # Allow override from env
    p = os.environ.get("SYM_DB_PATH", "").strip()
    if p:
        return Path(p).expanduser().resolve()
    return DEFAULT_DB_PATH

def connect(db_path: Optional[Path] = None) -> sqlite3.Connection:
    path = (db_path or _db_path())
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row

    # Recommended pragmas for a small local app
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
    except Exception:
        pass

    return conn

def init_db() -> None:
    """
    Minimal schema for Phase-B:
    - users table (for future credits/auth integration)
    - jobs table (if you want persistence later)
    This is safe even if unused right now.
    """
    conn = connect()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at TEXT DEFAULT (datetime('now')),
        username TEXT UNIQUE,
        credits INTEGER DEFAULT 0
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS jobs (
        id TEXT PRIMARY KEY,
        created_at TEXT DEFAULT (datetime('now')),
        status TEXT DEFAULT 'queued',
        prompt TEXT NOT NULL,
        out_path TEXT,
        error TEXT
    );
    """)

    conn.commit()
    conn.close()
