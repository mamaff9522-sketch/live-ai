"""SQLite storage. Every read and write of user data takes the owner's user_id and
filters by it, so one user can never read, change or delete another user's
conversations or memories (even by guessing ids)."""
from __future__ import annotations

import hashlib
import secrets
import sqlite3
import threading
import time
from pathlib import Path

SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    token_hash TEXT NOT NULL UNIQUE,
    created REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title TEXT NOT NULL DEFAULT '',
    created REAL NOT NULL,
    updated REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS conv_user ON conversations(user_id, updated);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY,
    conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('user','assistant')),
    content TEXT NOT NULL,
    created REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS msg_conv ON messages(conversation_id, id);
CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    created REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS mem_user ON memories(user_id);
"""


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


class NotFound(Exception):
    pass


class Store:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._local = threading.local()
        with self.conn() as c:
            c.executescript(SCHEMA)
            c.execute("INSERT OR IGNORE INTO meta(key, value) VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),))

    def conn(self) -> sqlite3.Connection:
        c = getattr(self._local, "c", None)
        if c is None:
            c = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
            c.row_factory = sqlite3.Row
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("PRAGMA foreign_keys=ON")
            c.execute("PRAGMA synchronous=NORMAL")
            self._local.c = c
        return c

    # users -----------------------------------------------------------------
    def create_user(self, name: str) -> tuple[int, str]:
        token = "lai_" + secrets.token_urlsafe(32)
        with self.conn() as c:
            cur = c.execute("INSERT INTO users(name, token_hash, created) VALUES (?,?,?)",
                            (name.strip()[:60] or "ผู้ใช้", hash_token(token), time.time()))
        return cur.lastrowid, token

    def user_by_token(self, token: str) -> dict | None:
        if not token:
            return None
        r = self.conn().execute("SELECT id, name, created FROM users WHERE token_hash=?", (hash_token(token),)).fetchone()
        return dict(r) if r else None

    def count_users(self) -> int:
        return self.conn().execute("SELECT COUNT(*) FROM users").fetchone()[0]

    # conversations ---------------------------------------------------------
    def create_conversation(self, user_id: int, title: str = "") -> int:
        now = time.time()
        with self.conn() as c:
            cur = c.execute("INSERT INTO conversations(user_id, title, created, updated) VALUES (?,?,?,?)",
                            (user_id, title[:80], now, now))
        return cur.lastrowid

    def get_conversation(self, user_id: int, conv_id: int) -> dict:
        r = self.conn().execute("SELECT id, title, created, updated FROM conversations WHERE id=? AND user_id=?",
                                (conv_id, user_id)).fetchone()
        if not r:
            raise NotFound("conversation")
        return dict(r)

    def list_conversations(self, user_id: int, limit: int = 100) -> list[dict]:
        rows = self.conn().execute(
            "SELECT id, title, created, updated FROM conversations WHERE user_id=? ORDER BY updated DESC LIMIT ?",
            (user_id, limit)).fetchall()
        return [dict(r) for r in rows]

    def rename_conversation(self, user_id: int, conv_id: int, title: str) -> None:
        with self.conn() as c:
            n = c.execute("UPDATE conversations SET title=? WHERE id=? AND user_id=?", (title[:80], conv_id, user_id)).rowcount
        if not n:
            raise NotFound("conversation")

    def delete_conversation(self, user_id: int, conv_id: int) -> None:
        with self.conn() as c:
            n = c.execute("DELETE FROM conversations WHERE id=? AND user_id=?", (conv_id, user_id)).rowcount
        if not n:
            raise NotFound("conversation")

    # messages --------------------------------------------------------------
    def add_message(self, user_id: int, conv_id: int, role: str, content: str) -> int:
        self.get_conversation(user_id, conv_id)  # ownership check
        now = time.time()
        with self.conn() as c:
            cur = c.execute("INSERT INTO messages(conversation_id, user_id, role, content, created) VALUES (?,?,?,?,?)",
                            (conv_id, user_id, role, content, now))
            c.execute("UPDATE conversations SET updated=? WHERE id=? AND user_id=?", (now, conv_id, user_id))
        return cur.lastrowid

    def messages(self, user_id: int, conv_id: int, limit: int = 500) -> list[dict]:
        self.get_conversation(user_id, conv_id)
        rows = self.conn().execute(
            "SELECT id, role, content, created FROM (SELECT * FROM messages WHERE conversation_id=? AND user_id=? "
            "ORDER BY id DESC LIMIT ?) ORDER BY id ASC", (conv_id, user_id, limit)).fetchall()
        return [dict(r) for r in rows]

    # memories --------------------------------------------------------------
    def add_memory(self, user_id: int, content: str) -> int:
        content = content.strip()[:500]
        existing = self.conn().execute("SELECT id FROM memories WHERE user_id=? AND content=?", (user_id, content)).fetchone()
        if existing:
            return existing[0]
        with self.conn() as c:
            cur = c.execute("INSERT INTO memories(user_id, content, created) VALUES (?,?,?)", (user_id, content, time.time()))
        return cur.lastrowid

    def memories(self, user_id: int) -> list[dict]:
        rows = self.conn().execute("SELECT id, content, created FROM memories WHERE user_id=? ORDER BY id", (user_id,)).fetchall()
        return [dict(r) for r in rows]

    def delete_memory(self, user_id: int, mem_id: int) -> None:
        with self.conn() as c:
            n = c.execute("DELETE FROM memories WHERE id=? AND user_id=?", (mem_id, user_id)).rowcount
        if not n:
            raise NotFound("memory")

    def delete_memories_matching(self, user_id: int, needle: str) -> int:
        with self.conn() as c:
            return c.execute("DELETE FROM memories WHERE user_id=? AND instr(content, ?) > 0", (user_id, needle)).rowcount

    # backups ---------------------------------------------------------------
    def backup_to(self, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".tmp")
        out = sqlite3.connect(tmp)
        try:
            self.conn().backup(out)
        finally:
            out.close()
        tmp.replace(dest)
