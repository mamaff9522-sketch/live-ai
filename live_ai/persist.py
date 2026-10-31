"""Keeps the user database safe on hosts with temporary disks (free Hugging Face Spaces).

When LIVEAI_PERSIST_REPO (a private Hugging Face *dataset* repo, e.g. "you/live-ai-data") and
HF_TOKEN are set:
  • at start-up, if there is no local database, the last saved copy is downloaded;
  • every few minutes (only when something changed) and at shutdown, a compressed snapshot of the
    database is uploaded.
Accounts, chats, memories and studio projects survive restarts this way. Video files are not
uploaded (too large for this route); download finished videos to keep them.
"""
from __future__ import annotations

import base64
import gzip
import json
import logging
import os
import sqlite3
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from . import logs

log = logs.get("persist")
FILE = "live_ai.sqlite3.gz"
MAX_BYTES = 9 * 2**20          # the plain (non-LFS) commit route takes files up to ~10 MB


def configured() -> bool:
    return bool(os.environ.get("LIVEAI_PERSIST_REPO") and os.environ.get("HF_TOKEN"))


def _req(url: str, data: bytes | None = None, ctype: str | None = None, method: str | None = None):
    h = {"Authorization": f"Bearer {os.environ['HF_TOKEN']}", "User-Agent": "live-ai"}
    if ctype:
        h["Content-Type"] = ctype
    return urllib.request.Request(url, data=data, headers=h, method=method)


def restore(db_path: Path) -> bool:
    if not configured() or db_path.exists():
        return False
    repo = os.environ["LIVEAI_PERSIST_REPO"]
    try:
        with urllib.request.urlopen(_req(f"https://huggingface.co/datasets/{repo}/resolve/main/{FILE}"), timeout=120) as r:
            raw = gzip.decompress(r.read())
    except urllib.error.HTTPError as e:
        if e.code == 404:
            logs.event(log, "no saved database yet", repo=repo)
            return False
        logs.event(log, "restore failed", logging.ERROR, error=str(e))
        return False
    except Exception as e:
        logs.event(log, "restore failed", logging.ERROR, error=str(e))
        return False
    tmp = db_path.with_suffix(".restore")
    tmp.write_bytes(raw)
    sqlite3.connect(tmp).execute("PRAGMA integrity_check").fetchone()
    tmp.replace(db_path)
    logs.event(log, "database restored from saved copy", repo=repo, size_kb=len(raw) // 1024)
    return True


def save(db_path: Path, reason: str) -> bool:
    if not configured() or not db_path.exists():
        return False
    repo = os.environ["LIVEAI_PERSIST_REPO"]
    snap = db_path.with_suffix(".snap")
    src, out = sqlite3.connect(db_path), sqlite3.connect(snap)
    try:
        src.backup(out)
    finally:
        out.close()
        src.close()
    data = gzip.compress(snap.read_bytes(), 6)
    snap.unlink(missing_ok=True)
    if len(data) > MAX_BYTES:
        logs.event(log, "database too large to save this way", logging.WARNING, size_mb=round(len(data) / 2**20, 1))
        return False
    body = "\n".join([
        json.dumps({"key": "header", "value": {"summary": f"Live AI backup ({reason})", "description": ""}}),
        json.dumps({"key": "file", "value": {"content": base64.b64encode(data).decode(), "path": FILE, "encoding": "base64"}}),
    ]).encode()
    try:
        with urllib.request.urlopen(_req(f"https://huggingface.co/api/datasets/{repo}/commit/main", body,
                                         "application/x-ndjson", "POST"), timeout=300) as r:
            r.read()
    except Exception as e:
        logs.event(log, "save failed", logging.WARNING, error=str(e)[:300])
        return False
    logs.event(log, "database saved", repo=repo, size_kb=len(data) // 1024, reason=reason)
    return True


def start(db_path: Path, every_s: int = 600) -> None:
    if not configured():
        return

    def loop():
        last = 0.0
        while True:
            time.sleep(every_s)
            try:
                m = max(p.stat().st_mtime for p in db_path.parent.glob(db_path.name + "*"))
                if m > last:
                    if save(db_path, "periodic"):
                        last = m
            except Exception as e:
                logs.event(log, "periodic save failed", logging.WARNING, error=str(e)[:200])
    threading.Thread(target=loop, name="persist", daemon=True).start()
