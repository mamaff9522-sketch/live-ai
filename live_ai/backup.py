"""Backups of the user database.

- A snapshot is taken every time Live AI starts (before anything can change) and once a day.
- The newest LIVEAI_BACKUPS_KEEP snapshots are kept.
- Roll back with:  python -m live_ai.backup restore <file>   (Live AI must be stopped)
  The current database is itself backed up first, so a restore can be undone too.
"""
from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
import threading
import time
from pathlib import Path

from . import logs
from .config import Settings

log = logs.get("backup")


def snapshot(s: Settings, reason: str) -> Path | None:
    if not s.db_path.exists():
        return None
    dest = s.data_dir / "backups" / f"live_ai-{time.strftime('%Y%m%d-%H%M%S')}-{reason}.sqlite3"
    src = sqlite3.connect(s.db_path)
    out = sqlite3.connect(dest)
    try:
        src.backup(out)
    finally:
        out.close()
        src.close()
    prune(s)
    logs.event(log, "backup saved", file=dest.name, reason=reason)
    return dest


def prune(s: Settings) -> None:
    files = sorted((s.data_dir / "backups").glob("live_ai-*.sqlite3"))
    for f in files[:-s.backups_to_keep] if len(files) > s.backups_to_keep else []:
        f.unlink(missing_ok=True)


def list_backups(s: Settings) -> list[Path]:
    return sorted((s.data_dir / "backups").glob("live_ai-*.sqlite3"))


def restore(s: Settings, file: Path) -> None:
    file = Path(file)
    if not file.is_file():
        file = s.data_dir / "backups" / file.name
    sqlite3.connect(file).execute("PRAGMA integrity_check").fetchone()  # refuses a broken file
    snapshot(s, "before-restore")
    for ext in ("-wal", "-shm"):
        Path(str(s.db_path) + ext).unlink(missing_ok=True)
    shutil.copy2(file, s.db_path)
    logs.event(log, "database restored", file=file.name)


def start_daily(s: Settings) -> None:
    def loop():
        while True:
            time.sleep(24 * 3600)
            try:
                snapshot(s, "daily")
            except Exception as e:  # never take the service down over a backup
                logs.event(log, "backup failed", 40, error=str(e))
    threading.Thread(target=loop, name="daily-backup", daemon=True).start()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m live_ai.backup")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("now")
    sub.add_parser("list")
    r = sub.add_parser("restore")
    r.add_argument("file")
    a = p.parse_args(argv)
    s = Settings()
    logs.setup(s.log_dir)
    if a.cmd == "now":
        print(snapshot(s, "manual"))
    elif a.cmd == "list":
        for f in list_backups(s):
            print(f.name, f.stat().st_size)
    else:
        restore(s, Path(a.file))
        print("restored", a.file)
    return 0


if __name__ == "__main__":
    sys.exit(main())
