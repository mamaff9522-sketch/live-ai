"""Logging: one JSON object per line in data/logs/live_ai.log (rotated), plus the console.

Message text is never written to the log, only sizes and ids, so one user's chat
can't leak to whoever reads the log.
"""
from __future__ import annotations

import json
import logging
import logging.handlers
import time
from pathlib import Path


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        out = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(record.created)),
            "level": record.levelname,
            "where": record.name,
            "msg": record.getMessage(),
        }
        extra = getattr(record, "data", None)
        if extra:
            out.update(extra)
        if record.exc_info:
            out["error"] = self.formatException(record.exc_info)
        return json.dumps(out, ensure_ascii=False)


_configured = False


def setup(log_dir: Path, level: int = logging.INFO) -> None:
    global _configured
    if _configured:
        return
    log_dir.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger("live_ai")
    root.setLevel(level)
    fh = logging.handlers.RotatingFileHandler(log_dir / "live_ai.log", maxBytes=5_000_000, backupCount=5, encoding="utf-8")
    fh.setFormatter(JsonFormatter())
    root.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(JsonFormatter())
    root.addHandler(ch)
    _configured = True


def get(name: str) -> logging.Logger:
    return logging.getLogger(f"live_ai.{name}")


def event(logger: logging.Logger, msg: str, level: int = logging.INFO, **data) -> None:
    logger.log(level, msg, extra={"data": data})
