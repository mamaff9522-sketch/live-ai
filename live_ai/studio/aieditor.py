"""The AI editor: a bigger vision model (Qwen3.5-9B) running on the owner's own free Colab GPU watches the
clips and decides the cuts. Live AI stays the place where the video is made; the model only answers
"keep from here to here, fade or not, why".

How it connects (no open ports needed on the phone or on Colab):
  1. In the studio the owner creates a worker key (shown once; only its hash is stored).
  2. The Colab notebook starts llama-server with Qwen3.5-9B on the GPU and runs scripts/ai_editor_worker.py
     with the server address and the key. The worker long-polls this server for jobs.
  3. When a project reaches AI 12 and the owner's worker has been seen in the last 90 s, a job is created:
     for every clip a few still frames (with their times), what is said and when, the scene mood, the
     picture-jump stretches the inspector found, and the rule-based plan. The worker returns per-clip
     choices; cutter.apply_plan keeps them only where they are safe (words are never cut).
  4. No worker, or no answer in time: the rule-based plan is used, and the project says so.
The job only ever contains this project's stills and text, and only the key's owner can take it.
"""
from __future__ import annotations

import hashlib
import json
import os
import secrets
import subprocess
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS editor_workers (
    user_id INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    key_hash TEXT NOT NULL,
    last_seen REAL NOT NULL DEFAULT 0,
    info TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS editor_jobs (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    project_id INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'queued',
    payload TEXT NOT NULL,
    result TEXT,
    created REAL NOT NULL,
    taken REAL,
    finished REAL
);
CREATE INDEX IF NOT EXISTS editor_jobs_user ON editor_jobs(user_id, status);
"""
ONLINE_SECONDS = 90
FRAMES_PER_CLIP = int(os.environ.get("LIVEAI_AIED_FRAMES", "6"))      # stills per clip (more on a fast GPU)
MAX_FRAMES = int(os.environ.get("LIVEAI_AIED_MAX_FRAMES", "12"))


def _hash(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


class AIEditor:
    def __init__(self, store, root: Path):
        self.store = store
        self.root = root
        with store.conn() as c:
            c.executescript(SCHEMA)

    # keys and presence ---------------------------------------------------------------------
    def new_key(self, user_id: int) -> str:
        key = "lew_" + secrets.token_urlsafe(24)
        with self.store.conn() as c:
            c.execute("INSERT INTO editor_workers(user_id,key_hash,last_seen,info) VALUES (?,?,0,'{}') "
                      "ON CONFLICT(user_id) DO UPDATE SET key_hash=excluded.key_hash, last_seen=0", (user_id, _hash(key)))
        return key

    def worker_user(self, key: str) -> int | None:
        if not key or not key.startswith("lew_"):
            return None
        r = self.store.conn().execute("SELECT user_id FROM editor_workers WHERE key_hash=?", (_hash(key),)).fetchone()
        return r["user_id"] if r else None

    def seen(self, user_id: int, info: dict | None = None) -> None:
        with self.store.conn() as c:
            if info is not None:
                c.execute("UPDATE editor_workers SET last_seen=?, info=? WHERE user_id=?",
                          (time.time(), json.dumps(info, ensure_ascii=False)[:2000], user_id))
            else:
                c.execute("UPDATE editor_workers SET last_seen=? WHERE user_id=?", (time.time(), user_id))

    def status(self, user_id: int) -> dict:
        r = self.store.conn().execute("SELECT last_seen, info FROM editor_workers WHERE user_id=?", (user_id,)).fetchone()
        if not r:
            return {"has_key": False, "online": False}
        return {"has_key": True, "online": time.time() - r["last_seen"] < ONLINE_SECONDS,
                "last_seen": r["last_seen"] or None, "info": json.loads(r["info"] or "{}")}

    def online(self, user_id: int) -> bool:
        return self.status(user_id)["online"]

    # jobs ---------------------------------------------------------------------------------------
    def make_frames(self, pid: int, clip: dict, times: list[float]) -> list[dict]:
        d = self.root / str(pid) / "frames"
        d.mkdir(parents=True, exist_ok=True)
        out = []
        for t in times:
            f = d / f"{clip['id']}_{t:.2f}.jpg"
            if not f.exists():
                subprocess.run(["ffmpeg", "-loglevel", "error", "-nostdin", "-y", "-ss", f"{t:.2f}", "-i", clip["path"], "-map", "0:v:0",
                                "-frames:v", "1", "-vf", "scale=448:-2", "-q:v", "4", str(f)], capture_output=True, timeout=120)
            if f.exists():
                out.append({"t": round(t, 2), "name": f.name})
        return out

    def create_job(self, user_id: int, pid: int, payload: dict) -> int:
        with self.store.conn() as c:
            c.execute("UPDATE editor_jobs SET status='expired' WHERE project_id=? AND status IN ('queued','taken')", (pid,))
            return c.execute("INSERT INTO editor_jobs(user_id,project_id,status,payload,created) VALUES (?,?,?,?,?)",
                             (user_id, pid, "queued", json.dumps(payload, ensure_ascii=False), time.time())).lastrowid

    def next_job(self, user_id: int) -> dict | None:
        with self.store.conn() as c:
            # a job taken but not answered for 10 min goes back to the queue (Colab may have restarted)
            c.execute("UPDATE editor_jobs SET status='queued' WHERE user_id=? AND status='taken' AND taken < ?",
                      (user_id, time.time() - 600))
            r = c.execute("SELECT id, payload FROM editor_jobs WHERE user_id=? AND status='queued' ORDER BY id LIMIT 1",
                          (user_id,)).fetchone()
            if not r:
                return None
            c.execute("UPDATE editor_jobs SET status='taken', taken=? WHERE id=?", (time.time(), r["id"]))
        return {"id": r["id"], **json.loads(r["payload"])}

    def frame_path(self, user_id: int, job_id: int, name: str) -> Path | None:
        r = self.store.conn().execute("SELECT project_id FROM editor_jobs WHERE id=? AND user_id=?", (job_id, user_id)).fetchone()
        if not r or "/" in name or not name.endswith(".jpg"):
            return None
        p = self.root / str(r["project_id"]) / "frames" / name
        return p if p.exists() else None

    def submit(self, user_id: int, job_id: int, result: dict) -> bool:
        with self.store.conn() as c:
            n = c.execute("UPDATE editor_jobs SET status='done', result=?, finished=? WHERE id=? AND user_id=? AND status='taken'",
                          (json.dumps(result, ensure_ascii=False)[:2_000_000], time.time(), job_id, user_id)).rowcount
        return bool(n)

    def fail(self, user_id: int, job_id: int, message: str) -> None:
        with self.store.conn() as c:
            c.execute("UPDATE editor_jobs SET status='failed', result=?, finished=? WHERE id=? AND user_id=?",
                      (json.dumps({"error": message[:500]}, ensure_ascii=False), time.time(), job_id, user_id))

    def wait(self, job_id: int, timeout: float, check=None) -> dict | None:
        """Waits for the worker's answer; `check` is called every few seconds (pause handling)."""
        end = time.time() + timeout
        while time.time() < end:
            r = self.store.conn().execute("SELECT status, result FROM editor_jobs WHERE id=?", (job_id,)).fetchone()
            if r["status"] == "done":
                return json.loads(r["result"] or "{}")
            if r["status"] in ("failed", "expired"):
                return None
            if check:
                check()
            time.sleep(3)
        with self.store.conn() as c:
            c.execute("UPDATE editor_jobs SET status='expired' WHERE id=? AND status IN ('queued','taken')", (job_id,))
        return None


def frame_times(duration: float, spans: list, jumps: list, n: int = FRAMES_PER_CLIP) -> list[float]:
    """Evenly spread stills, plus the edges of speech and of picture jumps (where cut decisions are made)."""
    d = max(0.2, float(duration))
    ts = {round(d * (i + 0.5) / n, 2) for i in range(n)}
    for a, b in (spans or [])[:1] + (spans or [])[-1:]:
        ts.update({round(max(0.0, a - 0.3), 2), round(min(d - 0.05, b + 0.3), 2)})
    for a, b in jumps or []:
        ts.update({round(min(d - 0.05, max(0.0, a)), 2), round(min(d - 0.05, b), 2)})
    return sorted(t for t in ts if 0 <= t < d)[:MAX_FRAMES]
