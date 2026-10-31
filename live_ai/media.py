"""User media files (videos, audio, images) for the video editor.

- Files live in data/media/<user_id>/ and are recorded in SQLite with their owner;
  every lookup takes the owner's user_id, like conversations.
- ffprobe checks every upload; a file ffprobe can't read is rejected.
- Videos are played/downloaded through short-lived signed links, because a <video>
  tag can't send the Authorization header.
- Old files are deleted after LIVEAI_MEDIA_TTL_HOURS; each user has a storage quota.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import shutil
import sqlite3
import subprocess
import threading
import time
from pathlib import Path

from . import logs
from .store import NotFound

log = logs.get("media")

MEDIA_SCHEMA = """
CREATE TABLE IF NOT EXISTS media (
    id TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    kind TEXT NOT NULL CHECK (kind IN ('video','audio','image')),
    name TEXT NOT NULL,
    path TEXT NOT NULL,
    size INTEGER NOT NULL,
    duration REAL NOT NULL DEFAULT 0,
    width INTEGER NOT NULL DEFAULT 0,
    height INTEGER NOT NULL DEFAULT 0,
    has_audio INTEGER NOT NULL DEFAULT 0,
    origin TEXT NOT NULL DEFAULT 'upload',
    created REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS media_user ON media(user_id, created);
"""

EXT_BY_TYPE = {
    "video/mp4": "mp4", "video/quicktime": "mov", "video/webm": "webm", "video/x-matroska": "mkv", "video/3gpp": "3gp",
    "audio/mpeg": "mp3", "audio/mp4": "m4a", "audio/x-m4a": "m4a", "audio/aac": "aac", "audio/wav": "wav",
    "audio/x-wav": "wav", "audio/ogg": "ogg", "audio/webm": "weba",
    "image/jpeg": "jpg", "image/png": "png", "image/webp": "webp",
}


class MediaError(ValueError):
    pass


def probe(path: Path) -> dict:
    """{'kind','duration','width','height','has_audio'} or MediaError."""
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-show_format", str(path)],
                             capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise MediaError(f"ตรวจไฟล์ไม่ได้: {e}") from e
    if out.returncode != 0:
        raise MediaError("ไฟล์เสียหายหรือไม่ใช่ไฟล์วิดีโอ/เสียง/ภาพที่รองรับ")
    info = json.loads(out.stdout or "{}")
    streams = info.get("streams", [])
    fmt = info.get("format", {})
    v = next((s for s in streams if s.get("codec_type") == "video" and not (s.get("disposition") or {}).get("attached_pic")), None)
    a = next((s for s in streams if s.get("codec_type") == "audio"), None)
    duration = float(fmt.get("duration") or (v or a or {}).get("duration") or 0)
    if v:
        w, h = int(v.get("width") or 0), int(v.get("height") or 0)
        rot = 0
        for sd in v.get("side_data_list", []) or []:
            if "rotation" in sd:
                rot = int(float(sd["rotation"]))
        rot = int((v.get("tags") or {}).get("rotate", rot))
        if abs(rot) % 180 == 90:      # phone videos: the player shows them rotated
            w, h = h, w
        image_like = fmt.get("format_name", "").split(",")[0] in ("image2", "png_pipe", "jpeg_pipe", "webp_pipe") or \
            v.get("codec_name") in ("png", "mjpeg", "webp") and duration < 0.1
        if image_like:
            return {"kind": "image", "duration": 0.0, "width": w, "height": h, "has_audio": False}
        if duration <= 0:
            raise MediaError("อ่านความยาววิดีโอไม่ได้")
        return {"kind": "video", "duration": round(duration, 3), "width": w, "height": h, "has_audio": a is not None}
    if a:
        if duration <= 0:
            raise MediaError("อ่านความยาวเสียงไม่ได้")
        return {"kind": "audio", "duration": round(duration, 3), "width": 0, "height": 0, "has_audio": True}
    raise MediaError("ไม่พบภาพหรือเสียงในไฟล์")


class MediaStore:
    def __init__(self, store, data_dir: Path, quota_mb: int = 2000, ttl_hours: float = 48):
        self.store = store
        self.root = data_dir / "media"
        self.root.mkdir(parents=True, exist_ok=True)
        self.quota = quota_mb * 2**20
        self.ttl = ttl_hours * 3600
        with store.conn() as c:
            c.executescript(MEDIA_SCHEMA)
        secret_file = data_dir / ".media_secret"
        if not secret_file.exists():
            secret_file.write_text(secrets.token_hex(32))
            secret_file.chmod(0o600)
        self._secret = secret_file.read_text().strip().encode()

    # files -----------------------------------------------------------------
    def used_bytes(self, user_id: int) -> int:
        return self.store.conn().execute("SELECT COALESCE(SUM(size),0) FROM media WHERE user_id=?", (user_id,)).fetchone()[0]

    def save_upload(self, user_id: int, name: str, content_type: str, stream, length: int) -> dict:
        ext = EXT_BY_TYPE.get(content_type.split(";")[0].strip().lower())
        if not ext:
            raise MediaError("รองรับเฉพาะวิดีโอ (MP4/MOV/WebM/MKV), เสียง (MP3/M4A/WAV/OGG) และภาพ (JPG/PNG/WebP)")
        if length <= 0:
            raise MediaError("ไม่มีข้อมูลไฟล์")
        if self.used_bytes(user_id) + length > self.quota:
            raise MediaError(f"พื้นที่เต็ม (สูงสุด {self.quota // 2**20} MB ต่อคน) ลบไฟล์เก่าก่อน")
        d = self.root / str(user_id)
        d.mkdir(parents=True, exist_ok=True)
        mid = secrets.token_hex(12)
        path = d / f"{mid}.{ext}"
        tmp = path.with_suffix(".part")
        left = length
        with open(tmp, "wb") as f:
            while left > 0:
                chunk = stream.read(min(1 << 20, left))
                if not chunk:
                    break
                f.write(chunk)
                left -= len(chunk)
        if left:
            tmp.unlink(missing_ok=True)
            raise MediaError("อัปโหลดไม่ครบ ลองใหม่อีกครั้ง")
        try:
            info = probe(tmp)
        except MediaError:
            tmp.unlink(missing_ok=True)
            raise
        tmp.replace(path)
        return self._record(user_id, mid, info, name, path, "upload")

    def add_file(self, user_id: int, path: Path, name: str, origin: str) -> dict:
        """Register a file Live AI made (an edit result)."""
        info = probe(path)
        return self._record(user_id, path.stem, info, name, path, origin)

    def _record(self, user_id, mid, info, name, path: Path, origin) -> dict:
        safe_name = "".join(ch for ch in (name or "file") if ch.isprintable() and ch not in '/\\"')[:80] or "file"
        with self.store.conn() as c:
            c.execute("INSERT INTO media(id,user_id,kind,name,path,size,duration,width,height,has_audio,origin,created) "
                      "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                      (mid, user_id, info["kind"], safe_name, str(path), path.stat().st_size, info["duration"],
                       info["width"], info["height"], int(info["has_audio"]), origin, time.time()))
        logs.event(log, "media saved", user=user_id, media=mid, kind=info["kind"], size_mb=round(path.stat().st_size / 2**20, 1),
                   seconds=info["duration"], origin=origin)
        return self.get(user_id, mid)

    def get(self, user_id: int, mid: str) -> dict:
        r = self.store.conn().execute("SELECT * FROM media WHERE id=? AND user_id=?", (mid, user_id)).fetchone()
        if not r:
            raise NotFound("media")
        return dict(r)

    def list(self, user_id: int) -> list[dict]:
        rows = self.store.conn().execute("SELECT * FROM media WHERE user_id=? ORDER BY created DESC", (user_id,)).fetchall()
        return [self.public(dict(r)) for r in rows]

    def delete(self, user_id: int, mid: str) -> None:
        m = self.get(user_id, mid)
        Path(m["path"]).unlink(missing_ok=True)
        with self.store.conn() as c:
            c.execute("DELETE FROM media WHERE id=? AND user_id=?", (mid, user_id))

    @staticmethod
    def public(m: dict) -> dict:
        return {k: m[k] for k in ("id", "kind", "name", "size", "duration", "width", "height", "has_audio", "origin", "created")}

    # signed links ----------------------------------------------------------
    def sign(self, mid: str, ttl: int = 6 * 3600) -> str:
        exp = int(time.time()) + ttl
        sig = hmac.new(self._secret, f"{mid}.{exp}".encode(), hashlib.sha256).hexdigest()[:32]
        return f"/api/media/{mid}/file?exp={exp}&sig={sig}"

    def verify(self, mid: str, exp: str, sig: str) -> dict | None:
        try:
            if int(exp) < time.time():
                return None
        except (TypeError, ValueError):
            return None
        good = hmac.new(self._secret, f"{mid}.{exp}".encode(), hashlib.sha256).hexdigest()[:32]
        if not hmac.compare_digest(good, sig or ""):
            return None
        r = self.store.conn().execute("SELECT * FROM media WHERE id=?", (mid,)).fetchone()
        return dict(r) if r else None

    # housekeeping ----------------------------------------------------------
    def cleanup(self) -> int:
        cutoff = time.time() - self.ttl
        rows = self.store.conn().execute("SELECT id, path FROM media WHERE created < ?", (cutoff,)).fetchall()
        for r in rows:
            Path(r["path"]).unlink(missing_ok=True)
        with self.store.conn() as c:
            c.execute("DELETE FROM media WHERE created < ?", (cutoff,))
        if rows:
            logs.event(log, "old media deleted", files=len(rows))
        return len(rows)

    def start_cleanup(self) -> None:
        def loop():
            while True:
                try:
                    self.cleanup()
                except (OSError, sqlite3.Error) as e:
                    logs.event(log, "cleanup failed", 40, error=str(e))
                time.sleep(3600)
        threading.Thread(target=loop, name="media-cleanup", daemon=True).start()


def disk_ok(path: Path, need_mb: int = 500) -> bool:
    return shutil.disk_usage(path).free > need_mb * 2**20
