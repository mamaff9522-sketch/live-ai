"""HTTP API + web page. Standard library only.

Auth: every /api call except /api/health and /api/register needs
"Authorization: Bearer <token>". The token is shown once at sign-up; only its
SHA-256 is stored.
"""
from __future__ import annotations

import json
import logging
import os
import re
import secrets
import signal
import threading
import time
from collections import defaultdict, deque
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from urllib.parse import parse_qs, unquote, urlparse

from . import backup, logs, persist, sysinfo
from .assistant import Assistant
from .editor import EditError, Editor
from .media import MediaError, MediaStore
from .studio.pipeline import Studio
from .growth.engine import Growth
from .growth.meta import MetaError, thai_reason
from .config import Settings
from .llm import LlamaServer
from .store import NotFound, Store

log = logs.get("http")
WEB_DIR = Path(__file__).resolve().parent.parent / "web"


class App:
    def __init__(self, s: Settings, start_model: bool = True):
        self.s = s
        logs.setup(s.log_dir)
        persist.restore(s.db_path)                 # free hosts: bring back the saved database first
        backup.snapshot(s, "startup")              # back up before anything can change
        self.store = Store(s.db_path)
        self.llm = LlamaServer(s)
        self.assistant = Assistant(s, self.store, self.llm)
        self.media = MediaStore(self.store, s.data_dir, quota_mb=int(os.environ.get("LIVEAI_MEDIA_QUOTA_MB", "3000")),
                                ttl_hours=float(os.environ.get("LIVEAI_MEDIA_TTL_HOURS", "72")))
        self.media.start_cleanup()
        self.editor = Editor(self.store, self.media, self.assistant)
        self.studio = Studio(self.store, self.media, s, self.assistant)
        self.growth = Growth(self.store, self.media, s, self.assistant)
        if os.environ.get("LIVEAI_GROWTH_LOOP", "1") == "1":
            self.growth.start()
        self.max_upload = int(os.environ.get("LIVEAI_MAX_UPLOAD_MB", "500")) * 2**20
        self.cpu = sysinfo.CpuMeter()
        self.started = time.time()
        self.admin_key = os.environ.get("LIVEAI_ADMIN_KEY", "")
        self._signups: dict[str, deque] = defaultdict(deque)
        if start_model:
            self.llm.start(background=True)
        backup.start_daily(s)
        persist.start(s.db_path)

    def health(self) -> dict:
        st = self.llm.status()
        pid = st.get("pid")
        return {
            "ok": st["state"] == "ready",
            "model": st,
            "uptime_s": int(time.time() - self.started),
            "threads": self.s.threads,
            "ctx": self.s.ctx_size,
            "ram_mb": {"model": sysinfo.rss_mb(pid), "app": sysinfo.rss_mb(os.getpid()), **sysinfo.memory_mb()},
            "cpu_percent": {"model": self.cpu.percent(pid), "cores": os.cpu_count()},
            "disk_free_mb": sysinfo.disk_free_mb(self.s.data_dir),
        }

    def signup_allowed(self, ip: str) -> bool:
        q = self._signups[ip]
        now = time.time()
        while q and now - q[0] > 3600:
            q.popleft()
        if len(q) >= int(os.environ.get("LIVEAI_SIGNUPS_PER_HOUR", "10")):
            return False
        q.append(now)
        return True


ROUTES: list[tuple[str, re.Pattern, str]] = []


def route(method: str, pattern: str):
    def deco(fn):
        ROUTES.append((method, re.compile("^" + pattern + "$"), fn.__name__))
        return fn
    return deco


class Handler(BaseHTTPRequestHandler):
    app: App = None  # set by serve()
    server_version = "LiveAI/1.0"
    protocol_version = "HTTP/1.1"

    # plumbing --------------------------------------------------------------
    def log_message(self, fmt, *args):  # quiet default access log; we log our own
        pass

    def _ip(self) -> str:
        fwd = self.headers.get("X-Forwarded-For", "")
        return fwd.split(",")[0].strip() or self.client_address[0]

    def _send(self, code: int, obj=None, ctype="application/json; charset=utf-8", body: bytes | None = None):
        if body is None:
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _err(self, code: int, msg: str):
        self._send(code, {"error": msg})

    def _body(self, limit: int) -> bytes:
        """The JSON request body (read once; _dispatch may already have read a small one)."""
        if getattr(self, "_pre", None) is not None:
            return self._pre
        n = int(self.headers.get("Content-Length") or 0)
        if n > limit:
            raise ValueError("body too large")
        self._pre = self.rfile.read(n) if n else b""
        return self._pre

    def _json(self) -> dict:
        raw = self._body(200_000)
        data = json.loads(raw or b"{}")
        if not isinstance(data, dict):
            raise ValueError("expected a JSON object")
        return data

    def _user(self) -> dict | None:
        auth = self.headers.get("Authorization", "")
        token = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
        return self.app.store.user_by_token(token)

    def _dispatch(self, method: str):
        path = self.path.split("?", 1)[0]
        self._pre = None
        # A small JSON body is read up front even when the handler ignores it (e.g. POST .../start with
        # "{}"): closing a connection with unread request bytes makes the OS reset it, and the client
        # can lose the response. Uploads (video, ZIP) are not JSON and are still streamed by their handler.
        n = int(self.headers.get("Content-Length") or 0)
        if 0 < n <= 65536 and self.headers.get("Content-Type", "").startswith("application/json"):
            self._pre = self.rfile.read(n)
        for m, rx, name in ROUTES:
            if m == method and (mt := rx.match(path)):
                started = time.time()
                try:
                    getattr(self, name)(*mt.groups())
                except NotFound:
                    self._err(404, "ไม่พบข้อมูล")
                except (MediaError, EditError) as e:
                    self._err(400, str(e))
                except MetaError as e:
                    self._err(400, thai_reason(e))
                except (ValueError, json.JSONDecodeError) as e:
                    self._err(400, f"คำขอไม่ถูกต้อง: {e}")
                except (BrokenPipeError, ConnectionResetError):
                    pass
                except Exception as e:
                    logs.event(log, "server error", logging.ERROR, path=path, error=repr(e)[:300])
                    try:
                        self._err(500, "ระบบขัดข้อง")
                    except Exception:
                        pass
                finally:
                    if path != "/api/health":
                        logs.event(log, "request", method=method, path=re.sub(r"/(\d+|[a-f0-9]{24})(?=/|$)", "/:id", path),
                                   ms=int((time.time() - started) * 1000))
                return
        self._err(404, "not found")

    def do_GET(self):
        self._dispatch("GET")

    def do_POST(self):
        self._dispatch("POST")

    def do_DELETE(self):
        self._dispatch("DELETE")

    def do_PATCH(self):
        self._dispatch("PATCH")

    def _auth(self) -> dict | None:
        u = self._user()
        if not u:
            self._err(401, "ต้องเข้าสู่ระบบ")
        return u

    # pages -----------------------------------------------------------------
    @route("GET", "/")
    def page(self):
        self._send(200, ctype="text/html; charset=utf-8", body=(WEB_DIR / "index.html").read_bytes())

    @route("GET", "/api/health")
    def health(self):
        h = self.app.health()
        self._send(200 if h["ok"] else 503, h)

    # users -----------------------------------------------------------------
    @route("POST", "/api/register")
    def register(self):
        data = self._json()
        key = self.headers.get("X-Admin-Key", "")
        is_admin = bool(self.app.admin_key) and secrets.compare_digest(key, self.app.admin_key)
        if not (self.app.s.open_registration or is_admin):
            return self._err(403, "ปิดรับสมัครผู้ใช้ใหม่")
        if not is_admin and not self.app.signup_allowed(self._ip()):
            return self._err(429, "สมัครบ่อยเกินไป ลองใหม่ภายหลัง")
        uid, token = self.app.store.create_user(str(data.get("name", "")))
        logs.event(log, "user created", user=uid)
        self._send(201, {"user_id": uid, "token": token})

    @route("GET", "/api/me")
    def me(self):
        if (u := self._auth()):
            self._send(200, u)

    # conversations ---------------------------------------------------------
    @route("GET", "/api/conversations")
    def conv_list(self):
        if (u := self._auth()):
            self._send(200, self.app.store.list_conversations(u["id"]))

    @route("POST", "/api/conversations")
    def conv_new(self):
        if (u := self._auth()):
            cid = self.app.store.create_conversation(u["id"], str(self._json().get("title", "")))
            self._send(201, {"id": cid})

    @route("PATCH", r"/api/conversations/(\d+)")
    def conv_rename(self, cid):
        if (u := self._auth()):
            self.app.store.rename_conversation(u["id"], int(cid), str(self._json().get("title", "")))
            self._send(200, {"ok": True})

    @route("DELETE", r"/api/conversations/(\d+)")
    def conv_delete(self, cid):
        if (u := self._auth()):
            self.app.store.delete_conversation(u["id"], int(cid))
            self._send(200, {"ok": True})

    @route("GET", r"/api/conversations/(\d+)/messages")
    def conv_messages(self, cid):
        if (u := self._auth()):
            self._send(200, self.app.store.messages(u["id"], int(cid)))

    # memories --------------------------------------------------------------
    @route("GET", "/api/memories")
    def mem_list(self):
        if (u := self._auth()):
            self._send(200, self.app.store.memories(u["id"]))

    @route("POST", "/api/memories")
    def mem_add(self):
        if (u := self._auth()):
            content = str(self._json().get("content", "")).strip()
            if not content:
                return self._err(400, "ข้อความว่าง")
            self._send(201, {"id": self.app.store.add_memory(u["id"], content)})

    @route("DELETE", r"/api/memories/(\d+)")
    def mem_delete(self, mid):
        if (u := self._auth()):
            self.app.store.delete_memory(u["id"], int(mid))
            self._send(200, {"ok": True})

    # chat ------------------------------------------------------------------
    @route("POST", "/api/chat")
    def chat(self):
        u = self._auth()
        if not u:
            return
        data = self._json()
        conv = data.get("conversation_id")
        conv = int(conv) if conv not in (None, "") else None
        if conv is not None:
            self.app.store.get_conversation(u["id"], conv)  # 404 before streaming starts
        events = self.app.assistant.reply(u["id"], conv, str(data.get("message", "")), bool(data.get("think")),
                                          voice=bool(data.get("voice")))
        if not data.get("stream", True):
            out, final = [], {}
            for ev in events:
                if ev["type"] == "token":
                    out.append(ev["text"])
                elif ev["type"] == "meta":
                    final["conversation_id"] = ev["conversation_id"]
                elif ev["type"] == "done":
                    final["stats"] = ev["stats"]
                elif ev["type"] == "error":
                    return self._send(503, {"error": ev["message"], **final})
            final["reply"] = "".join(out).strip()
            return self._send(200, final)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        try:
            for ev in events:
                self.wfile.write(f"data: {json.dumps(ev, ensure_ascii=False)}\n\n".encode("utf-8"))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            events.close()

    # media (uploads for the video editor) -----------------------------------
    @route("POST", "/api/media")
    def media_upload(self):
        u = self._auth()
        if not u:
            return
        n = int(self.headers.get("Content-Length") or 0)
        if n > self.app.max_upload:
            return self._err(413, f"ไฟล์ใหญ่เกิน {self.app.max_upload // 2**20} MB")
        name = unquote(self.headers.get("X-Filename", "file"))
        m = self.app.media.save_upload(u["id"], name, self.headers.get("Content-Type", ""), self.rfile, n)
        self._send(201, self.app.media.public(m))

    @route("GET", "/api/media")
    def media_list(self):
        if (u := self._auth()):
            self._send(200, self.app.media.list(u["id"]))

    @route("DELETE", r"/api/media/([a-f0-9]{24})")
    def media_delete(self, mid):
        if (u := self._auth()):
            self.app.media.delete(u["id"], mid)
            self._send(200, {"ok": True})

    @route("GET", r"/api/media/([a-f0-9]{24})/url")
    def media_url(self, mid):
        if (u := self._auth()):
            self.app.media.get(u["id"], mid)
            self._send(200, {"url": self.app.media.sign(mid)})

    @route("GET", r"/api/media/([a-f0-9]{24})/file")
    def media_file(self, mid):
        qs = parse_qs(urlparse(self.path).query)
        m = self.app.media.verify(mid, (qs.get("exp") or [""])[0], (qs.get("sig") or [""])[0])
        if not m or not Path(m["path"]).is_file():
            return self._err(404, "ไม่พบไฟล์หรือลิงก์หมดอายุ")
        ctype = {"video": "video/mp4", "audio": "audio/mpeg", "image": "image/jpeg", "text": "text/plain; charset=utf-8"}[m["kind"]]
        ext = Path(m["path"]).suffix.lower()
        ctype = {".mp4": "video/mp4", ".mov": "video/quicktime", ".webm": "video/webm", ".mkv": "video/x-matroska",
                 ".m4a": "audio/mp4", ".mp3": "audio/mpeg", ".wav": "audio/wav", ".ogg": "audio/ogg", ".png": "image/png",
                 ".webp": "image/webp", ".jpg": "image/jpeg", ".srt": "application/x-subrip; charset=utf-8"}.get(ext, ctype)
        self._send_file(Path(m["path"]), ctype, download=m["name"] if "download" in qs else None)

    def _send_file(self, path: Path, ctype: str, download: str | None = None):
        size = path.stat().st_size
        start, end, code = 0, size - 1, 200
        rng = self.headers.get("Range", "")
        if rng.startswith("bytes="):
            a, _, b = rng[6:].split(",")[0].partition("-")
            try:
                if a:
                    start, end = int(a), (int(b) if b else size - 1)
                else:
                    start = size - int(b)
                end = min(end, size - 1)
                if start < 0 or start > end:
                    raise ValueError
                code = 206
            except ValueError:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Cache-Control", "private, max-age=3600")
        if code == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        if download:
            from urllib.parse import quote
            self.send_header("Content-Disposition", f"attachment; filename*=UTF-8''{quote(download)}")
        self.end_headers()
        with open(path, "rb") as f:
            f.seek(start)
            left = end - start + 1
            while left > 0:
                chunk = f.read(min(1 << 20, left))
                if not chunk:
                    break
                self.wfile.write(chunk)
                left -= len(chunk)

    # quick edits (chat-style instruction) -----------------------------------
    @route("POST", "/api/edits")
    def edit_create(self):
        u = self._auth()
        if not u:
            return
        d = self._json()
        conv = d.get("conversation_id")
        base = d.get("base_edit")
        e = self.app.editor.create(u["id"], int(conv) if conv not in (None, "") else None, str(d.get("instruction", "")),
                                   [str(x) for x in d.get("media_ids") or []], int(base) if base not in (None, "") else None)
        e.pop("media_ids_raw", None)
        self._send(201, e)

    @route("GET", r"/api/edits/(\d+)")
    def edit_get(self, eid):
        if (u := self._auth()):
            self._send(200, self.app.editor.view(u["id"], int(eid)))

    # editing studio (script + many clips -> one long video) --------------------
    @route("GET", "/studio")
    def studio_page(self):
        self._send(200, ctype="text/html; charset=utf-8", body=(WEB_DIR / "studio.html").read_bytes())

    @route("GET", "/api/studio/projects")
    def st_list(self):
        if (u := self._auth()):
            self._send(200, self.app.studio.list(u["id"]))

    @route("POST", "/api/studio/projects")
    def st_create(self):
        if (u := self._auth()):
            self._send(201, self.app.studio.create(u["id"], str(self._json().get("name", ""))))

    @route("GET", r"/api/studio/projects/(\d+)")
    def st_get(self, pid):
        if (u := self._auth()):
            self._send(200, self.app.studio.get(u["id"], int(pid)))

    @route("PATCH", r"/api/studio/projects/(\d+)")
    def st_update(self, pid):
        if (u := self._auth()):
            d = self._json_big()
            self._send(200, self.app.studio.update(u["id"], int(pid), d.get("name"), d.get("script_text"), d.get("settings")))

    @route("DELETE", r"/api/studio/projects/(\d+)")
    def st_delete(self, pid):
        if (u := self._auth()):
            self.app.studio.delete(u["id"], int(pid))
            self._send(200, {"ok": True})

    @route("POST", r"/api/studio/projects/(\d+)/clips")
    def st_add_clips(self, pid):
        if (u := self._auth()):
            self._send(200, self.app.studio.add_clips(u["id"], int(pid), [str(x) for x in self._json().get("media_ids") or []]))

    @route("DELETE", r"/api/studio/projects/(\d+)/clips/([a-f0-9]{24})")
    def st_remove_clip(self, pid, mid):
        if (u := self._auth()):
            self.app.studio.remove_clip(u["id"], int(pid), mid)
            self._send(200, {"ok": True})

    @route("POST", r"/api/studio/projects/(\d+)/zip")
    def st_zip(self, pid):
        u = self._auth()
        if not u:
            return
        n = int(self.headers.get("Content-Length") or 0)
        if n > self.app.max_upload * 4:
            return self._err(413, f"ZIP ใหญ่เกิน {self.app.max_upload * 4 // 2**20} MB")
        self._send(200, self.app.studio.add_zip(u["id"], int(pid), self.rfile, n))

    @route("POST", r"/api/studio/projects/(\d+)/(start|pause|resume)")
    def st_control(self, pid, action):
        if (u := self._auth()):
            f = self.app.studio.pause if action == "pause" else self.app.studio.start
            self._send(200, f(u["id"], int(pid)))

    def _json_big(self) -> dict:
        try:
            raw = self._body(6_000_000)
        except ValueError:
            raise ValueError("ข้อมูลใหญ่เกินไป") from None
        data = json.loads(raw or b"{}")
        if not isinstance(data, dict):
            raise ValueError("expected a JSON object")
        return data

    # admin -----------------------------------------------------------------
    @route("POST", "/api/admin/backup")
    def admin_backup(self):
        key = self.headers.get("X-Admin-Key", "")
        if not self.app.admin_key or not secrets.compare_digest(key, self.app.admin_key):
            return self._err(403, "forbidden")
        p = backup.snapshot(self.app.s, "manual")
        saved = persist.save(self.app.s.db_path, "manual")
        self._send(200, {"file": p.name if p else None, "saved_remote": saved})

    # Facebook Page growth --------------------------------------------------------------
    @route("GET", "/privacy")
    def privacy_page(self):
        self._send(200, ctype="text/html; charset=utf-8", body=(WEB_DIR / "privacy.html").read_bytes())

    @route("GET", "/growth")
    def growth_page(self):
        self._send(200, ctype="text/html; charset=utf-8", body=(WEB_DIR / "growth.html").read_bytes())

    @route("GET", "/api/growth")
    def growth_state(self):
        if (u := self._auth()):
            self._send(200, self.app.growth.state(u["id"]))

    @route("POST", "/api/growth/connect")
    def growth_connect(self):
        if (u := self._auth()):
            d = self._json()
            self._send(200, self.app.growth.connect(u["id"], str(d.get("token", "")), str(d.get("app_id", "")),
                                                    str(d.get("app_secret", "")), str(d.get("page_id", ""))))

    @route("DELETE", "/api/growth/connect")
    def growth_disconnect(self):
        if (u := self._auth()):
            self.app.growth.disconnect(u["id"])
            self._send(200, {"ok": True})

    @route("PATCH", "/api/growth/settings")
    def growth_settings(self):
        if (u := self._auth()):
            self._send(200, self.app.growth.update_settings(u["id"], self._json()))

    @route("POST", "/api/growth/analyze")
    def growth_analyze(self):
        if (u := self._auth()):
            try:
                self._send(200, self.app.growth.analyze(u["id"]))
            except RuntimeError as e:
                self._err(503, str(e))

    @route("POST", "/api/growth/refresh")
    def growth_refresh(self):
        if (u := self._auth()):
            self.app.growth.snapshot(u["id"])
            self.app.growth.measure_posts(u["id"])
            self._send(200, {"ok": True})

    @route("POST", "/api/growth/items")
    def growth_new_item(self):
        if (u := self._auth()):
            self._send(201, {"id": self.app.growth.create_now(u["id"], str(self._json().get("kind", "")))})

    @route("POST", r"/api/growth/items/(\d+)/(post|regenerate|skip)")
    def growth_item_action(self, iid, action):
        if (u := self._auth()):
            {"post": self.app.growth.post_now, "regenerate": self.app.growth.regenerate,
             "skip": self.app.growth.skip}[action](u["id"], int(iid))
            self._send(200, {"ok": True})


def serve(s: Settings | None = None, start_model: bool = True) -> tuple[ThreadingHTTPServer, App]:
    s = s or Settings()
    app = App(s, start_model=start_model)
    Handler.app = app
    httpd = ThreadingHTTPServer((s.host, s.port), Handler)
    httpd.daemon_threads = True
    logs.event(log, "Live AI listening", host=s.host, port=s.port, threads=s.threads, ctx=s.ctx_size)
    return httpd, app


def main() -> None:
    httpd, app = serve()

    def shutdown(*_):
        logs.event(log, "shutting down")
        app.llm.stop()
        persist.save(app.s.db_path, "shutdown")
        threading.Thread(target=httpd.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
