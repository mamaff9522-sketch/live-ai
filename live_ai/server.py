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

from . import backup, logs, sysinfo
from .assistant import Assistant
from .config import Settings
from .llm import LlamaServer
from .store import NotFound, Store

log = logs.get("http")
WEB_DIR = Path(__file__).resolve().parent.parent / "web"


class App:
    def __init__(self, s: Settings, start_model: bool = True):
        self.s = s
        logs.setup(s.log_dir)
        backup.snapshot(s, "startup")              # back up before anything can change
        self.store = Store(s.db_path)
        self.llm = LlamaServer(s)
        self.assistant = Assistant(s, self.store, self.llm)
        self.cpu = sysinfo.CpuMeter()
        self.started = time.time()
        self.admin_key = os.environ.get("LIVEAI_ADMIN_KEY", "")
        self._signups: dict[str, deque] = defaultdict(deque)
        if start_model:
            self.llm.start(background=True)
        backup.start_daily(s)

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

    def _json(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        if n > 200_000:
            raise ValueError("body too large")
        raw = self.rfile.read(n) if n else b"{}"
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
        for m, rx, name in ROUTES:
            if m == method and (mt := rx.match(path)):
                started = time.time()
                try:
                    getattr(self, name)(*mt.groups())
                except NotFound:
                    self._err(404, "ไม่พบข้อมูล")
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
                        logs.event(log, "request", method=method, path=re.sub(r"/\d+", "/:id", path),
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
        events = self.app.assistant.reply(u["id"], conv, str(data.get("message", "")), bool(data.get("think")))
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

    # admin -----------------------------------------------------------------
    @route("POST", "/api/admin/backup")
    def admin_backup(self):
        key = self.headers.get("X-Admin-Key", "")
        if not self.app.admin_key or not secrets.compare_digest(key, self.app.admin_key):
            return self._err(403, "forbidden")
        p = backup.snapshot(self.app.s, "manual")
        self._send(200, {"file": p.name if p else None})


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
        threading.Thread(target=httpd.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
