"""A small stand-in for graph.facebook.com / rupload.facebook.com used by tests/test_growth.py.

Behaves like the real API for the calls Live AI makes, including errors:
  token "BAD..."        -> error 190 (expired)          token "USER..." -> a user token (has /me/accounts)
  token "PAGE..."       -> the page token               metric page_views_total -> error 100 (deprecated)
Every request is appended to FAKE_GRAPH_LOG as JSON (path, method, params) so tests can check
exactly what was sent.
"""
import json
import os
import sys
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LOG = os.environ.get("FAKE_GRAPH_LOG")
STATE = {"followers": 1200, "posts": {}, "n": 0, "videos": {}}
LOCK = threading.Lock()
PAGE = {"id": "1001", "name": "ละครสั้นทดสอบ", "category": "Media", "about": "ละครสั้นดราม่าไทย", "link": "https://facebook.com/1001"}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        b = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def _err(self, code, msg, http=400):
        self._send(http, {"error": {"message": msg, "type": "OAuthException", "code": code}})

    def _params(self):
        u = urllib.parse.urlparse(self.path)
        q = {k: v[0] for k, v in urllib.parse.parse_qs(u.query).items()}
        body = b""
        if self.command == "POST":
            body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            ct = self.headers.get("Content-Type", "")
            if ct.startswith("application/x-www-form-urlencoded"):
                q.update({k: v[0] for k, v in urllib.parse.parse_qs(body.decode()).items()})
            elif ct.startswith("multipart/form-data"):
                text = body.decode("latin-1")
                for name in ("caption", "access_token", "published"):
                    marker = f'name="{name}"\r\n\r\n'
                    if marker in text:
                        q[name] = text.split(marker, 1)[1].split("\r\n--", 1)[0].encode("latin-1").decode("utf-8")
                q["_file_bytes"] = str(len(body))
        return u.path, q, body

    def handle_any(self):
        path, q, body = self._params()
        parts = [p for p in path.split("/") if p]
        tok = q.get("access_token") or self.headers.get("Authorization", "").replace("OAuth ", "")
        if LOG:
            with LOCK, open(LOG, "a", encoding="utf-8") as f:
                f.write(json.dumps({"method": self.command, "path": path,
                                    "params": {k: v for k, v in q.items() if k != "access_token"},
                                    "token": tok[:5]}, ensure_ascii=False) + "\n")
        if path == "/trends.rss":                                  # Google Trends RSS stand-in
            rss = ('<?xml version="1.0" encoding="UTF-8"?><rss xmlns:ht="https://trends.google.com/trending/rss" version="2.0">'
                   '<channel><title>Daily Search Trends</title>'
                   '<item><title>ละครช่อง 3</title><ht:approx_traffic>50,000+</ht:approx_traffic>'
                   '<ht:news_item><ht:news_item_title>ละครเรื่องใหม่มาแรง</ht:news_item_title></ht:news_item></item>'
                   '<item><title>ฟุตบอลไทย</title><ht:approx_traffic>20,000+</ht:approx_traffic></item>'
                   '</channel></rss>').encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/rss+xml")
            self.send_header("Content-Length", str(len(rss)))
            self.end_headers()
            self.wfile.write(rss)
            return
        if parts and parts[0] == "upload":                          # rupload: raw video bytes
            STATE["videos"][parts[-1]] = len(body)
            return self._send(200, {"success": len(body) > 1000})
        if not parts or not parts[0].startswith("v"):
            return self._err(100, "Unknown path components")
        parts = parts[1:]
        if tok.startswith("BAD"):
            return self._err(190, "Error validating access token: Session has expired")
        if parts == ["oauth", "access_token"]:
            if q.get("client_secret") != "secret123":
                return self._err(101, "Error validating client secret.")
            return self._send(200, {"access_token": "USER-LONG-LIVED", "token_type": "bearer"})
        if parts == ["me", "accounts"]:
            if not tok.startswith("USER"):
                return self._err(100, "(#100) Tried accessing nonexisting field (accounts) on node type (Page)")
            pages = [{**PAGE, "access_token": "PAGE-" + tok, "tasks": ["ADVERTISE", "ANALYZE", "CREATE_CONTENT", "MODERATE", "MANAGE"]}]
            if "TWO" in tok:
                pages.append({"id": "2002", "name": "อีกเพจ", "access_token": "PAGE-2", "tasks": ["ANALYZE"]})
            return self._send(200, {"data": pages})
        if not tok.startswith("PAGE"):
            return self._err(200, "(#200) Requires pages_read_engagement permission")
        node = parts[0]
        if node in ("me", PAGE["id"]) and len(parts) == 1:
            with LOCK:
                STATE["followers"] += 7                              # the page grows a little each look
                return self._send(200, {**PAGE, "followers_count": STATE["followers"], "fan_count": STATE["followers"] - 50})
        if len(parts) == 2 and parts[1] == "insights":
            metric = q.get("metric", "")
            if metric in ("page_views_total", "page_video_views") or metric.startswith("page_impressions"):
                return self._err(100, f"(#100) The value must be a valid insights metric: {metric}")
            val = 40 if node == PAGE["id"] else 300
            return self._send(200, {"data": [{"name": metric, "period": q.get("period") or "lifetime",
                                              "values": [{"value": val, "end_time": "2026-11-01T07:00:00+0000"}]}]})
        if node == PAGE["id"] and parts[1:] == ["feed"] and self.command == "POST":
            if not q.get("message"):
                return self._err(100, "(#100) Missing message")
            return self._new_post("text", q["message"])
        if node == PAGE["id"] and parts[1:] == ["photos"] and self.command == "POST":
            if int(q.get("_file_bytes", "0")) < 2000:
                return self._err(324, "(#324) Requires upload file")
            r = self._new_post("photo", q.get("caption", ""))
            return r
        if node == PAGE["id"] and parts[1:] == ["video_reels"] and self.command == "POST":
            if q.get("upload_phase") == "start":
                vid = f"9{len(STATE['videos']) + 100}"
                STATE["videos"][vid] = 0
                host = self.headers.get("Host")
                return self._send(200, {"video_id": vid, "upload_url": f"http://{host}/upload/v25.0/{vid}"})
            if q.get("upload_phase") == "finish":
                vid = q.get("video_id")
                if not STATE["videos"].get(vid):
                    return self._err(6000, "There was a problem uploading your video file")
                with LOCK:
                    STATE["n"] += 1
                    pid = f"{PAGE['id']}_{700 + STATE['n']}"
                    STATE["posts"][pid] = {"kind": "reel", "text": q.get("description", ""), "ai": q.get("is_ai_generated")}
                return self._send(200, {"success": True, "post_id": pid})
        if node in STATE["posts"] and len(parts) == 1:
            p = STATE["posts"][node]
            boost = 3 if p["kind"] == "reel" else 1
            return self._send(200, {"id": node, "created_time": "2026-11-01T07:00:00+0000",
                                    "permalink_url": f"https://facebook.com/{node}",
                                    "reactions": {"data": [], "summary": {"total_count": 10 * boost}},
                                    "comments": {"data": [], "summary": {"total_count": 4 * boost}},
                                    "shares": {"count": 2 * boost}})
        return self._err(100, f"Unsupported request {self.command} {path}")

    def _new_post(self, kind, text):
        with LOCK:
            STATE["n"] += 1
            pid = f"{PAGE['id']}_{500 + STATE['n']}"
            STATE["posts"][pid] = {"kind": kind, "text": text}
        return self._send(200, {"id": pid} if kind == "text" else {"id": str(800 + STATE["n"]), "post_id": pid})

    do_GET = handle_any
    do_POST = handle_any


def start(port: int = 0) -> ThreadingHTTPServer:
    s = ThreadingHTTPServer(("127.0.0.1", port), H)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    return s


if __name__ == "__main__":
    srv = start(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
    print(srv.server_address[1], flush=True)
    threading.Event().wait()
