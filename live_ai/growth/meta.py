"""Facebook Page access through Meta's official Graph API (the page owner's own token only).

Only what the owner authorised is used: the Page's own posts, its own aggregated Insights and
publishing to the Page. No personal data about followers is fetched or stored.

Permissions the token needs: pages_show_list, pages_read_engagement, pages_manage_posts, read_insights.
Environment (for tests / future API versions): LIVEAI_GRAPH_URL, LIVEAI_GRAPH_UPLOAD_URL, LIVEAI_GRAPH_VERSION.
"""
from __future__ import annotations

import json
import os
import secrets
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


class MetaError(RuntimeError):
    def __init__(self, message: str, code: int | None = None, status: int | None = None):
        super().__init__(message)
        self.code = code
        self.status = status

    @property
    def token_problem(self) -> bool:
        return self.code in (190, 102, 463, 467)

    @property
    def permission_problem(self) -> bool:
        return self.code in (10, 200, 3, 294) or (self.code is not None and 200 <= self.code < 300)

    @property
    def rate_limited(self) -> bool:
        return self.code in (4, 17, 32, 613, 80001)


def thai_reason(e: MetaError) -> str:
    if e.token_problem:
        return "โทเคนของเพจหมดอายุหรือถูกยกเลิก — เชื่อมต่อเพจใหม่ในหน้า เพิ่มผู้ติดตาม"
    if e.permission_problem:
        return "สิทธิ์ของโทเคนไม่พอ (ต้องมี pages_manage_posts, pages_read_engagement, read_insights)"
    if e.rate_limited:
        return "Facebook จำกัดจำนวนคำขอชั่วคราว ระบบจะลองใหม่ภายหลัง"
    return f"Facebook ตอบกลับว่า: {e}"


class Graph:
    def __init__(self, token: str, base: str | None = None, upload_base: str | None = None, version: str | None = None):
        self.token = token
        self.version = version or os.environ.get("LIVEAI_GRAPH_VERSION", "v25.0")
        self.base = (base or os.environ.get("LIVEAI_GRAPH_URL", "https://graph.facebook.com")).rstrip("/")
        self.upload_base = (upload_base or os.environ.get("LIVEAI_GRAPH_UPLOAD_URL", "https://rupload.facebook.com")).rstrip("/")

    # plumbing ---------------------------------------------------------------
    def _url(self, path: str) -> str:
        return f"{self.base}/{self.version}/{path.lstrip('/')}"

    def _call(self, method: str, path: str, params: dict | None = None, data: bytes | None = None,
              headers: dict | None = None, url: str | None = None, timeout: int = 60) -> dict:
        params = {k: v for k, v in (params or {}).items() if v is not None}
        if not url:
            url = self._url(path)
        h = {"User-Agent": "LiveAI/1.0", **(headers or {})}
        if method == "GET":
            params["access_token"] = self.token
            url += ("&" if "?" in url else "?") + urllib.parse.urlencode(params)
        elif data is None:
            params["access_token"] = self.token
            data = urllib.parse.urlencode(params).encode()
            h["Content-Type"] = "application/x-www-form-urlencoded"
        req = urllib.request.Request(url, data=data, headers=h, method=method)
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    body = r.read()
                return json.loads(body or b"{}")
            except urllib.error.HTTPError as e:
                raw = e.read()
                try:
                    err = json.loads(raw).get("error", {})
                except Exception:
                    err = {"message": raw[:200].decode("utf-8", "replace")}
                me = MetaError(err.get("message") or f"HTTP {e.code}", err.get("code"), e.code)
                if e.code >= 500 and attempt < 2:
                    time.sleep(2 + attempt * 3)
                    continue
                raise me from None
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                if attempt < 2:
                    time.sleep(2 + attempt * 3)
                    continue
                raise MetaError(f"เชื่อมต่อ Facebook ไม่ได้: {e}") from None
        raise MetaError("unreachable")

    def get(self, path: str, **params) -> dict:
        return self._call("GET", path, params)

    def post(self, path: str, **params) -> dict:
        return self._call("POST", path, params)

    # tokens and pages ---------------------------------------------------------
    def exchange_long_lived(self, app_id: str, app_secret: str) -> str:
        """Short-lived user token -> long-lived user token (page tokens made from it do not expire)."""
        r = self._call("GET", "oauth/access_token", {"grant_type": "fb_exchange_token", "client_id": app_id,
                                                     "client_secret": app_secret, "fb_exchange_token": self.token})
        if not r.get("access_token"):
            raise MetaError("แลกโทเคนระยะยาวไม่สำเร็จ")
        return r["access_token"]

    def pages(self) -> list[dict]:
        """Pages this user token manages, each with its own page token. Raises if this is a page token."""
        r = self.get("me/accounts", fields="id,name,category,access_token,tasks", limit=100)
        return r.get("data", [])

    def page_info(self, page_id: str = "me") -> dict:
        return self.get(page_id, fields="id,name,category,about,link,followers_count,fan_count")

    # reading -------------------------------------------------------------------
    def insight(self, obj_id: str, metric: str, period: str | None = "day", since: int | None = None,
                until: int | None = None) -> list[dict]:
        r = self.get(f"{obj_id}/insights", metric=metric, period=period, since=since, until=until)
        out = []
        for m in r.get("data", []):
            for v in m.get("values", []):
                out.append({"metric": m.get("name"), "end_time": v.get("end_time"), "value": v.get("value")})
        return out

    def recent_posts(self, page_id: str, limit: int = 50) -> list[dict]:
        r = self.get(f"{page_id}/posts", limit=limit,
                     fields="id,message,created_time,permalink_url,shares,"
                            "reactions.summary(total_count).limit(0),comments.summary(total_count).limit(0)")
        return r.get("data", [])

    def post_counts(self, post_id: str) -> dict:
        r = self.get(post_id, fields="id,created_time,permalink_url,shares,"
                                     "reactions.summary(total_count).limit(0),comments.summary(total_count).limit(0)")
        return {"reactions": ((r.get("reactions") or {}).get("summary") or {}).get("total_count", 0),
                "comments": ((r.get("comments") or {}).get("summary") or {}).get("total_count", 0),
                "shares": (r.get("shares") or {}).get("count", 0),
                "permalink": r.get("permalink_url")}

    # publishing ------------------------------------------------------------------
    def publish_text(self, page_id: str, message: str) -> str:
        return self.post(f"{page_id}/feed", message=message, published="true")["id"]

    def publish_photo(self, page_id: str, image: Path, caption: str) -> str:
        boundary = "----liveai" + secrets.token_hex(8)
        parts = []
        for k, v in (("caption", caption), ("published", "true"), ("access_token", self.token)):
            parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode())
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="source"; filename="{image.name}"\r\n'
                     f'Content-Type: image/{"png" if image.suffix == ".png" else "jpeg"}\r\n\r\n'.encode())
        parts.append(image.read_bytes())
        parts.append(f"\r\n--{boundary}--\r\n".encode())
        r = self._call("POST", f"{page_id}/photos", data=b"".join(parts),
                       headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}, timeout=180)
        return r.get("post_id") or r["id"]

    def publish_reel(self, page_id: str, video: Path, description: str, ai_generated: bool = False) -> dict:
        """Reels publishing: start -> upload the file -> finish (published now)."""
        start = self.post(f"{page_id}/video_reels", upload_phase="start")
        vid = start["video_id"]
        upload_url = start.get("upload_url") or f"{self.upload_base}/video-upload/{self.version}/{vid}"
        data = video.read_bytes()
        up = self._call("POST", "", url=upload_url, data=data, timeout=600,
                        headers={"Authorization": f"OAuth {self.token}", "offset": "0", "file_size": str(len(data)),
                                 "Content-Type": "application/octet-stream"})
        if not up.get("success", True):
            raise MetaError("อัปโหลดวิดีโอไป Facebook ไม่สำเร็จ")
        fin = self.post(f"{page_id}/video_reels", upload_phase="finish", video_id=vid, video_state="PUBLISHED",
                        description=description, is_ai_generated="true" if ai_generated else None)
        if fin.get("success") is False:
            raise MetaError(fin.get("message") or "Facebook ไม่รับ Reels นี้")
        return {"video_id": vid, "post_id": fin.get("post_id")}

    def video_status(self, video_id: str) -> dict:
        return self.get(video_id, fields="status,permalink_url,created_time")
