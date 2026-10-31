"""Facebook Page growth: plan → write → render → post on schedule → measure → adapt.

One background thread does everything, a little at a time (tick every minute):
  plan      keep the next 36 hours filled with posts, within the owner's daily limit and posting window
  generate  about 2 hours before a post is due, Qwen writes it and ffmpeg renders the picture / Reel
  publish   when due, post it through the Graph API (one at a time, never faster than the minimum gap)
  measure   every 3 hours: follower count, Page Insights, and reactions / comments / shares / views of
            the posts from the last 14 days
  adapt     each post gets a score from its real results; the planner favours the kinds of post and
            the hours that scored best, while still trying the others now and then (20 %)
Nothing here buys ads, creates accounts, messages people or reads anyone's personal data.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import math
import os
import random
import re
import secrets
import threading
import time
from pathlib import Path

from .. import logs
from ..store import NotFound
from . import content as C
from . import trends as T
from .meta import Graph, MetaError, thai_reason

log = logs.get("growth")

SCHEMA = """
CREATE TABLE IF NOT EXISTS growth_pages (
    user_id INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    page_id TEXT NOT NULL,
    name TEXT NOT NULL DEFAULT '',
    token TEXT NOT NULL,
    info TEXT NOT NULL DEFAULT '{}',
    settings TEXT NOT NULL DEFAULT '{}',
    status TEXT NOT NULL DEFAULT 'ok',
    last_error TEXT NOT NULL DEFAULT '',
    connected REAL NOT NULL,
    marks TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS growth_items (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    kind TEXT NOT NULL,
    due REAL NOT NULL,
    status TEXT NOT NULL DEFAULT 'planned',
    content TEXT,
    media_id TEXT,
    source TEXT,
    fb_id TEXT,
    permalink TEXT,
    posted REAL,
    metrics TEXT,
    score REAL,
    attempts INTEGER NOT NULL DEFAULT 0,
    error TEXT NOT NULL DEFAULT '',
    origin TEXT NOT NULL DEFAULT 'auto',
    created REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS growth_items_user ON growth_items(user_id, due);
CREATE TABLE IF NOT EXISTS growth_stats (
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    day TEXT NOT NULL,
    followers INTEGER,
    fans INTEGER,
    insights TEXT NOT NULL DEFAULT '{}',
    updated REAL NOT NULL,
    PRIMARY KEY (user_id, day)
);
CREATE TABLE IF NOT EXISTS growth_plan (
    user_id INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    plan TEXT NOT NULL,
    created REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS growth_cache (key TEXT PRIMARY KEY, value TEXT NOT NULL, updated REAL NOT NULL);
"""

DEFAULTS = {
    "auto": True,                 # owner chose: post automatically on schedule
    "posts_per_day": 3,           # hard cap 6 — more looks like spam and hurts reach
    "min_gap_hours": 3,
    "window": [8, 22],            # local hours (Thailand) when posting is allowed
    "goal": 50000,
    "kinds": {k: True for k in C.KINDS},
    "story": "",                  # what the page's dramas are about, characters, tone (owner writes it)
    "ai_label": True,             # mark AI-made Reels as AI-generated (Meta's self-disclosure)
    "clip_media": [],             # videos allowed for clip Reels; empty = the studio's finished videos
    "tz": 7,
}
INSIGHT_METRICS = ["page_follows", "page_daily_follows_unique", "page_daily_unfollows_unique", "page_media_view",
                   "page_total_media_view_unique", "page_post_engagements", "page_views_total", "page_video_views"]
PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "audience": {"type": "array", "items": {"type": "string", "maxLength": 200}, "minItems": 2, "maxItems": 5},
        "content_pillars": {"type": "array", "items": {"type": "string", "maxLength": 200}, "minItems": 2, "maxItems": 5},
        "channels": {"type": "array", "items": {"type": "string", "maxLength": 220}, "minItems": 2, "maxItems": 6},
        "this_week": {"type": "array", "items": {"type": "string", "maxLength": 220}, "minItems": 2, "maxItems": 6},
    },
    "required": ["audience", "content_pillars", "channels", "this_week"],
}


# token sealing (the page token never sits in the database in plain text) ---------------------
class Sealer:
    def __init__(self, data_dir: Path):
        f = data_dir / ".growth_secret"
        if not f.exists():
            f.write_text(secrets.token_hex(32))
            f.chmod(0o600)
        self.key = bytes.fromhex(f.read_text().strip())

    def _stream(self, nonce: bytes, n: int) -> bytes:
        out, i = b"", 0
        while len(out) < n:
            out += hashlib.sha256(self.key + nonce + i.to_bytes(4, "big")).digest()
            i += 1
        return out[:n]

    def seal(self, text: str) -> str:
        nonce = secrets.token_bytes(16)
        raw = text.encode()
        ct = bytes(a ^ b for a, b in zip(raw, self._stream(nonce, len(raw))))
        mac = hmac.new(self.key, nonce + ct, hashlib.sha256).digest()
        return base64.b64encode(nonce + mac + ct).decode()

    def unseal(self, blob: str) -> str:
        d = base64.b64decode(blob)
        nonce, mac, ct = d[:16], d[16:48], d[48:]
        if not hmac.compare_digest(mac, hmac.new(self.key, nonce + ct, hashlib.sha256).digest()):
            raise ValueError("token damaged")
        return bytes(a ^ b for a, b in zip(ct, self._stream(nonce, len(ct)))).decode()


def similar(a: str, b: str) -> float:
    """Character-bigram overlap of two captions, hashtags ignored (0 = unrelated, 1 = identical)."""
    g = lambda s: {s[i:i + 2] for i in range(len(s) - 1)}
    norm = lambda s: re.sub(r"\s+", "", re.sub(r"#\S+", "", s))
    x, y = g(norm(a)), g(norm(b))
    return len(x & y) / max(1, len(x | y))


class Growth:
    def __init__(self, store, media, s, assistant, graph_factory=Graph):
        self.store = store
        self.media = media
        self.s = s
        self.assistant = assistant
        self.graph_factory = graph_factory
        self.sealer = Sealer(s.data_dir)
        with store.conn() as c:
            c.executescript(SCHEMA)
        self._lock = threading.Lock()
        self._wake = threading.Event()

    # helpers ------------------------------------------------------------------------
    def _page(self, user_id: int) -> dict:
        r = self.store.conn().execute("SELECT * FROM growth_pages WHERE user_id=?", (user_id,)).fetchone()
        if not r:
            raise NotFound("page")
        p = dict(r)
        p["settings"] = {**DEFAULTS, **json.loads(p["settings"] or "{}")}
        p["info"] = json.loads(p["info"] or "{}")
        p["marks"] = json.loads(p["marks"] or "{}")
        return p

    def _graph(self, page: dict) -> Graph:
        return self.graph_factory(self.sealer.unseal(page["token"]))

    def _mark(self, user_id: int, **kv) -> None:
        p = self._page(user_id)
        m = {**p["marks"], **kv}
        with self.store.conn() as c:
            c.execute("UPDATE growth_pages SET marks=? WHERE user_id=?", (json.dumps(m), user_id))

    def _set_page(self, user_id: int, **cols) -> None:
        if not cols:
            return
        with self.store.conn() as c:
            c.execute(f"UPDATE growth_pages SET {', '.join(k + '=?' for k in cols)} WHERE user_id=?", (*cols.values(), user_id))

    def _set_item(self, iid: int, **cols) -> None:
        with self.store.conn() as c:
            c.execute(f"UPDATE growth_items SET {', '.join(k + '=?' for k in cols)} WHERE id=?", (*cols.values(), iid))

    @staticmethod
    def local_hour(ts: float, tz: int) -> int:
        return int(((ts + tz * 3600) % 86400) // 3600)

    @staticmethod
    def local_day(ts: float, tz: int) -> str:
        return time.strftime("%Y-%m-%d", time.gmtime(ts + tz * 3600))

    # connecting a Page -------------------------------------------------------------------
    def connect(self, user_id: int, token: str, app_id: str = "", app_secret: str = "", page_id: str = "") -> dict:
        token = token.strip()
        if len(token) < 20:
            raise ValueError("โทเคนสั้นเกินไป")
        g = self.graph_factory(token)
        warnings = []
        if app_id and app_secret:
            g = self.graph_factory(g.exchange_long_lived(app_id.strip(), app_secret.strip()))
        try:
            pages = g.pages()
            is_user_token = True
        except MetaError as e:
            if e.token_problem:
                raise ValueError(thai_reason(e)) from None
            pages, is_user_token = [], False
        if is_user_token:
            if not pages:
                raise ValueError("บัญชีนี้ไม่มีเพจที่จัดการได้ หรือยังไม่ได้ให้สิทธิ์ pages_show_list")
            if page_id:
                pages = [p for p in pages if p["id"] == page_id] or pages
            if len(pages) > 1 and not page_id:
                return {"choose": [{"id": p["id"], "name": p.get("name", "")} for p in pages]}
            chosen = pages[0]
            page_token = chosen["access_token"]
            if not (app_id and app_secret):
                warnings.append("โทเคนนี้อาจหมดอายุใน 1–2 ชั่วโมง ใส่ App ID และ App Secret ด้วยเพื่อให้ได้โทเคนเพจที่ไม่หมดอายุ")
            tasks = set(chosen.get("tasks") or [])
            if tasks and not {"CREATE_CONTENT", "MANAGE"} & tasks:
                warnings.append("บัญชีนี้ไม่มีสิทธิ์สร้างโพสต์บนเพจ")
        else:
            page_token = token
        pg = self.graph_factory(page_token)
        try:
            info = pg.page_info("me")
        except MetaError as e:
            raise ValueError(thai_reason(e)) from None
        try:
            pg.insight(info["id"], "page_follows", period=None)
        except MetaError as e:
            if e.permission_problem or e.token_problem:
                warnings.append("อ่านสถิติเพจ (Insights) ไม่ได้ — ต้องมีสิทธิ์ read_insights ระบบจะใช้ยอดผู้ติดตามจากข้อมูลเพจแทน")
        with self.store.conn() as c:
            c.execute("INSERT INTO growth_pages(user_id,page_id,name,token,info,settings,status,last_error,connected) "
                      "VALUES (?,?,?,?,?,COALESCE((SELECT settings FROM growth_pages WHERE user_id=?),'{}'),'ok','',?) "
                      "ON CONFLICT(user_id) DO UPDATE SET page_id=excluded.page_id, name=excluded.name, token=excluded.token, "
                      "info=excluded.info, status='ok', last_error='', connected=excluded.connected",
                      (user_id, info["id"], info.get("name", ""), self.sealer.seal(page_token), json.dumps(info, ensure_ascii=False),
                       user_id, time.time()))
        logs.event(log, "page connected", user=user_id, page=info["id"], followers=info.get("followers_count"))
        self.snapshot(user_id)
        self._wake.set()
        return {"page": {"id": info["id"], "name": info.get("name"), "followers": info.get("followers_count"),
                         "category": info.get("category")}, "warnings": warnings}

    def disconnect(self, user_id: int) -> None:
        with self.store.conn() as c:
            c.execute("DELETE FROM growth_pages WHERE user_id=?", (user_id,))
            c.execute("UPDATE growth_items SET status='skipped' WHERE user_id=? AND status IN ('planned','ready')", (user_id,))

    def update_settings(self, user_id: int, new: dict) -> dict:
        p = self._page(user_id)
        st = dict(p["settings"])
        if "auto" in new:
            st["auto"] = bool(new["auto"])
        if "posts_per_day" in new:
            st["posts_per_day"] = max(1, min(6, int(new["posts_per_day"])))
        if "min_gap_hours" in new:
            st["min_gap_hours"] = max(2, min(12, float(new["min_gap_hours"])))
        if "window" in new:
            a, b = (int(x) for x in new["window"])
            if not (0 <= a < b <= 24 and b - a >= 4):
                raise ValueError("ช่วงเวลาโพสต์ต้องยาวอย่างน้อย 4 ชั่วโมง")
            st["window"] = [a, b]
        if "goal" in new:
            st["goal"] = max(100, int(new["goal"]))
        if "kinds" in new:
            st["kinds"] = {k: bool(new["kinds"].get(k, st["kinds"].get(k, True))) for k in C.KINDS}
            if not any(st["kinds"].values()):
                raise ValueError("ต้องเปิดอย่างน้อย 1 รูปแบบโพสต์")
        if "story" in new:
            st["story"] = str(new["story"])[:4000]
        if "ai_label" in new:
            st["ai_label"] = bool(new["ai_label"])
        if "clip_media" in new:
            st["clip_media"] = [str(x) for x in new["clip_media"]][:200]
        self._set_page(user_id, settings=json.dumps(st, ensure_ascii=False))
        # a changed plan should not keep posts written for the old one
        with self.store.conn() as c:
            c.execute("DELETE FROM growth_items WHERE user_id=? AND status='planned' AND origin='auto'", (user_id,))
        self._wake.set()
        return st

    # measuring ---------------------------------------------------------------------------
    def snapshot(self, user_id: int) -> None:
        p = self._page(user_id)
        g = self._graph(p)
        tz = p["settings"]["tz"]
        try:
            info = g.page_info(p["page_id"])
        except MetaError as e:
            self._fail_page(user_id, e)
            return
        ins = {}
        unavailable = set(p["marks"].get("no_metrics", []))
        until = int(time.time())
        for m in INSIGHT_METRICS:
            if m in unavailable:
                continue
            try:
                vals = g.insight(p["page_id"], m, period="day", since=until - 3 * 86400, until=until)
                if vals:
                    ins[m] = vals[-1]["value"]
            except MetaError as e:
                if e.token_problem:
                    self._fail_page(user_id, e)
                    return
                unavailable.add(m)        # deprecated or not allowed: do not ask again
        day = self.local_day(time.time(), tz)
        with self.store.conn() as c:
            c.execute("INSERT INTO growth_stats(user_id,day,followers,fans,insights,updated) VALUES (?,?,?,?,?,?) "
                      "ON CONFLICT(user_id,day) DO UPDATE SET followers=excluded.followers, fans=excluded.fans, "
                      "insights=excluded.insights, updated=excluded.updated",
                      (user_id, day, info.get("followers_count"), info.get("fan_count"), json.dumps(ins), time.time()))
            c.execute("UPDATE growth_pages SET info=?, name=? WHERE user_id=?",
                      (json.dumps(info, ensure_ascii=False), info.get("name", p["name"]), user_id))
        self._mark(user_id, no_metrics=sorted(unavailable), snapshot=time.time())

    def measure_posts(self, user_id: int) -> None:
        p = self._page(user_id)
        g = self._graph(p)
        followers = max(100, int(p["info"].get("followers_count") or 0))
        rows = self.store.conn().execute(
            "SELECT * FROM growth_items WHERE user_id=? AND status='posted' AND posted > ?",
            (user_id, time.time() - 14 * 86400)).fetchall()
        for r in rows:
            target = r["fb_id"]
            if not target:
                continue
            m = json.loads(r["metrics"] or "{}")
            try:
                m.update(g.post_counts(target))
            except MetaError as e:
                if e.token_problem:
                    self._fail_page(user_id, e)
                    return
                m["error"] = str(e)[:120]
            for metric in ("post_total_media_view_unique", "post_media_view"):
                try:
                    v = g.insight(target, metric, period="lifetime")
                    if v:
                        m["views"] = v[-1]["value"]
                        break
                except MetaError:
                    continue
            m["measured"] = time.time()
            eng = m.get("reactions", 0) + 2 * m.get("comments", 0) + 3 * m.get("shares", 0) + (m.get("views") or 0) / 50
            score = round(eng / followers * 1000, 3)
            self._set_item(r["id"], metrics=json.dumps(m), score=score,
                           permalink=r["permalink"] or m.get("permalink"))

    def _fail_page(self, user_id: int, e: MetaError) -> None:
        msg = thai_reason(e)
        status = "token" if e.token_problem else ("limited" if e.rate_limited else "error")
        self._set_page(user_id, status=status, last_error=msg)
        logs.event(log, "page problem", logging.WARNING, user=user_id, status=status, error=str(e)[:200])

    # learning from results -----------------------------------------------------------------
    def performance(self, user_id: int) -> dict:
        rows = self.store.conn().execute(
            "SELECT kind, posted, score FROM growth_items WHERE user_id=? AND status='posted' AND score IS NOT NULL",
            (user_id,)).fetchall()
        p = self._page(user_id)
        tz = p["settings"]["tz"]
        by_kind, by_hour = {}, {}
        for r in rows:
            by_kind.setdefault(r["kind"], []).append(r["score"])
            by_hour.setdefault(self.local_hour(r["posted"], tz), []).append(r["score"])
        avg = lambda xs: sum(xs) / len(xs)
        overall = avg([r["score"] for r in rows]) if rows else 0.0
        return {"overall": round(overall, 3), "posts": len(rows),
                "kinds": {k: {"n": len(v), "avg": round(avg(v), 3)} for k, v in by_kind.items()},
                "hours": {h: {"n": len(v), "avg": round(avg(v), 3)} for h, v in sorted(by_hour.items())}}

    def _choose_kind(self, user_id: int, settings: dict, perf: dict, has_clips: bool) -> str:
        kinds = [k for k in C.KINDS if settings["kinds"].get(k) and (k != "reel_clip" or has_clips)]
        if not kinds:
            kinds = ["question"]
        if random.random() < 0.2:                       # keep exploring
            return random.choice(kinds)
        prior, best, best_v = perf["overall"] or 1.0, kinds[0], -1e9
        for k in kinds:
            st = perf["kinds"].get(k, {"n": 0, "avg": prior})
            n = st["n"]
            mean = (st["avg"] * n + prior * 2) / (n + 2)        # shrink toward the page average
            sample = random.gauss(mean, max(0.05, prior) / math.sqrt(n + 1))
            if k.startswith("reel"):
                sample *= 1.15                               # Reels reach non-followers: slight preference
            if sample > best_v:
                best, best_v = k, sample
        return best

    def _hour_weights(self, settings: dict, perf: dict) -> dict[int, float]:
        a, b = settings["window"]
        prior = perf["overall"] or 1.0
        # before there is data: evenings and lunch are when Thai audiences scroll most
        base = {h: (1.4 if 19 <= h <= 21 else 1.2 if 11 <= h <= 13 else 1.0) for h in range(a, b)}
        for h in base:
            st = perf["hours"].get(h)
            if st:
                base[h] *= max(0.3, ((st["avg"] * st["n"] + prior * 2) / (st["n"] + 2)) / prior)
        return base

    # planning ------------------------------------------------------------------------------
    def clip_sources(self, user_id: int, settings: dict) -> list[dict]:
        if settings.get("clip_media"):
            out = []
            for mid in settings["clip_media"]:
                try:
                    m = self.media.get(user_id, mid)
                except NotFound:
                    continue
                if m["kind"] == "video" and Path(m["path"]).exists():
                    out.append(m)
            return out
        # default: the finished videos of the user's studio projects (the version with subtitles if there is one)
        if not self._has_table("projects"):
            return []
        out = []
        for r in self.store.conn().execute("SELECT result FROM projects WHERE user_id=? AND result IS NOT NULL "
                                           "ORDER BY updated DESC LIMIT 20", (user_id,)).fetchall():
            res = json.loads(r["result"] or "{}")
            v = res.get("video_subs") or res.get("video")
            if not v:
                continue
            try:
                m = self.media.get(user_id, v["id"])
            except NotFound:
                continue
            if m["kind"] == "video" and m["duration"] >= 15 and Path(m["path"]).exists():
                out.append(m)
        return out

    def plan(self, user_id: int, now: float | None = None, horizon_h: float = 36) -> int:
        now = now or time.time()
        p = self._page(user_id)
        st = p["settings"]
        if not st["auto"] or p["status"] == "token":
            return 0
        tz, gap = st["tz"], st["min_gap_hours"] * 3600
        perf = self.performance(user_id)
        weights = self._hour_weights(st, perf)
        has_clips = bool(self.clip_sources(user_id, st))
        rows = self.store.conn().execute(
            "SELECT due, status FROM growth_items WHERE user_id=? AND status IN ('planned','ready','posting','posted') "
            "AND due > ?", (user_id, now - 2 * 86400)).fetchall()
        taken = [r["due"] for r in rows]
        added = 0
        day0 = now - ((now + tz * 3600) % 86400)                # local midnight today (as UTC timestamp)
        for d in range(3):
            day_start = day0 + d * 86400
            if day_start > now + horizon_h * 3600:
                break
            in_day = [t for t in taken if day_start <= t < day_start + 86400]
            need = st["posts_per_day"] - len(in_day)
            tries = 0
            while need > 0 and tries < 60:
                tries += 1
                hours = list(weights)
                h = random.choices(hours, weights=[weights[x] for x in hours])[0]
                t = day_start + h * 3600 + random.randint(0, 40) * 60
                if t < now + 20 * 60 or t > now + horizon_h * 3600:
                    continue
                if any(abs(t - x) < gap for x in taken):
                    continue
                kind = self._choose_kind(user_id, st, perf, has_clips)
                with self.store.conn() as c:
                    c.execute("INSERT INTO growth_items(user_id,kind,due,status,created) VALUES (?,?,?,?,?)",
                              (user_id, kind, t, "planned", time.time()))
                taken.append(t)
                need -= 1
                added += 1
        return added

    # writing and rendering -----------------------------------------------------------------
    def _trend(self) -> str | None:
        r = self.store.conn().execute("SELECT value, updated FROM growth_cache WHERE key='trends_th'").fetchone()
        items = json.loads(r["value"]) if r else []
        if not items or random.random() > 0.35:
            return None
        return random.choice(items[:10])["title"]

    def refresh_trends(self) -> list[dict]:
        items = T.fetch()
        if items:
            with self.store.conn() as c:
                c.execute("INSERT INTO growth_cache(key,value,updated) VALUES ('trends_th',?,?) ON CONFLICT(key) DO UPDATE "
                          "SET value=excluded.value, updated=excluded.updated", (json.dumps(items, ensure_ascii=False), time.time()))
        return items

    def _story(self, user_id: int, settings: dict) -> str:
        rows = self.store.conn().execute(
            "SELECT name, script FROM projects WHERE user_id=? AND script IS NOT NULL ORDER BY updated DESC LIMIT 3",
            (user_id,)).fetchall() if self._has_table("projects") else []
        projects = [{"name": r["name"], "script": json.loads(r["script"])} for r in rows]
        return (settings.get("story", "") + "\n\n" + C.story_from_projects(projects)).strip()

    def _has_table(self, name: str) -> bool:
        return bool(self.store.conn().execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone())

    def _llm(self, msgs: list[dict]) -> dict:
        gate = self.assistant._gen
        if not gate.acquire(timeout=900):
            raise RuntimeError("โมเดลไม่ว่าง")
        try:
            if not self.assistant.server.wait_ready(300):
                raise RuntimeError("โมเดลยังไม่พร้อม")
            return self.assistant.llm.complete_json(msgs, C.SCHEMA, max_tokens=900, temperature=0.85)
        finally:
            gate.release()

    def generate(self, iid: int) -> None:
        r = dict(self.store.conn().execute("SELECT * FROM growth_items WHERE id=?", (iid,)).fetchone())
        uid = r["user_id"]
        p = self._page(uid)
        st = p["settings"]
        kind = r["kind"]
        source = {}
        if kind == "reel_clip":
            srcs = self.clip_sources(uid, st)
            if not srcs:
                kind = "reel_text"
            else:
                used = {}
                for u in self.store.conn().execute("SELECT source FROM growth_items WHERE user_id=? AND kind='reel_clip' "
                                                   "AND source IS NOT NULL", (uid,)).fetchall():
                    s = json.loads(u["source"])
                    used.setdefault(s["media_id"], []).append([s["start"], s["end"]])
                m = min(srcs, key=lambda x: (len(used.get(x["id"], [])), -x["created"]))
                a, b = C.pick_highlight(m["path"], m["duration"], want=min(45.0, max(15.0, m["duration"] * 0.3)),
                                        avoid=used.get(m["id"]))
                source = {"media_id": m["id"], "name": m["name"], "start": a, "end": b}
        best = [x["content"] for x in self.store.conn().execute(
            "SELECT content FROM growth_items WHERE user_id=? AND status='posted' AND score IS NOT NULL ORDER BY score DESC LIMIT 3",
            (uid,)).fetchall()]
        examples = [json.loads(b)["caption"] for b in best if b]
        recent = [json.loads(x["content"])["caption"] for x in self.store.conn().execute(
            "SELECT content FROM growth_items WHERE user_id=? AND content IS NOT NULL ORDER BY id DESC LIMIT 30", (uid,)).fetchall()]
        story = self._story(uid, st)
        if source:
            story += f"\n\nคลิปที่ใช้: {source['name']}"
        gen = None
        for attempt in range(2):
            g = C.clean(self._llm(C.prompt(kind, p["name"], story, examples, self._trend())), kind)
            if not any(similar(g["caption"], old) > 0.7 for old in recent):
                gen = g
                break
        if not gen:
            raise RuntimeError("เนื้อหาที่เขียนได้ซ้ำกับโพสต์เดิม")
        out_dir = self.media.root / str(uid)
        out_dir.mkdir(parents=True, exist_ok=True)
        name = os.urandom(12).hex()
        footer = p["name"] or ""
        media_id = None
        if kind in ("teaser", "quote", "question"):
            img = C.card_image(gen["lines"], out_dir / f"{name}.png", footer=footer, seed=iid)
            media_id = self.media.add_file(uid, img, f"growth-{iid}.png", "growth")["id"]
        elif kind == "reel_text":
            v = C.text_reel(gen["lines"], out_dir / f"{name}.mp4", footer=footer, seed=iid)
            media_id = self.media.add_file(uid, v, f"growth-{iid}.mp4", "growth")["id"]
        elif kind == "reel_clip":
            m = self.media.get(uid, source["media_id"])
            v = C.clip_reel(m["path"], source["start"], source["end"], gen["lines"][0], out_dir / f"{name}.mp4")
            media_id = self.media.add_file(uid, v, f"growth-{iid}.mp4", "growth")["id"]
        self._set_item(iid, kind=kind, status="ready", content=json.dumps(gen, ensure_ascii=False), media_id=media_id,
                       source=json.dumps(source, ensure_ascii=False) if source else None, error="")
        logs.event(log, "post written", user=uid, item=iid, kind=kind)

    # publishing ----------------------------------------------------------------------------
    def publish(self, iid: int, force: bool = False) -> None:
        r = dict(self.store.conn().execute("SELECT * FROM growth_items WHERE id=?", (iid,)).fetchone())
        uid = r["user_id"]
        p = self._page(uid)
        st = p["settings"]
        if not force:
            last = self.store.conn().execute("SELECT MAX(posted) FROM growth_items WHERE user_id=? AND status='posted'",
                                             (uid,)).fetchone()[0]
            if last and time.time() - last < st["min_gap_hours"] * 3600 * 0.9:
                self._set_item(iid, due=last + st["min_gap_hours"] * 3600)
                return
            today = self.local_day(time.time(), st["tz"])
            n_today = sum(1 for x in self.store.conn().execute(
                "SELECT posted FROM growth_items WHERE user_id=? AND status='posted' AND posted > ?",
                (uid, time.time() - 86400)).fetchall() if self.local_day(x["posted"], st["tz"]) == today)
            if n_today >= st["posts_per_day"]:
                self._set_item(iid, status="skipped", error="ครบจำนวนโพสต์ต่อวันแล้ว")
                return
        gen = json.loads(r["content"])
        g = self._graph(p)
        self._set_item(iid, status="posting")
        try:
            if r["kind"] == "question" and not r["media_id"]:
                fb_id = g.publish_text(p["page_id"], gen["caption"])
            elif r["kind"] in ("teaser", "quote", "question"):
                fb_id = g.publish_photo(p["page_id"], Path(self.media.get(uid, r["media_id"])["path"]), gen["caption"])
            else:
                res = g.publish_reel(p["page_id"], Path(self.media.get(uid, r["media_id"])["path"]), gen["caption"],
                                     ai_generated=st["ai_label"])
                fb_id = res.get("post_id") or res["video_id"]
        except NotFound:
            self._set_item(iid, status="planned", media_id=None, content=None, error="ไฟล์หายไป เขียนใหม่")
            return
        except MetaError as e:
            attempts = r["attempts"] + 1
            if e.token_problem or e.permission_problem:
                self._fail_page(uid, e)
                self._set_item(iid, status="ready", attempts=attempts, error=thai_reason(e))
            elif attempts < 3:
                self._set_item(iid, status="ready", attempts=attempts, due=time.time() + 1800 * attempts, error=thai_reason(e))
            else:
                self._set_item(iid, status="failed", attempts=attempts, error=thai_reason(e))
            logs.event(log, "post failed", logging.WARNING, user=uid, item=iid, error=str(e)[:200])
            return
        self._set_item(iid, status="posted", fb_id=fb_id, posted=time.time(), error="")
        if p["status"] != "ok":
            self._set_page(uid, status="ok", last_error="")
        logs.event(log, "posted", user=uid, item=iid, kind=r["kind"])

    # AI analysis and plan ---------------------------------------------------------------------
    def analyze(self, user_id: int) -> dict:
        p = self._page(user_id)
        st = p["settings"]
        perf = self.performance(user_id)
        stats = self.followers_series(user_id)[-30:]
        trends = [t["title"] for t in json.loads((self.store.conn().execute(
            "SELECT value FROM growth_cache WHERE key='trends_th'").fetchone() or {"value": "[]"})["value"])[:10]]
        top = [json.loads(x["content"])["caption"][:200] for x in self.store.conn().execute(
            "SELECT content FROM growth_items WHERE user_id=? AND status='posted' AND score IS NOT NULL ORDER BY score DESC LIMIT 3",
            (user_id,)).fetchall()]
        info = p["info"]
        data = {"page": {k: info.get(k) for k in ("name", "category", "about", "followers_count")},
                "goal": st["goal"], "followers_last_30_days": stats, "performance_by_kind": perf["kinds"],
                "performance_by_hour": perf["hours"], "top_posts": top, "thai_trends_today": trends,
                "story": self._story(user_id, st)[:1500]}
        msgs = [{"role": "system", "content": "คุณคือนักวางกลยุทธ์เพจ Facebook ละครสั้นไทย วิเคราะห์จากข้อมูลจริงที่ให้เท่านั้น "
                 "ห้ามแนะนำการซื้อผู้ติดตาม บัญชีปลอม สแปม หรือการเก็บข้อมูลส่วนตัว และห้ามแนะนำโฆษณาเสียเงิน ตอบเป็น JSON"},
                {"role": "user", "content": "ข้อมูลเพจ:\n" + json.dumps(data, ensure_ascii=False) +
                 "\n\nเขียน: audience = กลุ่มเป้าหมายที่น่าจะสนใจเพจ, content_pillars = แนวคอนเทนต์หลัก, "
                 "channels = ช่องทางโปรโมตฟรีที่เหมาะ (ทำเองได้ เช่น แชร์ Reels ไป TikTok/YouTube Shorts, ปักหมุดโพสต์, ร่วมมือกับเพจแนวเดียวกัน), "
                 "this_week = สิ่งที่ควรปรับสัปดาห์นี้จากผลลัพธ์จริง (ถ้ายังไม่มีข้อมูลให้บอกว่ากำลังเก็บข้อมูล)"}]
        gate = self.assistant._gen
        if not gate.acquire(timeout=900):
            raise RuntimeError("โมเดลไม่ว่าง")
        try:
            if not self.assistant.server.wait_ready(300):
                raise RuntimeError("โมเดลยังไม่พร้อม")
            plan = self.assistant.llm.complete_json(msgs, PLAN_SCHEMA, max_tokens=1200, temperature=0.4)
        finally:
            gate.release()
        plan["based_on"] = {"posts_measured": perf["posts"], "days_of_data": len(stats)}
        with self.store.conn() as c:
            c.execute("INSERT INTO growth_plan(user_id,plan,created) VALUES (?,?,?) ON CONFLICT(user_id) DO UPDATE "
                      "SET plan=excluded.plan, created=excluded.created", (user_id, json.dumps(plan, ensure_ascii=False), time.time()))
        return plan

    # dashboard -------------------------------------------------------------------------------
    def followers_series(self, user_id: int) -> list[dict]:
        return [dict(r) for r in self.store.conn().execute(
            "SELECT day, followers, fans FROM growth_stats WHERE user_id=? ORDER BY day", (user_id,)).fetchall()]

    def state(self, user_id: int) -> dict:
        try:
            p = self._page(user_id)
        except NotFound:
            return {"connected": False}
        st = p["settings"]
        series = self.followers_series(user_id)
        followers = p["info"].get("followers_count") or (series[-1]["followers"] if series else 0) or 0
        rate = None
        pts = [x for x in series if x["followers"] is not None]
        if len(pts) >= 2:
            recent = pts[-8:]
            d0 = time.mktime(time.strptime(recent[0]["day"], "%Y-%m-%d"))
            d1 = time.mktime(time.strptime(recent[-1]["day"], "%Y-%m-%d"))
            days = max(1.0, (d1 - d0) / 86400)
            rate = round((recent[-1]["followers"] - recent[0]["followers"]) / days, 1)
        eta = None
        if rate and rate > 0 and followers < st["goal"]:
            eta = math.ceil((st["goal"] - followers) / rate)
        items = [dict(r) for r in self.store.conn().execute(
            "SELECT * FROM growth_items WHERE user_id=? AND (status IN ('planned','ready','posting') OR due > ?) "
            "ORDER BY due DESC LIMIT 80", (user_id, time.time() - 30 * 86400)).fetchall()]
        for it in items:
            it["content"] = json.loads(it["content"]) if it["content"] else None
            it["metrics"] = json.loads(it["metrics"]) if it["metrics"] else None
            it["kind_th"] = C.KIND_TH.get(it["kind"], it["kind"])
            if it["media_id"]:
                try:
                    self.media.get(user_id, it["media_id"])
                    it["media_url"] = self.media.sign(it["media_id"])
                except NotFound:
                    it["media_id"] = None
        plan_row = self.store.conn().execute("SELECT plan, created FROM growth_plan WHERE user_id=?", (user_id,)).fetchone()
        last = series[-1] if series else None
        ins = {}
        if last:
            r = self.store.conn().execute("SELECT insights FROM growth_stats WHERE user_id=? AND day=?", (user_id, last["day"])).fetchone()
            ins = json.loads(r["insights"]) if r else {}
        trends = self.store.conn().execute("SELECT value, updated FROM growth_cache WHERE key='trends_th'").fetchone()
        return {
            "connected": True,
            "page": {"id": p["page_id"], "name": p["name"], "category": p["info"].get("category"), "link": p["info"].get("link"),
                     "status": p["status"], "last_error": p["last_error"]},
            "followers": followers, "goal": st["goal"], "progress": round(min(1.0, followers / st["goal"]), 4),
            "per_day": rate, "days_to_goal": eta, "series": series, "insights": ins,
            "unavailable_metrics": p["marks"].get("no_metrics", []),
            "settings": st, "kinds_th": C.KIND_TH, "performance": self.performance(user_id),
            "items": items, "plan": json.loads(plan_row["plan"]) if plan_row else None,
            "plan_created": plan_row["created"] if plan_row else None,
            "trends": json.loads(trends["value"])[:10] if trends else [],
            "clip_sources": [{"id": m["id"], "name": m["name"], "duration": m["duration"]} for m in self.clip_sources(user_id, st)],
            "last_snapshot": p["marks"].get("snapshot"),
        }

    # manual actions --------------------------------------------------------------------------
    def create_now(self, user_id: int, kind: str) -> int:
        self._page(user_id)
        if kind not in C.KINDS:
            raise ValueError("รูปแบบโพสต์ไม่ถูกต้อง")
        with self.store.conn() as c:
            iid = c.execute("INSERT INTO growth_items(user_id,kind,due,status,origin,created) VALUES (?,?,?,?,?,?)",
                            (user_id, kind, time.time() + 6 * 3600, "planned", "manual", time.time())).lastrowid
        threading.Thread(target=self._safe, args=(self.generate, iid), daemon=True).start()
        return iid

    def item(self, user_id: int, iid: int) -> dict:
        r = self.store.conn().execute("SELECT * FROM growth_items WHERE id=? AND user_id=?", (iid, user_id)).fetchone()
        if not r:
            raise NotFound("item")
        return dict(r)

    def skip(self, user_id: int, iid: int) -> None:
        it = self.item(user_id, iid)
        if it["status"] in ("posted", "posting"):
            raise ValueError("โพสต์ไปแล้ว ลบได้จากหน้าเพจ Facebook")
        self._set_item(iid, status="skipped")

    def post_now(self, user_id: int, iid: int) -> None:
        it = self.item(user_id, iid)
        if it["status"] != "ready":
            raise ValueError("โพสต์นี้ยังไม่พร้อม")
        threading.Thread(target=self._safe, args=(self.publish, iid, True), daemon=True).start()

    def regenerate(self, user_id: int, iid: int) -> None:
        it = self.item(user_id, iid)
        if it["status"] not in ("planned", "ready", "failed"):
            raise ValueError("แก้ไม่ได้แล้ว")
        self._set_item(iid, status="planned", content=None, media_id=None, error="")
        threading.Thread(target=self._safe, args=(self.generate, iid), daemon=True).start()

    def _safe(self, fn, *args) -> None:
        with self._lock:
            try:
                fn(*args)
            except Exception as e:
                logs.event(log, "growth step failed", logging.WARNING, step=fn.__name__, error=repr(e)[:300])
                if fn in (self.generate,) and args:
                    self._set_item(args[0], status="failed", error=f"สร้างเนื้อหาไม่สำเร็จ: {str(e)[:150]}")

    # the background loop ----------------------------------------------------------------------
    def tick(self, now: float | None = None) -> None:
        now = now or time.time()
        trends = self.store.conn().execute("SELECT updated FROM growth_cache WHERE key='trends_th'").fetchone()
        if not trends or now - trends["updated"] > 6 * 3600:
            try:
                if not self.refresh_trends():          # remember the attempt even when the feed is unreachable
                    with self.store.conn() as c:
                        c.execute("INSERT INTO growth_cache(key,value,updated) VALUES ('trends_th','[]',?) ON CONFLICT(key) "
                                  "DO UPDATE SET updated=excluded.updated", (now,))
            except Exception as e:
                logs.event(log, "trends failed", logging.WARNING, error=str(e)[:200])
        for row in self.store.conn().execute("SELECT user_id FROM growth_pages").fetchall():
            uid = row["user_id"]
            try:
                p = self._page(uid)
                if p["status"] == "token":
                    continue
                if now - p["marks"].get("snapshot", 0) > 3 * 3600:
                    self.snapshot(uid)
                    self.measure_posts(uid)
                if now - p["marks"].get("planned", 0) > 1800:
                    self.plan(uid, now)
                    self._mark(uid, planned=now)
                if p["settings"]["auto"]:
                    soon = self.store.conn().execute(
                        "SELECT id FROM growth_items WHERE user_id=? AND status='planned' AND due < ? ORDER BY due LIMIT 1",
                        (uid, now + 2 * 3600)).fetchone()
                    if soon:
                        self._safe(self.generate, soon["id"])
                    due = self.store.conn().execute(
                        "SELECT id FROM growth_items WHERE user_id=? AND status='ready' AND due <= ? ORDER BY due LIMIT 1",
                        (uid, now)).fetchone()
                    if due:
                        self._safe(self.publish, due["id"])
                if now - p["marks"].get("analyzed", 0) > 7 * 86400 and p["marks"].get("snapshot"):
                    self._mark(uid, analyzed=now)
                    self._safe(self.analyze, uid)
            except NotFound:
                continue
            except Exception as e:
                logs.event(log, "growth tick failed", logging.WARNING, user=uid, error=repr(e)[:300])

    def start(self) -> None:
        def loop():
            time.sleep(20)
            while True:
                try:
                    self.tick()
                except Exception as e:
                    logs.event(log, "growth loop error", logging.ERROR, error=repr(e)[:300])
                self._wake.wait(60)
                self._wake.clear()
        threading.Thread(target=loop, name="growth", daemon=True).start()
