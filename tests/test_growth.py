"""Facebook Page growth, end to end against a stand-in Graph API (tests/fake_graph_server.py):
connect → plan → write (fake model) → render (real ffmpeg) → publish → measure → adapt → dashboard.

Run:  python -m unittest tests.test_growth -v
"""
import json
import os
import random
import stat
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from test_offline import free_port  # noqa: E402
import fake_graph_server  # noqa: E402


class Growth(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cls.graph_log = cls.tmp / "graph.jsonl"
        os.environ["FAKE_GRAPH_LOG"] = str(cls.graph_log)
        fake_graph_server.LOG = str(cls.graph_log)
        cls.graph = fake_graph_server.start()
        gport = cls.graph.server_address[1]
        wrapper = cls.tmp / "fake-llama-server"
        wrapper.write_text(f"#!/bin/sh\nexec {sys.executable} {ROOT/'tests'/'fake_llama_server.py'} \"$@\"\n")
        wrapper.chmod(wrapper.stat().st_mode | stat.S_IEXEC)
        model = cls.tmp / "fake.gguf"
        model.write_bytes(b"GGUF")
        cls.port = free_port()
        os.environ.update({
            "FAKE_LLM_LOG": str(cls.tmp / "llm.jsonl"),
            "LIVEAI_DATA_DIR": str(cls.tmp / "data"), "LIVEAI_PORT": str(cls.port), "LIVEAI_HOST": "127.0.0.1",
            "LIVEAI_MODEL_PATH": str(model), "LIVEAI_LLAMA_SERVER": str(wrapper), "LIVEAI_LLM_PORT": str(free_port()),
            "LIVEAI_CTX": "2048", "LIVEAI_MAX_REPLY_TOKENS": "256", "LIVEAI_SIGNUPS_PER_HOUR": "1000",
            "LIVEAI_GRAPH_URL": f"http://127.0.0.1:{gport}", "LIVEAI_GRAPH_UPLOAD_URL": f"http://127.0.0.1:{gport}",
            "LIVEAI_TRENDS_URLS": f"http://127.0.0.1:{gport}/trends.rss", "LIVEAI_GROWTH_LOOP": "0",
        })
        import live_ai.growth.trends as T
        T.FEEDS = [os.environ["LIVEAI_TRENDS_URLS"]]
        from live_ai.config import Settings
        from live_ai.server import serve
        cls.httpd, cls.app = serve(Settings())
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()
        cls.base = f"http://127.0.0.1:{cls.port}"
        for _ in range(100):
            if cls.app.llm.state == "ready":
                break
            time.sleep(0.1)
        cls.g = cls.app.growth

    @classmethod
    def tearDownClass(cls):
        cls.app.llm.stop()
        cls.httpd.shutdown()
        cls.graph.shutdown()

    def call(self, method, path, body=None, token=None):
        req = urllib.request.Request(self.base + path, method=method, data=json.dumps(body).encode() if body is not None else None,
                                     headers={"Content-Type": "application/json", **({"Authorization": f"Bearer {token}"} if token else {})})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                raw = r.read()
                return r.status, (json.loads(raw) if r.headers.get_content_type() == "application/json" else raw)
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read() or b"{}")

    def user(self, name="u"):
        code, j = self.call("POST", "/api/register", {"name": name})
        self.assertEqual(code, 201)
        return j["token"], j["user_id"]

    def graph_calls(self):
        return [json.loads(x) for x in self.graph_log.read_text(encoding="utf-8").splitlines()] if self.graph_log.exists() else []

    def connected_user(self, name):
        tok, uid = self.user(name)
        code, j = self.call("POST", "/api/growth/connect", {"token": "USER-short-xxxxxxxxxxxxxxxx", "app_id": "42", "app_secret": "secret123"}, tok)
        self.assertEqual(code, 200, j)
        return tok, uid

    # tests ----------------------------------------------------------------------------------
    def test_01_connect_exchanges_token_and_seals_it(self):
        tok, uid = self.connected_user("owner")
        row = self.app.store.conn().execute("SELECT token, page_id FROM growth_pages WHERE user_id=?", (uid,)).fetchone()
        self.assertEqual(row["page_id"], "1001")
        self.assertNotIn("PAGE", row["token"])                       # never stored in plain text
        self.assertEqual(self.g.sealer.unseal(row["token"]), "PAGE-USER-LONG-LIVED")
        code, s = self.call("GET", "/api/growth", token=tok)
        self.assertTrue(s["connected"])
        self.assertGreater(s["followers"], 1000)
        self.assertEqual(s["goal"], 50000)
        self.assertGreater(s["progress"], 0)
        self.assertNotIn("secret123", self.app.store.conn().execute("SELECT * FROM growth_pages").fetchone()["settings"])

    def test_02_connect_errors_in_thai(self):
        tok, _ = self.user("e")
        code, j = self.call("POST", "/api/growth/connect", {"token": "BAD-token-xxxxxxxxxxxxxxxx"}, tok)
        self.assertEqual(code, 400)
        self.assertIn("หมดอายุ", j["error"])
        code, j = self.call("POST", "/api/growth/connect", {"token": "USER-short-xxxxxxxxxxxxxx", "app_id": "1", "app_secret": "nope"}, tok)
        self.assertEqual(code, 400)

    def test_03_several_pages_asks_which(self):
        tok, _ = self.user("multi")
        code, j = self.call("POST", "/api/growth/connect", {"token": "USER-TWO-xxxxxxxxxxxxxxxxx"}, tok)
        self.assertEqual(code, 200)
        self.assertEqual({p["id"] for p in j["choose"]}, {"1001", "2002"})
        code, j = self.call("POST", "/api/growth/connect", {"token": "USER-TWO-xxxxxxxxxxxxxxxxx", "page_id": "1001"}, tok)
        self.assertEqual(j["page"]["id"], "1001")
        self.assertTrue(any("หมดอายุ" in w for w in j["warnings"]))   # no app secret: warned about short token

    def test_04_plan_respects_daily_cap_gap_and_window(self):
        tok, uid = self.connected_user("planner")
        self.call("PATCH", "/api/growth/settings", {"posts_per_day": 3, "min_gap_hours": 3, "window": [9, 22]}, tok)
        now = time.time()
        self.g.plan(uid, now)
        rows = self.app.store.conn().execute("SELECT due, kind FROM growth_items WHERE user_id=? ORDER BY due", (uid,)).fetchall()
        self.assertGreaterEqual(len(rows), 3)
        days = {}
        for r in rows:
            h = self.g.local_hour(r["due"], 7)
            self.assertTrue(9 <= h < 22, h)
            self.assertGreater(r["due"], now)
            days.setdefault(self.g.local_day(r["due"], 7), []).append(r["due"])
        for d, ts in days.items():
            self.assertLessEqual(len(ts), 3, d)
        ts = [r["due"] for r in rows]
        self.assertTrue(all(b - a >= 3 * 3600 for a, b in zip(ts, ts[1:])))
        n = len(rows)
        self.g.plan(uid, now)                                         # planning again adds nothing new
        self.assertEqual(n, self.app.store.conn().execute("SELECT COUNT(*) FROM growth_items WHERE user_id=?", (uid,)).fetchone()[0])

    def _video(self, seconds=40):
        p = self.tmp / f"drama{random.randint(0, 99999)}.mp4"
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", f"testsrc2=s=1280x720:d={seconds}:r=25",
                        "-f", "lavfi", "-i", f"sine=f=300:d={seconds}", "-af", "volume='if(between(mod(t,6),1,4),1,0.001)':eval=frame",
                        "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", "-shortest", str(p)], check=True)
        return p

    def _upload(self, tok, path):
        req = urllib.request.Request(self.base + "/api/media", data=path.read_bytes(), method="POST",
                                     headers={"Authorization": f"Bearer {tok}", "Content-Type": "video/mp4", "X-Filename": "drama.mp4"})
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.load(r)

    def test_05_every_kind_written_rendered_and_published(self):
        from live_ai.media import probe
        tok, uid = self.connected_user("maker")
        m = self._upload(tok, self._video())
        code, st = self.call("PATCH", "/api/growth/settings", {"clip_media": [m["id"]], "story": "เรื่อง ไฟแค้นซ่อนรัก น้ำฟ้า กับ ภาคิน"}, tok)
        self.assertEqual(code, 200, st)
        before = len(self.graph_calls())
        sizes = {"question": (1080, 1350), "teaser": (1080, 1350), "quote": (1080, 1350), "reel_text": (720, 1280), "reel_clip": (720, 1280)}
        for kind, (w, h) in sizes.items():
            with self.app.store.conn() as c:
                iid = c.execute("INSERT INTO growth_items(user_id,kind,due,status,created) VALUES (?,?,?,?,?)",
                                (uid, kind, time.time(), "planned", time.time())).lastrowid
            self.g.generate(iid)
            it = self.g.item(uid, iid)
            self.assertEqual(it["status"], "ready", (kind, it["error"]))
            self.assertEqual(it["kind"], kind)
            content = json.loads(it["content"])
            self.assertIn("กดติดตาม", content["caption"])
            f = self.app.media.get(uid, it["media_id"])
            info = probe(Path(f["path"]))
            self.assertEqual((info["width"], info["height"]), (w, h), kind)
            if kind.startswith("reel"):
                self.assertTrue(3 <= info["duration"] <= 90, info["duration"])
            if kind == "reel_clip":
                src = json.loads(it["source"])
                self.assertEqual(src["media_id"], m["id"])
                self.assertLessEqual(src["end"] - src["start"], 45.1)
            self.g.publish(iid, force=True)
            it = self.g.item(uid, iid)
            self.assertEqual(it["status"], "posted", (kind, it["error"]))
            self.assertTrue(it["fb_id"].startswith("1001_"))
        calls = self.graph_calls()[before:]
        posts = [c for c in calls if c["method"] == "POST"]
        paths = [c["path"].split("/", 2)[-1] for c in posts]
        self.assertIn("1001/photos", paths)
        self.assertEqual(paths.count("1001/video_reels"), 4)          # start + finish for two Reels
        self.assertEqual(sum(1 for c in posts if c["path"].startswith("/upload/")), 2)
        fin = [c for c in posts if c["params"].get("upload_phase") == "finish"]
        self.assertTrue(all(c["params"].get("is_ai_generated") == "true" for c in fin))
        self.assertTrue(all(c["params"].get("video_state") == "PUBLISHED" for c in fin))
        # nothing except the owner's own page and its own posts was touched
        nodes = {c["path"].split("/")[2] for c in self.graph_calls() if c["path"].startswith("/v")}
        self.assertTrue(all(n in ("me", "oauth", "1001") or n.startswith("1001_") for n in nodes), nodes)

    def test_06_measure_score_and_adapt(self):
        tok, uid = self.connected_user("learner")
        for kind in ("question", "reel_text", "question", "reel_text"):
            with self.app.store.conn() as c:
                iid = c.execute("INSERT INTO growth_items(user_id,kind,due,status,created) VALUES (?,?,?,?,?)",
                                (uid, kind, time.time(), "planned", time.time())).lastrowid
            self.g.generate(iid)
            self.g.publish(iid, force=True)
        self.g.measure_posts(uid)
        perf = self.g.performance(uid)
        self.assertEqual(perf["posts"], 4)
        self.assertGreater(perf["kinds"]["reel_text"]["avg"], perf["kinds"]["question"]["avg"])   # Reels did better (fake data)
        st = self.g._page(uid)["settings"]
        random.seed(1)
        picks = [self.g._choose_kind(uid, st, perf, False) for _ in range(400)]
        self.assertGreater(picks.count("reel_text"), picks.count("question"))
        self.assertGreater(picks.count("question"), 0)                    # still explores
        code, s = self.call("GET", "/api/growth", token=tok)
        posted = [i for i in s["items"] if i["status"] == "posted"]
        self.assertTrue(all(i["metrics"]["reactions"] > 0 and i["score"] is not None for i in posted))
        self.assertTrue(all(i["permalink"] for i in posted))

    def test_07_snapshot_insights_and_deprecated_metrics(self):
        tok, uid = self.connected_user("stats")
        self.g.snapshot(uid)
        p = self.g._page(uid)
        self.assertIn("page_views_total", p["marks"]["no_metrics"])        # deprecated metric remembered, not retried
        row = self.app.store.conn().execute("SELECT * FROM growth_stats WHERE user_id=?", (uid,)).fetchone()
        self.assertGreater(row["followers"], 1000)
        self.assertEqual(json.loads(row["insights"]).get("page_follows"), 40)
        n = sum(1 for c in self.graph_calls() if "page_views_total" in json.dumps(c))
        self.g.snapshot(uid)
        self.assertEqual(n, sum(1 for c in self.graph_calls() if "page_views_total" in json.dumps(c)))

    def test_08_anti_spam_gap_and_daily_cap(self):
        tok, uid = self.connected_user("spam")
        self.call("PATCH", "/api/growth/settings", {"posts_per_day": 2, "min_gap_hours": 3}, tok)
        ids = []
        for _ in range(3):
            with self.app.store.conn() as c:
                ids.append(c.execute("INSERT INTO growth_items(user_id,kind,due,status,created) VALUES (?,?,?,?,?)",
                                     (uid, "question", time.time(), "planned", time.time())).lastrowid)
            self.g.generate(ids[-1])
        self.g.publish(ids[0])
        self.assertEqual(self.g.item(uid, ids[0])["status"], "posted")
        self.g.publish(ids[1])                                           # too soon after the first
        it = self.g.item(uid, ids[1])
        self.assertEqual(it["status"], "ready")
        self.assertGreater(it["due"], time.time() + 2.5 * 3600)
        with self.app.store.conn() as c:                                 # pretend the first was 4 h ago, a second one posted
            c.execute("UPDATE growth_items SET posted=? WHERE id=?", (time.time() - 4 * 3600, ids[0]))
            c.execute("UPDATE growth_items SET status='posted', posted=?, fb_id='x' WHERE id=?", (time.time() - 3.5 * 3600, ids[1]))
        self.g.publish(ids[2])
        it = self.g.item(uid, ids[2])
        if self.g.local_day(time.time() - 4 * 3600, 7) == self.g.local_day(time.time(), 7):
            self.assertEqual(it["status"], "skipped")                     # third post on the same day: over the cap of 2

    def test_09_expired_token_stops_posting(self):
        tok, uid = self.connected_user("expired")
        with self.app.store.conn() as c:
            c.execute("UPDATE growth_pages SET token=? WHERE user_id=?", (self.g.sealer.seal("BAD-expired"), uid))
            iid = c.execute("INSERT INTO growth_items(user_id,kind,due,status,content,created) VALUES (?,?,?,?,?,?)",
                            (uid, "question", time.time() - 10, "ready", json.dumps({"caption": "x", "lines": ["x"], "hashtags": []}),
                             time.time())).lastrowid
        self.g.publish(iid, force=True)
        p = self.g._page(uid)
        self.assertEqual(p["status"], "token")
        self.assertIn("หมดอายุ", p["last_error"])
        before = len(self.graph_calls())
        self.g.tick()
        self.assertFalse(any(c["token"].startswith("BAD") and c["method"] == "POST" for c in self.graph_calls()[before:]))

    def test_10_users_are_isolated(self):
        tok_a, uid_a = self.connected_user("iso-a")
        tok_b, _ = self.user("iso-b")
        with self.app.store.conn() as c:
            iid = c.execute("INSERT INTO growth_items(user_id,kind,due,status,created) VALUES (?,?,?,?,?)",
                            (uid_a, "question", time.time() + 9999, "planned", time.time())).lastrowid
        code, s = self.call("GET", "/api/growth", token=tok_b)
        self.assertFalse(s["connected"])
        code, _ = self.call("POST", f"/api/growth/items/{iid}/skip", token=tok_b)
        self.assertEqual(code, 404)
        code, _ = self.call("GET", "/api/growth")
        self.assertEqual(code, 401)

    def test_11_trends_dashboard_and_analysis(self):
        tok, uid = self.connected_user("dash")
        items = self.g.refresh_trends()
        self.assertEqual(items[0]["title"], "ละครช่อง 3")
        self.assertEqual(items[0]["traffic"], 50000)
        code, plan = self.call("POST", "/api/growth/analyze", token=tok)
        self.assertEqual(code, 200, plan)
        self.assertTrue(plan["audience"] and plan["channels"])
        code, s = self.call("GET", "/api/growth", token=tok)
        self.assertEqual(s["plan"]["audience"], plan["audience"])
        self.assertEqual(s["trends"][0]["title"], "ละครช่อง 3")
        code, html = self.call("GET", "/growth")
        self.assertEqual(code, 200)
        self.assertIn("เพิ่มผู้ติดตามเพจ", html.decode())
        # the model was told not to suggest paid ads, fake followers or spam
        prompts = [json.loads(x) for x in Path(os.environ["FAKE_LLM_LOG"]).read_text(encoding="utf-8").splitlines()]
        system = [m["content"] for b in prompts for m in b["messages"] if m["role"] == "system"]
        self.assertTrue(any("ห้ามแนะนำการซื้อผู้ติดตาม" in x and "โฆษณาเสียเงิน" in x for x in system))
        self.assertTrue(any("ห้ามข้อความแบบสแปม" in x for x in system))

    def test_12_tick_runs_the_whole_cycle(self):
        tok, uid = self.connected_user("cycle")
        with self.app.store.conn() as c:
            c.execute("DELETE FROM growth_items WHERE user_id=?", (uid,))
            c.execute("UPDATE growth_pages SET marks='{}' WHERE user_id=?", (uid,))
        self.g.tick()                                                   # snapshot + plan + write the first post
        rows = self.app.store.conn().execute("SELECT * FROM growth_items WHERE user_id=?", (uid,)).fetchall()
        self.assertGreater(len(rows), 0)
        with self.app.store.conn() as c:                                # make the earliest one due now
            first = min(rows, key=lambda r: r["due"])
            c.execute("UPDATE growth_items SET due=? WHERE id=?", (time.time() - 5, first["id"]))
        self.g.tick()
        self.g.tick()
        it = self.g.item(uid, first["id"])
        self.assertEqual(it["status"], "posted", it["error"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
