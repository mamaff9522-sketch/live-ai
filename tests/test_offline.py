"""Offline tests: the whole app with a fake llama-server (no model download).

Run:  python -m unittest tests.test_offline -v
"""
import json
import os
import signal
import socket
import stat
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


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Live(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cls.llm_log = cls.tmp / "llm_requests.jsonl"
        wrapper = cls.tmp / "fake-llama-server"
        wrapper.write_text(f"#!/bin/sh\nexec {sys.executable} {ROOT/'tests'/'fake_llama_server.py'} \"$@\"\n")
        wrapper.chmod(wrapper.stat().st_mode | stat.S_IEXEC)
        model = cls.tmp / "fake.gguf"
        model.write_bytes(b"GGUF")
        cls.port = free_port()
        os.environ.update({
            "LIVEAI_DATA_DIR": str(cls.tmp / "data"), "LIVEAI_PORT": str(cls.port), "LIVEAI_HOST": "127.0.0.1",
            "LIVEAI_MODEL_PATH": str(model), "LIVEAI_LLAMA_SERVER": str(wrapper), "LIVEAI_LLM_PORT": str(free_port()),
            "LIVEAI_CTX": "2048", "LIVEAI_MAX_REPLY_TOKENS": "256", "LIVEAI_ADMIN_KEY": "test-admin",
            "FAKE_LLM_LOG": str(cls.llm_log), "LIVEAI_SIGNUPS_PER_HOUR": "1000",
        })
        from live_ai.config import Settings
        from live_ai.server import serve
        cls.httpd, cls.app = serve(Settings())
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()
        cls.base = f"http://127.0.0.1:{cls.port}"
        for _ in range(100):
            if cls.app.llm.state == "ready":
                break
            time.sleep(0.1)

    @classmethod
    def tearDownClass(cls):
        cls.app.llm.stop()
        cls.httpd.shutdown()

    # helpers ---------------------------------------------------------------
    def call(self, method, path, body=None, token=None, headers=None):
        req = urllib.request.Request(self.base + path, method=method,
                                     data=json.dumps(body).encode() if body is not None else None,
                                     headers={"Content-Type": "application/json", **({"Authorization": f"Bearer {token}"} if token else {}), **(headers or {})})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                raw = r.read().decode()
                return r.status, (json.loads(raw) if r.headers.get_content_type() == "application/json" else raw)
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read() or b"{}")

    def user(self, name="u"):
        code, j = self.call("POST", "/api/register", {"name": name})
        self.assertEqual(code, 201)
        return j["token"]

    def chat(self, token, text, conv=None, think=False):
        code, j = self.call("POST", "/api/chat", {"message": text, "conversation_id": conv, "stream": False, "think": think}, token)
        return code, j

    def last_llm_request(self):
        return json.loads(self.llm_log.read_text(encoding="utf-8").strip().splitlines()[-1])

    # tests -----------------------------------------------------------------
    def test_01_health(self):
        code, h = self.call("GET", "/api/health")
        self.assertEqual(code, 200)
        self.assertTrue(h["ok"])
        self.assertEqual(h["model"]["state"], "ready")
        self.assertGreater(h["ram_mb"]["model"], 0)

    def test_02_page(self):
        code, html = self.call("GET", "/")
        self.assertEqual(code, 200)
        self.assertIn("Live AI", html)

    def test_03_auth_required(self):
        for m, p in [("GET", "/api/conversations"), ("GET", "/api/memories"), ("POST", "/api/chat")]:
            code, _ = self.call(m, p, {} if m == "POST" else None)
            self.assertEqual(code, 401, p)
        code, _ = self.call("GET", "/api/conversations", token="lai_wrong")
        self.assertEqual(code, 401)

    def test_04_chat_saves_history(self):
        t = self.user()
        code, j = self.chat(t, "สวัสดีครับ")
        self.assertEqual(code, 200, j)
        self.assertEqual(j["reply"], "ตอบ: สวัสดีครับ")
        cid = j["conversation_id"]
        code, j2 = self.chat(t, "ข้อความที่สอง", cid)
        self.assertEqual(code, 200)
        req = self.last_llm_request()
        self.assertEqual([m["role"] for m in req["messages"]], ["system", "user", "assistant", "user"])
        self.assertEqual(req["messages"][1]["content"], "สวัสดีครับ")
        code, msgs = self.call("GET", f"/api/conversations/{cid}/messages", token=t)
        self.assertEqual([m["content"] for m in msgs], ["สวัสดีครับ", "ตอบ: สวัสดีครับ", "ข้อความที่สอง", "ตอบ: ข้อความที่สอง"])

    def test_05_users_are_isolated(self):
        a, b = self.user("A"), self.user("B")
        _, ja = self.chat(a, "จำไว้ว่า รหัสลับของเอคือ 4826")
        cid = ja["conversation_id"]
        # B can't see, read, write to, rename or delete A's conversation
        self.assertEqual(self.call("GET", "/api/conversations", token=b)[1], [])
        self.assertEqual(self.call("GET", f"/api/conversations/{cid}/messages", token=b)[0], 404)
        self.assertEqual(self.chat(b, "แอบเขียน", cid)[0], 404)
        self.assertEqual(self.call("PATCH", f"/api/conversations/{cid}", {"title": "x"}, b)[0], 404)
        self.assertEqual(self.call("DELETE", f"/api/conversations/{cid}", token=b)[0], 404)
        # memories are per user, and B's prompt never contains A's memory
        self.assertEqual(self.call("GET", "/api/memories", token=b)[1], [])
        mem_a = self.call("GET", "/api/memories", token=a)[1]
        self.assertEqual([m["content"] for m in mem_a], ["รหัสลับของเอคือ 4826"])
        self.assertEqual(self.call("DELETE", f"/api/memories/{mem_a[0]['id']}", token=b)[0], 404)
        self.chat(b, "รหัสลับของเอคืออะไร")
        sent = json.dumps(self.last_llm_request(), ensure_ascii=False)
        self.assertNotIn("4826", sent)
        # A still has everything
        self.assertEqual(len(self.call("GET", f"/api/conversations/{cid}/messages", token=a)[1]), 2)

    def test_06_memory_used_in_new_conversation(self):
        t = self.user()
        self.chat(t, "จำไว้ว่า ปิยะชอบสีม่วง")
        self.chat(t, "ปิยะชอบสีอะไร")  # new conversation
        system = self.last_llm_request()["messages"][0]["content"]
        self.assertIn("ปิยะชอบสีม่วง", system)

    def test_07_forget_is_strict(self):
        t = self.user()
        self.chat(t, "จำไว้ว่า ฉันแพ้กุ้ง")
        self.chat(t, "ลืมรหัสผ่านอีเมลทำยังไงดี")          # a question, must not delete anything
        self.assertEqual(len(self.call("GET", "/api/memories", token=t)[1]), 1)
        self.chat(t, "ลืมเรื่องแพ้กุ้งไปเลยนะ")
        self.assertEqual(self.call("GET", "/api/memories", token=t)[1], [])

    def test_08_streaming(self):
        t = self.user()
        req = urllib.request.Request(self.base + "/api/chat", data=json.dumps({"message": "ทดสอบสตรีม", "think": True}).encode(),
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {t}"})
        with urllib.request.urlopen(req, timeout=30) as r:
            events = [json.loads(l[5:]) for l in r.read().decode().split("\n\n") if l.startswith("data:")]
        types = [e["type"] for e in events]
        self.assertEqual(types[0], "meta")
        self.assertIn("thinking", types)
        self.assertEqual(types[-1], "done")
        self.assertEqual("".join(e["text"] for e in events if e["type"] == "token"), "ตอบ: ทดสอบสตรีม")
        self.assertTrue(self.last_llm_request()["chat_template_kwargs"]["enable_thinking"])

    def test_09_history_trimmed_to_context(self):
        t = self.user()
        _, j = self.chat(t, "เริ่ม")
        cid = j["conversation_id"]
        for i in range(30):
            self.chat(t, f"ข้อความยาว {i} " + "ก" * 300, cid)
        req = self.last_llm_request()
        total = sum(len(m["content"]) for m in req["messages"])
        self.assertLess(len(req["messages"]), 62)
        self.assertLess(total, 2048 * 2)
        self.assertEqual(req["messages"][1]["role"], "user")

    def test_10_context_overflow_retries_with_less_history(self):
        t = self.user()
        _, j = self.chat(t, "ข้อความแรก")
        code, j2 = self.chat(t, "TOO_LONG กรุณาตอบ", j["conversation_id"])
        self.assertEqual(code, 200, j2)
        self.assertEqual(len(self.last_llm_request()["messages"]), 2)

    def test_11_one_reply_at_a_time_per_user(self):
        t = self.user()
        results = []
        th = threading.Thread(target=lambda: results.append(self.chat(t, "SLOW ข้อความช้า")))
        th.start()
        time.sleep(0.3)
        code, j = self.chat(t, "ส่งซ้อน")
        th.join()
        self.assertEqual(code, 503)
        self.assertIn("ก่อนหน้า", j["error"])
        self.assertEqual(results[0][0], 200)

    def test_12_recovers_when_model_dies(self):
        llm = self.app.llm
        before = llm.restarts
        os.killpg(llm.proc.pid, signal.SIGKILL)
        t = self.user()
        code, j = self.chat(t, "ยังอยู่ไหม")      # waits for the watchdog restart
        self.assertEqual(code, 200, j)
        self.assertEqual(j["reply"], "ตอบ: ยังอยู่ไหม")
        self.assertEqual(llm.restarts, before + 1)
        log = (self.tmp / "data" / "logs" / "live_ai.log").read_text(encoding="utf-8")
        self.assertIn("model stopped, restarting", log)
        self.assertIn("model recovered", log)

    def test_13_logs_have_no_message_text(self):
        t = self.user()
        self.chat(t, "ความลับสุดยอด ABCXYZ")
        log = (self.tmp / "data" / "logs" / "live_ai.log").read_text(encoding="utf-8")
        self.assertNotIn("ABCXYZ", log)
        self.assertNotIn(t, log)
        for line in log.splitlines():
            json.loads(line)

    def test_14_backup_and_restore(self):
        from live_ai import backup
        s = self.app.s
        self.assertTrue(any("startup" in f.name for f in backup.list_backups(s)) or s.db_path.exists())
        code, j = self.call("POST", "/api/admin/backup", {}, headers={"X-Admin-Key": "test-admin"})
        self.assertEqual(code, 200)
        self.assertEqual(self.call("POST", "/api/admin/backup", {}, headers={"X-Admin-Key": "nope"})[0], 403)
        snap = s.data_dir / "backups" / j["file"]
        import sqlite3
        n = sqlite3.connect(snap).execute("SELECT COUNT(*) FROM users").fetchone()[0]
        self.assertGreater(n, 0)

    def test_15_bad_input(self):
        t = self.user()
        self.assertEqual(self.chat(t, "")[0], 503)
        self.assertEqual(self.chat(t, "x" * 9000)[0], 503)
        self.assertEqual(self.chat(t, "hi", 999999)[0], 404)
        req = urllib.request.Request(self.base + "/api/chat", data=b"not json", headers={"Authorization": f"Bearer {t}"})
        with self.assertRaises(urllib.error.HTTPError) as cm:
            urllib.request.urlopen(req)
        self.assertEqual(cm.exception.code, 400)


class RestoreTest(unittest.TestCase):
    def test_restore_roundtrip(self):
        tmp = Path(tempfile.mkdtemp())
        os.environ["LIVEAI_DATA_DIR"] = str(tmp)
        from live_ai.config import Settings
        from live_ai.store import Store
        from live_ai import backup
        s = Settings()
        st = Store(s.db_path)
        uid, _ = st.create_user("before")
        snap = backup.snapshot(s, "test")
        st.create_user("after")
        st.conn().close()
        st._local.c = None
        backup.restore(s, snap)
        st2 = Store(s.db_path)
        self.assertEqual(st2.count_users(), 1)
        # the restore itself was backed up first, so it can be undone
        self.assertTrue(any("before-restore" in f.name for f in backup.list_backups(s)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
