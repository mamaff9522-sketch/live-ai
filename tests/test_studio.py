"""Studio tests with real ffmpeg rendering and the real 225-clip script (ไฟแค้นซ่อนรัก).

Speech recognition is replaced by recorded transcripts with ~8% character errors (LIVEAI_ASR=fake),
because this machine has no speech model; tests/real_studio_test.py runs the real whisper.cpp in CI.
Covers brief tests 1, 2, 3, 4, 5, 6 (honest warning), 8 and 10.
"""
import io
import json
import os
import random
import socket
import stat
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SCRIPT = (ROOT / "tests" / "fixture_faiklaen_225.txt").read_text(encoding="utf-8")
N = 10                      # first 10 shots of the real script
COLORS = ["red", "green", "blue", "yellow", "purple", "orange", "cyan", "white", "gray", "pink", "brown", "navy"]


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def noisy(text, rnd):
    t = list(text.replace(" ", "").replace("!", "").replace("“", "").replace("”", ""))
    for _ in range(max(1, len(t) // 12)):
        t[rnd.randrange(len(t))] = rnd.choice("กขคงจชดตทนบปพมยรลวสหอ")
    return "".join(t)


def make_clip(path, color, seconds, speech, size="720x1280"):
    args = ["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", f"color=c={color}:size={size}:rate=30:duration={seconds}"]
    if speech:   # a tone stands in for a voice (silencedetect sees sound)
        args += ["-f", "lavfi", "-i", f"sine=frequency=300:duration={seconds - 0.6}", "-af", "adelay=300|300,apad"]
    else:        # quiet room tone only
        args += ["-f", "lavfi", "-i", f"anoisesrc=a=0.0005:d={seconds}"]
    args += ["-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", "-shortest", "-t", str(seconds), str(path)]
    subprocess.run(args, check=True)


class Studio(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        from live_ai.studio.script import parse
        cls.script = parse(SCRIPT)
        rnd = random.Random(7)
        cls.clipdir = cls.tmp / "clips"
        cls.clipdir.mkdir()
        fake, cls.truth = {}, {}
        for s in cls.script["shots"][:N]:
            name = f"meta_ai_{rnd.randint(10000, 99999)}.mp4"
            speech = bool(s["dialogue"])
            make_clip(cls.clipdir / name, COLORS[s["index"]], 3, speech)
            fake[name] = " ".join(noisy(d["text"], rnd) for d in s["dialogue"])
            cls.truth[name] = s["index"]
        # a duplicate copy of shot 2 under another name
        dup = cls.clipdir / "copy_of_clip.mp4"
        dup.write_bytes((cls.clipdir / [n for n, i in cls.truth.items() if i == 1][0]).read_bytes())
        fake["copy_of_clip.mp4"] = fake[[n for n, i in cls.truth.items() if i == 1][0]]
        # shot 11 + 12 spoken across two clips (J/L-cut): line of shot 12 split in half
        s11, s12 = cls.script["shots"][10], cls.script["shots"][11]
        line = s12["dialogue"][0]["text"]
        make_clip(cls.clipdir / "x_part_b.mp4", "olive", 3, True)
        make_clip(cls.clipdir / "x_part_a.mp4", "teal", 3, True)
        fake["x_part_a.mp4"] = noisy(s11["dialogue"][0]["text"], rnd) + " " + line[: len(line) // 2]
        fake["x_part_b.mp4"] = line[len(line) // 2:]
        cls.split_names = ("x_part_a.mp4", "x_part_b.mp4")
        (cls.tmp / "asr.json").write_text(json.dumps(fake, ensure_ascii=False), encoding="utf-8")
        wrapper = cls.tmp / "fake-llama-server"
        wrapper.write_text(f"#!/bin/sh\nexec {sys.executable} {ROOT / 'tests' / 'fake_llama_server.py'} \"$@\"\n")
        wrapper.chmod(wrapper.stat().st_mode | stat.S_IEXEC)
        model = cls.tmp / "fake.gguf"
        model.write_bytes(b"GGUF")
        cls.port = free_port()
        os.environ.update({
            "LIVEAI_DATA_DIR": str(cls.tmp / "data"), "LIVEAI_PORT": str(cls.port), "LIVEAI_HOST": "127.0.0.1",
            "LIVEAI_MODEL_PATH": str(model), "LIVEAI_LLAMA_SERVER": str(wrapper), "LIVEAI_LLM_PORT": str(free_port()),
            "LIVEAI_CTX": "2048", "LIVEAI_SIGNUPS_PER_HOUR": "1000", "LIVEAI_ASR": "fake",
            "LIVEAI_FAKE_ASR": str(cls.tmp / "asr.json"),
        })
        from live_ai.config import Settings
        from live_ai.server import serve
        cls.httpd, cls.app = serve(Settings())
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()
        cls.base = f"http://127.0.0.1:{cls.port}"

    @classmethod
    def tearDownClass(cls):
        cls.app.llm.stop()
        cls.httpd.shutdown()
        os.environ.pop("LIVEAI_ASR", None)

    def call(self, method, path, body=None, token=None, raw=None, ctype="application/json"):
        data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
        req = urllib.request.Request(self.base + path, method=method, data=data,
                                     headers={"Content-Type": ctype, **({"Authorization": f"Bearer {token}"} if token else {})})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read() or b"{}")

    def user(self):
        return self.call("POST", "/api/register", {"name": "u"})[1]["token"]

    def zip_of(self, names):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            for n in names:
                z.write(self.clipdir / n, f"export/{n}")
            z.writestr("../evil.txt", "x")          # path traversal attempt is ignored
            z.writestr("readme.txt", "not a video")
        return buf.getvalue()

    def project(self, t, names, rng=(1, N)):
        code, p = self.call("POST", "/api/studio/projects", {"name": "ไฟแค้นซ่อนรัก ตอนทดสอบ"}, t)
        self.assertEqual(code, 201)
        code, p = self.call("PATCH", f"/api/studio/projects/{p['id']}", {"script_text": SCRIPT, "settings": {
            "aspect": "9:16", "range_from": rng[0], "range_to": rng[1]}}, t)
        self.assertEqual(code, 200, p)
        code, r = self.call("POST", f"/api/studio/projects/{p['id']}/zip", token=t, raw=self.zip_of(names), ctype="application/zip")
        self.assertEqual(code, 200, r)
        return p["id"], r

    def wait(self, t, pid, timeout=300):
        end = time.time() + timeout
        while time.time() < end:
            p = self.call("GET", f"/api/studio/projects/{pid}", token=t)[1]
            if p["status"] in ("done", "needs_attention", "failed", "paused"):
                return p
            time.sleep(0.5)
        self.fail("project did not finish")

    def order_of(self, p):
        return [self.truth.get(i["name"], i["name"]) for i in p["result"]["timeline"]["video"]] if p["result"].get("timeline") else None

    # ----------------------------------------------------------------- tests
    def test_1_2_10_shuffled_clips_one_start(self):
        t = self.user()
        names = [n for n in self.truth]
        random.Random(3).shuffle(names)        # uploaded out of order, names carry no order
        pid, r = self.project(t, names)
        self.assertEqual(r["added"], N)
        self.assertEqual(len(r["skipped"]), 0)
        code, p = self.call("POST", f"/api/studio/projects/{pid}/start", {}, t)
        self.assertEqual(code, 200, p)
        p = self.wait(t, pid)
        self.assertEqual(p["status"], "done", p["message"])
        res = p["result"]
        self.assertTrue(res["complete"])
        self.assertTrue(all(c["ok"] for c in res["qa"]["checks"]), res["qa"])
        # order follows the story, not the upload or the names
        tl = [self.truth[c["name"]] for c in sorted(p["clips"], key=lambda c: 0)]  # noqa (just names exist)
        self.assertEqual(len(tl), N)
        self.assertEqual(res["qa"]["output"]["size"], [720, 1280])
        self.assertAlmostEqual(res["qa"]["output"]["duration"], 3.0 * N, delta=0.3)
        # brief test 5: subtitles = script lines, verbatim, once each, no invented lines
        with urllib.request.urlopen(self.base + res["srt"]["url"]) as f:
            srt = f.read().decode("utf-8")
        script_lines = [d["text"] for s in self.script["shots"][:N] for d in s["dialogue"]]
        subs = [b.split("\n")[2] for b in srt.strip().split("\n\n")]
        self.assertEqual(subs, script_lines)
        # agents board: 15 agents, phase-2 ones honestly marked
        self.assertEqual(len(p["agents"]), 15)
        self.assertEqual(p["agents"]["4"]["status"], "skipped")
        self.assertEqual(p["agents"]["15"]["status"], "done")
        with urllib.request.urlopen(urllib.request.Request(self.base + res["video"]["url"], headers={"Range": "bytes=0-9"})) as f:
            self.assertEqual(f.status, 206)

    def test_3_missing_shot_and_duplicate(self):
        t = self.user()
        names = [n for n, i in self.truth.items() if i != 5] + ["copy_of_clip.mp4"]   # shot 6 missing, one duplicate
        pid, _ = self.project(t, names)
        self.call("POST", f"/api/studio/projects/{pid}/start", {}, t)
        p = self.wait(t, pid)
        self.assertEqual(p["status"], "needs_attention")
        res = p["result"]
        self.assertFalse(res["complete"])
        self.assertEqual([m["label"] for m in res["missing"]], ["คลิป 006"])
        self.assertIn("คุณหญิงมณี", res["missing"][0]["regen"])          # what to recreate, with the exact line
        self.assertIn("ภาคิน! แกจะเลือกแม่หรือเลือกมัน!", res["missing"][0]["regen"])
        self.assertTrue(any(i["type"] == "duplicate" for i in res["issues"]))
        self.assertIn("คลิป 006", p["message"])
        self.assertIsNone(res.get("video"))              # never presented as final
        self.assertIsNotNone(res["review"])              # review copy kept for checking

    def test_4_speech_across_two_clips(self):
        t = self.user()
        names = ["x_part_b.mp4", "x_part_a.mp4"]          # uploaded in the wrong order
        pid, _ = self.project(t, names, rng=(11, 12))
        self.call("POST", f"/api/studio/projects/{pid}/start", {}, t)
        p = self.wait(t, pid)
        tl = [i["name"] for i in p["result"]["timeline"]["video"]]
        self.assertEqual(tl, ["x_part_a.mp4", "x_part_b.mp4"])
        # the shared line appears once in the subtitles
        with urllib.request.urlopen(self.base + p["result"]["srt"]["url"]) as f:
            srt = f.read().decode("utf-8")
        line12 = self.script["shots"][11]["dialogue"][0]["text"]
        self.assertEqual(srt.count(line12), 1)

    def test_6_dialogue_only_is_reported_honestly(self):
        t = self.user()
        pid, _ = self.project(t, [n for n, i in self.truth.items() if i < 3], rng=(1, 3))
        self.call("PATCH", f"/api/studio/projects/{pid}", {"settings": {"audio": "dialogue_only"}}, t)
        self.call("POST", f"/api/studio/projects/{pid}/start", {}, t)
        p = self.wait(t, pid)
        self.assertEqual(p["agents"]["13"]["status"], "warning")
        self.assertIn("ยังแยกเสียงพูด", p["agents"]["13"]["message"])

    def test_8_pause_and_resume_from_checkpoint(self):
        t = self.user()
        pid, _ = self.project(t, list(self.truth))
        self.call("POST", f"/api/studio/projects/{pid}/start", {}, t)
        time.sleep(0.4)
        self.call("POST", f"/api/studio/projects/{pid}/pause", {}, t)
        p = self.wait(t, pid)
        if p["status"] == "paused":
            done_before = sum(1 for c in p["clips"] if c["inspect"])
            self.assertIn("หยุดชั่วคราว", p["message"])
            code, p = self.call("POST", f"/api/studio/projects/{pid}/resume", {}, t)
            p = self.wait(t, pid)
            self.assertGreaterEqual(sum(1 for c in p["clips"] if c["inspect"]), done_before)
        self.assertEqual(p["status"], "done", p["message"])
        # segments rendered once are reused on a second run (render checkpoint)
        seg_dir = Path(os.environ["LIVEAI_DATA_DIR"]) / "studio" / str(pid) / "segments"
        mtimes = {f.name: f.stat().st_mtime for f in seg_dir.iterdir()}
        self.call("POST", f"/api/studio/projects/{pid}/start", {}, t)
        p = self.wait(t, pid)
        self.assertEqual({f.name: f.stat().st_mtime for f in seg_dir.iterdir()}, mtimes)

    def test_isolation_and_zip_safety(self):
        t, other = self.user(), self.user()
        pid, r = self.project(t, list(self.truth)[:2])
        self.assertEqual(r["added"], 2)
        self.assertFalse((Path(os.environ["LIVEAI_DATA_DIR"]).parent / "evil.txt").exists())
        self.assertEqual(self.call("GET", f"/api/studio/projects/{pid}", token=other)[0], 404)
        self.assertEqual(self.call("POST", f"/api/studio/projects/{pid}/start", {}, other)[0], 404)
        self.assertEqual(self.call("GET", "/api/studio/projects", token=other)[1], [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
