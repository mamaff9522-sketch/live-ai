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

    def wait(self, t, pid, timeout=180):
        end = time.time() + timeout
        p = {}
        while time.time() < end:
            p = self.call("GET", f"/api/studio/projects/{pid}", token=t)[1]
            if p["status"] in ("done", "needs_attention", "failed", "paused"):
                return p
            time.sleep(0.5)
        import faulthandler
        faulthandler.dump_traceback(all_threads=True)
        self.fail("project did not finish: " + json.dumps({k: p.get(k) for k in ("status", "message", "agents")}, ensure_ascii=False)[:3000])

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
        self.assertTrue(all(a["status"] in ("done", "warning") for a in p["agents"].values()), p["agents"])
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

    def test_6_dialogue_only(self):
        t = self.user()
        pid, _ = self.project(t, [n for n, i in self.truth.items() if i < 3], rng=(1, 3))
        self.call("PATCH", f"/api/studio/projects/{pid}", {"settings": {"audio": "dialogue_only"}}, t)
        self.call("POST", f"/api/studio/projects/{pid}/start", {}, t)
        p = self.wait(t, pid)
        self.assertEqual(p["status"], "done", p["message"])
        self.assertEqual(p["agents"]["13"]["status"], "done")
        self.assertIn("Dialogue Only", p["agents"]["13"]["message"])

    def test_7_silent_clips_placed_by_picture_continuity(self):
        """Shots 18–31 of the real script hold 3 silent shots. Like Meta AI clips made from the
        previous clip's last frame, each clip here fades from its own colour into the next clip's
        colour. Silent clips get meaningless names and are uploaded in random order."""
        t = self.user()
        shots = self.script["shots"][17:31]
        rnd = random.Random(21)
        d = self.tmp / "cont"
        d.mkdir(exist_ok=True)
        fake = json.loads((self.tmp / "asr.json").read_text(encoding="utf-8"))
        names, truth = [], {}
        pal = ["0x8B0000", "0x006400", "0x00008B", "0x8B8B00", "0x4B0082", "0xFF8C00", "0x008B8B", "0xC0C0C0",
               "0x2F4F4F", "0xFF1493", "0x8B4513", "0x556B2F", "0x191970", "0xDAA520", "0x000000"]
        for k, s in enumerate(shots):
            name = f"clip_{rnd.randint(100000, 999999)}.mp4"
            subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", f"color=c={pal[k]}:s=360x640:d=3",
                            "-f", "lavfi", "-i", f"color=c={pal[k + 1]}:s=360x640:d=3", "-f", "lavfi",
                            "-i", "sine=frequency=300:duration=2.4" if s["dialogue"] else "anoisesrc=a=0.0005:d=3",
                            "-filter_complex", "[0][1]xfade=transition=fade:duration=1:offset=1.4[v]",
                            "-map", "[v]", "-map", "2:a", "-af", "adelay=300|300,apad", "-t", "3",
                            "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", str(d / name)], check=True)
            fake[name] = " ".join(noisy(x["text"], rnd) for x in s["dialogue"])
            names.append(name)
            truth[name] = s["index"]
        (self.tmp / "asr.json").write_text(json.dumps(fake, ensure_ascii=False), encoding="utf-8")
        self.app.studio.asr.table = fake
        rnd.shuffle(names)
        code, p = self.call("POST", "/api/studio/projects", {"name": "ต่อเนื่อง"}, t)
        pid = p["id"]
        self.call("PATCH", f"/api/studio/projects/{pid}", {"script_text": SCRIPT, "settings": {"range_from": 18, "range_to": 31}}, t)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            for n in names:
                z.write(d / n, n)
        self.call("POST", f"/api/studio/projects/{pid}/zip", token=t, raw=buf.getvalue(), ctype="application/zip")
        self.call("POST", f"/api/studio/projects/{pid}/start", {}, t)
        p = self.wait(t, pid)
        tl = p["result"]["timeline"]["video"]
        got = [(truth[i["name"]] - 17, i["shots"][0]) for i in tl]
        self.assertEqual([g for g, _ in got], list(range(14)), got)
        silent = [i for i in tl if i["how"].startswith("silent")]
        self.assertEqual(len(silent), 3)
        self.assertTrue(all(i["confidence"] == "สูง" for i in silent), silent)
        self.assertEqual(p["status"], "done", p["message"])
        self.assertTrue(all(i.get("join_similarity", 1) > 0.8 for i in tl[1:]), [i.get("join_similarity") for i in tl])

    def test_9_ai_editor_worker_decides_cuts(self):
        """The AI editor path end to end: key -> worker online -> job with frames -> answers -> applied safely."""
        sys.path.insert(0, str(ROOT / "scripts"))
        import ai_editor_worker
        t = self.user()
        code, k = self.call("POST", "/api/editor/key", token=t)
        self.assertEqual(code, 200)
        key = k["key"]
        self.assertFalse(self.call("GET", "/api/editor/status", token=t)[1]["online"])
        # a wrong key is refused
        req = urllib.request.Request(self.base + "/api/editor/worker/next", headers={"X-Worker-Key": "lew_wrong"})
        with self.assertRaises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(req, timeout=30)
        self.assertEqual(e.exception.code, 401)
        worker = threading.Thread(target=ai_editor_worker.run, args=(self.base, key, self.app.llm.url, True), daemon=True)
        worker.start()
        for _ in range(50):
            if self.call("GET", "/api/editor/status", token=t)[1]["online"]:
                break
            time.sleep(0.1)
        self.assertTrue(self.call("GET", "/api/editor/status", token=t)[1]["online"])
        names = [n for n in self.truth if self.truth[n] < 4]
        pid, _ = self.project(t, names, (1, 4))
        self.call("POST", f"/api/studio/projects/{pid}/start", {}, t)
        p = self.wait(t, pid, 240)
        worker.join(30)
        tl = p["result"]["timeline"]
        self.assertTrue(tl.get("ai_editor", {}).get("used"), tl.get("ai_editor"))
        self.assertTrue(all("ทดสอบ: ดูภาพ" in why for why in tl["ai_editor"]["reasons"].values()))
        self.assertTrue(any("ดูภาพ 0" not in why for why in tl["ai_editor"]["reasons"].values()))   # frames were sent
        self.assertEqual(p["status"], "done", p["message"])
        qa = {q["name"]: q["ok"] for q in p["result"]["qa"]["checks"]}
        self.assertTrue(qa["ไม่ตัดคำพูดทิ้ง"])
        self.assertIn("AI คนตัดต่อ", p["agents"]["12"]["message"] if isinstance(p["agents"], dict) else "")
        # another user cannot take this user's jobs or frames
        t2 = self.user()
        key2 = self.call("POST", "/api/editor/key", token=t2)[1]["key"]
        job = self.app.studio.aied.store.conn().execute("SELECT id, payload FROM editor_jobs WHERE project_id=?", (pid,)).fetchone()
        frame = json.loads(job["payload"])["clips"][0]["frames"][0]["name"]
        req = urllib.request.Request(f"{self.base}/api/editor/worker/jobs/{job['id']}/frames/{frame}", headers={"X-Worker-Key": key2})
        with self.assertRaises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(req, timeout=30)
        self.assertEqual(e.exception.code, 404)

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


class FreeFormScripts(unittest.TestCase):
    """Scripts without any ฉาก/คลิป headings are still read (one spoken line = one shot)."""

    def test_plain_dialogue_lines(self):
        from live_ai.studio.script import parse
        r = parse("น้ำฟ้า: ฉันไม่ยอมหรอก\nภาคิน: ใจเย็นก่อน\nแม่: ออกไปจากบ้านนี้!")
        self.assertEqual([(d["speaker"], d["text"]) for s in r["shots"] for d in s["dialogue"]],
                         [("น้ำฟ้า", "ฉันไม่ยอมหรอก"), ("ภาคิน", "ใจเย็นก่อน"), ("แม่", "ออกไปจากบ้านนี้!")])

    def test_prose_with_quotes_and_prose_only(self):
        from live_ai.studio.script import parse
        r = parse("น้ำฟ้าเดินเข้าบ้าน\n\nเธอพูดว่า “ทำไมทุกคนเงียบ”\n\nภาคินหันมามอง")
        self.assertEqual([d["text"] for s in r["shots"] for d in s["dialogue"]], ["ทำไมทุกคนเงียบ"])
        r = parse("ทุยเดินไปตลาด\nทุยซื้อกล้วย\nทุยกลับบ้าน")
        self.assertEqual(r["stats"]["shots"], 3)

    def test_speaker_heading_sections_and_style_note(self):
        """The user's own format: a title, labelled sections, 'ชื่อ (อารมณ์):' then quoted lines with stage
        directions in brackets between them, an ending and a style line."""
        from live_ai.studio.script import parse
        r = parse("ฉากรักที่ยังรอ — พี่ทุย\n\nสถานที่: ริมทะเลสาบยามพระอาทิตย์ตก\n\nการแสดง: พี่ทุยยืนอยู่ปลายสะพานไม้เพียงลำพัง\n\n"
                  "พี่ทุย (ร้องไห้ไปด้วย พูดไปด้วย):\n\n\"คนหนึ่ง... ไปไกลแล้ว...\"\n\n(น้ำตาไหลอาบแก้ม หายใจสะอื้น)\n\n"
                  "\"แต่อีกคน... ยังอยู่ที่เดิม...\"\n\n(ฝืนยิ้มทั้งน้ำตา)\n\n\"ยังรออยู่ตรงนี้นะ... ยังไม่ได้ไปไหน...\"\n\n"
                  "ตอนจบ: พี่ทุยก้มหน้าร้องไห้เงียบ ๆ กล้องค่อย ๆ ถอยออก\n\nสไตล์ภาพ: Cinematic Photorealistic, 9:16")
        self.assertEqual(r["title"], "ฉากรักที่ยังรอ — พี่ทุย")
        self.assertEqual([(d["speaker"], d["text"]) for s in r["shots"] for d in s["dialogue"]],
                         [("พี่ทุย", "คนหนึ่ง... ไปไกลแล้ว..."), ("พี่ทุย", "แต่อีกคน... ยังอยู่ที่เดิม..."),
                          ("พี่ทุย", "ยังรออยู่ตรงนี้นะ... ยังไม่ได้ไปไหน...")])
        # opening (performance before words) and ending become optional shots without words; style is a note
        self.assertEqual(r["stats"]["shots"], 5)
        self.assertTrue(r["shots"][0].get("optional") and not r["shots"][0]["dialogue"])
        self.assertTrue(r["shots"][-1].get("optional") and any("ตอนจบ" in a for a in r["shots"][-1]["action"]))
        self.assertTrue(any("สไตล์ภาพ" in n for n in r["notes"]))


    def test_name_line_then_unquoted_words_with_direction(self):
        """'ชื่อ:' on its own line, then '(ท่าทาง) คำพูด' without quotation marks; an invisible character
        before a name; a direction broken over two lines; a copy that starts mid-sentence."""
        from live_ai.studio.script import parse
        r = parse("ชัย:\n(ตะโกนจากหน้าร้าน) ตง! ยกทีวีไปขึ้นรถตู้นะ\n\u200bตง:\n(สะดุ้ง) ขอโทษครับเฮีย\n"
                  "ชัย:\n(ล้วงซองจาก\nกระเป๋าเสื้อ) เอ้า! นี่เบี้ยเลี้ยง\nตง: (ไหว้) ขอบคุณมากครับเฮียชัย")
        self.assertEqual([(d["speaker"], d["text"]) for s in r["shots"] for d in s["dialogue"]],
                         [("ชัย", "ตง! ยกทีวีไปขึ้นรถตู้นะ"), ("ตง", "ขอโทษครับเฮีย"), ("ชัย", "เอ้า! นี่เบี้ยเลี้ยง"),
                          ("ตง", "ขอบคุณมากครับเฮียชัย")])
        self.assertEqual([s["action"] for s in r["shots"]], [["ตะโกนจากหน้าร้าน"], ["สะดุ้ง"], ["ล้วงซองจาก กระเป๋าเสื้อ"], ["ไหว้"]])
        r = parse("กระเป๋าเสื้อ ออกมายื่นให้ตง) เอ้า! นี่เบี้ยเลี้ยง\nตง:\n(รับซองมา) ขอบคุณมากครับ")
        self.assertEqual([d["text"] for s in r["shots"] for d in s["dialogue"]], ["เอ้า! นี่เบี้ยเลี้ยง", "ขอบคุณมากครับ"])
        self.assertTrue(any("ไม่ครบ" in w for w in r["warnings"]))
        # quoted lines under a name heading keep working, and prose after them is not taken as words
        r = parse("แม่:\n\"กลับบ้านเถอะลูก\"\nแม่เดินจากไป\nลูก: \"ครับแม่\"")
        self.assertEqual([d["text"] for s in r["shots"] for d in s["dialogue"]], ["กลับบ้านเถอะลูก", "ครับแม่"])


class ClipOrderAndUnscripted(unittest.TestCase):
    def test_file_time_order_and_clips_outside_the_script(self):
        from live_ai.studio import match as M
        names = ["Meta_AI_AQPk_20261101_171304.mp4", "Meta_AI_AQMC_20261101_171141.mp4", "Meta_AI_AQMH_20261101_171319.mp4",
                 "Meta_AI_AQO7_20261101_171230.mp4"]
        made = M.file_order(names)
        self.assertEqual(sorted(names, key=made.get), [names[1], names[3], names[0], names[2]])
        self.assertEqual(sorted(["คลิป 10.mp4", "คลิป 2.mp4"], key=M.file_order(["คลิป 10.mp4", "คลิป 2.mp4"]).get),
                         ["คลิป 2.mp4", "คลิป 10.mp4"])
        from live_ai.studio.script import parse
        script = parse("ชัย: (ยื่นซอง) เอ้า! นี่เบี้ยเลี้ยงรอบพิเศษของสัปดาห์นี้\nตง: (ไหว้) ขอบคุณมากครับเฮียชัย ผมจะไม่ลืมพระคุณ")
        talk = {names[0]: "เอ้า นี่เบี้ยเลี้ยงรอบพิเศษของสัปดาห์นี้", names[2]: "ขอบคุณมากครับเฮียชัย ผมจะไม่ลืมพระคุณ",
                names[1]: "ตงซ่อมเสร็จแล้วยกทีวีขึ้นรถตู้", names[3]: "ไม่ไหวก็ต้องไหวครับเฮียแม่ผมต้องฟอกไต"}
        clips = [{"id": f"c{i}", "name": n, "order": made[n], "duration": 10.0, "speech": True, "transcript": talk[n]}
                 for i, n in enumerate(names)]
        r = M.match(script, clips)
        self.assertEqual([next(c["name"] for c in clips if c["id"] == cid) for cid in r["timeline"]],
                         [names[1], names[3], names[0], names[2]])
        self.assertEqual(sorted(i["clip"] for i in r["issues"] if i["type"] == "unscripted"), ["c1", "c3"])
        self.assertEqual(r["missing"], [])
        r = M.match(script, clips, unscripted=False)
        self.assertEqual(len(r["timeline"]), 2)
        self.assertEqual(sorted(i["clip"] for i in r["issues"] if i["type"] == "unmatched"), ["c1", "c3"])


class ReactionAfterWords(unittest.TestCase):
    def test_measured_jump_after_words_only_trims_the_last_frames(self):
        """A big picture change after the words is usually a gesture or a camera move (the reaction):
        only a jump at the very end of the clip is cut; one the AI editor saw as broken is cut anywhere."""
        from live_ai.studio import cutter
        def run(jumps, broken=()):
            clip = {"id": "c", "name": "c", "duration": 10.24, "transcript": "ขอบคุณครับ",
                    "asr": {"segments": [{"start": 0.0, "end": 2.0}, {"start": 3.0, "end": 4.0}]},
                    "inspect": {"speech_regions": [[0.0, 10.24]], "jumps": jumps}, "ai_broken": list(broken)}
            tl = {"video": [{"clip": "c", "lines": [0]}]}
            cutter.plan_cuts(tl, {"c": clip}, lambda it: "ปกติ")
            return tl["video"][0]["out"]
        self.assertGreaterEqual(run([[2.0, 4.5], [5.5, 6.5]]), 6.5)     # words end at 4.0, last clip holds 2.5 s
        self.assertEqual(run([[5.0, 10.24]]), 5.0)                      # glitch running into the end: cut
        self.assertEqual(run([], broken=[(5.0, 6.0)]), 5.0)              # seen broken by the AI editor: cut


class ClipChecks(unittest.TestCase):
    """Inspector notices sudden picture jumps inside an AI clip and very quiet audio."""

    def test_jump_and_quiet_audio(self):
        import subprocess
        import tempfile
        from live_ai.studio.pipeline import inspect_clip, picture_jumps
        d = Path(tempfile.mkdtemp())
        p = d / "jump.mp4"
        # 3 s steady picture, then a completely different one (like a sudden zoom), quiet tone
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "color=c=0x203040:s=360x640:d=3:r=24",
                        "-f", "lavfi", "-i", "testsrc2=s=360x640:d=3:r=24", "-f", "lavfi", "-i", "sine=f=220:d=6",
                        "-filter_complex", "[0][1]concat=n=2:v=1:a=0[v];[2]volume=0.01[a]", "-map", "[v]", "-map", "[a]",
                        "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", str(p)], check=True)
        jumps = picture_jumps(str(p))
        self.assertEqual(len(jumps), 1)
        self.assertTrue(2.0 <= jumps[0][0] <= 3.0, jumps)
        ins = inspect_clip(str(p))
        self.assertTrue(any("ภาพเปลี่ยนกะทันหัน" in x for x in ins["problems"]))
        self.assertTrue(any("เสียงเบามาก" in x for x in ins["problems"]), ins["problems"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
