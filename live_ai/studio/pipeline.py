"""AI 1 (Master Director) and the working agents of the editing studio.

One background worker runs projects step by step. Every result is saved in SQLite as it is
produced (per clip for the slow parts), so a project interrupted at clip 80 of 100 continues at
clip 81. A step that fails is retried at most twice, then the Director stops just that project
and says what is wrong. Nothing is exported as "complete" unless QA (AI 15) passes.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import queue
import shutil
import subprocess
import threading
import time
import zipfile
from pathlib import Path

from .. import logs
from ..media import MediaError, probe
from ..store import NotFound
from . import cutter
from .aieditor import AIEditor, frame_times
from . import match as M
from .asr import ASRUnavailable, FakeASR, loudness, make_asr, speech_regions
from .script import label, parse

log = logs.get("studio")

SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    script_text TEXT NOT NULL DEFAULT '',
    script TEXT,
    settings TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'draft',
    message TEXT NOT NULL DEFAULT '',
    agents TEXT NOT NULL DEFAULT '{}',
    result TEXT,
    pause INTEGER NOT NULL DEFAULT 0,
    created REAL NOT NULL,
    updated REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS proj_user ON projects(user_id, updated);
CREATE TABLE IF NOT EXISTS project_clips (
    project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    media_id TEXT NOT NULL,
    upload_order INTEGER NOT NULL,
    inspect TEXT,
    asr TEXT,
    PRIMARY KEY (project_id, media_id)
);
"""

AGENTS = [
    (1, "ผู้กำกับใหญ่", "Master Director"),
    (2, "วิเคราะห์บทละคร", "Script Analyzer"),
    (3, "จัดการฉากและลำดับเรื่อง", "Scene Manager"),
    (4, "วิเคราะห์ภาพวิดีโอ", "Visual Analyzer"),
    (5, "วิเคราะห์บทพูดและเสียง", "Dialogue Analyzer"),
    (6, "จับคู่บทละครกับคลิป", "Script–Video Matcher"),
    (7, "เรียงลำดับคลิป", "Sequence Editor"),
    (8, "ตรวจสอบคลิปผิดพลาด", "Clip Inspector"),
    (9, "ตรวจสอบความต่อเนื่อง", "Continuity Checker"),
    (10, "วิเคราะห์อารมณ์และจังหวะ", "Emotion Director"),
    (11, "ตัดต่อวิดีโอ", "Video Editor"),
    (12, "กำกับภาพและรอยต่อ", "Cinematic Editor"),
    (13, "วิศวกรเสียง", "Audio Engineer"),
    (14, "สร้างคำบรรยาย", "Subtitle Generator"),
    (15, "ตรวจคุณภาพขั้นสุดท้าย", "Final QA"),
]
NOT_YET: dict[int, str] = {}
DEFAULT_SETTINGS = {"aspect": "9:16", "height": 720, "fps": 30, "subtitles": "file", "audio": "dialogue_only", "match_light": False,
                    "pacing": "auto", "ai_editor": True, "unscripted": True, "speech_speed": "auto",
                    "speaker_in_subs": False, "range_from": None, "range_to": None}
VIDEO_EXT = {".mp4", ".mov", ".m4v", ".webm", ".mkv", ".3gp"}
RNNOISE_MODEL = Path(__file__).resolve().parent.parent / "assets" / "rnnoise-std.rnnn"
_DENOISE: tuple[str, str] | None = None


def denoise_filter() -> tuple[str, str]:
    """(ffmpeg filter, name) for Dialogue Only. Each candidate is first tried on exactly the kind of job
    a real clip gets, because some ffmpeg builds hang on some chains (seen with ffmpeg 8.0.1 on
    Ubuntu 26.04): RNNoise, else ffmpeg's FFT denoiser, else no denoiser (gap lowering only)."""
    global _DENOISE
    if _DENOISE is None:
        cands = []
        if RNNOISE_MODEL.is_file():
            cands.append((f"arnndn=m={RNNOISE_MODEL}", "RNNoise"))
        cands += [("afftdn=nr=20:nf=-35", "ตัวกรองเสียงรบกวน FFT ของ ffmpeg"), ("anull", "")]
        import tempfile
        chosen = cands[-1]
        with tempfile.TemporaryDirectory() as d:
            try:
                subprocess.run(["ffmpeg", "-loglevel", "error", "-nostdin", "-y", "-f", "lavfi", "-i", "color=c=gray:s=320x240:d=2",
                                "-f", "lavfi", "-i", "sine=frequency=300:duration=2", "-c:v", "libx264", "-preset", "ultrafast",
                                "-c:a", "aac", "-t", "2", f"{d}/in.mp4"], capture_output=True, timeout=30, check=True)
            except Exception:
                cands = cands[-1:]
            for flt, name in cands[:-1]:
                try:
                    r = subprocess.run(["ffmpeg", "-loglevel", "error", "-nostdin", "-y", "-i", f"{d}/in.mp4", "-vn",
                                        "-af", f"aresample=48000,aformat=channel_layouts=mono,{flt},"
                                               "volume='if(gt(between(t,0.2,1.8),0),1,0.1)':eval=frame",
                                        "-c:a", "pcm_s16le", f"{d}/t.wav"], capture_output=True, timeout=20)
                    if r.returncode == 0:
                        chosen = (flt, name)
                        break
                except subprocess.TimeoutExpired:
                    continue
        _DENOISE = chosen
        logs.event(log, "speech denoiser", engine=chosen[1] or "none")
    return _DENOISE


MAX_RETRIES = 2


class StudioError(RuntimeError):
    pass


class Paused(Exception):
    pass


def out_size(settings: dict, first: dict | None) -> tuple[int, int]:
    h = int(settings.get("height", 720))
    asp = settings.get("aspect", "9:16")
    if asp == "9:16":
        return _vertical(h)
    if asp == "16:9":
        return _horizontal(h)
    if first and first.get("width"):          # original shape, short side = chosen height (never upscaled)
        w, hh = first["width"], first["height"]
        sc = min(1.0, h / min(w, hh))
        return int(w * sc) // 2 * 2, int(hh * sc) // 2 * 2
    return _vertical(h)


def _vertical(h: int) -> tuple[int, int]:
    return {720: (720, 1280), 1080: (1080, 1920), 1440: (1440, 2560), 2160: (2160, 3840)}.get(h, (720, 1280))


def _horizontal(h: int) -> tuple[int, int]:
    return {720: (1280, 720), 1080: (1920, 1080), 1440: (2560, 1440), 2160: (3840, 2160)}.get(h, (1280, 720))


def srt_time(t: float) -> str:
    ms = int(round(t * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


class Studio:
    def __init__(self, store, media, s, assistant=None):
        self.store = store
        self.media = media
        self.s = s
        self.assistant = assistant
        self.root = s.data_dir / "studio"
        self.root.mkdir(parents=True, exist_ok=True)
        self.q: queue.Queue[int] = queue.Queue()
        self.asr = make_asr(s.data_dir, s.threads)
        self.aied = AIEditor(store, self.root)
        with store.conn() as c:
            c.executescript(SCHEMA)
            cols = {r["name"] for r in c.execute("PRAGMA table_info(project_clips)")}
            if "visual" not in cols:
                c.execute("ALTER TABLE project_clips ADD COLUMN visual TEXT")
            rows = c.execute("SELECT id FROM projects WHERE status IN ('queued','running')").fetchall()
        for r in rows:   # resume after a restart, from the saved checkpoints
            self.q.put(r["id"])
        threading.Thread(target=self._worker, name="studio", daemon=True).start()

    # ------------------------------------------------------------------ projects
    def create(self, user_id: int, name: str) -> dict:
        now = time.time()
        with self.store.conn() as c:
            cur = c.execute("INSERT INTO projects(user_id,name,settings,agents,created,updated) VALUES (?,?,?,?,?,?)",
                            (user_id, (name or "โปรเจกต์ใหม่").strip()[:80], json.dumps(DEFAULT_SETTINGS),
                             json.dumps(self._fresh_agents()), now, now))
        return self.get(user_id, cur.lastrowid)

    def _row(self, user_id: int, pid: int) -> dict:
        r = self.store.conn().execute("SELECT * FROM projects WHERE id=? AND user_id=?", (pid, user_id)).fetchone()
        if not r:
            raise NotFound("project")
        return dict(r)

    def list(self, user_id: int) -> list[dict]:
        rows = self.store.conn().execute("SELECT id,name,status,message,updated FROM projects WHERE user_id=? ORDER BY updated DESC",
                                         (user_id,)).fetchall()
        return [dict(r) for r in rows]

    def get(self, user_id: int, pid: int) -> dict:
        p = self._row(user_id, pid)
        out = {k: p[k] for k in ("id", "name", "status", "message", "created", "updated", "pause")}
        out["settings"] = json.loads(p["settings"])
        out["agents"] = json.loads(p["agents"])
        out["script"] = json.loads(p["script"]) if p["script"] else None
        out["has_script"] = bool(p["script_text"].strip())
        out["script_text"] = p["script_text"]
        out["result"] = json.loads(p["result"]) if p["result"] else None
        if out["result"]:
            for k in ("video", "video_subs", "srt", "review"):
                m = out["result"].get(k)
                if m:
                    try:
                        self.media.get(user_id, m["id"])
                        m["url"] = self.media.sign(m["id"])
                    except NotFound:
                        m["url"] = None
        clips = []
        for r in self.store.conn().execute("SELECT * FROM project_clips WHERE project_id=? ORDER BY upload_order", (pid,)):
            try:
                m = self.media.get(user_id, r["media_id"])
            except NotFound:
                continue
            clips.append({"id": m["id"], "name": m["name"], "duration": m["duration"], "width": m["width"], "height": m["height"],
                          "thumb": self.media.sign(m["id"]).replace("/file?", "/thumb?"),
                          "inspect": json.loads(r["inspect"]) if r["inspect"] else None,
                          "asr": json.loads(r["asr"]) if r["asr"] else None})
        out["clips"] = clips
        return out

    def update(self, user_id: int, pid: int, name=None, script_text=None, settings=None) -> dict:
        p = self._row(user_id, pid)
        if p["status"] in ("queued", "running"):
            raise ValueError("กำลังประมวลผลอยู่ กดหยุดชั่วคราวก่อนแก้ไข")
        sets, vals = [], []
        if name is not None:
            sets.append("name=?"), vals.append(str(name)[:80])
        if script_text is not None:
            if len(script_text) > 2_000_000:
                raise ValueError("บทยาวเกิน 2 ล้านตัวอักษร")
            parsed = parse(script_text) if script_text.strip() else None
            sets += ["script_text=?", "script=?"]
            vals += [script_text, json.dumps(parsed, ensure_ascii=False) if parsed else None]
        if settings is not None:
            cur = json.loads(p["settings"])
            for k, v in settings.items():
                if k == "aspect" and v in ("9:16", "16:9", "original"):
                    cur[k] = v
                elif k == "height" and int(v) in (720, 1080, 1440, 2160):
                    cur[k] = int(v)
                elif k == "fps" and int(v) in (24, 25, 30, 60):
                    cur[k] = int(v)
                elif k == "subtitles" and v in ("none", "file", "burn"):
                    cur[k] = v
                elif k == "speech_speed" and v in ("auto", "original"):
                    cur[k] = v
                elif k == "pacing" and v in ("auto", "original"):
                    cur[k] = v
                elif k == "order" and (v is None or (isinstance(v, list) and len(v) <= 2000 and all(isinstance(x, str) and len(x) <= 40 for x in v))):
                    cur[k] = v or None          # the owner's own clip order (None = by the script)
                elif k in ("ai_editor", "unscripted") and isinstance(v, bool):
                    cur[k] = v
                elif k == "audio" and v in ("original", "dialogue_only"):
                    cur[k] = v
                elif k in ("speaker_in_subs", "match_light"):
                    cur[k] = bool(v)
                elif k in ("range_from", "range_to"):
                    cur[k] = int(v) if v not in (None, "", 0) else None
            sets.append("settings=?"), vals.append(json.dumps(cur))
        if sets:
            sets.append("updated=?"), vals.append(time.time())
            with self.store.conn() as c:
                c.execute(f"UPDATE projects SET {', '.join(sets)} WHERE id=? AND user_id=?", (*vals, pid, user_id))
        return self.get(user_id, pid)

    def delete(self, user_id: int, pid: int) -> None:
        p = self._row(user_id, pid)
        if p["status"] in ("queued", "running"):
            raise ValueError("กำลังประมวลผลอยู่ กดหยุดก่อน")
        with self.store.conn() as c:
            c.execute("DELETE FROM projects WHERE id=? AND user_id=?", (pid, user_id))
        shutil.rmtree(self.root / str(pid), ignore_errors=True)   # cache only; uploaded originals are kept

    # ------------------------------------------------------------------ clips
    def add_clips(self, user_id: int, pid: int, media_ids: list[str]) -> dict:
        self._row(user_id, pid)
        with self.store.conn() as c:
            n = c.execute("SELECT COALESCE(MAX(upload_order),-1)+1 FROM project_clips WHERE project_id=?", (pid,)).fetchone()[0]
            for mid in media_ids:
                m = self.media.get(user_id, mid)
                if m["kind"] != "video":
                    raise ValueError(f"“{m['name']}” ไม่ใช่ไฟล์วิดีโอ")
                c.execute("INSERT OR IGNORE INTO project_clips(project_id,media_id,upload_order) VALUES (?,?,?)", (pid, mid, n))
                c.execute("UPDATE media SET origin='studio' WHERE id=? AND user_id=?", (mid, user_id))
                n += 1
        return self.get(user_id, pid)

    def remove_clip(self, user_id: int, pid: int, mid: str) -> None:
        self._row(user_id, pid)
        with self.store.conn() as c:
            c.execute("DELETE FROM project_clips WHERE project_id=? AND media_id=?", (pid, mid))

    def add_zip(self, user_id: int, pid: int, stream, length: int) -> dict:
        """Unpacks the videos in a ZIP (names are flattened; nothing is written outside our folder)."""
        self._row(user_id, pid)
        tmp = self.root / f"upload-{os.urandom(6).hex()}.zip"
        left = length
        with open(tmp, "wb") as f:
            while left > 0:
                chunk = stream.read(min(1 << 20, left))
                if not chunk:
                    break
                f.write(chunk)
                left -= len(chunk)
        added, skipped = [], []
        try:
            if left:
                raise ValueError("อัปโหลด ZIP ไม่ครบ")
            try:
                z = zipfile.ZipFile(tmp)
            except zipfile.BadZipFile as e:
                raise ValueError("ไฟล์ ZIP เสียหาย") from e
            members = [i for i in z.infolist() if not i.is_dir() and Path(i.filename).suffix.lower() in VIDEO_EXT
                       and not Path(i.filename).name.startswith((".", "__MACOSX"))]
            if len(members) > 600:
                raise ValueError("ZIP มีวิดีโอเกิน 600 ไฟล์")
            if sum(i.file_size for i in members) + self.media.used_bytes(user_id) > self.media.quota:
                raise ValueError(f"พื้นที่ไม่พอ (สูงสุด {self.media.quota // 2**20} MB)")
            dest = self.media.root / str(user_id)
            dest.mkdir(parents=True, exist_ok=True)
            for i in sorted(members, key=lambda i: M.natural_key(Path(i.filename).name)):
                if i.compress_size and i.file_size / max(i.compress_size, 1) > 200:
                    skipped.append(f"{Path(i.filename).name}: อัตราบีบอัดผิดปกติ")
                    continue
                name = Path(i.filename).name
                out = dest / f"{os.urandom(12).hex()}{Path(name).suffix.lower()}"
                with z.open(i) as src, open(out, "wb") as dst:
                    shutil.copyfileobj(src, dst, 1 << 20)
                try:
                    m = self.media.add_file(user_id, out, name, "studio")
                    if m["kind"] != "video":
                        raise MediaError("ไม่ใช่วิดีโอ")
                    added.append(m["id"])
                except MediaError as e:
                    out.unlink(missing_ok=True)
                    skipped.append(f"{name}: {e}")
        finally:
            tmp.unlink(missing_ok=True)
        if added:
            self.add_clips(user_id, pid, added)
        logs.event(log, "zip added", project=pid, added=len(added), skipped=len(skipped))
        return {"added": len(added), "skipped": skipped, "project": self.get(user_id, pid)}

    # ------------------------------------------------------------------ control
    def start(self, user_id: int, pid: int) -> dict:
        p = self._row(user_id, pid)
        if p["status"] in ("queued", "running"):
            return self.get(user_id, pid)
        if not p["script_text"].strip():
            raise ValueError("ยังไม่มีบทละคร")
        n = self.store.conn().execute("SELECT COUNT(*) FROM project_clips WHERE project_id=?", (pid,)).fetchone()[0]
        if not n:
            raise ValueError("ยังไม่มีคลิปวิดีโอ")
        self._set(pid, status="queued", message="รอคิวประมวลผล", pause=0)
        self.q.put(pid)
        logs.event(log, "project queued", project=pid, clips=n)
        return self.get(user_id, pid)

    def pause(self, user_id: int, pid: int) -> dict:
        self._row(user_id, pid)
        self._set(pid, pause=1)
        return self.get(user_id, pid)

    # ------------------------------------------------------------------ internals
    @staticmethod
    def _fresh_agents() -> dict:
        return {str(n): {"name": th, "en": en, "status": "skipped" if n in NOT_YET else "waiting",
                         "message": NOT_YET.get(n, ""), "updated": 0} for n, th, en in AGENTS}

    def _set(self, pid: int, **kw) -> None:
        kw["updated"] = time.time()
        with self.store.conn() as c:
            c.execute(f"UPDATE projects SET {', '.join(f'{k}=?' for k in kw)} WHERE id=?", (*kw.values(), pid))

    def _agent(self, pid: int, n: int, status: str, message: str = "") -> None:
        with self.store.conn() as c:
            row = c.execute("SELECT agents FROM projects WHERE id=?", (pid,)).fetchone()
            a = json.loads(row["agents"])
            a[str(n)].update(status=status, message=message[:400], updated=time.time())
            c.execute("UPDATE projects SET agents=?, updated=? WHERE id=?", (json.dumps(a, ensure_ascii=False), time.time(), pid))

    def _check_pause(self, pid: int) -> None:
        if self.store.conn().execute("SELECT pause FROM projects WHERE id=?", (pid,)).fetchone()["pause"]:
            raise Paused()

    def _worker(self) -> None:
        while True:
            pid = self.q.get()
            try:
                self.run(pid)
            except Exception as e:   # never kill the worker
                logs.event(log, "project crashed", logging.ERROR, project=pid, error=repr(e)[:400])
                self._set(pid, status="failed", message=f"ระบบขัดข้อง: {str(e)[:200]} (กด Start เพื่อทำต่อจากจุดเดิม)")

    def _step(self, pid: int, n: int, fn, *a):
        """Run one agent with a bounded number of retries."""
        for attempt in range(MAX_RETRIES + 1):
            self._check_pause(pid)
            self._agent(pid, n, "running", "กำลังทำงาน" + (f" (ลองครั้งที่ {attempt + 1})" if attempt else ""))
            try:
                return fn(pid, *a)
            except (Paused, StudioError, ASRUnavailable):
                raise
            except Exception as e:
                logs.event(log, "agent error", logging.WARNING, project=pid, agent=n, attempt=attempt, error=repr(e)[:300])
                if attempt == MAX_RETRIES:
                    self._agent(pid, n, "failed", str(e)[:300])
                    raise StudioError(f"AI {n} ทำงานไม่สำเร็จหลังลอง {MAX_RETRIES + 1} ครั้ง: {str(e)[:200]}")
                time.sleep(1 + attempt)

    def run(self, pid: int) -> None:
        p = dict(self.store.conn().execute("SELECT * FROM projects WHERE id=?", (pid,)).fetchone())
        uid = p["user_id"]
        started = time.time()
        self._set(pid, status="running", message="กำลังทำงาน", pause=0)
        agents = json.loads(p["agents"])
        for k, v in self._fresh_agents().items():   # agents added later / status reset per run
            agents.setdefault(k, v)
            if agents[k]["status"] != "skipped":
                agents[k]["status"] = "waiting"
        self._set(pid, agents=json.dumps(agents, ensure_ascii=False))
        self._agent(pid, 1, "running", "ตรวจข้อมูลและแจกงาน")
        try:
            script = self._step(pid, 2, self._script, p)
            self._step(pid, 3, self._scenes, script)
            clips = self._step(pid, 8, self._inspect, uid)
            clips = self._step(pid, 4, self._visual, clips)
            clips = self._step(pid, 5, self._dialogue, uid, clips, script)
            plan = self._step(pid, 6, self._match, script, clips)
            plan = self._manual_order(pid, script, clips, plan)
            timeline = self._step(pid, 7, self._sequence, script, clips, plan)
            self._step(pid, 9, self._continuity, script, clips, timeline)
            self._step(pid, 10, self._emotion, script)
            self._step(pid, 12, self._cinematic, script, clips, timeline)
            self._step(pid, 13, self._audio, clips, timeline)
            out = self._step(pid, 11, self._render, uid, clips, timeline)
            subs = self._step(pid, 14, self._subtitles, uid, script, clips, timeline, out)
            qa = self._step(pid, 15, self._qa, uid, script, clips, timeline, out, subs, plan)
        except Paused:
            self._set(pid, status="paused", message="หยุดชั่วคราว งานที่ทำแล้วถูกเก็บไว้ กด ดำเนินการต่อ เพื่อทำต่อ")
            self._agent(pid, 1, "paused", "หยุดตามคำสั่ง")
            return
        except (StudioError, ASRUnavailable) as e:
            self._set(pid, status="failed", message=str(e)[:500])
            self._agent(pid, 1, "failed", str(e)[:300])
            logs.event(log, "project stopped", logging.WARNING, project=pid, reason=str(e)[:300])
            return
        secs = round(time.time() - started, 1)
        complete = qa["passed"]
        result = {**json.loads(self.store.conn().execute("SELECT result FROM projects WHERE id=?", (pid,)).fetchone()["result"] or "{}"),
                  "qa": qa, "seconds": secs, "missing": plan["missing"],
                  "issues": plan["issues"] + timeline["issues"] + clip_notes(clips),
                  "timeline": timeline,
                  "complete": complete}
        unused = [c["name"] for c in clips if c["id"] in {i["clip"] for i in plan["issues"] if i.get("type") == "extra"}]
        outside = [i["name"] for i in timeline["video"] if i.get("unscripted") and not i.get("confirmed")]
        if complete:
            msg = f"เสร็จสมบูรณ์ ผ่าน QA ({out['duration']:.1f} วินาที, ใช้เวลา {secs:.0f} วินาที)"
            if outside:
                msg += (f" — มี {len(outside)} คลิปที่พูดนอกบท ใส่ตามลำดับที่สร้างคลิปและไม่มีซับ"
                        " (วางบทให้ครบทั้งเรื่องแล้วกด Start ใหม่ จะได้ซับครบ)")
            if unused:
                msg += f" — ไม่ได้ใช้ {len(unused)} คลิปที่ไม่ตรงกับบท: {', '.join(unused[:5])} (ดูแท็บ ตรวจข้อผิดพลาด)"
        else:
            msg = "ยังไม่สมบูรณ์: " + "; ".join(qa["problems"][:3])
        self._set(pid, status="done" if complete else "needs_attention", message=msg[:500], result=json.dumps(result, ensure_ascii=False))
        self._agent(pid, 1, "done" if complete else "warning", msg)
        logs.event(log, "project finished", project=pid, complete=complete, seconds=secs, clips=len(clips),
                   missing=len(plan["missing"]))

    # ------------------------------------------------------------------ agents
    def _script(self, pid, p) -> dict:
        if not (p["script_text"] or "").strip():
            raise StudioError("ยังไม่ได้ใส่บทละคร: ไปที่แท็บ บทละคร วางบทแล้วกด บันทึกและวิเคราะห์บท")
        script = parse(p["script_text"])
        if not script["shots"]:
            raise StudioError("อ่านบทละครไม่ได้: ไม่พบฉาก คลิป หรือบทพูด")
        st = json.loads(p["settings"])
        a, b = st.get("range_from"), st.get("range_to")
        if a or b:
            def num(sh):
                return sh["number"] if sh.get("number") else sh["index"] + 1
            keep = [sh for sh in script["shots"] if (not a or num(sh) >= a) and (not b or num(sh) <= b)]
            if not keep:
                raise StudioError(f"ไม่มีช็อตในช่วง {a or 1}–{b or 'จบ'} ที่เลือกไว้")
            for k, sh in enumerate(keep):
                sh["pos"] = sh["index"]
                sh["index"] = k
            script["shots"] = keep
            script["range"] = [a, b]
        with self.store.conn() as c:
            c.execute("UPDATE projects SET script=? WHERE id=?", (json.dumps(script, ensure_ascii=False), pid))
        st = script["stats"]
        self._agent(pid, 2, "warning" if script["warnings"] else "done",
                    f"{st['shots']} ช็อต, บทพูด {st['dialogue_lines']} บรรทัด (คัดลอกตรงต้นฉบับ), ช็อตไม่มีบทพูด {st['silent_shots']}"
                    + (" — คำเตือน: " + " / ".join(script["warnings"][:2]) if script["warnings"] else ""))
        return script

    def _scenes(self, pid, script) -> None:
        eps = len(script["episodes"]) or 1
        self._agent(pid, 3, "done", f"{eps} ตอน, {len(script['scenes'])} ฉาก, {len(script['shots'])} ช็อต — ลำดับตามต้นฉบับ")

    def _clip_rows(self, pid: int) -> list[dict]:
        return [dict(r) for r in self.store.conn().execute("SELECT * FROM project_clips WHERE project_id=? ORDER BY upload_order", (pid,))]

    def _inspect(self, pid, uid) -> list[dict]:
        rows = self._clip_rows(pid)
        out, problems = [], 0
        for i, r in enumerate(rows):
            self._check_pause(pid)
            try:
                m = self.media.get(uid, r["media_id"])
            except NotFound:
                continue
            ins = json.loads(r["inspect"]) if r["inspect"] else None
            if ins is None:
                ins = inspect_clip(m["path"])
                with self.store.conn() as c:
                    c.execute("UPDATE project_clips SET inspect=? WHERE project_id=? AND media_id=?",
                              (json.dumps(ins, ensure_ascii=False), pid, r["media_id"]))
            problems += bool(ins["problems"])
            out.append({"id": m["id"], "name": m["name"], "path": m["path"], "duration": ins.get("duration", m["duration"]),
                        "width": m["width"], "height": m["height"], "has_audio": bool(m["has_audio"]),
                        "inspect": ins, "row": r})
            if i % 5 == 0:
                self._agent(pid, 8, "running", f"ตรวจไฟล์ {i + 1}/{len(rows)}")
        # exact duplicates (same file content)
        seen: dict[str, str] = {}
        for c in out:
            h = c["inspect"].get("sha256")
            if h in seen:
                c["duplicate_of"] = seen[h]
                c["inspect"]["problems"].append("ไฟล์ซ้ำกับคลิปอื่นทุกไบต์")
            elif h:
                seen[h] = c["id"]
        bad = [c for c in out if c["inspect"].get("broken")]
        msg = f"ตรวจ {len(out)} คลิป: เปิดไม่ได้ {len(bad)}, มีข้อสังเกต {problems}"
        self._agent(pid, 8, "warning" if problems else "done", msg)
        return [c for c in out if not c["inspect"].get("broken")]

    def _dialogue(self, pid, uid, clips, script) -> list[dict]:
        need_asr = any(c["inspect"].get("speech") for c in clips)
        if need_asr:
            try:
                self.asr.check()
            except ASRUnavailable as e:
                self._agent(pid, 5, "failed", str(e))
                raise
        done = 0
        for i, c in enumerate(clips):
            self._check_pause(pid)
            r = c["row"]
            asr = json.loads(r["asr"]) if r["asr"] else None
            if asr is None:
                if not c["inspect"].get("speech"):
                    asr = {"text": "", "segments": [], "engine": "silence"}
                elif isinstance(self.asr, FakeASR):
                    asr = self.asr.transcribe_named(c["name"], c["duration"])
                else:
                    t0 = time.time()
                    names = [ch["name"] for ch in script.get("characters", [])] or sorted(
                        {d["speaker"] for sh in script["shots"] for d in sh["dialogue"] if d["speaker"]})
                    asr = self.asr.transcribe(c["path"], [tuple(x) for x in c["inspect"]["speech_regions"]], names)
                    asr["seconds"] = round(time.time() - t0, 1)
                with self.store.conn() as cn:
                    cn.execute("UPDATE project_clips SET asr=? WHERE project_id=? AND media_id=?",
                               (json.dumps(asr, ensure_ascii=False), pid, c["id"]))
                done += 1
            c["asr"] = asr
            c["transcript"] = asr["text"]
            c["speech"] = bool(asr["text"])
            if i % 3 == 0:
                self._agent(pid, 5, "running", f"ถอดเสียงพูด {i + 1}/{len(clips)}")
        spoken = sum(1 for c in clips if c["speech"])
        self._agent(pid, 5, "done", f"มีเสียงพูด {spoken}/{len(clips)} คลิป (ถอดเสียงใหม่ {done} คลิป, ที่เหลือใช้ผลที่บันทึกไว้)")
        return clips

    def _visual(self, pid, clips) -> list[dict]:
        """AI 4: first/middle/last frame thumbnails (cached), used for continuity and silent clips."""
        from .visual import thumbs
        for i, c in enumerate(clips):
            self._check_pause(pid)
            v = c["row"].get("visual")
            v = json.loads(v) if v else None
            if v is None:
                v = thumbs(c["path"], c["duration"])
                with self.store.conn() as cn:
                    cn.execute("UPDATE project_clips SET visual=? WHERE project_id=? AND media_id=?", (json.dumps(v), pid, c["id"]))
            c["visual"] = v
            if i % 10 == 0:
                self._agent(pid, 4, "running", f"อ่านภาพ {i + 1}/{len(clips)}")
        vision = bool(self.assistant and self.assistant.server.mmproj)
        self._agent(pid, 4, "done", f"เก็บภาพต้นและท้ายของ {len(clips)} คลิปไว้ตรวจความต่อเนื่อง"
                    + ("; Qwen ดูภาพได้ ใช้ช่วยวางคลิปที่ไม่มีเสียงพูด" if vision else "; ยังไม่ได้เปิดโหมดดูภาพของ Qwen"))
        return clips

    def _vl_votes(self, pid, script, clips, cands: dict) -> dict:
        """Ask Qwen (vision) which silent shot a silent clip shows, among its few candidates."""
        if not (self.assistant and self.assistant.server.mmproj) or not cands:
            return {}
        from .visual import keyframe_jpeg
        by_id = {c["id"]: c for c in clips}
        votes = {}
        gate = self.assistant._gen
        for k, (cid, shot_ids) in enumerate(cands.items()):
            if len(shot_ids) < 2:
                continue
            self._check_pause(pid)
            c = by_id[cid]
            opts = [" ".join(script["shots"][si]["action"])[:160] or label(script["shots"][si]) for si in shot_ids[:6]]
            if not gate.acquire(timeout=900):
                break
            try:
                t0 = time.time()
                choice = self.assistant.llm.choose_for_image(keyframe_jpeg(c["path"], c["duration"]),
                                                             "ภาพนี้มาจากช็อตไหนของละคร", opts)
                if choice is not None:
                    votes[cid] = shot_ids[choice]
                logs.event(log, "vision vote", project=pid, clip=cid, choice=choice, seconds=round(time.time() - t0, 1))
            except Exception as e:
                logs.event(log, "vision vote failed", logging.WARNING, project=pid, error=str(e)[:200])
            finally:
                gate.release()
            self._agent(pid, 6, "running", f"ให้ AI ดูภาพคลิปที่ไม่มีเสียงพูด {k + 1}/{len(cands)}")
        return votes

    def _continuity(self, pid, script, clips, timeline) -> None:
        """AI 9: consecutive clips in the same scene should continue visually; big jumps are flagged."""
        from .visual import similarity
        by_id = {c["id"]: c for c in clips}
        flagged = []
        v = timeline["video"]
        for a, b in zip(v, v[1:]):
            sa, sb = script["shots"][a["shots"][-1]], script["shots"][b["shots"][0]]
            if sa["scene"] != sb["scene"]:
                continue
            sim = similarity((by_id[a["clip"]].get("visual") or {}).get("last"), (by_id[b["clip"]].get("visual") or {}).get("first"))
            b["join_similarity"] = sim
            if sim < 0.35:
                flagged.append({"type": "continuity", "clip": b["clip"],
                                "message": f"รอยต่อ {label(sa)} → {label(sb)} ภาพเปลี่ยนมาก (คล้ายกัน {int(sim * 100)}%) ทั้งที่อยู่ฉากเดียวกัน "
                                           f"— ตรวจหน้า/ชุด/สถานที่ ว่าต่อเนื่องหรือไม่"})
        timeline["issues"] += flagged
        self._agent(pid, 9, "warning" if flagged else "done",
                    (f"พบรอยต่อที่ภาพเปลี่ยนมาก {len(flagged)} จุด (ดูแท็บตรวจข้อผิดพลาด)" if flagged else "รอยต่อในฉากเดียวกันต่อเนื่องดี")
                    + " — ตรวจจากภาพรวม ยังแยกแยะใบหน้า/เสื้อผ้าทีละจุดไม่ได้")

    EMOTIONS = {"เศร้า": ("ร้องไห้", "น้ำตา", "เสียใจ", "สะอื้น", "เศร้า", "อาลัย"),
                "โกรธ": ("โกรธ", "ตะคอก", "ชี้หน้า", "ตบ", "ด่า", "เกรี้ยวกราด", "ขึ้นเสียง"),
                "ระทึก": ("ตกใจ", "วิ่ง", "หนี", "ไล่", "สะดุ้ง", "ตำรวจ", "ชะงัก", "ระทึก"),
                "โรแมนติก": ("กอด", "จับมือ", "หอม", "รัก", "สบตา"),
                "ตลก": ("หัวเราะ", "ขำ", "ตลก", "แซว", "ยิ้มขำ")}

    def _emotion(self, pid, script) -> None:
        """AI 10: mood of each scene from the words of the script. Clips are never trimmed, so dialogue
        and reactions keep their full length; the mood is shown for the editor and QA."""
        moods = {}
        for sc in script["scenes"]:
            text = " ".join(" ".join(sh["action"]) + " ".join(d["text"] for d in sh["dialogue"])
                            for sh in script["shots"] if sh["scene"] == sc["id"])
            score = {m: sum(text.count(w) for w in ws) for m, ws in self.EMOTIONS.items()}
            best = max(score, key=score.get)
            moods[sc["id"]] = best if score[best] else "ปกติ"
            sc["mood"] = moods[sc["id"]]
        counts = {}
        for m in moods.values():
            counts[m] = counts.get(m, 0) + 1
        self._agent(pid, 10, "done", "อารมณ์รายฉาก: " + ", ".join(f"{k} {v}" for k, v in counts.items())
                    + " — ใช้กำหนดจังหวะตัดต่อ (ค้างภาพหลังพูด, รอยต่อ)")

    def _cinematic(self, pid, script, clips, timeline) -> None:
        """AI 12: brightness of each clip vs the rest of its scene; optional gentle correction."""
        st = json.loads(self.store.conn().execute("SELECT settings FROM projects WHERE id=?", (pid,)).fetchone()["settings"])
        by_id = {c["id"]: c for c in clips}
        scene_l: dict[str, list] = {}
        for it in timeline["video"]:
            l = (by_id[it["clip"]].get("visual") or {}).get("luma")
            if l is not None:
                scene_l.setdefault(script["shots"][it["shots"][0]]["scene"], []).append((it, l))
        off = 0
        for sc, items in scene_l.items():
            if len(items) < 3:
                continue
            med = sorted(l for _, l in items)[len(items) // 2]
            for it, l in items:
                diff = l - med
                if abs(diff) > 25:
                    off += 1
                    if st.get("match_light"):
                        it["brightness"] = round(max(-0.15, min(0.15, -diff / 255)), 3)
                    timeline["issues"].append({"type": "light", "clip": it["clip"], "message":
                        f"“{it['name']}” {'สว่าง' if diff > 0 else 'มืด'}กว่าคลิปอื่นในฉากเดียวกัน ({int(l)} เทียบ {int(med)})"
                        + (" — ปรับให้ใกล้เคียงแล้ว" if st.get("match_light") else " — เปิด 'ปรับแสงให้ใกล้กัน' ได้ในหน้าตั้งค่า")})
        scene_mood = {sc["id"]: sc.get("mood", "ปกติ") for sc in script["scenes"]}
        for it in timeline["video"]:
            it["scene"] = script["shots"][it["shots"][0]]["scene"]
        before = timeline["duration"]
        notes = cutter.plan_cuts(timeline, by_id, lambda it: scene_mood.get(it.get("scene"), "ปกติ"), st.get("pacing", "auto"))
        timeline["cuts"] = notes
        ai = self._ai_editor(pid, script, by_id, timeline, scene_mood, st)
        fades = sum(1 for it in timeline["video"][1:] if it.get("fade_in"))
        if st.get("pacing", "auto") == "original":
            how = "ใช้คลิปเต็มความยาวตามต้นฉบับ ต่อแบบ Hard Cut"
        else:
            how = (f"ตัดช่วงว่าง/ภาพเสียตามจังหวะอารมณ์: {before:.1f} → {timeline['duration']:.1f} วินาที, "
                   f"รอยต่อแบบจางผ่านดำ {fades} จุด (ไม่ตัดคำพูด)")
        how = (ai + " — " if ai else "") + how
        self._agent(pid, 12, "warning" if off and not st.get("match_light") else "done",
                    how + (f"; แสงต่างจากฉาก {off} คลิป" if off else "; แสงใกล้เคียงกันทั้งฉาก")
                    + (" (ปรับให้แล้ว)" if off and st.get("match_light") else ""))

    def _ai_editor(self, pid, script, by_id, timeline, scene_mood, st) -> str:
        """Asks the owner's AI editor (Qwen3.5-9B on their Colab GPU) for the cuts, when it is online."""
        if st.get("pacing", "auto") == "original" or not st.get("ai_editor", True) or not timeline["video"]:
            return ""
        uid = self.store.conn().execute("SELECT user_id FROM projects WHERE id=?", (pid,)).fetchone()["user_id"]
        if not self.aied.online(uid):
            timeline["ai_editor"] = {"used": False, "why": "offline"}
            return ""
        self._agent(pid, 12, "running", "ส่งภาพและบทพูดให้ AI คนตัดต่อ (บน Colab ของคุณ) …")
        lines = M.dialogue_lines(script["shots"])
        clips_out = []
        n = len(timeline["video"])
        for k, it in enumerate(timeline["video"]):
            c = by_id[it["clip"]]
            spans = cutter.speech_spans(c) if it.get("lines") or it.get("unscripted") else []
            jumps = (c.get("inspect") or {}).get("jumps") or []
            frames = self.aied.make_frames(pid, c, frame_times(c["duration"], spans, jumps))
            shot = script["shots"][it["shots"][0]]
            clips_out.append({"id": c["id"], "name": c["name"], "position": k + 1, "of": n, "duration": round(c["duration"], 2),
                              "mood": scene_mood.get(it.get("scene"), "ปกติ"),
                              "lines": [{"speaker": lines[li]["speaker"], "text": lines[li]["text"]} for li in it.get("lines", [])],
                              "action": " ".join(shot.get("action", []))[:400], "speech": spans, "jumps": jumps,
                              "rule": {k2: it.get(k2) for k2 in ("in", "out", "fade_in", "fade_out")}, "frames": frames})
        payload = {"version": 1, "project": self.store.conn().execute("SELECT name FROM projects WHERE id=?", (pid,)).fetchone()["name"],
                   "title": script.get("title", ""), "clips": clips_out}
        job = self.aied.create_job(uid, pid, payload)
        per_clip = float(os.environ.get("LIVEAI_AIED_SECONDS_PER_CLIP", "60"))     # Colab T4: well under a minute
        res = self.aied.wait(job, min(float(os.environ.get("LIVEAI_AIED_MAX_WAIT", "3600")), 180 + per_clip * n),
                             check=lambda: self._check_pause(pid))
        if not res or not isinstance(res.get("plan"), dict):
            timeline["ai_editor"] = {"used": False, "why": "no answer"}
            return "AI คนตัดต่อไม่ตอบในเวลาที่กำหนด จึงใช้กฎตัดต่อแทน"
        labelled = {cid: p for cid, p in res["plan"].items() if isinstance(p, dict) and isinstance(p.get("frames"), list) and cid in by_id}
        if labelled:
            # the model judged each still (broken? how expressive?); the cutting rules turn that into cuts
            seen = {cid: cutter.labels_to_clip(by_id[cid], p["frames"]) for cid, p in labelled.items()}
            notes = cutter.plan_cuts(timeline, by_id, lambda it: scene_mood.get(it.get("scene"), "ปกติ"), st.get("pacing", "auto"))
            for cid, v in seen.items():
                if v["broken"]:
                    notes.insert(0, f"“{by_id[cid]['name']}”: AI เห็นภาพเสียช่วง " + ", ".join(f"{a:.1f}–{b:.1f}" for a, b in v["broken"]) + " วิ")
        else:
            notes = cutter.apply_plan(timeline, by_id, res["plan"])
        timeline["ai_editor"] = {"used": True, "model": str(res.get("model", ""))[:80], "notes": notes,
                                 "reasons": {cid: str(p.get("why", ""))[:300] for cid, p in res["plan"].items() if isinstance(p, dict)}}
        timeline["cuts"] = [f"“{by_id[cid]['name']}”: {why}" for cid, why in timeline["ai_editor"]["reasons"].items()
                            if why and cid in by_id] + notes
        return f"AI คนตัดต่อ ({res.get('model') or 'Qwen3.5-9B'}) ดูภาพแล้วกำหนดจุดตัด {len(res['plan'])} คลิป"

    def _audio(self, pid, clips, timeline) -> None:
        """AI 13: Dialogue Only = local RNNoise speech denoiser + quieting the gaps between lines."""
        st = json.loads(self.store.conn().execute("SELECT settings FROM projects WHERE id=?", (pid,)).fetchone()["settings"])
        if st["audio"] != "dialogue_only":
            self._agent(pid, 13, "done", "ใช้เสียงจากคลิปต้นฉบับตามเดิม")
            return
        engine = denoise_filter()[1]
        lead = f"ลดเสียงลม/รถ/ฝน/เสียงรบกวนด้วย {engine} บนเครื่อง และ" if engine else "(ตัวลดเสียงรบกวนใช้กับ ffmpeg เครื่องนี้ไม่ได้) "
        self._agent(pid, 13, "done" if engine else "warning", f"Dialogue Only: {lead}ลดเสียงช่วงที่ไม่มีคนพูดลง 20 dB "
                    "— เพลงที่ดังซ้อนใต้เสียงพูดพร้อมกันอาจยังได้ยินบ้าง ระบบไม่สร้างหรือแก้คำพูด")

    def _match(self, pid, script, clips) -> dict:
        made = M.file_order([c["name"] for c in clips])
        for c in clips:
            c["order"] = (made[c["name"]], c["row"]["upload_order"])
        slim = [{k: c.get(k) for k in ("id", "name", "order", "duration", "speech", "transcript", "duplicate_of")} for c in clips]
        visual = {c["id"]: c["visual"] for c in clips if c.get("visual")}
        st = json.loads(self.store.conn().execute("SELECT settings FROM projects WHERE id=?", (pid,)).fetchone()["settings"])
        extra = bool(st.get("unscripted", True))
        r = M.match(script, slim, visual, unscripted=extra)
        unsure = {cid: r["silent_candidates"][cid] for cid, a in r["assign"].items()
                  if a["how"].startswith("silent") and a["confidence"] != "สูง" and cid in r["silent_candidates"]}
        votes = self._vl_votes(pid, script, clips, unsure)
        if votes:
            r = M.match(script, slim, visual, votes, unscripted=extra)
        names = {c["id"]: c["name"] for c in clips}
        for c in clips:
            if c.get("duplicate_of") and not any(i.get("clip") == c["id"] for i in r["issues"]):
                r["issues"].append({"type": "duplicate", "clip": c["id"], "of": c["duplicate_of"],
                                    "message": f"คลิป “{c['name']}” เป็นไฟล์เดียวกับ “{names.get(c['duplicate_of'], '')}” จะไม่ถูกใช้ซ้ำ"})
        for c in clips:
            for i in r["issues"]:
                if i.get("clip") == c["id"] and i["type"] == "duplicate":
                    c["duplicate_of"] = i["of"]
        hi = sum(1 for a in r["assign"].values() if a["confidence"] == "สูง")
        outside = sum(1 for a in r["assign"].values() if a.get("unscripted"))
        msg = f"จับคู่ได้ {len(r['assign']) - outside}/{len(clips)} คลิป (มั่นใจสูง {hi}), ช็อตที่ขาด {len(r['missing'])}"
        if outside:
            msg += f", คลิปที่พูดนอกบท {outside} คลิป ใส่ตามลำดับที่สร้าง"
        self._agent(pid, 6, "warning" if r["issues"] or r["missing"] else "done", msg)
        for m in r["missing"]:
            m["regen"] = M.regen_prompt(script["shots"][m["shot"]], script)
        return r

    def _manual_order(self, pid, script, clips, plan) -> dict:
        """The owner arranged the clips (Timeline tab): exactly those clips in exactly that order. The
        script still gives each clip its lines (subtitles) and its scene (pacing, transitions)."""
        st = json.loads(self.store.conn().execute("SELECT settings FROM projects WHERE id=?", (pid,)).fetchone()["settings"])
        order = [cid for cid in (st.get("order") or []) if cid in {c["id"] for c in clips}]
        if not order:
            return plan
        assign, last_shot = plan["assign"], 0
        for cid in order:
            a = assign.get(cid)
            if a and a["shots"]:
                last_shot = a["shots"][-1]
            else:                     # a clip the matching had left out: the owner wants it here
                c = next(x for x in clips if x["id"] == cid)
                assign[cid] = {"lines": [], "shots": [last_shot], "score": 0.0, "margin": 0.0, "how": "manual", "confidence": "สูง",
                               # it speaks: its words are kept whole like any spoken clip (no subtitles: not matched to a line)
                               **({"unscripted": True} if c.get("speech") and c.get("transcript") else {})}
            assign[cid]["confirmed"] = True
        for cid in [k for k in assign if k not in order]:
            del assign[cid]
        plan["timeline"] = list(dict.fromkeys(order))
        plan["manual"] = True
        plan["issues"] = [i for i in plan["issues"] if i.get("clip") not in order or i.get("type") not in ("uncertain", "alternate", "duplicate", "wrong_place", "extra", "unmatched")]
        self._agent(pid, 6, "done", f"ใช้ลำดับที่คุณจัดเอง {len(order)} คลิป (บทใช้ทำซับและจังหวะตัดต่อ)")
        return plan

    def _sequence(self, pid, script, clips, plan) -> dict:
        by_id = {c["id"]: c for c in clips}
        lines = M.dialogue_lines(script["shots"])
        st = json.loads(self.store.conn().execute("SELECT settings FROM projects WHERE id=?", (pid,)).fetchone()["settings"])
        items, t = [], 0.0
        issues = []
        slowed = 0
        for cid in plan["timeline"]:
            c, a = by_id[cid], plan["assign"][cid]
            shot = script["shots"][a["shots"][0]]
            if shot.get("duration") and not a.get("unscripted") and (c["duration"] < shot["duration"] * 0.5 or c["duration"] > shot["duration"] * 2):
                issues.append({"type": "duration", "clip": cid, "message": f"“{c['name']}” ยาว {c['duration']:.1f} วิ แต่บทกำหนด "
                                                                           f"{shot['duration']:g} วิ ({label(shot)})"})
            items.append({"clip": cid, "name": c["name"], "shots": a["shots"], "lines": a["lines"], "start": round(t, 3),
                          "in": 0.0, "out": c["duration"], "confidence": a["confidence"], "how": a["how"],
                          # how long its words take to say: the cuts never keep less than that
                          "say_seconds": round(sum(cutter.say_seconds(lines[li]["text"]) for li in a["lines"]) if a["lines"]
                                               else cutter.say_seconds(c.get("transcript") or ""), 2),
                          **({"unscripted": True} if a.get("unscripted") else {}),
                          **({"confirmed": True} if a.get("confirmed") else {})})
            if st.get("speech_speed", "auto") == "auto" and (a["lines"] or a.get("unscripted")):
                # words said too fast (AI voices sometimes rush a long line): play the clip a little slower
                text = "".join(lines[li]["text"] for li in a["lines"]) if a["lines"] else c.get("transcript") or ""
                rate = cutter.speech_rate(c, text)
                f = cutter.slow_factor(rate)
                if f > 1.0:
                    items[-1].update(slow=f, rate=rate)
                    slowed += 1
            t += c["duration"] * float(items[-1].get("slow") or 1.0)
        tl = {"video": items, "duration": round(t, 3), "issues": issues}
        self._agent(pid, 7, "done", f"เรียง {len(items)} คลิป ยาวรวม {t:.1f} วินาที"
                    + (f" — คลิปที่พูดเร็วเกินไป {slowed} คลิป เล่นช้าลงให้ฟังเป็นปกติ (ไม่เกิน 25%)" if slowed else ""))
        return tl

    def _render(self, pid, uid, clips, timeline) -> dict:
        p = self.store.conn().execute("SELECT settings FROM projects WHERE id=?", (pid,)).fetchone()
        st = json.loads(p["settings"])
        by_id = {c["id"]: c for c in clips}
        first = by_id[timeline["video"][0]["clip"]] if timeline["video"] else None
        if not first:
            raise StudioError("ไม่มีคลิปที่จับคู่ได้เลย จึงตัดต่อไม่ได้")
        W, H = out_size(st, first)
        fps = st["fps"]
        dialogue_only = st["audio"] == "dialogue_only"
        cache = self.root / str(pid) / "segments"
        cache.mkdir(parents=True, exist_ok=True)
        segs = []
        for i, item in enumerate(timeline["video"]):
            self._check_pause(pid)
            c = by_id[item["clip"]]
            bright = item.get("brightness", 0.0)
            cut = (item.get("in", 0.0), item.get("out", c["duration"]))
            fades = (item.get("fade_in", 0.0), item.get("fade_out", 0.0))
            speed = float(item.get("slow") or 1.0)
            key = hashlib.sha1(f"{W}x{H}@{fps}|{dialogue_only}|{bright}|{cut}|{fades}|{speed}".encode()).hexdigest()[:8]
            seg = cache / f"{c['id']}-{key}.mp4"
            if not seg.exists():    # render checkpoint: finished segments are reused
                # where people speak: the recogniser's segments are more reliable than loudness in noisy clips
                spoken = [(x["start"], x["end"]) for x in (c.get("asr") or {}).get("segments") or []]
                normalize_segment(c, seg, W, H, fps, dialogue_only=dialogue_only, brightness=bright,
                                  regions=spoken or (c.get("inspect") or {}).get("speech_regions") or [], cut=cut, fades=fades,
                                  speed=speed)
            segs.append(seg)
            self._agent(pid, 11, "running", f"เตรียมคลิป {i + 1}/{len(timeline['video'])} ({W}x{H} {fps}fps)")
        out_dir = self.media.root / str(uid)
        out = out_dir / f"{os.urandom(12).hex()}.mp4"
        lst = cache.parent / "concat.txt"
        lst.write_text("".join(f"file '{s}'\n" for s in segs), encoding="utf-8")
        # one loudness for the whole video (social platforms: about -14 LUFS, peaks under -1.5 dB): AI clips
        # come out at very different levels (seen: -31 LUFS next to -8 LUFS)
        r = subprocess.run(["ffmpeg", "-loglevel", "error", "-nostdin", "-y", "-f", "concat", "-safe", "0", "-i", str(lst),
                            "-map", "0:v:0", "-map", "0:a:0", "-c:v", "copy", "-af", "loudnorm=I=-14:TP=-1.5:LRA=11",
                            "-ar", "48000", "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(out)],
                           capture_output=True, text=True, timeout=7200)
        if r.returncode != 0 or not out.exists():
            raise RuntimeError("รวมคลิปไม่สำเร็จ: " + r.stderr[-300:])
        name = self.store.conn().execute("SELECT name FROM projects WHERE id=?", (pid,)).fetchone()["name"]
        m = self.media.add_file(uid, out, f"{name}.mp4", "studio")
        self._save_result(pid, video={"id": m["id"], "name": m["name"], "duration": m["duration"]})
        self._agent(pid, 11, "done", f"ส่งออก MP4 H.264/AAC {W}x{H} {fps}fps ยาว {m['duration']:.1f} วินาที")
        return {**m, "W": W, "H": H}

    def _subtitles(self, pid, uid, script, clips, timeline, out) -> dict:
        st = json.loads(self.store.conn().execute("SELECT settings FROM projects WHERE id=?", (pid,)).fetchone()["settings"])
        lines = M.dialogue_lines(script["shots"])
        by_id = {c["id"]: c for c in clips}
        cues = []
        for item in timeline["video"]:
            c = by_id[item["clip"]]
            ls = [lines[li] for li in item["lines"]]
            if not ls:
                continue
            segs = (c.get("asr") or {}).get("segments") or []
            spans = align_lines(ls, segs, c["duration"])
            t_in, keep = item.get("in", 0.0), item.get("out", c["duration"]) - item.get("in", 0.0)
            f = float(item.get("slow") or 1.0)
            spans = [(min(max(0.0, a - t_in), keep) * f, min(max(0.0, b - t_in), keep) * f) for a, b in spans]
            for l, (a, b) in zip(ls, spans):
                if cues and any(c["li"] == l["li"] for c in cues):
                    prev = next(c for c in cues if c["li"] == l["li"])      # line continues from the previous clip
                    prev["end"] = item["start"] + b
                    continue
                text = (f"{l['speaker']}: " if st.get("speaker_in_subs") and l["speaker"] else "") + l["text"]
                cues.append({"start": item["start"] + a, "end": item["start"] + b, "text": text, "speaker": l["speaker"], "li": l["li"]})
        for i in range(len(cues) - 1):       # no overlaps
            cues[i]["end"] = min(cues[i]["end"], cues[i + 1]["start"] - 0.05)
            if cues[i]["end"] <= cues[i]["start"]:
                cues[i]["end"] = cues[i]["start"] + 0.5
        if st["subtitles"] == "none":
            self._agent(pid, 14, "done", "ปิดซับตามที่ตั้งค่าไว้")
            return {"cues": cues, "srt": None}
        srt = "".join(f"{i + 1}\n{srt_time(c['start'])} --> {srt_time(c['end'])}\n{c['text']}\n\n" for i, c in enumerate(cues))
        path = self.media.root / str(uid) / f"{os.urandom(12).hex()}.srt"
        path.write_text(srt, encoding="utf-8")
        name = self.store.conn().execute("SELECT name FROM projects WHERE id=?", (pid,)).fetchone()["name"]
        srt_rec = self._register_text(uid, path, f"{name}.srt")
        self._save_result(pid, srt=srt_rec)
        msg = f"สร้างซับ {len(cues)} บรรทัด (ข้อความตรงตามบท จับเวลาจากเสียงพูดจริง) ส่งเป็นไฟล์ .srt แยก"
        if st["subtitles"] == "burn":
            from ..editor import write_ass
            ass = path.with_suffix(".ass")
            write_ass([{"text": c["text"], "start": c["start"], "end": c["end"], "position": "bottom", "size": "medium"}
                       for c in cues], out["W"], out["H"], ass)
            from ..editor import FONT_DIR
            burned = path.with_name(f"{os.urandom(12).hex()}.mp4")
            r = subprocess.run(["nice", "-n", "10", "ffmpeg", "-loglevel", "error", "-nostdin", "-y", "-i", out["path"], "-vf",
                                f"subtitles=filename={ass}:fontsdir={FONT_DIR}", "-c:v", "libx264", "-preset", "veryfast",
                                "-crf", "21", "-c:a", "copy", "-movflags", "+faststart", str(burned)],
                               capture_output=True, text=True, timeout=14400)
            ass.unlink(missing_ok=True)
            if r.returncode != 0:
                raise RuntimeError("ฝังซับไม่สำเร็จ: " + r.stderr[-300:])
            m = self.media.add_file(uid, burned, f"{name} (มีซับ).mp4", "studio")
            self._save_result(pid, video_subs={"id": m["id"], "name": m["name"], "duration": m["duration"]})
            msg += " และฝังซับเป็นอีกไฟล์ (ไฟล์ไม่มีซับยังอยู่)"
        self._agent(pid, 14, "done", msg)
        return {"cues": cues, "srt": srt_rec}

    def _register_text(self, uid: int, path: Path, name: str) -> dict:
        mid = path.stem
        with self.store.conn() as c:
            c.execute("INSERT INTO media(id,user_id,kind,name,path,size,duration,width,height,has_audio,origin,created) "
                      "VALUES (?,?,?,?,?,?,0,0,0,0,'studio',?)", (mid, uid, "text", name, str(path), path.stat().st_size, time.time()))
        return {"id": mid, "name": name}

    def _qa(self, pid, uid, script, clips, timeline, out, subs, plan) -> dict:
        problems, checks = [], []
        info = av_info(out["path"])
        expect = timeline["duration"]
        checks.append(("ไฟล์ MP4 เปิดและถอดรหัสได้ครบทั้งไฟล์", info["decodes"]))
        checks.append((f"ความยาวรวม {info['duration']:.2f} วิ ตรงกับ Timeline {expect:.2f} วิ", abs(info["duration"] - expect) <= 0.25))
        checks.append(("มีทั้งภาพและเสียง", info["video"] and info["audio"]))
        checks.append((f"ภาพและเสียงยาวเท่ากัน (ต่างกัน {info['av_diff']:.2f} วิ)", info["av_diff"] <= 0.2))
        used = [i["clip"] for i in timeline["video"]]
        checks.append(("ไม่มีคลิปซ้ำใน Timeline", len(used) == len(set(used))))
        manual = plan.get("manual")
        checks.append((f"ครบทุกช็อตตามบท (ขาด {len(plan['missing'])})" + (" — คุณจัดลำดับเอง" if manual and plan["missing"] else ""),
                       manual or not plan["missing"]))
        lines = M.dialogue_lines(script["shots"])
        cue_lines = [c["li"] for c in (subs or {}).get("cues") or []]
        checks.append(("บทพูดไม่ซ้ำ", len(cue_lines) == len(set(cue_lines))))
        order_ok = all(timeline["video"][k]["shots"][0] <= timeline["video"][k + 1]["shots"][0] for k in range(len(timeline["video"]) - 1))
        checks.append(("ลำดับตามที่คุณจัด" if manual else "ลำดับเนื้อเรื่องเดินหน้าตามบท", manual or order_ok))
        if subs and subs.get("cues"):
            texts_ok = all(c["text"].endswith(lines[c["li"]]["text"]) for c in subs["cues"])
            checks.append(("ซับตรงตามบททุกตัวอักษร", texts_ok))
            in_range = all(0 <= c["start"] < c["end"] <= info["duration"] + 0.05 for c in subs["cues"])
            checks.append(("เวลาซับอยู่ในวิดีโอ ไม่ทับกัน", in_range))
        by_id = {c["id"]: c for c in clips}
        cut_words = [i["name"] for i in timeline["video"] if i.get("lines") and any(
            a < i.get("in", 0) - 0.05 or b > i.get("out", 1e9) + 0.05 for a, b in cutter.speech_spans(by_id[i["clip"]]))]
        checks.append(("ไม่ตัดคำพูดทิ้ง" + (f" (ตัดโดน: {', '.join(cut_words[:3])})" if cut_words else ""), not cut_words))
        low = [i for i in timeline["video"] if i["confidence"] == "ต่ำ" and not i.get("unscripted") and not manual]
        checks.append((f"คลิปที่จับคู่ความมั่นใจต่ำ {len(low)} คลิป (ต้องให้คนตรวจ)", not low))
        for name, ok in checks:
            if not ok:
                problems.append(name)
        qa = {"passed": not problems, "checks": [{"name": n, "ok": bool(o)} for n, o in checks], "problems": problems,
              "output": info}
        if plan["missing"] and not plan.get("manual"):
            problems.insert(0, "พบคลิปที่ขาด " + str(len(plan["missing"])) + " ช็อต ได้แก่ " + ", ".join(m["label"] for m in plan["missing"][:8]))
        self._agent(pid, 15, "done" if qa["passed"] else "warning",
                    "ผ่านทุกข้อ" if qa["passed"] else f"ไม่ผ่าน {len(problems)} ข้อ: " + "; ".join(problems[:3]))
        if not qa["passed"]:
            # the file is kept as a review copy, never presented as the final version
            r = json.loads(self.store.conn().execute("SELECT result FROM projects WHERE id=?", (pid,)).fetchone()["result"] or "{}")
            r["review"] = r.pop("video", None)
            r["video"] = None
            with self.store.conn() as c:
                c.execute("UPDATE projects SET result=? WHERE id=?", (json.dumps(r, ensure_ascii=False), pid))
        return qa

    def _save_result(self, pid: int, **kw) -> None:
        with self.store.conn() as c:
            r = json.loads(c.execute("SELECT result FROM projects WHERE id=?", (pid,)).fetchone()["result"] or "{}")
            r.update(kw)
            c.execute("UPDATE projects SET result=? WHERE id=?", (json.dumps(r, ensure_ascii=False), pid))


# ---------------------------------------------------------------------- ffmpeg helpers
def inspect_clip(path: str) -> dict:
    """AI 8 per-clip checks: opens? video/audio present? black picture? speech present?"""
    out = {"problems": [], "broken": False}
    try:
        info = probe(Path(path))
    except MediaError as e:
        return {"problems": [f"ไฟล์เปิดไม่ได้: {e}"], "broken": True}
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    out.update(sha256=h.hexdigest(), duration=info["duration"], has_audio=info["has_audio"])
    if info["kind"] != "video":
        return {**out, "problems": ["ไม่ใช่ไฟล์วิดีโอ"], "broken": True}
    if info["duration"] < 0.5:
        out["problems"].append("สั้นผิดปกติ (ไม่ถึงครึ่งวินาที)")
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostdin", "-i", path, "-vf", "blackdetect=d=0.5:pix_th=0.10", "-an", "-f", "null", "-"],
                       capture_output=True, text=True, timeout=600)
    black = sum(float(x) for x in __import__("re").findall(r"black_duration:([\d.]+)", r.stderr))
    out["black_seconds"] = round(black, 2)
    if black >= max(0.5, info["duration"] * 0.5):
        out["problems"].append(f"ภาพดำ {black:.1f} วินาที")
    if "Error" in r.stderr and "decode" in r.stderr.lower():
        out["problems"].append("มีเฟรมเสีย ถอดรหัสไม่ได้บางส่วน")
    jumps = picture_jumps(path)
    out["jumps"] = jumps
    for a, b in jumps:
        out["problems"].append(f"ภาพเปลี่ยนกะทันหันช่วง {a:.1f}–{b:.1f} วินาที (ซูม/ภาพกระโดด/ภาพซ้อน — ควรดูช่วงนี้)")
    if info["has_audio"]:
        regions = speech_regions(path, info["duration"])
        out["audio_db"] = list(loudness(path))
        lufs = integrated_lufs(path)
        out["lufs"] = lufs
        if lufs is not None and lufs < -26:
            out["problems"].append(f"เสียงเบามาก ({lufs:.0f} LUFS) เทียบกับระดับปกติ — วิดีโอที่ส่งออกถูกปรับความดังให้เท่ากันทั้งเรื่อง")
        if out["audio_db"][1] > -1.0:
            out["problems"].append(f"เสียงดังจนเกือบแตก (พีก {out['audio_db'][1]:.1f} dB)")
        out["speech_regions"] = regions
        out["speech"] = sum(b - a for a, b in regions) >= 0.4
    else:
        out["speech_regions"], out["speech"] = [], False
        out["problems"].append("ไม่มีเสียง")
    return out


def clip_notes(clips: list[dict]) -> list[dict]:
    """What the clip inspector noticed, per clip, for the error report (does not stop the export)."""
    out = []
    for c in clips:
        for p in (c.get("inspect") or {}).get("problems", []):
            out.append({"type": "clip", "clip": c["id"], "message": f"คลิป “{c['name']}”: {p}"})
        if (c.get("inspect") or {}).get("speech") and not c.get("transcript"):
            out.append({"type": "clip", "clip": c["id"], "message": f"คลิป “{c['name']}”: มีเสียงแต่ไม่พบคำพูด "
                        "(เป็นเพลง เสียงร้องไห้ หรือบทพูดไม่ออกมาตอนสร้างคลิป)"})
    return out


def picture_jumps(path: str, threshold: float = 0.8) -> list[tuple[float, float]]:
    """Stretches where the picture changes a lot within half a second (sudden zoom, jump, see-through
    figure — common in AI-generated clips). Compares tiny 24x24 frames twice per second."""
    try:
        raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-nostdin", "-i", path, "-map", "0:v:0", "-vf",
                              "fps=2,scale=24:24", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                             capture_output=True, timeout=600).stdout
    except (OSError, subprocess.TimeoutExpired):
        return []
    n = 24 * 24 * 3
    frames = [raw[i:i + n] for i in range(0, len(raw) - n + 1, n)]
    bad = [i for i, (a, b) in enumerate(zip(frames, frames[1:])) if 1 - sum(abs(x - y) for x, y in zip(a, b)) / n / 64 < threshold]
    spans: list[list[float]] = []
    for i in bad:
        a, b = i / 2, (i + 1) / 2
        if spans and a <= spans[-1][1] + 0.5:
            spans[-1][1] = b
        else:
            spans.append([a, b])
    return [(round(a, 1), round(b, 1)) for a, b in spans]


def integrated_lufs(path: str) -> float | None:
    try:
        r = subprocess.run(["ffmpeg", "-hide_banner", "-nostdin", "-i", path, "-vn", "-af", "ebur128", "-f", "null", "-"],
                           capture_output=True, text=True, timeout=600)
    except (OSError, subprocess.TimeoutExpired):
        return None
    m = __import__("re").findall(r"I:\s+(-?[\d.]+) LUFS", r.stderr)
    return float(m[-1]) if m else None


def normalize_segment(c: dict, seg: Path, W: int, H: int, fps: int, dialogue_only: bool = False,
                      brightness: float = 0.0, regions: list | None = None, cut: tuple | None = None,
                      fades: tuple = (0.0, 0.0), speed: float = 1.0) -> None:
    """Same size/fps/codec for every clip, picture kept whole (padded, never cropped).
    cut = (in, out) seconds kept from the clip; fades = (in, out) dip-to-black seconds.
    speed > 1 plays the clip that much slower (speech too fast): picture and sound together, the voice
    keeps its pitch (atempo), so the lips still match.
    dialogue_only: RNNoise speech denoiser + gaps between spoken lines lowered by 20 dB."""
    t0, t1 = cut or (0.0, c["duration"])
    t0, t1 = max(0.0, float(t0)), min(float(c["duration"]), float(t1))
    D = round(t1 - t0, 3)
    speed = max(1.0, min(1.3, float(speed or 1.0)))
    O = round(D * speed, 3)                       # length of the finished segment
    ss = ["-ss", f"{t0:.3f}"] if t0 > 0 else []
    regions = [(max(0.0, a - t0), min(D, b - t0)) for a, b in (regions or []) if b > t0 and a < t1]
    tmp = seg.with_suffix(".part.mp4")
    vf = (f"scale={W}:{H}:force_original_aspect_ratio=decrease:force_divisible_by=2,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:black,"
          f"setsar=1,{f'setpts=PTS*{speed:.4f},' if speed > 1 else ''}fps={fps},format=yuv420p,tpad=stop_mode=clone:stop_duration={O:.3f}")
    if brightness:
        vf += f",eq=brightness={brightness:.3f}"
    af = f"aresample=48000,{f'atempo={1 / speed:.4f},' if speed > 1 else ''}apad=whole_dur={O:.3f},aformat=channel_layouts=stereo"
    fi, fo = (round(min(float(x), O / 3), 2) for x in fades)
    if fi > 0:
        vf += f",fade=t=in:st=0:d={fi}"
        af += f",afade=t=in:st=0:d={fi}"
    if fo > 0:
        vf += f",fade=t=out:st={O - fo:.3f}:d={fo}"
        af += f",afade=t=out:st={O - fo:.3f}:d={fo}"
    audio_src = None
    if dialogue_only and c["has_audio"]:
        # pass 1, audio only: denoise + lower the gaps between lines. Kept separate from the video job
        # because some ffmpeg builds (8.0.1) hang when a denoiser and apad share one job.
        spans = [(max(0.0, a - 0.25), b + 0.25) for a, b in (regions or [])]
        gate = "+".join(f"between(t,{a:.2f},{b:.2f})" for a, b in spans) or "0"
        flt = denoise_filter()[0]
        audio_src = seg.with_suffix(".clean.wav")
        try:
            r1 = subprocess.run(["nice", "-n", "10", "ffmpeg", "-loglevel", "error", "-nostdin", "-y", *ss, "-t", f"{D:.3f}", "-i", c["path"], "-vn",
                                 "-af", f"aresample=48000,aformat=channel_layouts=mono,{flt},volume='if(gt({gate},0),1,0.1)':eval=frame",
                                 "-c:a", "pcm_s16le", str(audio_src)], capture_output=True, text=True, timeout=max(120, int(c["duration"] * 10)))
            if r1.returncode != 0:
                raise RuntimeError(r1.stderr[-200:])
        except (subprocess.TimeoutExpired, RuntimeError):
            # last resort: gap lowering only, never a silent failure
            subprocess.run(["ffmpeg", "-loglevel", "error", "-nostdin", "-y", *ss, "-t", f"{D:.3f}", "-i", c["path"], "-vn",
                            "-af", f"aresample=48000,volume='if(gt({gate},0),1,0.1)':eval=frame", "-c:a", "pcm_s16le", str(audio_src)],
                           capture_output=True, timeout=120, check=True)
    args = ["nice", "-n", "10", "ffmpeg", "-loglevel", "error", "-nostdin", "-y", *ss, "-i", c["path"]]
    if audio_src:
        args += ["-i", str(audio_src)]
        amap = "1:a:0"
    elif not c["has_audio"]:
        args += ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo"]
        amap = "1:a:0"
    else:
        amap = "0:a:0"
    args += ["-map", "0:v:0", "-map", amap, "-vf", vf,
             "-af", af, "-t", f"{O:.3f}",
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-g", str(fps * 2), "-c:a", "aac", "-b:a", "160k", "-ar", "48000",
             "-video_track_timescale", "90000", "-movflags", "+faststart", "-threads", os.environ.get("LIVEAI_FFMPEG_THREADS", "2"), str(tmp)]
    r = subprocess.run(args, capture_output=True, text=True, timeout=max(300, int(c["duration"] * 30)))
    if audio_src:
        audio_src.unlink(missing_ok=True)
    if r.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"แปลงคลิป {c['name']} ไม่สำเร็จ: {r.stderr[-200:]}")
    tmp.replace(seg)


def av_info(path: str) -> dict:
    r = subprocess.run(["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-show_format", path],
                       capture_output=True, text=True, timeout=120)
    d = json.loads(r.stdout or "{}")
    v = next((s for s in d.get("streams", []) if s.get("codec_type") == "video"), None)
    a = next((s for s in d.get("streams", []) if s.get("codec_type") == "audio"), None)
    dv = float((v or {}).get("duration") or 0)
    da = float((a or {}).get("duration") or 0)
    dec = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-i", path, "-f", "null", "-"], capture_output=True, text=True, timeout=7200)
    return {"duration": float(d.get("format", {}).get("duration") or 0), "video": v is not None, "audio": a is not None,
            "av_diff": round(abs(dv - da), 3) if v and a else 99, "decodes": dec.returncode == 0 and not dec.stderr.strip(),
            "codec": (v or {}).get("codec_name"), "size": [(v or {}).get("width"), (v or {}).get("height")]}


def align_lines(lines: list[dict], segs: list[dict], duration: float) -> list[tuple[float, float]]:
    """Times for each script line inside one clip, from the recognised speech segments."""
    n = len(lines)
    if segs:
        if n == 1:
            return [(segs[0]["start"], segs[-1]["end"])]
        # split the segments between lines in proportion to line length, keeping order
        total = sum(len(l["text"]) for l in lines)
        a, b = segs[0]["start"], segs[-1]["end"]
        out, t = [], a
        for l in lines:
            d = (b - a) * len(l["text"]) / max(total, 1)
            out.append((round(t, 2), round(t + d, 2)))
            t += d
        return out
    step = max(duration - 0.4, 0.5) / n
    return [(round(0.2 + i * step, 2), round(0.2 + (i + 1) * step, 2)) for i in range(n)]
