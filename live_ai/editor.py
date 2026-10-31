"""AI video editor.

1. Qwen turns a Thai (or English) instruction into an edit PLAN — JSON constrained by a
   schema, so the model can only choose from safe, known operations.
2. The plan is checked and clamped here (times, file numbers, speeds…). The model never
   writes ffmpeg commands.
3. ffmpeg renders the plan on the CPU: cut, join, speed, aspect ratio (crop / pad /
   blurred background), Thai text (libass, proper Thai shaping), background music,
   volume, fades.
A follow-up instruction ("ตัดให้สั้นลง", "เปลี่ยนเพลง") is applied to the previous plan.
"""
from __future__ import annotations

import json
import logging
import os
import queue
import re
import subprocess
import tempfile
import threading
import time
from pathlib import Path

from . import logs
from .store import NotFound

log = logs.get("editor")

EDIT_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS edits (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    instruction TEXT NOT NULL,
    media_ids TEXT NOT NULL,
    base_edit INTEGER,
    plan TEXT,
    summary TEXT,
    status TEXT NOT NULL,
    progress REAL NOT NULL DEFAULT 0,
    output_id TEXT,
    error TEXT,
    seconds REAL,
    created REAL NOT NULL,
    updated REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS edit_user ON edits(user_id, id);
"""

ASPECTS = {"9:16": (720, 1280), "16:9": (1280, 720), "1:1": (720, 720), "4:5": (720, 900)}
MAX_OUTPUT_S = float(os.environ.get("LIVEAI_MAX_EDIT_SECONDS", "600"))
FONT_DIR = os.environ.get("LIVEAI_FONT_DIR", "/usr/share/fonts/opentype/tlwg")
FONT_NAME = os.environ.get("LIVEAI_FONT_NAME", "Loma")

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "clips": {"type": "array", "items": {"type": "object", "properties": {
            "file": {"type": "integer"},
            "start": {"type": "number"},
            "end": {"type": "number"},
            "speed": {"type": "number"},
            "mute": {"type": "boolean"}},
            "required": ["file", "start", "end", "speed", "mute"]}},
        "aspect": {"enum": ["original", "9:16", "16:9", "1:1", "4:5"]},
        "fit": {"enum": ["blur", "crop", "pad"]},
        "texts": {"type": "array", "items": {"type": "object", "properties": {
            "text": {"type": "string"},
            "start": {"type": "number"},
            "end": {"type": "number"},
            "position": {"enum": ["top", "center", "bottom"]},
            "size": {"enum": ["small", "medium", "large"]}},
            "required": ["text", "start", "end", "position", "size"]}},
        "music_file": {"type": "integer"},
        "music_volume": {"type": "number"},
        "original_volume": {"type": "number"},
        "transition": {"enum": ["none", "fade"]},
        "fade_in": {"type": "number"},
        "fade_out": {"type": "number"},
    },
    "required": ["clips", "aspect", "fit", "texts", "music_file", "music_volume", "original_volume", "transition",
                 "fade_in", "fade_out"],
}

PLANNER_PROMPT = """คุณคือผู้ช่วยตัดต่อวิดีโอ หน้าที่คือแปลงคำสั่งของผู้ใช้เป็นแผนตัดต่อในรูปแบบ JSON เท่านั้น

กติกา:
- clips คือคลิปที่จะนำมาต่อกันตามลำดับ file คือหมายเลขไฟล์ (เริ่มที่ 1) ใช้ได้เฉพาะไฟล์วิดีโอหรือภาพ
- start และ end เป็นวินาทีของไฟล์นั้น ถ้าต้องการถึงท้ายคลิปให้ใส่ end เป็น -1 ถ้าผู้ใช้ไม่ได้บอกให้ตัด ให้ใช้ทั้งคลิป (start 0, end -1)
- เวลาแบบ 1:05 หมายถึง 65 วินาที
- ภาพนิ่งให้ใส่ start 0 และ end เท่ากับจำนวนวินาทีที่จะแสดง (ปกติ 3)
- speed 1 คือความเร็วปกติ 2 คือเร็วขึ้น 2 เท่า 0.5 คือช้าลงครึ่งหนึ่ง mute true คือปิดเสียงคลิปนั้น
- aspect: "9:16" แนวตั้ง (TikTok/Reels), "16:9" แนวนอน (YouTube), "1:1" สี่เหลี่ยม, "original" ขนาดเดิม
- fit: "blur" พื้นหลังเบลอ, "crop" ตัดขอบให้เต็มจอ, "pad" ขอบดำ
- texts คือข้อความบนวิดีโอ ใช้เฉพาะข้อความที่ผู้ใช้สั่งเท่านั้น ห้ามแต่งเอง เวลา start/end นับจากวิดีโอที่ตัดต่อเสร็จแล้ว ถ้าผู้ใช้บอกว่าตลอดคลิปให้ใส่ end เป็น -1
- music_file คือหมายเลขไฟล์เสียงที่ใช้เป็นเพลงประกอบ ถ้าไม่มีให้ใส่ 0 music_volume 0-1 (เบา ๆ ประมาณ 0.3)
- original_volume คือระดับเสียงเดิมของคลิป 1 คือปกติ 0 คือเงียบ
- transition "fade" คือเฟดดำระหว่างคลิป fade_in/fade_out คือวินาทีที่เฟดตอนต้น/ท้ายวิดีโอ (0 ถ้าไม่ได้สั่ง)
- ถ้ามีแผนเดิม ให้แก้เฉพาะสิ่งที่ผู้ใช้สั่ง ส่วนอื่นคงไว้เหมือนเดิม"""

FIT_TH = {"blur": "พื้นหลังเบลอ", "crop": "ตัดขอบเต็มจอ", "pad": "ขอบดำ"}
ASPECT_TH = {"9:16": "แนวตั้ง 9:16", "16:9": "แนวนอน 16:9", "1:1": "สี่เหลี่ยม 1:1", "4:5": "แนวตั้ง 4:5", "original": "ขนาดเดิม"}


class EditError(RuntimeError):
    pass


# --------------------------------------------------------------------------- plan
def default_plan(files: list[dict]) -> dict:
    clips = [{"file": i + 1, "start": 0, "end": -1 if f["kind"] == "video" else 3, "speed": 1, "mute": False}
             for i, f in enumerate(files) if f["kind"] in ("video", "image")]
    music = next((i + 1 for i, f in enumerate(files) if f["kind"] == "audio"), 0)
    return {"clips": clips, "aspect": "original", "fit": "blur", "texts": [], "music_file": music,
            "music_volume": 0.3, "original_volume": 1, "transition": "none", "fade_in": 0, "fade_out": 0}


def clean_plan(raw: dict, files: list[dict]) -> dict:
    """Clamp every value to something ffmpeg can safely do with these files."""
    def num(v, lo, hi, default):
        try:
            v = float(v)
        except (TypeError, ValueError):
            return default
        return min(max(v, lo), hi)

    plan = default_plan(files)
    clips = []
    for c in raw.get("clips") or []:
        idx = int(num(c.get("file"), 0, 10**6, 0))
        if not 1 <= idx <= len(files) or files[idx - 1]["kind"] not in ("video", "image"):
            continue
        f = files[idx - 1]
        speed = num(c.get("speed", 1), 0.25, 4, 1) or 1
        if f["kind"] == "image":
            start, end = 0.0, num(c.get("end", 3), 0.5, 60, 3)
            if end <= 0.5:
                end = 3.0
        else:
            dur = f["duration"]
            start = num(c.get("start", 0), 0, max(dur - 0.1, 0), 0)
            end = num(c.get("end", -1), -1, dur, -1)
            if end <= start:
                end = dur
        clips.append({"file": idx, "start": round(start, 3), "end": round(end, 3), "speed": round(speed, 3),
                      "mute": bool(c.get("mute", False))})
    if clips:
        plan["clips"] = clips
    if not plan["clips"]:
        raise EditError("ต้องมีไฟล์วิดีโอหรือภาพอย่างน้อย 1 ไฟล์")
    plan["aspect"] = raw.get("aspect") if raw.get("aspect") in (*ASPECTS, "original") else "original"
    plan["fit"] = raw.get("fit") if raw.get("fit") in FIT_TH else "blur"
    plan["transition"] = raw.get("transition") if raw.get("transition") in ("none", "fade") else "none"
    total = output_duration(plan)
    if total > MAX_OUTPUT_S:
        raise EditError(f"วิดีโอยาวเกิน {int(MAX_OUTPUT_S // 60)} นาที")
    texts = []
    for t in raw.get("texts") or []:
        text = str(t.get("text", "")).strip()[:200]
        if not text:
            continue
        start = num(t.get("start", 0), 0, total, 0)
        end = num(t.get("end", -1), -1, total, -1)
        if end <= start:
            end = total
        texts.append({"text": text, "start": round(start, 2), "end": round(end, 2),
                      "position": t.get("position") if t.get("position") in ("top", "center", "bottom") else "bottom",
                      "size": t.get("size") if t.get("size") in ("small", "medium", "large") else "medium"})
    plan["texts"] = texts[:20]
    mf = int(num(raw.get("music_file", 0), 0, 10**6, 0))
    plan["music_file"] = mf if 1 <= mf <= len(files) and files[mf - 1]["has_audio"] and files[mf - 1]["kind"] in ("audio", "video") \
        and mf not in [c["file"] for c in plan["clips"]] else 0
    plan["music_volume"] = round(num(raw.get("music_volume", 0.3), 0, 1.5, 0.3), 2)
    plan["original_volume"] = round(num(raw.get("original_volume", 1), 0, 2, 1), 2)
    plan["fade_in"] = round(num(raw.get("fade_in", 0), 0, min(5, total / 3), 0), 2)
    plan["fade_out"] = round(num(raw.get("fade_out", 0), 0, min(5, total / 3), 0), 2)
    return plan


def clip_length(c: dict, f: dict | None = None) -> float:
    return max(0.1, (c["end"] - c["start"]) / c["speed"])


def output_duration(plan: dict) -> float:
    return round(sum(clip_length(c) for c in plan["clips"]), 3)


def resolve_ends(plan: dict, files: list[dict]) -> dict:
    """Replace end=-1 with the real end of each file."""
    p = json.loads(json.dumps(plan))
    for c in p["clips"]:
        if c["end"] < 0:
            c["end"] = files[c["file"] - 1]["duration"] or 3
    total = output_duration(p)
    for t in p["texts"]:
        if t["end"] < 0 or t["end"] > total:
            t["end"] = total
    return p


def summarize(plan: dict, files: list[dict]) -> str:
    def sec(x):
        return f"{x:g}"
    parts = []
    for c in plan["clips"]:
        f = files[c["file"] - 1]
        if f["kind"] == "image":
            s = f"ภาพ {c['file']} ({sec(c['end'])} วิ)"
        elif c["start"] == 0 and abs(c["end"] - f["duration"]) < 0.05:
            s = f"คลิป {c['file']} (ทั้งคลิป)"
        else:
            s = f"คลิป {c['file']} ({sec(c['start'])}–{sec(c['end'])} วิ)"
        extra = []
        if c["speed"] != 1:
            extra.append(f"ความเร็ว {sec(c['speed'])} เท่า")
        if c["mute"]:
            extra.append("ปิดเสียง")
        parts.append(s + (" " + ", ".join(extra) if extra else ""))
    lines = ["ต่อ " + " → ".join(parts) if len(parts) > 1 else "ใช้ " + parts[0]]
    lines.append(ASPECT_TH[plan["aspect"]] + ("" if plan["aspect"] == "original" else f" ({FIT_TH[plan['fit']]})"))
    for t in plan["texts"]:
        lines.append(f"ข้อความ “{t['text']}” {sec(t['start'])}–{sec(t['end'])} วิ")
    if plan["music_file"]:
        lines.append(f"เพลงจากไฟล์ {plan['music_file']} (ระดับเสียง {int(plan['music_volume'] * 100)}%)")
    if plan["original_volume"] != 1:
        lines.append(f"เสียงเดิม {int(plan['original_volume'] * 100)}%")
    if plan["transition"] == "fade":
        lines.append("เฟดระหว่างคลิป")
    if plan["fade_in"] or plan["fade_out"]:
        lines.append(f"เฟดเข้า {sec(plan['fade_in'])} วิ / เฟดออก {sec(plan['fade_out'])} วิ")
    lines.append(f"ความยาวรวม {output_duration(plan):.1f} วินาที")
    return "\n".join(f"• {x}" for x in lines)


def files_for_prompt(files: list[dict]) -> str:
    out = []
    for i, f in enumerate(files, 1):
        if f["kind"] == "video":
            out.append(f"ไฟล์ {i}: วิดีโอ \"{f['name']}\" ยาว {f['duration']:.1f} วินาที {f['width']}x{f['height']}"
                       + (" มีเสียง" if f["has_audio"] else " ไม่มีเสียง"))
        elif f["kind"] == "audio":
            out.append(f"ไฟล์ {i}: เสียง/เพลง \"{f['name']}\" ยาว {f['duration']:.1f} วินาที")
        else:
            out.append(f"ไฟล์ {i}: ภาพนิ่ง \"{f['name']}\" {f['width']}x{f['height']}")
    return "\n".join(out)


# --------------------------------------------------------------------------- render
def _even(x: float) -> int:
    return max(2, int(round(x / 2)) * 2)


def target_size(plan: dict, files: list[dict]) -> tuple[int, int]:
    if plan["aspect"] in ASPECTS:
        return ASPECTS[plan["aspect"]]
    f = files[plan["clips"][0]["file"] - 1]
    w, h = f["width"] or 1280, f["height"] or 720
    scale = min(1.0, 1280 / max(w, h))
    return _even(w * scale), _even(h * scale)


def _atempo(speed: float) -> str:
    parts, s = [], speed
    while s > 2.0:
        parts.append("atempo=2.0")
        s /= 2.0
    while s < 0.5:
        parts.append("atempo=0.5")
        s /= 0.5
    parts.append(f"atempo={s:.4f}")
    return ",".join(parts)


def _ass_time(t: float) -> str:
    cs = int(round(t * 100))
    return f"{cs // 360000}:{cs // 6000 % 60:02d}:{cs // 100 % 60:02d}.{cs % 100:02d}"


THAI_MARKS = set(chr(c) for c in [0x0E31, *range(0x0E34, 0x0E3B), *range(0x0E47, 0x0E4F)])
THAI_LEADING = set("เแโใไ")
THAI_FOLLOWING = set("ะาำๅฯ")


def _units(s: str) -> int:
    return sum(1 for ch in s if ch not in THAI_MARKS)


def wrap_thai(text: str, max_units: int) -> str:
    """Thai has no spaces between words, and libass only wraps at spaces, so long lines would run
    off the screen. Break at spaces when possible, otherwise between character clusters (never
    before a tone mark/vowel sign, never right after a leading vowel)."""
    out = []
    for para in text.split("\n"):
        line = ""
        for word in para.split(" "):
            cand = (line + " " + word) if line else word
            if _units(cand) <= max_units:
                line = cand
                continue
            if line:
                out.append(line)
                line = ""
            while _units(word) > max_units:
                cut, good, n = 0, 0, 0
                for i, ch in enumerate(word):
                    if ch not in THAI_MARKS:
                        n += 1
                    if n > max_units:
                        break
                    nxt = word[i + 1] if i + 1 < len(word) else ""
                    nxt2 = word[i + 2] if i + 2 < len(word) else ""
                    if nxt and nxt not in THAI_MARKS and nxt not in THAI_FOLLOWING and ch not in THAI_LEADING and nxt2 != "์":
                        cut = i + 1
                        # a leading vowel always starts a syllable: the nicest place to break
                        if nxt in THAI_LEADING and n >= max_units * 0.6:
                            good = i + 1
                cut = good or cut
                if cut <= 0:
                    break
                out.append(word[:cut])
                word = word[cut:]
            line = word
        if line:
            out.append(line)
    return "\n".join(out)


def write_ass(texts: list[dict], w: int, h: int, path: Path) -> None:
    sizes = {"small": 0.035, "medium": 0.05, "large": 0.07}
    align = {"bottom": 2, "center": 5, "top": 8}
    base = min(w, h) if w > h else w
    styles = []
    for pos, a in align.items():
        for sz, k in sizes.items():
            fs = max(16, int(base * k * (1.6 if w <= h else 1.0)))
            styles.append(f"Style: {pos}_{sz},{FONT_NAME},{fs},&H00FFFFFF,&H00FFFFFF,&H00000000,&H64000000,1,0,0,0,"
                          f"100,100,0,0,1,{max(2, fs // 14)},1,{a},{int(w * 0.06)},{int(w * 0.06)},{int(h * 0.08)},1")
    events = []
    for t in texts:
        fs = max(16, int(base * sizes[t["size"]] * (1.6 if w <= h else 1.0)))
        usable = w - 2 * int(w * 0.06)
        wrapped = wrap_thai(t["text"], max(6, int(usable / (fs * 0.56))))
        txt = wrapped.replace("\\", "＼").replace("{", "(").replace("}", ")").replace("\n", "\\N")
        events.append(f"Dialogue: 0,{_ass_time(t['start'])},{_ass_time(t['end'])},{t['position']}_{t['size']},,0,0,0,,{txt}")
    path.write_text("[Script Info]\nScriptType: v4.00+\nWrapStyle: 0\nScaledBorderAndShadow: yes\n"
                    f"PlayResX: {w}\nPlayResY: {h}\n\n[V4+ Styles]\n"
                    "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, "
                    "Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, "
                    "MarginR, MarginV, Encoding\n" + "\n".join(styles) +
                    "\n\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
                    + "\n".join(events) + "\n", encoding="utf-8")


def build_command(plan: dict, files: list[dict], out: Path, work: Path) -> tuple[list[str], float]:
    """ffmpeg argv for a cleaned, resolved plan. Returns (argv, output_seconds)."""
    W, H = target_size(plan, files)
    args = ["ffmpeg", "-hide_banner", "-nostdin", "-y", "-loglevel", "error", "-progress", "pipe:1", "-nostats"]
    filters, vlabels = [], []
    fade_d = 0.4
    for k, c in enumerate(plan["clips"]):
        f = files[c["file"] - 1]
        seg = c["end"] - c["start"]
        length = seg / c["speed"]
        if f["kind"] == "image":
            args += ["-loop", "1", "-framerate", "30", "-t", f"{seg:.3f}", "-i", f["path"]]
        else:
            args += ["-ss", f"{c['start']:.3f}", "-t", f"{seg:.3f}", "-i", f["path"]]
        v = f"[{k}:v]setpts=(PTS-STARTPTS)/{c['speed']:.4f},"
        if plan["aspect"] == "original" or plan["fit"] == "pad":
            v += (f"scale={W}:{H}:force_original_aspect_ratio=decrease:force_divisible_by=2,"
                  f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:black,")
        elif plan["fit"] == "crop":
            v += f"scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},"
        else:  # blurred background
            filters.append(f"[{k}:v]setpts=(PTS-STARTPTS)/{c['speed']:.4f},split=2[bg{k}][fg{k}]")
            filters.append(f"[bg{k}]scale={W // 4}:{H // 4}:force_original_aspect_ratio=increase,crop={W // 4}:{H // 4},"
                           f"boxblur=10:2,scale={W}:{H},setsar=1[bgb{k}]")
            filters.append(f"[fg{k}]scale={W}:{H}:force_original_aspect_ratio=decrease:force_divisible_by=2,setsar=1[fgs{k}]")
            filters.append(f"[bgb{k}][fgs{k}]overlay=(W-w)/2:(H-h)/2[ov{k}]")
            v = f"[ov{k}]"
        v += f"fps=30,format=yuv420p,setsar=1,trim=duration={length:.3f}"
        if plan["transition"] == "fade" and len(plan["clips"]) > 1:
            d = min(fade_d, length / 3)
            if k > 0:
                v += f",fade=t=in:st=0:d={d:.3f}"
            if k < len(plan["clips"]) - 1:
                v += f",fade=t=out:st={length - d:.3f}:d={d:.3f}"
        filters.append(v + f"[v{k}]")
        if f["kind"] == "video" and f["has_audio"] and not c["mute"]:
            filters.append(f"[{k}:a]asetpts=PTS-STARTPTS,{_atempo(c['speed'])},aresample=48000,"
                           f"aformat=sample_fmts=fltp:channel_layouts=stereo,apad,atrim=duration={length:.3f}[a{k}]")
        else:
            filters.append(f"anullsrc=r=48000:cl=stereo,atrim=duration={length:.3f}[a{k}]")
        vlabels.append(f"[v{k}][a{k}]")
    total = output_duration(plan)
    filters.append("".join(vlabels) + f"concat=n={len(plan['clips'])}:v=1:a=1[cv][ca]")
    vcur, acur = "[cv]", "[ca]"
    vf = []
    if plan["fade_in"]:
        vf.append(f"fade=t=in:st=0:d={plan['fade_in']}")
    if plan["fade_out"]:
        vf.append(f"fade=t=out:st={max(0, total - plan['fade_out']):.3f}:d={plan['fade_out']}")
    if plan["texts"]:
        ass = work / "texts.ass"
        write_ass(plan["texts"], W, H, ass)
        vf.append(f"subtitles=filename={ass}:fontsdir={FONT_DIR}")
    if vf:
        filters.append(f"{vcur}{','.join(vf)}[vout]")
        vcur = "[vout]"
    af = [f"volume={plan['original_volume']}"]
    if plan["fade_in"]:
        af.append(f"afade=t=in:st=0:d={plan['fade_in']}")
    if plan["fade_out"]:
        af.append(f"afade=t=out:st={max(0, total - plan['fade_out']):.3f}:d={plan['fade_out']}")
    filters.append(f"{acur}{','.join(af)}[aorig]")
    acur = "[aorig]"
    if plan["music_file"]:
        m = files[plan["music_file"] - 1]
        mi = len(plan["clips"])
        args += ["-stream_loop", "-1", "-i", m["path"]]
        mf = min(2.0, total / 4)
        filters.append(f"[{mi}:a]atrim=duration={total:.3f},asetpts=PTS-STARTPTS,aresample=48000,"
                       f"aformat=sample_fmts=fltp:channel_layouts=stereo,volume={plan['music_volume']},"
                       f"afade=t=out:st={max(0, total - mf):.3f}:d={mf:.3f}[mus]")
        filters.append(f"{acur}[mus]amix=inputs=2:duration=first:normalize=0[amix]")
        acur = "[amix]"
    args += ["-filter_complex", ";".join(filters), "-map", vcur, "-map", acur,
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
             "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", "-threads", os.environ.get("LIVEAI_FFMPEG_THREADS", "2"),
             "-t", f"{total:.3f}", str(out)]
    return args, total


def render(plan: dict, files: list[dict], out: Path, on_progress=None, timeout: float = 3600) -> float:
    """Runs ffmpeg (low CPU priority so chat stays responsive). Returns render seconds."""
    with tempfile.TemporaryDirectory(prefix="edit-") as tmp:
        argv, total = build_command(plan, files, out, Path(tmp))
        started = time.time()
        cmd = ["nice", "-n", "10", *argv] if os.path.exists("/usr/bin/nice") else argv
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        err_lines: list[str] = []
        t = threading.Thread(target=lambda: err_lines.extend(p.stderr.readlines()), daemon=True)
        t.start()
        for line in p.stdout:
            if line.startswith("out_time_us=") and on_progress:
                try:
                    on_progress(min(0.99, int(line.split("=")[1]) / 1e6 / max(total, 0.1)))
                except ValueError:
                    pass
            if time.time() - started > timeout:
                p.kill()
                raise EditError("ตัดต่อนานเกินกำหนด")
        p.wait()
        t.join(5)
        p.stdout.close()
        p.stderr.close()
        if p.returncode != 0 or not out.exists():
            out.unlink(missing_ok=True)
            raise EditError("ffmpeg ล้มเหลว: " + "".join(err_lines)[-400:])
        return time.time() - started


# --------------------------------------------------------------------------- jobs
class Editor:
    def __init__(self, store, media, assistant):
        self.store = store
        self.media = media
        self.assistant = assistant
        self.q: queue.Queue[int] = queue.Queue()
        with store.conn() as c:
            c.executescript(EDIT_SCHEMA_SQL)
            c.execute("UPDATE edits SET status='failed', error='ระบบเริ่มใหม่ระหว่างตัดต่อ กรุณาสั่งใหม่' "
                      "WHERE status IN ('queued','planning','rendering')")
        threading.Thread(target=self._worker, name="video-editor", daemon=True).start()

    # api -------------------------------------------------------------------
    def create(self, user_id: int, conv_id: int | None, instruction: str, media_ids: list[str], base_edit: int | None) -> dict:
        instruction = instruction.strip()[:2000]
        if not instruction:
            raise ValueError("กรุณาพิมพ์คำสั่งตัดต่อ")
        if base_edit is not None:
            base = self.get(user_id, base_edit)
            if base["status"] != "done":
                raise ValueError("งานตัดต่อก่อนหน้ายังไม่เสร็จ")
            media_ids = json.loads(base["media_ids_raw"]) + [m for m in media_ids if m not in json.loads(base["media_ids_raw"])]
            conv_id = base["conversation_id"]
        if not media_ids:
            raise ValueError("กรุณาแนบไฟล์วิดีโอ เสียง หรือภาพก่อน")
        if len(media_ids) > 12:
            raise ValueError("แนบได้สูงสุด 12 ไฟล์ต่อครั้ง")
        for mid in media_ids:
            self.media.get(user_id, mid)          # ownership check (NotFound)
        if conv_id is None:
            conv_id = self.store.create_conversation(user_id, "🎬 " + instruction.replace("\n", " ")[:38])
        else:
            self.store.get_conversation(user_id, conv_id)
        pending = self.store.conn().execute("SELECT COUNT(*) FROM edits WHERE user_id=? AND status IN ('queued','planning','rendering')",
                                            (user_id,)).fetchone()[0]
        if pending:
            raise ValueError("มีงานตัดต่อที่ยังทำอยู่ รอให้เสร็จก่อน")
        now = time.time()
        with self.store.conn() as c:
            cur = c.execute("INSERT INTO edits(user_id,conversation_id,instruction,media_ids,base_edit,status,created,updated) "
                            "VALUES (?,?,?,?,?,'queued',?,?)", (user_id, conv_id, instruction, json.dumps(media_ids), base_edit, now, now))
        self.q.put(cur.lastrowid)
        logs.event(log, "edit queued", user=user_id, edit=cur.lastrowid, files=len(media_ids), refine=base_edit is not None)
        return self.get(user_id, cur.lastrowid)

    def get(self, user_id: int, edit_id: int) -> dict:
        r = self.store.conn().execute("SELECT * FROM edits WHERE id=? AND user_id=?", (edit_id, user_id)).fetchone()
        if not r:
            raise NotFound("edit")
        e = dict(r)
        out = {k: e[k] for k in ("id", "conversation_id", "instruction", "base_edit", "summary", "status", "progress",
                                 "error", "seconds", "created", "updated")}
        out["plan"] = json.loads(e["plan"]) if e["plan"] else None
        out["output"] = None
        out["media_ids_raw"] = e["media_ids"]
        if e["output_id"]:
            try:
                m = self.media.get(user_id, e["output_id"])
                out["output"] = {**self.media.public(m), "url": self.media.sign(m["id"])}
            except NotFound:
                out["error"] = out["error"] or "ไฟล์ผลลัพธ์ถูกลบแล้ว (เก็บไว้ชั่วคราว)"
        return out

    def view(self, user_id: int, edit_id: int) -> dict:
        e = self.get(user_id, edit_id)
        e.pop("media_ids_raw", None)
        return e

    # worker ----------------------------------------------------------------
    def _set(self, eid: int, **kw) -> None:
        kw["updated"] = time.time()
        cols = ", ".join(f"{k}=?" for k in kw)
        with self.store.conn() as c:
            c.execute(f"UPDATE edits SET {cols} WHERE id=?", (*kw.values(), eid))

    def _worker(self) -> None:
        while True:
            eid = self.q.get()
            try:
                self._run(eid)
            except Exception as e:  # never kill the worker
                logs.event(log, "edit crashed", logging.ERROR, edit=eid, error=repr(e)[:300])
                self._set(eid, status="failed", error="ระบบตัดต่อขัดข้อง กรุณาลองใหม่")

    def _files(self, user_id: int, media_ids: list[str]) -> list[dict]:
        return [self.media.get(user_id, mid) for mid in media_ids]

    def plan_with_llm(self, instruction: str, files: list[dict], base_plan: dict | None) -> dict:
        msg = "ไฟล์ที่มี:\n" + files_for_prompt(files)
        if base_plan:
            msg += "\n\nแผนเดิม:\n" + json.dumps(base_plan, ensure_ascii=False)
        msg += f"\n\nคำสั่งของผู้ใช้: {instruction}"
        msgs = [{"role": "system", "content": PLANNER_PROMPT}, {"role": "user", "content": msg}]
        gate = self.assistant._gen
        if not gate.acquire(timeout=600):
            raise EditError("โมเดลไม่ว่าง ลองใหม่อีกครั้ง")
        try:
            if not self.assistant.server.wait_ready(240):
                raise EditError("โมเดลยังไม่พร้อม")
            return self.assistant.llm.complete_json(msgs, PLAN_SCHEMA)
        finally:
            gate.release()

    def _run(self, eid: int) -> None:
        row = dict(self.store.conn().execute("SELECT * FROM edits WHERE id=?", (eid,)).fetchone())
        uid = row["user_id"]
        started = time.time()
        try:
            files = self._files(uid, json.loads(row["media_ids"]))
            base_plan = None
            if row["base_edit"]:
                b = self.store.conn().execute("SELECT plan FROM edits WHERE id=? AND user_id=?", (row["base_edit"], uid)).fetchone()
                base_plan = json.loads(b["plan"]) if b and b["plan"] else None
            self._set(eid, status="planning", progress=0.02)
            try:
                raw = self.plan_with_llm(row["instruction"], files, base_plan)
            except EditError:
                raise
            except Exception as e:
                logs.event(log, "planner failed, using default plan", logging.WARNING, edit=eid, error=str(e)[:200])
                raw = base_plan or default_plan(files)
            plan = clean_plan(raw, files)
            resolved = resolve_ends(plan, files)
            summary = summarize(resolved, files)
            self._set(eid, status="rendering", plan=json.dumps(resolved, ensure_ascii=False), summary=summary, progress=0.05)
            out_dir = self.media.root / str(uid)
            out_dir.mkdir(parents=True, exist_ok=True)
            out = out_dir / f"{os.urandom(12).hex()}.mp4"
            render_s = render(resolved, files, out, on_progress=lambda p: self._set(eid, progress=round(0.05 + 0.95 * p, 3)))
            m = self.media.add_file(uid, out, f"liveai-edit-{eid}.mp4", "edit")
            secs = round(time.time() - started, 1)
            self._set(eid, status="done", progress=1.0, output_id=m["id"], seconds=secs)
            self.store.add_message(uid, row["conversation_id"], "user", f"🎬 {row['instruction']}")
            self.store.add_message(uid, row["conversation_id"], "assistant",
                                   f"ตัดต่อเสร็จแล้ว ({m['duration']:.1f} วินาที)\n{summary}\n[[video:{m['id']}]]")
            logs.event(log, "edit done", user=uid, edit=eid, seconds=secs, render_seconds=round(render_s, 1),
                       output_seconds=m["duration"], clips=len(resolved["clips"]))
        except (EditError, NotFound, ValueError) as e:
            msg = str(e) if not isinstance(e, NotFound) else "ไม่พบไฟล์ที่แนบ (อาจถูกลบไปแล้ว)"
            self._set(eid, status="failed", error=msg[:500])
            logs.event(log, "edit failed", logging.WARNING, user=uid, edit=eid, error=msg[:300])
            try:
                self.store.add_message(uid, row["conversation_id"], "user", f"🎬 {row['instruction']}")
                self.store.add_message(uid, row["conversation_id"], "assistant", f"ตัดต่อไม่สำเร็จ: {msg[:300]}")
            except NotFound:
                pass
