"""Makes the Page's content on this machine: Thai text written by Qwen (no paid AI API), picture cards
and 9:16 Reels rendered with ffmpeg.

Kinds:
  question   – a short story situation + a question that invites comments (text post)
  teaser     – a cliffhanger for the user's drama, on a picture card
  quote      – one strong line of dialogue on a picture card
  reel_text  – a 9:16 Reel: a mini story told in 3–5 lines of text
  reel_clip  – a 9:16 Reel cut from one of the user's own videos, with a hook title on top
"""
from __future__ import annotations

import random
import re
import subprocess
from pathlib import Path

from ..editor import FONT_DIR, write_ass
from ..media import probe
from ..studio.asr import speech_regions

KINDS = ["question", "teaser", "quote", "reel_text", "reel_clip"]
KIND_TH = {"question": "โพสต์คำถามชวนคุย", "teaser": "ภาพตัวอย่างตอนต่อไป", "quote": "ภาพคำพูดเด็ด",
           "reel_text": "Reels เล่าเรื่องสั้น", "reel_clip": "Reels ตัดจากคลิปละคร"}

# Every kind asks for the same JSON so one schema serves all.
SCHEMA = {
    "type": "object",
    "properties": {
        "caption": {"type": "string", "minLength": 20, "maxLength": 900},
        "lines": {"type": "array", "items": {"type": "string", "minLength": 2, "maxLength": 90}, "minItems": 1, "maxItems": 5},
        "hashtags": {"type": "array", "items": {"type": "string", "maxLength": 30}, "minItems": 1, "maxItems": 4},
    },
    "required": ["caption", "lines", "hashtags"],
}

SYSTEM = """คุณคือแอดมินเพจละครสั้นภาษาไทยที่เขียนโพสต์เก่ง เป้าหมายคือให้คนดูอยากกดติดตามเพจด้วยเนื้อหาที่ดีจริง
กติกา:
- ภาษาไทยเป็นธรรมชาติ อารมณ์ชัด ไม่ยาวเกินไป ประโยคแรกต้องดึงความสนใจ
- ห้ามหลอกลวง ห้ามขอไลก์แลกของรางวัล ห้ามข้อความแบบสแปม ห้ามแท็กคนอื่น ห้ามพาดพิงคนจริงหรือข่าวจริงในทางเสียหาย
- ปิดท้ายด้วยการชวนกดติดตามหรือชวนคอมเมนต์อย่างสุภาพ 1 ประโยค
- ตอบเป็น JSON ตามรูปแบบที่กำหนดเท่านั้น"""

TASK = {
    "question": "เขียนโพสต์คำถามชวนคุย: เล่าสถานการณ์ดราม่าสั้น ๆ 2–3 ประโยคจากเรื่องของเพจ แล้วถามคนดูว่า \"ถ้าเป็นคุณจะทำอย่างไร\" "
                "ให้มีตัวเลือก A/B ใน lines (2 บรรทัด) caption คือโพสต์ทั้งหมด",
    "teaser": "เขียนโพสต์ตัวอย่างตอนต่อไปแบบค้างคา lines คือข้อความบนภาพ 1–2 บรรทัด (สั้นมาก ไม่เกิน 40 ตัวอักษรต่อบรรทัด) "
              "caption คือข้อความโพสต์ที่ชวนให้รอดูตอนต่อไปและกดติดตามเพจ",
    "quote": "เลือกหรือแต่งประโยคเด็ดของตัวละคร 1 ประโยคที่กินใจ lines คือคำพูดนั้น (1–2 บรรทัด) และชื่อผู้พูดขึ้นต้นด้วย — "
             "caption คือข้อความโพสต์สั้น ๆ ที่ชวนคนดูแชร์ความรู้สึก",
    "reel_text": "เขียนเรื่องสั้นหักมุมสำหรับ Reels 3–5 บรรทัด (บรรทัดละไม่เกิน 60 ตัวอักษร) อ่านทีละบรรทัดแล้วอยากรู้ตอนจบ "
                 "บรรทัดสุดท้ายเป็นจุดหักมุมหรือคำถาม caption คือคำบรรยาย Reels สั้น ๆ",
    "reel_clip": "เขียนข้อความพาดหัวบนคลิปละคร 1 บรรทัด (ไม่เกิน 40 ตัวอักษร) ที่ทำให้อยากดูต่อ ใส่ใน lines "
                 "caption คือคำบรรยาย Reels ที่บอกว่าเป็นฉากจากเรื่องไหนและชวนกดติดตามเพื่อดูตอนต่อไป",
}


def prompt(kind: str, page_name: str, story: str, style_examples: list[str], trend: str | None) -> list[dict]:
    msg = f"ชื่อเพจ: {page_name or 'เพจละครสั้น'}\n"
    msg += f"ข้อมูลเรื่องและตัวละครของเพจ:\n{story.strip()[:2500] or '(ยังไม่มี — แต่งเรื่องดราม่ารักแค้นครอบครัวแบบละครไทย)'}\n"
    if style_examples:
        msg += "\nโพสต์ของเพจที่คนดูตอบรับดีที่สุด (ใช้เป็นแนวทางสไตล์ ห้ามลอก):\n" + "\n---\n".join(e[:300] for e in style_examples[:3]) + "\n"
    if trend:
        msg += f"\nเรื่องที่คนไทยค้นหามากวันนี้: {trend}\nใช้เป็นมุมหรือแฮชแท็กได้เฉพาะเมื่อเข้ากับเรื่องอย่างเป็นธรรมชาติ ถ้าไม่เข้าให้ข้ามไป\n"
    msg += "\nงาน: " + TASK[kind]
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": msg}]


CTAS = ["กดติดตามเพจไว้ แล้วมาลุ้นกันว่าเรื่องจะเป็นอย่างไรต่อ", "ติดตามเพจไว้ ตอนต่อไปเข้มข้นกว่านี้แน่นอน",
        "คิดเห็นอย่างไรคอมเมนต์บอกกันหน่อย และกดติดตามเพจไว้ไม่พลาดตอนใหม่", "ใครอยากรู้ตอนจบ กดติดตามเพจไว้เลย"]


def _sim(a: str, b: str) -> float:
    g = lambda t: {t[i:i + 2] for i in range(len(t) - 1)}
    x, y = g(re.sub(r"\s+", "", a)), g(re.sub(r"\s+", "", b))
    return len(x & y) / max(1, len(x | y))


def clean(gen: dict, kind: str) -> dict:
    """Keeps what the model wrote within safe, postable bounds: hashtags only at the end (max 4),
    no repeated lines, and always one polite invitation to follow or comment."""
    tags = []
    for t in list(gen.get("hashtags") or []) + re.findall(r"#[\w\u0E00-\u0E7F]+", str(gen.get("caption", ""))):
        t = "#" + re.sub(r"[^\w\u0E00-\u0E7F]", "", str(t))
        if len(t) > 2 and t.lower() not in [x.lower() for x in tags]:
            tags.append(t)
    cap = re.sub(r"#[\w\u0E00-\u0E7F]+", "", str(gen.get("caption", "")))
    cap = re.sub(r"[ \t]+", " ", re.sub(r"[ \t]*\n[ \t]*", "\n", cap))
    cap = re.sub(r"\n{3,}", "\n\n", cap).strip()
    lines = []
    for x in gen.get("lines") or []:
        x = re.sub(r"#[\w\u0E00-\u0E7F]+", "", str(x))
        x = re.sub(r"\s+", " ", x).strip().lstrip("-–—•* ").strip()
        x = re.sub(r"(.{3,}?)(?:\s*[:：,]?\s*\1){1,}", r"\1", x)       # "X: X: X" (small models repeat) -> "X"
        if kind in ("teaser", "quote", "reel_clip") and len(x) > 50:
            m = re.match(r"^(.{12,50}?[!?…]+|.{12,50}?\.\.\.)", x)     # a card line: keep the first sentence
            if m:
                x = m.group(1)
        if x and not any(_sim(x, y) > 0.6 for y in lines):
            lines.append(x)
    if kind == "question":                 # "A. ... B. ..." on one line -> one choice per line
        split = []
        for x in lines:
            m = re.match(r"^(.*?\bA[.)：:]\s*.+?)\s+(B[.)：:]\s*.+)$", x)
            split += [m.group(1), m.group(2)] if m else [x]
        lines = split
    if not lines:
        lines = [cap.split("\n")[0][:60]]
    limit = {"teaser": 2, "quote": 2, "reel_clip": 1, "question": 3}.get(kind, 5)
    lines = lines[:limit]
    if not re.search(r"ติดตาม|คอมเมนต์|เม้น|ความเห็น|ถ้าเป็นคุณ", cap):
        cap += "\n\n" + random.choice(CTAS)
    full = cap + ("\n\n" + " ".join(tags[:4]) if tags else "")
    return {"caption": full[:2000], "lines": lines, "hashtags": tags[:4]}


# rendering --------------------------------------------------------------------------
PALETTES = [("0x1a0b12", "0x5c1424"), ("0x0b1320", "0x23395b"), ("0x140f0a", "0x5a3a16"), ("0x0f1a14", "0x1f4d3a")]


def _run(args: list[str], timeout: int = 600) -> None:
    r = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError("ffmpeg: " + r.stderr[-400:])


def _bg(w: int, h: int, seconds: float, seed: int) -> str:
    c0, c1 = PALETTES[seed % len(PALETTES)]
    return (f"gradients=s={w}x{h}:c0={c0}:c1={c1}:x0=0:y0=0:x1={w}:y1={h}:d={seconds:.2f}:speed=0.01:r=30,"
            f"vignette=PI/5,format=yuv420p")


def card_image(lines: list[str], out: Path, footer: str = "", seed: int = 0) -> Path:
    """A 1080x1350 picture with the text (for teaser / quote / question posts)."""
    w, h = 1080, 1350
    ass = out.with_suffix(".ass")
    size = "large" if sum(len(x) for x in lines) <= 45 else "medium"
    texts = [{"text": "\n\n".join(lines), "start": 0, "end": 5, "position": "center", "size": size}]
    if footer:
        texts.append({"text": footer, "start": 0, "end": 5, "position": "bottom", "size": "small"})
    write_ass(texts, w, h, ass)
    _run(["ffmpeg", "-hide_banner", "-nostdin", "-y", "-loglevel", "error", "-f", "lavfi", "-i", _bg(w, h, 1, seed),
          "-vf", f"subtitles=filename={ass}:fontsdir={FONT_DIR}", "-frames:v", "1", str(out)])
    ass.unlink(missing_ok=True)
    return out


def text_reel(lines: list[str], out: Path, footer: str = "", seed: int = 0) -> Path:
    """9:16 720x1280 Reel: the lines appear one after another (about 3 s each), silent audio track."""
    w, h = 720, 1280
    per = 3.2
    total = round(1.0 + per * len(lines) + 2.0, 2)
    texts, y = [], 0.0
    shown = []
    for i, line in enumerate(lines):
        shown.append(line)
        start = 0.8 + per * i
        end = total if i == len(lines) - 1 else start + per
        texts.append({"text": "\n\n".join(shown), "start": start, "end": end, "position": "center", "size": "medium"})
    if footer:
        texts.append({"text": footer, "start": 0.0, "end": total, "position": "bottom", "size": "small"})
    ass = out.with_suffix(".ass")
    write_ass(texts, w, h, ass)
    _run(["ffmpeg", "-hide_banner", "-nostdin", "-y", "-loglevel", "error", "-f", "lavfi", "-i", _bg(w, h, total, seed),
          "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo", "-t", f"{total}",
          "-vf", f"subtitles=filename={ass}:fontsdir={FONT_DIR}", "-c:v", "libx264", "-preset", "veryfast", "-crf", "21",
          "-pix_fmt", "yuv420p", "-r", "30", "-c:a", "aac", "-b:a", "96k", "-shortest", "-movflags", "+faststart", str(out)])
    ass.unlink(missing_ok=True)
    return out


def pick_highlight(path: str, duration: float, want: float = 30.0, avoid: list[list[float]] | None = None) -> tuple[float, float]:
    """The stretch (about `want` s) with the most speech, not overlapping stretches already used."""
    if duration <= want + 1:
        return 0.0, duration
    regions = speech_regions(path, duration)
    avoid = avoid or []
    best, best_score = (0.0, want), -1.0
    step = 2.0
    t = 0.0
    while t + want <= duration:
        a, b = t, t + want
        if not any(a < y and x < b for x, y in avoid):
            speech = sum(max(0.0, min(b, e) - max(a, s)) for s, e in regions)
            # start just before a line begins, not in the middle of one
            starts_clean = 0.3 if any(abs(s - a) < 0.6 for s, _ in regions) else 0.0
            score = speech / want + starts_clean
            if score > best_score:
                best, best_score = (a, b), score
        t += step
    a, b = best
    near = [s for s, _ in regions if a - 1.5 <= s <= a + 1.5]
    if near:
        a = max(0.0, min(near) - 0.3)
        b = min(duration, a + want)
    return round(a, 2), round(b, 2)


def clip_reel(src: str, start: float, end: float, title: str, out: Path) -> Path:
    """Cuts start..end from the user's video into a 9:16 Reel (no cropping: blurred fill), title on top."""
    w, h = 720, 1280
    dur = end - start
    ass = out.with_suffix(".ass")
    write_ass([{"text": title, "start": 0, "end": dur, "position": "top", "size": "medium"}], w, h, ass)
    info = probe(Path(src))
    fc = (f"[0:v]split[a][b];[a]scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},boxblur=20:2[bg];"
          f"[b]scale={w}:{h}:force_original_aspect_ratio=decrease[fg];[bg][fg]overlay=(W-w)/2:(H-h)/2,"
          f"subtitles=filename={ass}:fontsdir={FONT_DIR},fps=30,format=yuv420p[v]")
    args = ["ffmpeg", "-hide_banner", "-nostdin", "-y", "-loglevel", "error", "-ss", f"{start:.2f}", "-t", f"{dur:.2f}",
            "-i", src]
    if info["has_audio"]:
        amap = ["-map", "0:a:0"]
    else:
        args += ["-f", "lavfi", "-t", f"{dur:.2f}", "-i", "anullsrc=r=48000:cl=stereo"]
        amap = ["-map", "1:a:0"]
    _run(args + ["-filter_complex", fc, "-map", "[v]", *amap, "-c:v", "libx264", "-preset", "veryfast", "-crf", "21",
                 "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", "-shortest", str(out)], timeout=1200)
    ass.unlink(missing_ok=True)
    return out


def story_from_projects(projects: list[dict]) -> str:
    """Characters and a few scenes from the user's studio scripts: real material for the posts."""
    parts = []
    for p in projects[:3]:
        sc = p.get("script") or {}
        chars = [c.get("name") for c in sc.get("characters", []) if c.get("name")]
        lines = []
        for shot in sc.get("shots", [])[:400]:
            for d in shot.get("dialogue", [])[:2]:
                if d.get("text"):
                    lines.append(f"{d.get('speaker') or ''}: {d['text']}")
        random.shuffle(lines)
        parts.append(f"เรื่อง {sc.get('title') or p.get('name')}: ตัวละคร {', '.join(chars[:8]) or '-'}\n"
                     + "\n".join(lines[:8]))
    return "\n\n".join(parts)
