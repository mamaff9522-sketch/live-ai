"""AI 2 (Script Analyzer) + AI 3 (Scene Manager): Thai drama scripts -> structure.

Understands the formats used with Sala AI / Meta AI:
  • clip-by-clip:  "คลิป 001/225 | 8 วินาที | 9:16" / "คลิป 3 — ชื่อ" blocks, "ภาพ/การกระทำ:",
                   "บทพูด [ชื่อ]: “…”", "บท: ชื่อ: "…"", "บทพูด: ไม่มี", "มุมกล้อง:", "ตัวละครในฉาก:"
  • story:         "ละครเรื่อง “…”", "ตอนที่ 1: …", "ตัวละคร" list, "ฉาก: …" / "ฉากที่ 1 — สถานที่ / เวลา",
                   action lines and `ชื่อ: "…"` dialogue; each spoken line becomes one shot.

Rules from the brief: dialogue is copied character-for-character (never rewritten), narration
and picture directions are never turned into dialogue, and the result is checked against the
raw text so no spoken line is lost.
"""
from __future__ import annotations

import re
import unicodedata

QUOTE_PAIRS = [("“", "”"), ('"', '"'), ("‘", "’"), ("«", "»"), ("「", "」")]
OPEN_Q = "“\"‘«「"
CLOSE_Q = "”\"’»」"

RE_TITLE = re.compile(r"^(?:ละครเรื่อง\s*[:：]?|ชื่อเรื่อง\s*[:：]?|เรื่อง\s*[:：])\s*[“\"']?(.+?)[”\"']?\s*$")
RE_CHAR_INLINE = re.compile(r"^ตัวละคร(?:หลัก)?\s*[:：]\s*(.+)$")
RE_DIALOGUE_HEADER = re.compile(r"^(?:บทพูด|บท|dialogue)\s*[:：]?\s*$", re.I)
RE_EPISODE = re.compile(r"^ตอนที่\s*(\d+)\s*[:：\-—–]?\s*(.*)$")
RE_CLIP = re.compile(r"^คลิป(?:ที่)?\s*(\d+)(?:\s*/\s*(\d+))?\s*(.*)$")
RE_SCENE_NUM = re.compile(r"^ฉาก(?:ที่)?\s*(\d+)\s*[:：\-—–]\s*(.*)$")
RE_SCENE = re.compile(r"^ฉาก\s*[:：]\s*(.+)$")
RE_DURATION = re.compile(r"(\d+(?:\.\d+)?)\s*วินาที")
RE_CHAR_HEADER = re.compile(r"^(?:ตัวละคร|รายชื่อตัวละคร)\s*[:：]?\s*$")
RE_CAMERA = re.compile(r"^มุมกล้อง\s*[:：]\s*(.+)$")
RE_ACTION = re.compile(r"^(?:ภาพ\s*/\s*การกระทำ|ภาพ|การกระทำ|แอ็กชัน|action)\s*[:：]\s*(.+)$", re.I)
RE_IN_SCENE = re.compile(r"^ตัวละครในฉาก\s*[:：]\s*(.+)$")
RE_NO_DIALOGUE = re.compile(r"^(?:บทพูด|บท|dialogue)\s*[:：]\s*(?:ไม่มี|—|-|none)", re.I)
RE_NARRATION = re.compile(r"^(?:\(?\s*(?:เสียงบรรยาย|บรรยาย|ผู้บรรยาย|narrator|narration)\s*\)?)\s*[:：]?\s*(.+)$", re.I)
# บทพูด [ชื่อ]: “…”   |   บท: ชื่อ: "…"   |   ชื่อ (อารมณ์): "…"   |   ชื่อ: …
RE_DIALOGUE_BRACKET = re.compile(r"^(?:บทพูด|บท)\s*\[([^\]]+)\]\s*(?:\(([^)]*)\))?\s*[:：]\s*(.+)$")
RE_DIALOGUE_PREFIX = re.compile(r"^(?:บทพูด|บท)\s*[:：]\s*(.+)$")
RE_DIALOGUE = re.compile(r"^([^\s:：“\"‘(][^:：“\"‘]{0,40}?)\s*(?:\(([^)]*)\))?\s*[:：]\s*(.+)$")
NOT_SPEAKERS = {"ฉาก", "ภาพ", "การกระทำ", "มุมกล้อง", "เวลา", "สถานที่", "หมายเหตุ", "แนว", "อัตราส่วนภาพ", "ความยาวเป้าหมาย",
                "คุณภาพภาพที่ต้องการ", "ตัวละครในฉาก", "บรรยากาศ", "เพลง", "เสียง", "ข้อกำหนด", "note", "สไตล์"}


def norm(s: str) -> str:
    return unicodedata.normalize("NFC", s).replace("​", "").strip()


def split_quoted(text: str) -> tuple[str | None, str]:
    """('ข้อความในเครื่องหมายคำพูด', 'ส่วนที่เหลือ') — the quoted part is returned exactly as written."""
    t = text.strip()
    if t and t[0] in OPEN_Q:
        close = CLOSE_Q[OPEN_Q.index(t[0])]
        end = t.rfind(close)
        if end > 0:
            return t[1:end], t[end + 1:].strip()
        return t[1:].rstrip("".join(CLOSE_Q)), ""
    return None, t


def quoted_segments(raw: str) -> list[str]:
    out = []
    for o, c in QUOTE_PAIRS:
        if o == c:
            out += re.findall(re.escape(o) + r"([^" + re.escape(o) + r"\n]+)" + re.escape(c), raw)
        else:
            out += re.findall(re.escape(o) + r"([^" + re.escape(c) + r"\n]+)" + re.escape(c), raw)
    return out


class Script:
    def __init__(self):
        self.title = ""
        self.episodes: list[dict] = []
        self.characters: list[dict] = []
        self.scenes: list[dict] = []
        self.shots: list[dict] = []
        self.notes: list[str] = []
        self.warnings: list[str] = []
        self.format = "story"

    def to_dict(self) -> dict:
        return {"title": self.title, "format": self.format, "episodes": self.episodes, "characters": self.characters,
                "scenes": self.scenes, "shots": self.shots, "notes": self.notes[:30], "warnings": self.warnings,
                "stats": stats(self.shots)}


def stats(shots: list[dict]) -> dict:
    return {"shots": len(shots), "dialogue_lines": sum(len(s["dialogue"]) for s in shots),
            "silent_shots": sum(1 for s in shots if not s["dialogue"]),
            "narration_lines": sum(len(s["narration"]) for s in shots),
            "expected_seconds": round(sum(s["duration"] or 0 for s in shots), 1)}


def parse(text: str) -> dict:
    lines = [norm(x) for x in text.replace("\r\n", "\n").split("\n")]
    sc = Script()
    sc.format = "clips" if any(RE_CLIP.match(x) for x in lines) else "story"
    known: set[str] = set()
    cur_ep = None
    cur_scene = None
    cur_shot = None
    pending_actions: list[str] = []
    in_chars = False
    chars_in_block = 0
    started = False                # story content begins at the first scene/clip

    def new_scene(title: str, number=None) -> dict:
        nonlocal cur_scene
        title = title.strip(" -—–")
        loc, _, time_ = title.partition("/")
        cur_scene = {"id": f"S{len(sc.scenes) + 1}", "number": number, "title": title, "location": loc.strip(),
                     "time": time_.strip(), "episode": cur_ep}
        sc.scenes.append(cur_scene)
        return cur_scene

    def new_shot(number=None, duration=None, title="") -> dict:
        nonlocal cur_shot
        if cur_scene is None:
            new_scene("ฉากที่ 1")
        cur_shot = {"id": f"SH{len(sc.shots) + 1:03d}", "index": len(sc.shots), "number": number, "title": title,
                    "scene": cur_scene["id"], "episode": cur_ep, "duration": duration, "action": [], "dialogue": [],
                    "narration": [], "camera": "", "characters": [], "location": "", "declared_silent": False}
        sc.shots.append(cur_shot)
        return cur_shot

    def add_dialogue(speaker: str, emotion: str | None, rest: str) -> bool:
        speaker = speaker.strip().strip("*").strip()
        quote, after = split_quoted(rest)
        text = quote if quote is not None else rest.strip()
        if not text:
            return False
        if sc.format == "story":
            # one spoken line = one shot; picture directions written before it belong to that shot
            new_shot()
            cur_shot["action"] += pending_actions
            pending_actions.clear()
        elif cur_shot is None:
            new_shot()
        cur_shot["dialogue"].append({"speaker": speaker, "text": text, "emotion": (emotion or "").strip(),
                                     "quoted": quote is not None})
        if after:
            cur_shot["action"].append(after)
        if speaker not in [c["name"] for c in sc.characters]:
            known.add(speaker)
        return True

    for i, line in enumerate(lines):
        if not line:
            if in_chars and chars_in_block:
                in_chars = False
            continue
        bare = line.lstrip("#*•- ").strip()
        if not started and (m := RE_CHAR_INLINE.match(bare)):
            for part in re.split(r"[,，、]", m.group(1)):
                name = part.split("(")[0].strip()
                role = part[part.find("(") + 1:part.rfind(")")].strip() if "(" in part else ""
                if name and name not in [c["name"] for c in sc.characters]:
                    sc.characters.append({"name": name, "description": role})
                    known.add(name)
            sc.notes.append(line)
            continue
        if not sc.title and not started and i == next(k for k, x in enumerate(lines) if x) and len(bare) <= 60 \
                and not RE_TITLE.match(bare) and not RE_CLIP.match(bare) and ":" not in bare and "ตัวละคร" != bare:
            sc.title = bare.strip("“”\"' ")
            continue
        if not sc.title and (m := RE_TITLE.match(bare)):
            sc.title = m.group(1).strip("“”\"' ")
            continue
        if (m := RE_EPISODE.match(bare)):
            cur_ep = {"number": int(m.group(1)), "title": m.group(2).strip()}
            sc.episodes.append(cur_ep)
            continue
        if RE_CHAR_HEADER.match(bare):
            in_chars, chars_in_block = True, 0
            continue
        if in_chars and not started and not RE_SCENE.match(bare) and not RE_SCENE_NUM.match(bare) and not RE_CLIP.match(bare):
            raw = line.lstrip("-•* ").strip()
            name, sep, desc = re.split(r"\s*[—–:]\s*|\s+-\s+", raw, maxsplit=1)[0], None, ""
            parts = re.split(r"\s*[—–:]\s*|\s+-\s+", raw, maxsplit=1)
            if len(parts) == 2:
                name, desc = parts
            else:
                bits = raw.split(maxsplit=1)
                name, desc = bits[0], (bits[1] if len(bits) > 1 else "")
                # names like "คุณหญิงมณี", "ทนายสมชาย" are one token in Thai; keep as written
            sc.characters.append({"name": name.strip(), "description": desc.strip()})
            known.add(name.strip())
            chars_in_block += 1
            continue
        if (m := RE_CLIP.match(bare)):
            started = True
            rest = m.group(3)
            dur = RE_DURATION.search(rest)
            title = re.sub(r"^\s*[\-—–|:]\s*", "", rest)
            title = title if "|" not in rest else ""
            new_shot(int(m.group(1)), float(dur.group(1)) if dur else None, title.strip())
            pending_actions.clear()
            continue
        if (m := RE_SCENE_NUM.match(bare)):
            started = True
            new_scene(m.group(2), int(m.group(1)))
            # "ฉาก 1: การเผชิญหน้า" may be followed by "วันที่ 1 ช่วงกลางวัน / ห้องโถงคฤหาสน์"
            nxt = lines[i + 1] if i + 1 < len(lines) else ""
            if nxt and "/" in nxt and not RE_CLIP.match(nxt) and ":" not in nxt and "“" not in nxt and '"' not in nxt:
                when, _, where = nxt.partition("/")
                cur_scene.update(time=when.strip(), location=where.strip())
                lines[i + 1] = ""
            cur_shot = None if sc.format == "clips" else cur_shot
            continue
        if (m := RE_SCENE.match(bare)):
            started = True
            if sc.format == "clips" and cur_shot is not None:
                cur_shot["location"] = m.group(1).strip()      # place of this clip
            else:
                new_scene(m.group(1))
                if sc.format == "story" and pending_actions and sc.shots:
                    sc.shots[-1]["action"] += pending_actions
                    pending_actions.clear()
            continue
        if not started:
            sc.notes.append(line)
            continue
        if (m := RE_CAMERA.match(bare)):
            if cur_shot is not None:
                cur_shot["camera"] = m.group(1).strip()
            continue
        if (m := RE_IN_SCENE.match(bare)):
            if cur_shot is not None:
                cur_shot["characters"] = [x.strip() for x in re.split(r"[,、，และ]+", m.group(1)) if x.strip()]
            continue
        if RE_DIALOGUE_HEADER.match(bare):
            continue
        if RE_NO_DIALOGUE.match(bare):
            if cur_shot is not None:
                cur_shot["declared_silent"] = True
            continue
        if (m := RE_NARRATION.match(bare)):
            q, after = split_quoted(m.group(1))
            target = cur_shot if sc.format == "clips" else None
            if target is None:
                if sc.format == "story":
                    new_shot()
                    cur_shot["action"] += pending_actions
                    pending_actions.clear()
                target = cur_shot
            target["narration"].append(q if q is not None else m.group(1).strip())
            continue
        if (m := RE_DIALOGUE_BRACKET.match(bare)):
            if add_dialogue(m.group(1), m.group(2), m.group(3)):
                continue
        if (m := RE_DIALOGUE_PREFIX.match(bare)):
            inner = m.group(1)
            if inner.strip()[:1] in OPEN_Q and add_dialogue("", None, inner):
                sc.warnings.append(f"{label(cur_shot)} มีบทพูดที่ไม่ระบุผู้พูด: “{cur_shot['dialogue'][-1]['text'][:40]}”")
                continue
            if (d := RE_DIALOGUE.match(inner)) and d.group(1).strip() not in NOT_SPEAKERS:
                if add_dialogue(d.group(1), d.group(2), d.group(3)):
                    continue
        if (m := RE_DIALOGUE.match(bare)):
            speaker = m.group(1).strip()
            q, _ = split_quoted(m.group(3))
            looks_spoken = q is not None or speaker in known
            if speaker in known and re.fullmatch(r"\(?\s*(?:ไม่มี|ไม่พูด|เงียบ|—|-)\s*\)?", m.group(3).strip()):
                continue
            if speaker not in NOT_SPEAKERS and looks_spoken and len(speaker) <= 30:
                if add_dialogue(speaker, m.group(2), m.group(3)):
                    continue
        if (m := RE_ACTION.match(bare)):
            line = m.group(1).strip()
        if sc.format == "clips":
            if cur_shot is None:
                new_shot()
            cur_shot["action"].append(line)
        else:
            pending_actions.append(line)

    if pending_actions:
        if sc.format == "story":
            new_shot()
        cur_shot["action"] += pending_actions
    for s in sc.shots:
        names = {d["speaker"] for d in s["dialogue"]}
        if not s["characters"]:
            s["characters"] = sorted(names | {c["name"] for c in sc.characters
                                              if any(c["name"] in a for a in s["action"])})
        if s["declared_silent"] and s["dialogue"]:
            sc.warnings.append(f"{label(s)} ระบุว่าไม่มีบทพูด แต่พบบทพูด")
    check_complete(text, sc)
    return sc.to_dict()


def label(shot: dict) -> str:
    return f"คลิป {shot['number']:03d}" if shot.get("number") else f"ช็อต {shot.get('pos', shot['index']) + 1}"


def check_complete(raw: str, sc: Script) -> None:
    """Self-check: every quoted line in the story part must be in the parsed dialogue, verbatim."""
    spoken = [d["text"] for s in sc.shots for d in s["dialogue"]] + [n for s in sc.shots for n in s["narration"]]
    joined = "\n".join(spoken)
    notes = "\n".join(sc.notes)
    for q in quoted_segments(raw):
        q = norm(q)
        if q and q not in joined and q not in notes and q != sc.title and not any(q in a for s in sc.shots for a in s["action"]):
            sc.warnings.append(f"พบข้อความในเครื่องหมายคำพูดที่ไม่ได้ถูกนับเป็นบทพูด: “{q[:60]}”")
    for d in spoken:
        if norm(d) not in norm(raw):
            sc.warnings.append(f"บทพูดไม่ตรงต้นฉบับ: “{d[:60]}”")
