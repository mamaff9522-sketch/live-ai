"""AI 6 (Script–Video Matcher) + AI 7 (Sequence Editor) + the missing/duplicate part of AI 8.

How clips are matched (file names are never the main evidence):
1. A clip with speech is matched by WHAT IS SAID: its transcript is compared with every window
   of 1–4 consecutive script lines (Thai-friendly character-bigram F1). Best window wins.
2. Two clips claiming the same lines: near-identical transcripts = duplicate clip; otherwise the
   line is spoken across both clips (J/L-cut) and the one holding the start of the line goes first.
3. Clips without speech go into silent shots between their speaking neighbours. Upload/file-name
   order is used ONLY to decide among those gaps, and such placements are marked low-confidence
   for the user to check (picture analysis, AI 4, will replace this in phase 2).
4. Shots with no clip are reported as missing with the scene, shot number and what happens in it.
"""
from __future__ import annotations

import re
import unicodedata

RE_KEEP = re.compile(r"[^฀-๿a-z0-9]+")
MATCH_OK = 0.42         # bigram F1 needed to call a clip matched
MATCH_WEAK = 0.22       # below this the clip is treated as not matching any line


def norm(s: str) -> str:
    s = unicodedata.normalize("NFC", s).lower()
    s = s.replace("ๆ", "")
    return RE_KEEP.sub("", s)


def bigrams(s: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for i in range(len(s) - 1):
        g = s[i:i + 2]
        out[g] = out.get(g, 0) + 1
    return out


def f1(a: str, b: str) -> float:
    A, B = bigrams(norm(a)), bigrams(norm(b))
    if not A or not B:
        return 0.0
    inter = sum(min(v, B.get(k, 0)) for k, v in A.items())
    if not inter:
        return 0.0
    p, r = inter / sum(B.values()), inter / sum(A.values())
    return 2 * p * r / (p + r)


def containment(part: str, whole: str) -> float:
    A, B = bigrams(norm(part)), bigrams(norm(whole))
    if not A:
        return 0.0
    return sum(min(v, B.get(k, 0)) for k, v in A.items()) / sum(A.values())


def natural_key(name: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def dialogue_lines(shots: list[dict]) -> list[dict]:
    out = []
    for s in shots:
        for k, d in enumerate(s["dialogue"]):
            out.append({"li": len(out), "shot": s["index"], "k": k, "speaker": d["speaker"], "text": d["text"]})
    return out


def best_window(transcript: str, lines: list[dict], exclude: set[int] = frozenset(), max_len: int = 4) -> dict:
    best = {"score": 0.0, "lines": []}
    second = 0.0
    n = len(lines)
    for a in range(n):
        if a in exclude:
            continue
        text = ""
        for b in range(a, min(n, a + max_len)):
            if b in exclude:
                break
            text += lines[b]["text"]
            sc = f1(text, transcript)
            if sc > best["score"]:
                if best["lines"] and best["lines"][0] != a:
                    second = max(second, best["score"])
                best = {"score": sc, "lines": list(range(a, b + 1))}
            elif best["lines"] and best["lines"][0] != a:
                second = max(second, sc)
    best["margin"] = round(best["score"] - second, 3)
    best["score"] = round(best["score"], 3)
    return best


def match(script: dict, clips: list[dict]) -> dict:
    """clips: [{"id","name","order","duration","speech": bool, "transcript": str, "duplicate_of": id|None}]
    Returns {"assign": {clip_id: {...}}, "timeline": [...], "missing": [...], "extra": [...], "issues": [...]}"""
    shots = script["shots"]
    lines = dialogue_lines(shots)
    shot_of_line = {l["li"]: l["shot"] for l in lines}
    assign: dict[str, dict] = {}
    issues: list[dict] = []

    # 1. clips with speech -> line windows
    speaking = [c for c in clips if c.get("speech") and c.get("transcript") and not c.get("duplicate_of")]
    for c in speaking:
        w = best_window(c["transcript"], lines)
        if w["score"] >= MATCH_WEAK and w["lines"]:
            assign[c["id"]] = {"lines": w["lines"], "score": w["score"], "margin": w["margin"], "how": "dialogue"}

    # 2. conflicts: several clips on the same first line
    by_line: dict[int, list[str]] = {}
    for cid, a in assign.items():
        by_line.setdefault(a["lines"][0], []).append(cid)
    clip_by_id = {c["id"]: c for c in clips}
    for li, cids in by_line.items():
        if len(cids) < 2:
            continue
        cids.sort(key=lambda x: -assign[x]["score"])
        keep = cids[0]
        for other in cids[1:]:
            t1, t2 = clip_by_id[keep]["transcript"], clip_by_id[other]["transcript"]
            if f1(t1, t2) >= 0.8 and abs(clip_by_id[keep]["duration"] - clip_by_id[other]["duration"]) < 1.0:
                assign.pop(other)
                clip_by_id[other]["duplicate_of"] = keep
                issues.append({"type": "duplicate", "clip": other, "of": keep,
                               "message": f"คลิป “{clip_by_id[other]['name']}” ซ้ำกับ “{clip_by_id[keep]['name']}” (เสียงพูดเหมือนกัน) จะไม่ถูกใช้"})
                continue
            # the same line spoken across two clips (continuation): order by who has the start of the line
            line = lines[li]["text"]
            half = max(2, len(norm(line)) // 2)
            head, tail = line[: len(line) // 2], line[len(line) // 2:]
            for cid in (keep, other):
                assign[cid]["pos"] = containment(head, clip_by_id[cid]["transcript"]) - containment(tail, clip_by_id[cid]["transcript"])
            assign[other]["continuation"] = True
            del half

    # 3. order speaking clips, then fit silent clips into silent shots between them
    def anchor(cid):
        a = assign[cid]
        return (shot_of_line[a["lines"][0]], a["lines"][0], -a.get("pos", 0.0))

    for cid, a in assign.items():
        a["shots"] = sorted({shot_of_line[li] for li in a["lines"]})
        a["confidence"] = "สูง" if a["score"] >= 0.6 and a["margin"] >= 0.1 else ("กลาง" if a["score"] >= MATCH_OK else "ต่ำ")
        if a["score"] < MATCH_OK:
            issues.append({"type": "uncertain", "clip": cid,
                           "message": f"คลิป “{clip_by_id[cid]['name']}” เสียงพูดตรงกับบทเพียง {int(a['score'] * 100)}% — กรุณาตรวจว่าเป็นช็อต "
                                      f"{', '.join(label(shots[s]) for s in a['shots'])} จริงหรือไม่"})
    used_shots = {s for a in assign.values() for s in a["shots"]}
    silent_shots = [s["index"] for s in shots if not s["dialogue"] and s["index"] not in used_shots]
    rest = [c for c in clips if c["id"] not in assign and not c.get("duplicate_of")]
    order = sorted(clips, key=lambda c: (c["order"],))
    rank = {c["id"]: i for i, c in enumerate(order)}
    placed_silent: set[int] = set()
    silent_clips = [c for c in rest if not (c.get("speech") and c.get("transcript"))]
    if len(silent_clips) == 1 and len(silent_shots) == 1:      # only one possible place: no guessing needed
        c, sidx = silent_clips[0], silent_shots[0]
        assign[c["id"]] = {"lines": [], "shots": [sidx], "score": 0.0, "margin": 0.0, "how": "silent-only-gap", "confidence": "กลาง"}
        issues.append({"type": "check_silent", "clip": c["id"],
                       "message": f"คลิป “{c['name']}” ไม่มีเสียงพูด และเป็นช่องเงียบช่องเดียวที่เหลือ จึงวางไว้ที่ {label(shots[sidx])} "
                                  f"({' '.join(shots[sidx]['action'])[:60]}) — กรุณาตรวจภาพ"})
        rest = [x for x in rest if x["id"] != c["id"]]
    for c in sorted(rest, key=lambda c: rank[c["id"]]):
        if c.get("speech") and c.get("transcript"):
            issues.append({"type": "unmatched", "clip": c["id"],
                           "message": f"คลิป “{c['name']}” มีเสียงพูด (“{c['transcript'][:50]}”) แต่ไม่ตรงกับบทใดเลย — อาจเป็นคลิปที่สร้างผิดหรือไม่ใช่ของเรื่องนี้"})
            continue
        before = [assign[x]["shots"][-1] for x in assign if rank[x] < rank[c["id"]]]
        after = [assign[x]["shots"][0] for x in assign if rank[x] > rank[c["id"]]]
        lo = max(before) if before else -1
        hi = min(after) if after else len(shots)
        cands = [s for s in silent_shots if lo < s < hi and s not in placed_silent]
        if not cands:
            cands = [s for s in silent_shots if s not in placed_silent]
            how = "silent-any"
        else:
            how = "silent-between"
        if not cands:
            issues.append({"type": "extra", "clip": c["id"],
                           "message": f"คลิป “{c['name']}” ไม่มีเสียงพูด และไม่มีช็อตเงียบที่ว่างให้ใส่ — อาจเป็นคลิปเกิน"})
            continue
        s = cands[0]
        placed_silent.add(s)
        assign[c["id"]] = {"lines": [], "shots": [s], "score": 0.0, "margin": 0.0, "how": how,
                           "confidence": "ต่ำ" if how == "silent-any" or len(cands) > 1 else "กลาง"}
        issues.append({"type": "check_silent", "clip": c["id"],
                       "message": f"คลิป “{c['name']}” ไม่มีเสียงพูด จึงวางไว้ที่ {label(shots[s])} ({' '.join(shots[s]['action'])[:60]}) "
                                  f"ตามตำแหน่งระหว่างคลิปข้างเคียง — กรุณาตรวจภาพ"})

    # 4. timeline + missing shots
    timeline = sorted(assign, key=lambda cid: (assign[cid]["shots"][0], assign[cid]["lines"][0] if assign[cid]["lines"] else -1,
                                               -assign[cid].get("pos", 0.0), rank[cid]))
    covered = {s for a in assign.values() for s in a["shots"]}
    covered_lines = {li for a in assign.values() for li in a["lines"]}
    missing = []
    for s in shots:
        lines_here = [l for l in lines if l["shot"] == s["index"]]
        if s["index"] in covered and all(l["li"] in covered_lines for l in lines_here):
            continue
        if s["index"] in covered and lines_here:
            lost = [l for l in lines_here if l["li"] not in covered_lines]
            missing.append({"shot": s["index"], "label": label(s), "partial": True,
                            "dialogue": [{"speaker": l["speaker"], "text": l["text"]} for l in lost]})
            continue
        missing.append({"shot": s["index"], "label": label(s), "partial": False, "action": s["action"],
                        "dialogue": s["dialogue"], "scene": s["scene"], "camera": s.get("camera", "")})
    return {"assign": assign, "timeline": timeline, "missing": missing, "issues": issues}


def label(shot: dict) -> str:
    return f"คลิป {shot['number']:03d}" if shot.get("number") else f"ช็อต {shot.get('pos', shot['index']) + 1}"


def regen_prompt(shot: dict, script: dict) -> str:
    """What to give the video generator to recreate a missing shot (no invented dialogue)."""
    scene = next((x for x in script["scenes"] if x["id"] == shot["scene"]), {})
    parts = [f"{label(shot)}", f"ฉาก: {scene.get('title', '')}".strip()]
    if shot.get("location"):
        parts.append(f"สถานที่: {shot['location']}")
    if shot.get("camera"):
        parts.append(f"มุมกล้อง: {shot['camera']}")
    if shot["action"]:
        parts.append("ภาพ/การกระทำ: " + " ".join(shot["action"]))
    if shot["dialogue"]:
        parts += [f"บทพูด [{d['speaker'] or 'ไม่ระบุ'}]: “{d['text']}”" for d in shot["dialogue"]]
    else:
        parts.append("บทพูด: ไม่มี (แสดงสีหน้า/บรรยากาศเท่านั้น)")
    if shot.get("duration"):
        parts.append(f"ความยาว {shot['duration']:g} วินาที")
    return "\n".join(parts)
