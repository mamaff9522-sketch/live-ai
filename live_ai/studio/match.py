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


RE_STAMP = re.compile(r"(20\d\d)[-_.]?(\d\d)[-_.]?(\d\d)[-_. T]?(\d\d)[-_.:]?(\d\d)[-_.:]?(\d\d)")


def file_order(names: list[str]) -> dict[str, tuple]:
    """Order of clips as made: by the date-time in the file names when every name has one (phones and
    Meta AI / Grok name downloads "…_20261101_171141.mp4"; the random part before it means nothing),
    else by natural name order ("คลิป 2" before "คลิป 10")."""
    stamps = {n: RE_STAMP.search(n) for n in names}
    if names and all(stamps.values()):
        return {n: (0, "".join(stamps[n].groups()), natural_key(n)) for n in names}
    # numbered names ("คลิป 2", "scene_10"): the numbers give the order. Names that differ in more than
    # their numbers (random ids) say nothing about the order: then the upload order decides.
    shapes = [re.sub(r"\d+", "#", n) for n in names]
    if names and all(re.search(r"\d", n) for n in names) \
            and (len(names) == 1 or 2 <= max(shapes.count(x) for x in shapes) >= len(names) / 2):
        return {n: (1, "", natural_key(n)) for n in names}
    return {n: (2, "", []) for n in names}


def match(script: dict, clips: list[dict], visual: dict | None = None, vl_votes: dict | None = None,
          unscripted: bool = True) -> dict:
    """clips: [{"id","name","order","duration","speech": bool, "transcript": str, "duplicate_of": id|None}]
    Returns {"assign": {clip_id: {...}}, "timeline": [...], "missing": [...], "extra": [...], "issues": [...]}"""
    shots = script["shots"]
    lines = dialogue_lines(shots)
    shot_of_line = {l["li"]: l["shot"] for l in lines}
    assign: dict[str, dict] = {}
    issues: list[dict] = []

    # 1. clips with speech -> line windows. Strongest matches first; a clip whose lines are already
    #    taken tries the remaining lines (the script may repeat a sentence word for word) and only
    #    shares a line when nothing else fits nearly as well (a line continuing into the next clip).
    speaking = [c for c in clips if c.get("speech") and c.get("transcript") and not c.get("duplicate_of")]
    full = {c["id"]: best_window(c["transcript"], lines) for c in speaking}
    taken: set[int] = set()
    holder: dict[int, float] = {}              # line -> score of the clip that took it
    holder_text: dict[int, str] = {}           # line -> what that clip says
    alternates: dict[str, str] = {}            # clip -> the clip that already says its line (another take)
    for c in sorted(speaking, key=lambda c: (-full[c["id"]]["score"], c["order"])):
        w = full[c["id"]]
        if w["score"] < MATCH_WEAK or not w["lines"]:
            continue
        if taken & set(w["lines"]):
            free = best_window(c["transcript"], lines, exclude=taken)
            shared = sorted(taken & set(w["lines"]))
            # the clip that has this line already says all of it (not only its first half)
            whole = containment("".join(lines[li]["text"] for li in shared), "".join(holder_text[li] for li in shared)) >= 0.7
            if free["lines"] and free["score"] >= max(MATCH_WEAK, 0.85 * w["score"]):
                w = free
            elif whole and free["lines"] and free["score"] >= max(0.12, 0.6 * w["score"]):
                # its best line is said in full by another clip, and it fits a line no clip has yet: it says
                # that line, badly recognised (seen: a clip for the one missing shot sat on another line)
                w = free
            elif w["score"] < 0.5 * max(holder[li] for li in shared):
                # a line another clip says clearly; this one only shares a few sounds with it ("ครับเฮีย"),
                # which is not the second half of that line: it says something else
                continue
            elif whole:
                # another take of a line a clip already says in full: kept out of the video
                alternates[c["id"]] = next(k for k, v in assign.items() if set(v["lines"]) & set(shared))
                continue
        assign[c["id"]] = {"lines": w["lines"], "score": w["score"], "margin": w["margin"], "how": "dialogue"}
        taken |= set(w["lines"])
        for li in w["lines"]:
            if w["score"] > holder.get(li, 0.0):
                holder[li], holder_text[li] = w["score"], c["transcript"]

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
    wrong_place(assign, clip_by_id, shots, lines, visual or {}, issues)
    used_shots = {s for a in assign.values() for s in a["shots"]}
    silent_shots = [s["index"] for s in shots if not s["dialogue"] and s["index"] not in used_shots]
    rest = [c for c in clips if c["id"] not in assign and not c.get("duplicate_of")]
    order = sorted(clips, key=lambda c: (c["order"],))
    rank = {c["id"]: i for i, c in enumerate(order)}
    silent_clips, talking = [], []
    for c in sorted(rest, key=lambda c: rank[c["id"]]):
        if c["id"] in alternates:
            o = clip_by_id[alternates[c["id"]]]
            issues.append({"type": "alternate", "clip": c["id"], "of": o["id"],
                           "message": f"คลิป “{c['name']}” พูดบทเดียวกับ “{o['name']}” (อีกเทค) — ใช้คลิปที่เสียงพูดตรงบทกว่า "
                                      "ถ้าอยากใช้เทคนี้แทน ให้ลบอีกคลิปออกแล้วกด Start ใหม่"})
            continue
        if c.get("speech") and c.get("transcript"):
            if any(f1(c["transcript"], o["transcript"]) >= 0.8 and abs(c["duration"] - o["duration"]) < 1.0 for o in talking):
                o = next(o for o in talking if f1(c["transcript"], o["transcript"]) >= 0.8)
                c["duplicate_of"] = o["id"]
                issues.append({"type": "duplicate", "clip": c["id"], "of": o["id"],
                               "message": f"คลิป “{c['name']}” ซ้ำกับ “{o['name']}” (เสียงพูดเหมือนกัน) จะไม่ถูกใช้"})
            elif unscripted:
                talking.append(c)
            else:
                issues.append({"type": "unmatched", "clip": c["id"],
                               "message": f"คลิป “{c['name']}” มีเสียงพูด (“{c['transcript'][:50]}”) แต่ไม่ตรงกับบทใดเลย — ไม่ได้ใส่ (ตั้งค่าไว้ว่าไม่ใส่คลิปที่ไม่อยู่ในบท)"})
        else:
            silent_clips.append(c)
    placements = place_silent(silent_clips, silent_shots, assign, clip_by_id, rank, shots, visual or {}, vl_votes or {})
    candidates = {}
    for c in silent_clips:
        p = placements.get(c["id"])
        candidates[c["id"]] = p["candidates"] if p else silent_shots[:6]
        if not p or p.get("shot") is None:
            issues.append({"type": "extra", "clip": c["id"],
                           "message": f"คลิป “{c['name']}” ไม่มีเสียงพูด และไม่มีช็อตเงียบที่ว่างให้ใส่ — อาจเป็นคลิปเกิน"})
            continue
        sidx = p["shot"]
        assign[c["id"]] = {"lines": [], "shots": [sidx], "score": p["score"], "margin": p["margin"], "how": p["how"],
                           "confidence": p["confidence"], "evidence": p["evidence"]}
        if p["confidence"] != "สูง":
            issues.append({"type": "check_silent", "clip": c["id"],
                           "message": f"คลิป “{c['name']}” ไม่มีเสียงพูด วางไว้ที่ {label(shots[sidx])} "
                                      f"({' '.join(shots[sidx]['action'])[:60]}) จาก{p['evidence']} — ความมั่นใจ{p['confidence']} กรุณาตรวจภาพ"})

    # 4. timeline + missing shots
    timeline = sorted(assign, key=lambda cid: (assign[cid]["shots"][0], assign[cid]["lines"][0] if assign[cid]["lines"] else -1,
                                               -assign[cid].get("pos", 0.0), rank[cid]))
    # clips whose words are not in the script (the script was pasted incomplete, or the clip says it
    # differently): an editor would still use them, in the order they were made, next to the clip made
    # just before them. Their words get no subtitles (subtitles are copied from the script only).
    for c in talking:
        before = [cid for cid in timeline if rank[cid] < rank[c["id"]]]
        at = timeline.index(max(before, key=lambda cid: rank[cid])) + 1 if before else 0
        if at < len(timeline) and not before:
            shot = assign[timeline[0]]["shots"][0]
        elif at:
            shot = assign[timeline[at - 1]]["shots"][-1]
        else:
            shot = 0
        assign[c["id"]] = {"lines": [], "shots": [shot], "score": 0.0, "margin": 0.0, "how": "unscripted",
                           "confidence": "ต่ำ", "unscripted": True}
        timeline.insert(at, c["id"])
        issues.append({"type": "unscripted", "clip": c["id"],
                       "message": f"คลิป “{c['name']}” พูดว่า “{c['transcript'][:50]}” ซึ่งไม่อยู่ในบท — ใส่ไว้ตามลำดับที่สร้างคลิป "
                                  "(ไม่มีซับสำหรับคลิปนี้ ถ้าวางบทครบทั้งเรื่องจะได้ซับและลำดับตามบท; ถ้าไม่ใช่ของเรื่องนี้ ให้ลบคลิปออกแล้วกด Start ใหม่)"})
    covered = {s for a in assign.values() for s in a["shots"] if not a.get("unscripted")}
    covered_lines = {li for a in assign.values() for li in a["lines"]}
    missing = []
    for s in shots:
        if s.get("optional") and s["index"] not in covered:
            continue                     # e.g. an opening / closing shot without words in a free-form script
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
    return {"assign": assign, "timeline": timeline, "missing": missing, "issues": issues, "silent_candidates": candidates}


def label(shot: dict) -> str:
    return f"คลิป {shot['number']:03d}" if shot.get("number") else f"ช็อต {shot.get('pos', shot['index']) + 1}"


PLACE_MARGIN = 0.08     # how much more a clip must look like another scene than its own to be moved


def wrong_place(assign: dict, clip_by_id: dict, shots: list[dict], lines: list[dict], visual: dict, issues: list) -> None:
    """A clip matched by its words only weakly, whose picture looks like another scene (seen: a night
    hospital clip in the middle of the daytime shop scene, matched on a few shared words), is moved to
    a free line of the scene it looks like, or left out when there is none. Clips matched with high
    confidence are the scenes' reference pictures and are never moved."""
    from .visual import similarity
    if not visual:
        return
    scene_of = {s["index"]: s.get("scene") for s in shots}
    refs: dict = {}
    for cid, a in assign.items():
        if a.get("confidence") == "สูง" and a["shots"] and visual.get(cid):
            refs.setdefault(scene_of[a["shots"][0]], []).append(cid)
    if len(refs) < 2:
        return

    def look(cid, scene):
        v = visual.get(cid) or {}
        return max((similarity(v.get(k), (visual.get(r) or {}).get(k2)) for r in refs.get(scene, []) if r != cid
                    for k in ("first", "mid", "last") for k2 in ("first", "mid", "last")), default=0.0)

    taken = {li for a in assign.values() for li in a["lines"]}
    for cid in list(assign):
        a = assign[cid]
        if a.get("confidence") == "สูง" or not a["shots"] or not visual.get(cid):
            continue
        own = scene_of[a["shots"][0]]
        scores = {sc: look(cid, sc) for sc in refs}
        best = max(scores, key=scores.get)
        if best == own or scores[best] < scores.get(own, 0.0) + PLACE_MARGIN:
            continue
        c = clip_by_id[cid]
        here = [l for l in lines if scene_of[l["shot"]] == best and l["li"] not in taken]
        w = best_window(c.get("transcript") or "", here and lines, exclude={l["li"] for l in lines} - {l["li"] for l in here}) if here else None
        mine = set(a["lines"])
        if w and w["lines"] and w["score"] >= 0.12:
            taken -= mine
            taken |= set(w["lines"])
            a.update(lines=w["lines"], shots=sorted({lines[li]["shot"] for li in w["lines"]}), score=w["score"],
                     margin=w.get("margin", 0.0), confidence="ต่ำ", how="dialogue+picture")
            issues.append({"type": "moved", "clip": cid,
                           "message": f"คลิป “{c['name']}” ภาพดูเป็นอีกฉาก จึงย้ายไปช็อต {', '.join(label(shots[s]) for s in a['shots'])} "
                                      "ที่ยังไม่มีคลิป — กรุณาตรวจ"})
        else:
            taken -= mine
            del assign[cid]
            issues.append({"type": "wrong_place", "clip": cid,
                           "message": f"คลิป “{c['name']}” เสียงพูดตรงบทแค่บางคำ และภาพดูเป็นอีกฉาก (คนละสถานที่/เวลา) จึงไม่ใส่ "
                                      "— ถ้าเป็นคลิปของเรื่องนี้จริง ให้ตรวจว่าพูดตรงบทไหม"})
        for i in [i for i in issues if i.get("clip") == cid and i["type"] == "uncertain"]:
            issues.remove(i)


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


def place_silent(silent_clips, silent_shots, assign, clip_by_id, rank, shots, visual, vl_votes) -> dict:
    """Puts clips without speech into silent shots.

    Evidence, strongest first: the picture continues from the clip before / into the clip after
    (generated clips start from the previous clip's last frame), what the picture shows (the vision
    model picked this shot's description), and only then upload / file-name order."""
    from .visual import similarity
    shot_clip = {}
    for cid, a in assign.items():
        for s in a["shots"]:
            shot_clip.setdefault(s, cid)
    remaining_shots = list(silent_shots)
    out = {}
    if not silent_clips:
        return out

    def neighbours(s):
        prev = next((shot_clip[k] for k in range(s - 1, -1, -1) if k in shot_clip), None)
        nxt = next((shot_clip[k] for k in range(s + 1, len(shots)) if k in shot_clip), None)
        return prev, nxt

    def score(c, s):
        ev, parts = [], []
        prev, nxt = neighbours(s)
        v = visual.get(c["id"])
        if v:
            if prev is not None and shot_clip.get(s - 1) == prev and visual.get(prev):
                parts.append(similarity(visual[prev]["last"], v["first"]))
            if nxt is not None and shot_clip.get(s + 1) == nxt and visual.get(nxt):
                parts.append(similarity(v["last"], visual[nxt]["first"]))
        cont = max(parts) if parts else None
        sc = 0.0
        if cont is not None:
            sc += cont
            ev.append(f"ภาพต่อเนื่องกับคลิปข้างเคียง {int(cont * 100)}%")
        if vl_votes.get(c["id"]) == s:
            sc += 0.5
            ev.append("AI ดูภาพแล้วตรงกับคำบรรยายช็อตนี้")
        lo = max([rank[x] for x in ([prev] if prev else [])] or [-1])
        hi = min([rank[x] for x in ([nxt] if nxt else [])] or [10**9])
        if lo < rank[c["id"]] < hi:
            sc += 0.1
            ev.append("ลำดับไฟล์อยู่ระหว่างคลิปข้างเคียง")
        return sc, cont, ev

    # greedy over the best (clip, shot) pairs; recompute after each placement because a placed
    # silent clip becomes the neighbour of the next silent shot
    pending = list(silent_clips)
    for _ in range(len(silent_clips)):
        if not pending or not remaining_shots:
            break
        table = []
        for c in pending:
            row = sorted(((score(c, s), s) for s in remaining_shots), key=lambda x: -x[0][0])
            best, second = row[0], (row[1] if len(row) > 1 else ((0.0, None, []), None))
            table.append((best[0][0], best[0][0] - second[0][0], c, best, [s for _, s in row[:6]]))
        table.sort(key=lambda t: (-t[0], -t[1]))
        sc, margin, c, ((_, cont, ev), s), cands = table[0]
        only_one = len(remaining_shots) == 1 and len(pending) == 1
        if (cont is not None and cont >= 0.85 and margin >= 0.15) or (vl_votes.get(c["id"]) == s and (cont or 0) >= 0.5):
            conf = "สูง"
        elif only_one or margin >= 0.1 or vl_votes.get(c["id"]) == s:
            conf = "กลาง"
        else:
            conf = "ต่ำ"
        if only_one and not ev:
            ev = ["เป็นช่องเงียบช่องเดียวที่เหลือ"]
        out[c["id"]] = {"shot": s, "score": round(sc, 3), "margin": round(margin, 3), "confidence": conf,
                        "how": "silent-visual" if cont is not None else "silent-order", "evidence": ", ".join(ev) or "ลำดับไฟล์",
                        "candidates": cands}
        shot_clip[s] = c["id"]
        remaining_shots.remove(s)
        pending.remove(c)
    return out
