"""AI 12 (Cinematic Editor), the cutting rules: how much of each clip to keep and how clips join.

What an editor does first with generated clips, written as rules that need no big model:
  • spoken clips start a breath before the first word and end a beat after the last one; the beat is
    longer in sad / tender scenes and shorter in angry / tense ones, and the last clip of the video
    gets a longer ending;
  • stretches where the picture jumps (the inspector's `jumps`: sudden zoom, see-through figure …) are
    cut when they fall before the first word or after the last one;
  • clips without words keep a few seconds of their steadiest part (longer in sad scenes);
  • joins are hard cuts, or a short dip to black in sad / tender scenes and between scenes.
Words are never cut: every recognised speech segment stays inside the kept part.

A plan from a bigger model (the AI editor) can replace these numbers per clip; `apply_plan` checks
it against the same safety rules (no cut words, sane lengths) before using it.
"""
from __future__ import annotations

HOLD_AFTER = {"เศร้า": 1.6, "โรแมนติก": 1.4, "ปกติ": 0.8, "ตลก": 0.6, "โกรธ": 0.45, "ระทึก": 0.4}
LEAD_IN = {"เศร้า": 0.5, "โรแมนติก": 0.45, "ปกติ": 0.3, "ตลก": 0.25, "โกรธ": 0.15, "ระทึก": 0.15}
SILENT_KEEP = {"เศร้า": 4.0, "โรแมนติก": 3.5, "ปกติ": 3.0, "ตลก": 2.5, "โกรธ": 2.0, "ระทึก": 2.0}
SOFT = {"เศร้า", "โรแมนติก"}
MIN_KEEP = 1.0


def speech_spans(clip: dict) -> list[tuple[float, float]]:
    """Where words are spoken in the clip. The recogniser's segments say which stretch holds the words
    but often run on to the end of the clip; the loudness regions say where sound really stops, so the
    end of the words is the end of the last loud region inside the recogniser's span."""
    d = float(clip["duration"])
    segs = (clip.get("asr") or {}).get("segments") or []
    asr = [(max(0.0, float(s["start"])), min(d, float(s["end"]))) for s in segs if float(s["end"]) > float(s["start"])]
    regions = [(float(a), float(b)) for a, b in (clip.get("inspect") or {}).get("speech_regions") or []]
    if not asr:
        return regions if clip.get("transcript") else []
    a0, b0 = asr[0][0], asr[-1][1]
    inside = [(a, b) for a, b in regions if b > a0 - 0.3 and a < b0]
    if inside:
        s0 = max(a0 - 0.3, min(a for a, _ in inside))
        s1 = min(b0, max(b for _, b in inside))
        if s1 - s0 >= 0.3:
            return [(round(s0, 2), round(s1, 2))]
    return asr


def _clear_of_jumps(a: float, b: float, jumps: list) -> tuple[float, float]:
    """Moves a kept window off jump stretches at its edges (only where that leaves something)."""
    for x, y in jumps:
        if x <= a < y:
            a = y
        if x < b <= y:
            b = x
    return a, b


def plan_cuts(timeline: dict, clips_by_id: dict, mood_of_item, pacing: str = "auto") -> list[str]:
    """Sets in/out/fade_in/fade_out on every timeline item and recomputes start times.
    Returns human-readable notes (Thai) about what was cut."""
    items = timeline["video"]
    notes = []
    for k, it in enumerate(items):
        c = clips_by_id[it["clip"]]
        d = float(c["duration"])
        it["in"], it["out"] = 0.0, d
        it["fade_in"] = it["fade_out"] = 0.0
        if pacing == "original":
            continue
        mood = mood_of_item(it)
        jumps = [tuple(j) for j in (c.get("inspect") or {}).get("jumps") or []] + list(c.get("ai_broken") or [])
        peaks = sorted(c.get("ai_peaks") or [])          # times the AI editor rated most expressive
        spans = speech_spans(c) if it.get("lines") or it.get("unscripted") else []
        first, last = k == 0, k == len(items) - 1
        if spans:
            s0, s1 = spans[0][0], spans[-1][1]
            lead = max(LEAD_IN.get(mood, 0.3), 1.0 if first else 0.0)
            hold = HOLD_AFTER.get(mood, 0.8)
            if last:
                hold = max(hold * 2, 2.5)
            a, b = max(0.0, s0 - lead), min(d, s1 + hold)
            late = [t for t in peaks if s1 < t <= s1 + 3.0]
            if late:                                     # a strong reaction after the words: let it play
                b = min(d, max(b, late[-1] + 0.5))
            for x, y in jumps:
                if x < s0 and y > a:                     # jump before the first word: start after it
                    a = min(max(a, y), max(0.0, s0 - 0.1))
                if y > s1 and x < b:                     # jump after the last word: end before it
                    b = max(min(b, x), min(d, s1 + 0.2))
                if x < s1 and y > s0:
                    notes.append(f"“{c['name']}” ภาพกระโดดช่วง {x:.1f}–{y:.1f} วิ ทับช่วงที่พูด จึงตัดออกไม่ได้")
        else:
            keep = SILENT_KEEP.get(mood, 3.0) + (1.0 if first or last else 0.0)
            a, b = 0.0, d
            a, b = _clear_of_jumps(a, b, jumps)
            if b - a < MIN_KEEP:
                a, b = 0.0, d
            good = [t for t in peaks if a <= t <= b]
            if good and b - a > keep:                    # centre the kept seconds on the most expressive moment
                centre = good[0]
                a = max(a, min(centre - keep / 2, b - keep))
            b = min(b, a + keep)
        if b - a < MIN_KEEP:
            a, b = max(0.0, min(a, d - MIN_KEEP)), min(d, max(b, a + MIN_KEEP))
        cut = (a - 0.0) + (d - b)
        it["in"], it["out"] = round(a, 2), round(b, 2)
        if cut >= 0.5:
            parts = []
            if a >= 0.3:
                parts.append(f"ต้น {a:.1f} วิ")
            if d - b >= 0.3:
                parts.append(f"ท้าย {d - b:.1f} วิ")
            notes.append(f"“{c['name']}” ตัด{' และ'.join(parts)} (อารมณ์{mood})")
    # joins: dip to black in sad / tender scenes and between scenes
    for prev, nxt in zip(items, items[1:]):
        if pacing == "original":
            break
        m1, m2 = mood_of_item(prev), mood_of_item(nxt)
        new_scene = prev.get("scene") and nxt.get("scene") and prev["scene"] != nxt["scene"]
        if new_scene or m1 in SOFT or m2 in SOFT:
            dur = 0.5 if new_scene else 0.35
            prev["fade_out"] = nxt["fade_in"] = dur
    if items and pacing != "original":
        items[0]["fade_in"] = max(items[0]["fade_in"], 0.4)
        items[-1]["fade_out"] = max(items[-1]["fade_out"], 0.8 if mood_of_item(items[-1]) in SOFT else 0.4)
    t = 0.0
    for it in items:
        it["start"] = round(t, 3)
        t += it["out"] - it["in"]
    timeline["duration"] = round(t, 3)
    return notes


def apply_plan(timeline: dict, clips_by_id: dict, plan: dict) -> list[str]:
    """Uses an AI editor's per-clip choices ({clip_id: {in, out, fade_in, fade_out}}) where they are safe."""
    notes = []
    for it in timeline["video"]:
        p = plan.get(it["clip"])
        if not p:
            continue
        c = clips_by_id[it["clip"]]
        d = float(c["duration"])
        spans = speech_spans(c) if it.get("lines") or it.get("unscripted") else []
        a = max(0.0, min(float(p.get("in", it["in"])), d))
        b = max(0.0, min(float(p.get("out", it["out"])), d))
        if spans:
            a, b = min(a, spans[0][0]), max(b, spans[-1][1])        # never cut words
            s0, s1 = spans[0][0], spans[-1][1]
            # frames the inspector measured as broken (sudden jump / see-through) stay out even if the
            # model kept them, as long as no word is spoken there
            for x, y in [tuple(j) for j in (c.get("inspect") or {}).get("jumps") or []]:
                if x < s0 and y > a:
                    a2 = min(max(a, y), max(0.0, s0 - 0.1))
                    if a2 > a:
                        notes.append(f"“{c['name']}”: ตัดช่วงภาพเสีย {x:.1f}–{y:.1f} วิ ที่ AI เก็บไว้ออก")
                        a = a2
                if y > s1 and x < b:
                    b2 = max(min(b, x), min(d, s1 + 0.2))
                    if b2 < b:
                        notes.append(f"“{c['name']}”: ตัดช่วงภาพเสีย {x:.1f}–{y:.1f} วิ ที่ AI เก็บไว้ออก")
                        b = b2
        else:
            jumps = [tuple(j) for j in (c.get("inspect") or {}).get("jumps") or []]
            a, b = _clear_of_jumps(a, b, jumps) if b - a > MIN_KEEP else (a, b)
        if b - a < MIN_KEEP:
            notes.append(f"“{c['name']}”: แผนของ AI สั้นเกินไป ใช้ค่าจากกฎแทน")
            continue
        it["in"], it["out"] = round(a, 2), round(b, 2)
        for k in ("fade_in", "fade_out"):
            if k in p:
                it[k] = round(max(0.0, min(1.0, float(p[k]))), 2)
    t = 0.0
    for it in timeline["video"]:
        it["start"] = round(t, 3)
        t += it["out"] - it["in"]
    timeline["duration"] = round(t, 3)
    return notes


def labels_to_clip(clip: dict, frames: list[dict]) -> dict:
    """AI editor frame labels [{t, broken, emotion 1-5}] -> broken stretches and expressive moments on the clip.
    A broken frame marks the stretch from the previous to the next labelled frame (we only saw stills)."""
    fr = sorted((f for f in frames if isinstance(f, dict) and "t" in f), key=lambda f: float(f["t"]))
    d = float(clip["duration"])
    broken = []
    for i, f in enumerate(fr):
        if f.get("broken"):
            a = (float(fr[i - 1]["t"]) + float(f["t"])) / 2 if i else 0.0
            b = (float(f["t"]) + float(fr[i + 1]["t"])) / 2 if i + 1 < len(fr) else d
            broken.append((round(a, 2), round(b, 2)))
    top = max([int(f.get("emotion", 0)) for f in fr] or [0])
    peaks = [round(float(f["t"]), 2) for f in fr if not f.get("broken") and top >= 4 and int(f.get("emotion", 0)) >= top]
    clip["ai_broken"], clip["ai_peaks"] = broken, peaks
    return {"broken": broken, "peaks": peaks}
