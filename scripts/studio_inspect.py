"""Runs inside the Live AI container (read-only): how the newest studio projects were edited. The
repository is public, so nothing the owner wrote or said is printed: no names, no script text, no
transcripts (quoted text becomes “…”). Counts, positions, scores and the studio's own state only."""
import json, re, sqlite3
c = sqlite3.connect("file:/data/live_ai.sqlite3?mode=ro", uri=True)
c.row_factory = sqlite3.Row


def hide(s):
    return re.sub(r"“[^”]*”|\"[^\"]*\"", "“…”", str(s or ""))


for p in c.execute("SELECT * FROM projects ORDER BY updated DESC LIMIT 3").fetchall():
    clips = c.execute("SELECT media_id, upload_order, inspect, asr FROM project_clips WHERE project_id=? ORDER BY upload_order", (p["id"],)).fetchall()
    up = {r["media_id"]: r["upload_order"] for r in clips}
    print(f"\n===== project {p['id']} | status {p['status']} | clips {len(clips)} | script chars {len(p['script_text'] or '')}")
    print("message:", hide(p["message"]))
    print("settings:", p["settings"])
    sc = json.loads(p["script"] or "null")
    if sc:
        print("script:", sc.get("format"), sc.get("stats"), "| scenes", len(sc.get("scenes") or []))
        print("script warnings:", [hide(w)[:160] for w in sc.get("warnings", [])][:6])
        print("shots (speakers, lines, optional):", [(len({d['speaker'] for d in s['dialogue']}), len(s['dialogue']), bool(s.get('optional'))) for s in sc.get("shots", [])])
    for k, a in json.loads(p["agents"] or "{}").items():
        print(f"agent {k}:", a.get("state"), hide(a.get("message"))[:300])
    r = json.loads(p["result"] or "{}")
    tl = (r.get("timeline") or {}).get("video", [])
    print("timeline:", len(tl), "items, duration", (r.get("timeline") or {}).get("duration"))
    for v in tl:
        print("  up#", up.get(v.get("clip"), "?"), "| shots", v.get("shots"), "| lines", v.get("lines"), "|", v.get("how"), v.get("confidence"),
              "| keep", v.get("in"), "-", v.get("out"), "| fade", v.get("fade_in"), v.get("fade_out"))
    used = {v.get("clip") for v in tl}
    print("clips not used (upload #):", sorted(up[m] for m in up if m not in used))
    for r2 in clips:
        a = json.loads(r2["asr"] or "{}"); ins = json.loads(r2["inspect"] or "{}")
        print("  clip up#", r2["upload_order"], "| speech", bool((a.get("text") or "").strip()), "| words", len((a.get("text") or "").split()),
              "| chars", len(a.get("text") or ""), "| jumps", ins.get("jumps"), "| lufs", ins.get("lufs"))
    kinds = {}
    for i in r.get("issues", []):
        kinds[i.get("type")] = kinds.get(i.get("type"), 0) + 1
    print("issues by type:", kinds)
    print("missing shots:", [m.get("label") for m in r.get("missing", [])])
    print("qa:", [(q["ok"], hide(q["name"])) for q in (r.get("qa") or {}).get("checks", [])])
