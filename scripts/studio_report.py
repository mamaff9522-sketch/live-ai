"""Runs inside the Live AI container: the studio's own result for project $PID (states, messages, timeline
shape, QA). No script text, no media."""
import json, os, sqlite3
c = sqlite3.connect("/data/live_ai.sqlite3")
c.row_factory = sqlite3.Row
p = c.execute("SELECT status, message, result FROM projects WHERE id=?", (int(os.environ["PID"]),)).fetchone()
print("status:", p["status"])
print("message:", p["message"])
r = json.loads(p["result"] or "{}")
for v in (r.get("timeline") or {}).get("video", []):
    print("timeline:", v.get("name"), "shots", v.get("shots"), "confidence", v.get("confidence"), v.get("how"))
print("missing:", [m.get("label") for m in r.get("missing", [])])
for i in r.get("issues", []):
    print("issue:", i.get("message"))
for q in (r.get("qa") or {}).get("checks", []):
    print("qa:", "OK " if q["ok"] else "NO ", q["name"])
print("video:", bool(r.get("video")), "| subtitles file:", bool(r.get("srt")), "| burned subs:", bool(r.get("video_subs")))
