"""Runs inside the Live AI container: the studio's own result for project $PID (states, counts, timeline
shape, QA). The repository is public, so nothing the owner wrote or said is printed: no script text, no
clip names, no transcripts (quoted text in messages is replaced by “…”), no media."""
import json, os, re, sqlite3
c = sqlite3.connect("/data/live_ai.sqlite3")
c.row_factory = sqlite3.Row
pid = int(os.environ["PID"])
p = c.execute("SELECT status, message, result FROM projects WHERE id=?", (pid,)).fetchone()
up = {r["media_id"]: r["upload_order"] for r in c.execute("SELECT media_id, upload_order FROM project_clips WHERE project_id=?", (pid,))}


def hide(s):
    return re.sub(r"“[^”]*”", "“…”", str(s or ""))


print("status:", p["status"])
print("message:", hide(p["message"]))
r = json.loads(p["result"] or "{}")
for v in (r.get("timeline") or {}).get("video", []):
    print("timeline: uploaded #", up.get(v.get("clip"), "?"), "| shots", v.get("shots"), "| kept", v.get("in"), "-", v.get("out"),
          "s |", v.get("how"), v.get("confidence"))
print("missing shots:", len(r.get("missing", [])))
kinds = {}
for i in r.get("issues", []):
    kinds[i.get("type")] = kinds.get(i.get("type"), 0) + 1
print("issues by type:", kinds)
for q in (r.get("qa") or {}).get("checks", []):
    print("qa:", "OK " if q["ok"] else "NO ", hide(q["name"]))
print("video:", bool(r.get("video")), "| review copy:", bool(r.get("review")), "| subtitles file:", bool(r.get("srt")),
      "| burned subs:", bool(r.get("video_subs")))
