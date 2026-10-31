"""Runs inside the Live AI container: queue the newest project called $PROJECT again (fresh clip checks,
saved speech transcripts kept). Prints counts only."""
import json, os, sqlite3, sys
sys.path.insert(0, "/app")
from live_ai.studio.script import parse
c = sqlite3.connect("/data/live_ai.sqlite3")
c.row_factory = sqlite3.Row
r = c.execute("SELECT id, status, script_text FROM projects WHERE name=? ORDER BY updated DESC LIMIT 1", (os.environ["PROJECT"],)).fetchone()
if not r:
    sys.exit("no project with that name")
clips = c.execute("SELECT COUNT(*) FROM project_clips WHERE project_id=?", (r["id"],)).fetchone()[0]
st = parse(r["script_text"] or "")["stats"] if (r["script_text"] or "").strip() else None
print("project", r["id"], "was", r["status"], "| clips", clips, "| script chars", len(r["script_text"] or ""), "| parsed", st)
if not st or not st["shots"]:
    sys.exit("the script is empty or unreadable: paste it in the บทละคร tab first")
if not clips:
    sys.exit("no clips in the project")
with c:
    c.execute("UPDATE project_clips SET inspect=NULL WHERE project_id=?", (r["id"],))
    c.execute("UPDATE projects SET status='queued', message='รอคิวประมวลผล (สั่งให้ทำใหม่)', pause=0 WHERE id=?", (r["id"],))
print("QUEUED", r["id"])
