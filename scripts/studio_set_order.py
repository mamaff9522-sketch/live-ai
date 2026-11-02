"""Runs inside the Live AI container: the owner's clip order for the newest project called $PROJECT,
given as upload numbers ($ORDER, comma separated; empty = back to the script order). Then the
project is queued again. Prints counts only."""
import json, os, sqlite3
c = sqlite3.connect("/data/live_ai.sqlite3")
c.row_factory = sqlite3.Row
p = c.execute("SELECT id, settings FROM projects WHERE name=? ORDER BY updated DESC LIMIT 1", (os.environ["PROJECT"],)).fetchone()
if not p:
    raise SystemExit("no project with that name")
by_up = {r["upload_order"]: r["media_id"] for r in c.execute("SELECT upload_order, media_id FROM project_clips WHERE project_id=?", (p["id"],))}
ups = [int(x) for x in os.environ.get("ORDER", "").replace(" ", "").split(",") if x]
missing = [u for u in ups if u not in by_up]
if missing:
    raise SystemExit(f"upload numbers not in the project: {missing}")
st = json.loads(p["settings"])
st["order"] = [by_up[u] for u in ups] or None
with c:
    c.execute("UPDATE projects SET settings=? WHERE id=?", (json.dumps(st), p["id"]))
    c.execute("UPDATE project_clips SET inspect=NULL WHERE project_id=?", (p["id"],))
    c.execute("UPDATE projects SET status='queued', message='รอคิวประมวลผล (ลำดับที่จัดให้)', pause=0 WHERE id=?", (p["id"],))
print("order set:", len(ups), "clips | QUEUED", p["id"])
