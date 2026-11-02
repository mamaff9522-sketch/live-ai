"""Read-only, inside the Live AI container: for the newest project, each clip's upload number, whether its
file name carries a date-time and that time's rank (the order the clips were made), how big the file is,
and what the last run did with it. No names, no words."""
import hashlib, json, re, sqlite3, sys
sys.path.insert(0, "/app")
from live_ai.studio.match import RE_STAMP
c = sqlite3.connect("file:/data/live_ai.sqlite3?mode=ro", uri=True)
c.row_factory = sqlite3.Row
p = c.execute("SELECT * FROM projects ORDER BY updated DESC LIMIT 1").fetchone()
rows = c.execute("SELECT pc.media_id, pc.upload_order, m.name, m.size, m.duration, m.path FROM project_clips pc JOIN media m ON m.id = pc.media_id "
                 "WHERE pc.project_id=? ORDER BY pc.upload_order", (p["id"],)).fetchall()
stamps = {r["media_id"]: (RE_STAMP.search(r["name"] or "") and "".join(RE_STAMP.search(r["name"]).groups())) for r in rows}
have = sorted(v for v in stamps.values() if v)
rank = {k: (have.index(v) if v else None) for k, v in stamps.items()}
nums = {r["media_id"]: int(max(re.findall(r"\d+", r["name"] or "") or ["-1"], key=len)) for r in rows}
numsorted = sorted(set(nums.values()))
nrank = {k: numsorted.index(v) for k, v in nums.items()}
shape = {r["media_id"]: re.sub(r"\d+", "#", r["name"] or "")[-12:] for r in rows}
res = json.loads(p["result"] or "{}")
pos = {v["clip"]: i for i, v in enumerate((res.get("timeline") or {}).get("video", []))}
kinds = {}
for i in res.get("issues", []):
    if i.get("clip") and i.get("type") in ("duplicate", "alternate", "unscripted", "wrong_place", "moved", "unmatched", "extra"):
        kinds[i["clip"]] = i["type"] + (f" of up#{next((r['upload_order'] for r in rows if r['media_id'] == i.get('of')), '?')}" if i.get("of") else "")
def digest(path):
    try:
        h = hashlib.sha1()
        with open(path, "rb") as f:
            h.update(f.read(2_000_000))
        return h.hexdigest()[:6]
    except OSError:
        return "-"
print("project", p["id"], "clips", len(rows), "| names with a date-time:", len(have))
for r in rows:
    print(f"up#{r['upload_order']:>2} made-rank {rank[r['media_id']]!s:>4} number-rank {nrank[r['media_id']]:>2} shape {shape[r['media_id']]!r} | {r['duration']:.2f}s {r['size'] // 1000}KB head-hash {digest(r['path'])} "
          f"| timeline pos {pos.get(r['media_id'], '-')!s:>2} | {kinds.get(r['media_id'], '')}")
