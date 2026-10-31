"""Page-growth test with the REAL model (Qwen3.5-2B on CPU), in CI.

Live AI runs normally; only Facebook is replaced by tests/fake_graph_server.py (same API shape), because
CI has no access to a real Page. Checks that the real model writes usable Thai posts for every kind,
that pictures/Reels render, that everything is published through the right Graph calls, measured
and scored, and that the dashboard data is complete. Captions and frames are saved for review.

usage: python tests/real_growth_test.py <base-url> <out-dir> <fake-graph-log>
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE, OUT, GLOG = sys.argv[1], Path(sys.argv[2]) / "growth", Path(sys.argv[3])
OUT.mkdir(parents=True, exist_ok=True)
REPORT: dict = {"checks": [], "posts": []}
BANNED = ["ซื้อไลก์", "ซื้อผู้ติดตาม", "แชร์เพื่อรับ", "แจกฟรี", "คลิกลิงก์", "โอนเงิน", "@"]


def check(name, ok, detail=""):
    REPORT["checks"].append({"name": name, "passed": bool(ok), "detail": str(detail)[:600]})
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {'' if ok else '— ' + str(detail)[:300]}", flush=True)


def call(method, path, body=None, token=None, timeout=120):
    req = urllib.request.Request(BASE + path, method=method, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json", **({"Authorization": f"Bearer {token}"} if token else {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def thai_ratio(s: str) -> float:
    letters = [c for c in s if c.isalpha()]
    return sum(1 for c in letters if "฀" <= c <= "๿") / max(1, len(letters))


def main():
    for _ in range(120):
        try:
            with urllib.request.urlopen(BASE + "/api/health", timeout=5) as r:
                if json.load(r)["ok"]:
                    break
        except Exception:
            pass
        time.sleep(5)
    code, u = call("POST", "/api/register", {"name": "growth-ci"})
    tok = u["token"]
    code, j = call("POST", "/api/growth/connect", {"token": "USER-ci-short-token-xxxxxxxx", "app_id": "42", "app_secret": "secret123"}, tok)
    check("connect page (token exchanged, page token sealed)", code == 200 and j.get("page", {}).get("id") == "1001", j)
    story = ("เรื่อง ไฟแค้นซ่อนรัก: น้ำฟ้า ลูกสะใภ้ที่ถูกแม่สามี (คุณหญิงมณี) กลั่นแกล้ง ภาคิน สามีที่ต้องเลือกระหว่างแม่กับเมีย "
             "แพรวา ผู้หญิงที่แม่สามีอยากให้แต่งงานกับภาคิน ความลับเรื่องพินัยกรรมของพ่อภาคิน")
    code, st = call("PATCH", "/api/growth/settings", {"story": story, "auto": False}, tok)
    check("settings saved", code == 200 and st.get("story", "").startswith("เรื่อง"), st)

    kinds = ["question", "teaser", "quote", "reel_text"]
    ids = {}
    for k in kinds:
        code, j = call("POST", "/api/growth/items", {"kind": k}, tok)
        ids[k] = j.get("id")
    started = time.time()
    items = {}
    while time.time() - started < 1500:
        code, s = call("GET", "/api/growth", token=tok)
        items = {i["id"]: i for i in s.get("items", [])}
        if all(items.get(i, {}).get("status") in ("ready", "failed") for i in ids.values()):
            break
        time.sleep(10)
    gen_s = round(time.time() - started, 1)
    for k, iid in ids.items():
        it = items.get(iid, {})
        c = it.get("content") or {}
        cap = c.get("caption", "")
        ok = it.get("status") == "ready"
        check(f"{k}: real model wrote it and it rendered", ok, it.get("error"))
        if not ok:
            continue
        REPORT["posts"].append({"kind": k, "caption": cap, "lines": c.get("lines"), "hashtags": c.get("hashtags")})
        check(f"{k}: Thai text", thai_ratio(cap) > 0.6, cap[:200])
        check(f"{k}: sensible length", 40 <= len(cap) <= 1200, len(cap))
        check(f"{k}: invites to follow or comment", bool(re.search(r"ติดตาม|คอมเมนต์|เม้น|ความเห็น|คิดว่า|เลือก|ถ้าเป็นคุณ|แชร์", cap)), cap[-150:])
        check(f"{k}: no spam / scam wording", not any(b in cap for b in BANNED), [b for b in BANNED if b in cap])
        check(f"{k}: lines fit on screen", all(len(x) <= 90 for x in c.get("lines", [])), c.get("lines"))
        check(f"{k}: uses the page's own story", any(n in cap + " ".join(c.get("lines", [])) for n in ("น้ำฟ้า", "ภาคิน", "แพรวา", "มณี", "แม่สามี", "พินัยกรรม", "สะใภ้")), cap[:200])
        if it.get("media_url"):
            try:
                with urllib.request.urlopen(BASE + it["media_url"], timeout=60) as r:
                    data = r.read()
                ext = ".mp4" if k.startswith("reel") else ".png"
                f = OUT / f"{k}{ext}"
                f.write_bytes(data)
                if ext == ".mp4":
                    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-sseof", "-1", "-i", str(f), "-frames:v", "1",
                                    "-vf", "scale=360:-1", str(OUT / f"{k}_frame.png")], check=False)
                    f.unlink()
                else:
                    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(f), "-vf", "scale=540:-1", str(OUT / f"{k}_small.png")], check=False)
                    f.unlink()
                check(f"{k}: media downloadable", len(data) > 5000, len(data))
            except Exception as e:
                check(f"{k}: media downloadable", False, e)
    REPORT["generation_seconds_all"] = gen_s

    for k, iid in ids.items():
        if items.get(iid, {}).get("status") == "ready":
            call("POST", f"/api/growth/items/{iid}/post", token=tok)
    time.sleep(20)
    code, s = call("GET", "/api/growth", token=tok)
    posted = {i["id"]: i for i in s["items"] if i["status"] == "posted"}
    check("all ready posts published", all(iid in posted for k, iid in ids.items() if items.get(iid, {}).get("status") == "ready"),
          [(i["kind"], i["status"], i["error"]) for i in s["items"]])
    calls = [json.loads(x) for x in GLOG.read_text(encoding="utf-8").splitlines()] if GLOG.exists() else []
    posts = [c for c in calls if c["method"] == "POST"]
    check("pictures via /photos, Reels via /video_reels (+ file upload)",
          any(c["path"].endswith("/1001/photos") for c in posts) and any(c["path"].endswith("/1001/video_reels") for c in posts)
          and any(c["path"].startswith("/upload/") for c in posts), [c["path"] for c in posts])
    check("Reels marked as AI-generated", all(c["params"].get("is_ai_generated") == "true" for c in posts
                                              if c["params"].get("upload_phase") == "finish"))
    code, _ = call("POST", "/api/growth/refresh", token=tok)
    code, s = call("GET", "/api/growth", token=tok)
    measured = [i for i in s["items"] if i["status"] == "posted" and i.get("score") is not None]
    check("posts measured and scored", len(measured) == len(posted) and len(measured) > 0, len(measured))
    check("dashboard: followers, goal 50,000, progress", s["followers"] > 0 and s["goal"] == 50000 and 0 < s["progress"] < 1,
          {k: s.get(k) for k in ("followers", "goal", "progress")})
    t0 = time.time()
    code, plan = call("POST", "/api/growth/analyze", token=tok, timeout=900)
    REPORT["analysis_seconds"] = round(time.time() - t0, 1)
    REPORT["analysis"] = plan
    check("AI audience + promotion analysis (real model)", code == 200 and len(plan.get("audience", [])) >= 2 and len(plan.get("channels", [])) >= 2, plan)
    text = json.dumps(plan, ensure_ascii=False)
    check("analysis suggests no paid ads / fake followers", not re.search(r"ซื้อ(ผู้ติดตาม|ไลก์|โฆษณา)|ยิงแอด|บูสต์โพสต์", text), text[:300])
    passed = sum(c["passed"] for c in REPORT["checks"])
    REPORT["summary"] = f"{passed}/{len(REPORT['checks'])} checks passed"
    (OUT / "growth_report.json").write_text(json.dumps(REPORT, ensure_ascii=False, indent=1), encoding="utf-8")
    print(REPORT["summary"])
    sys.exit(0 if passed == len(REPORT["checks"]) else 1)


if __name__ == "__main__":
    main()
