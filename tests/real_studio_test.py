"""Studio test with REAL Thai speech and the REAL local speech recognizer (whisper.cpp), in CI.

Real recorded Thai sentences (Google FLEURS, th_th, CC-BY 4.0) become the dialogue of a 12-clip
script in the Sala AI format. Each clip = a coloured picture + one recorded sentence (2 clips are
silent shots). The clips are uploaded shuffled with meaningless names, one is left out, one is
uploaded twice. Live AI must: recognise the speech, match every clip to its shot, put them in
story order, report the missing shot and the duplicate, write subtitles that are the script text,
and keep the result out of "complete" because a shot is missing. Then the missing clip is added
and the second run must finish complete, reusing the saved work.

usage: python tests/real_studio_test.py <base-url> <out-dir>
"""
from __future__ import annotations

import csv
import io
import json
import random
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

BASE = sys.argv[1]
OUT = Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
WORK = OUT / "studio_work"
WORK.mkdir(exist_ok=True)
REPORT: dict = {"checks": []}
HF = "https://huggingface.co"


def check(name, ok, detail=""):
    REPORT["checks"].append({"name": name, "passed": bool(ok), "detail": str(detail)[:800]})
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {'' if ok else '— ' + str(detail)[:300]}", flush=True)


def get(url, timeout=600):
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "live-ai-ci"}), timeout=timeout) as r:
        return r.read()


def fleurs_thai(n=10):
    """n real Thai utterances: [(wav_bytes, transcript)] from FLEURS th_th."""
    tree = json.loads(get(f"{HF}/api/datasets/google/fleurs/tree/main/data/th_th?recursive=true"))
    paths = [t["path"] for t in tree]
    split = next(s for s in ("dev", "test", "train") if any(p.endswith(f"{s}.tsv") for p in paths))
    tsv = get(f"{HF}/datasets/google/fleurs/resolve/main/data/th_th/{split}.tsv").decode("utf-8")
    rows = list(csv.reader(io.StringIO(tsv), delimiter="\t"))
    # columns: id, file name, raw transcription, normalized, words, num_samples, gender
    want, texts = {}, set()
    for r in rows:     # FLEURS has several speakers per sentence: keep each sentence once
        if len(r) >= 6 and 4 <= int(r[5]) / 16000 <= 9 and r[1] not in want and r[2].strip() not in texts:
            want[r[1]] = r[2].strip()
            texts.add(r[2].strip())
    tar = get(f"{HF}/datasets/google/fleurs/resolve/main/data/th_th/audio/{split}.tar.gz", timeout=1800)
    out = []
    with tarfile.open(fileobj=io.BytesIO(tar), mode="r:gz") as t:
        for m in t:
            name = m.name.rsplit("/", 1)[-1]
            if name in want and m.isfile():
                out.append((t.extractfile(m).read(), want[name]))
                if len(out) == n:
                    break
    REPORT["speech_source"] = f"google/fleurs th_th {split} ({len(out)} utterances)"
    return out


def make_clip(path: Path, color: str, audio: bytes | None, seconds: float = 4.0):
    if audio:
        wav = path.with_suffix(".wav")
        wav.write_bytes(audio)
        dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(wav)],
                                   capture_output=True, text=True).stdout.strip() or 5)
        seconds = dur + 0.8
        args = ["-f", "lavfi", "-i", f"color=c={color}:size=720x1280:rate=30:duration={seconds}", "-i", str(wav),
                "-af", "adelay=400|400,apad"]
    else:
        args = ["-f", "lavfi", "-i", f"color=c={color}:size=720x1280:rate=30:duration={seconds}",
                "-f", "lavfi", "-i", f"anoisesrc=a=0.0005:d={seconds}"]
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", *args, "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac",
                    "-t", f"{seconds:.2f}", str(path)], check=True)
    path.with_suffix(".wav").unlink(missing_ok=True)


def call(method, path, body=None, token=None, raw=None, ctype="application/json", timeout=900):
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    req = urllib.request.Request(BASE + path, method=method, data=data,
                                 headers={"Content-Type": ctype, **({"Authorization": f"Bearer {token}"} if token else {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def wait(token, pid, timeout=3600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        p = call("GET", f"/api/studio/projects/{pid}", token=token)[1]
        if p["status"] in ("done", "needs_attention", "failed", "paused"):
            return p, round(time.time() - t0, 1)
        time.sleep(3)
    raise TimeoutError("studio did not finish")


def main():
    utts = fleurs_thai(10)
    colors = ["red", "green", "blue", "yellow", "purple", "orange", "cyan", "white", "gray", "pink", "brown", "olive"]
    # 12 shots: 10 spoken (real sentences) + 2 silent, in the Sala AI clip format
    plan, k = [], 0
    for i in range(12):
        if i in (4, 9):
            plan.append(None)
        else:
            plan.append(utts[k])
            k += 1
    lines = ["ทดสอบสตูดิโอ", "", "ฉาก 1: ทดสอบเสียงพูดจริง", "วันที่ 1 กลางวัน / ห้องทดสอบ", ""]
    for i, u in enumerate(plan, 1):
        lines.append(f"คลิป {i:03d}/012 | 5 วินาที | 9:16")
        lines.append(f"ภาพ/การกระทำ: ฉากสี {colors[i - 1]}")
        lines.append(f"บทพูด [พิธีกร]: “{u[1]}”" if u else "บทพูด: ไม่มี (แสดงสีหน้า/บรรยากาศเท่านั้น)")
        lines.append("")
    script = "\n".join(lines)
    (WORK / "script.txt").write_text(script, encoding="utf-8")
    rnd = random.Random(11)
    files = {}
    for i, u in enumerate(plan):
        name = f"download_{rnd.randint(100000, 999999)}.mp4"
        make_clip(WORK / name, colors[i], u[0] if u else None)
        files[i] = name
    missing_idx = 6                                      # clip 007 left out on the first run
    upload = [files[i] for i in range(12) if i != missing_idx]
    rnd.shuffle(upload)
    dup_name = "same_again.mp4"
    (WORK / dup_name).write_bytes((WORK / files[2]).read_bytes())
    upload.append(dup_name)

    tok = call("POST", "/api/register", {"name": "studio-ci"})[1]["token"]
    code, p = call("POST", "/api/studio/projects", {"name": "CI เสียงพูดไทยจริง"}, tok)
    pid = p["id"]
    call("PATCH", f"/api/studio/projects/{pid}", {"script_text": script, "settings": {"aspect": "9:16", "subtitles": "burn"}}, tok)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for n in upload:
            z.write(WORK / n, n)
    code, r = call("POST", f"/api/studio/projects/{pid}/zip", token=tok, raw=buf.getvalue(), ctype="application/zip")
    check("อัปโหลด ZIP 12 ไฟล์", code == 200 and r.get("added") == 12, r)
    call("POST", f"/api/studio/projects/{pid}/start", {}, tok)
    p, secs = wait(tok, pid)
    REPORT["run1_seconds"] = secs
    res = p.get("result") or {}
    asr = {c["name"]: (c.get("asr") or {}) for c in p["clips"]}
    REPORT["transcripts"] = [{"clip": n, "expected": plan[i][1] if plan[i] else "", "heard": asr.get(n, {}).get("text", ""),
                              "seconds": asr.get(n, {}).get("seconds")} for i, n in files.items() if n in asr]
    truth = {n: i for i, n in files.items()}
    tl = [(v["name"], v["shots"]) for v in (res.get("timeline") or {}).get("video", [])]
    REPORT["timeline_run1"] = tl
    right = sum(1 for n, s in tl if n in truth and s == [truth[n]])
    check("รอบ 1: ถอดเสียงไทยจริงและจับคู่คลิปกับช็อตถูกต้อง", right == 11, f"ถูก {right}/11 {tl}")
    check("รอบ 1: เรียงตามบท ไม่ใช่ตามลำดับอัปโหลด", [s for _, s in tl] == sorted(s for _, s in tl), tl)
    check("รอบ 1: แจ้งช็อตที่ขาด = คลิป 007", [m["label"] for m in res.get("missing", [])] == ["คลิป 007"], res.get("missing"))
    check("รอบ 1: พบคลิปซ้ำ", any(i["type"] == "duplicate" for i in res.get("issues", [])), res.get("issues"))
    check("รอบ 1: ไม่รายงานว่าเสร็จสมบูรณ์เมื่อยังขาดช็อต", p["status"] == "needs_attention" and not res.get("video"), p["status"])

    # add the missing clip and run again: must finish, reusing saved analysis
    code, m = 0, None
    data = (WORK / files[missing_idx]).read_bytes()
    req = urllib.request.Request(BASE + "/api/media", data=data, method="POST",
                                 headers={"Content-Type": "video/mp4", "X-Filename": files[missing_idx], "Authorization": f"Bearer {tok}"})
    m = json.loads(urllib.request.urlopen(req, timeout=300).read())
    call("POST", f"/api/studio/projects/{pid}/clips", {"media_ids": [m["id"]]}, tok)
    call("POST", f"/api/studio/projects/{pid}/start", {}, tok)
    p, secs2 = wait(tok, pid)
    REPORT["run2_seconds"] = secs2
    res = p.get("result") or {}
    tl = [(v["name"], v["shots"]) for v in (res.get("timeline") or {}).get("video", [])]
    REPORT["timeline_run2"] = tl
    check("รอบ 2: เพิ่มคลิปที่ขาดแล้วเสร็จสมบูรณ์ ผ่าน QA", p["status"] == "done" and res.get("complete"), p["message"])
    check("รอบ 2: ทำต่อจากผลที่บันทึกไว้ (เร็วกว่ารอบแรก)", secs2 < secs, f"{secs2}s vs {secs}s")
    check("รอบ 2: ครบ 12 ช็อตตามลำดับ", [s for _, s in tl] == [[i] for i in range(12)], tl)
    if res.get("srt", {}).get("url"):
        srt = get(BASE + res["srt"]["url"]).decode("utf-8")
        subs = [b.split("\n", 2)[2].strip() for b in srt.strip().split("\n\n")]
        expect = [u[1] for u in plan if u]
        check("ซับตรงตามบททุกตัวอักษร ไม่มีแต่งเพิ่ม", subs == expect, subs)
        (OUT / "studio.srt").write_text(srt, encoding="utf-8")
    check("มีไฟล์ฝังซับแยก และไฟล์ไม่มีซับยังอยู่", bool(res.get("video_subs")) and bool(res.get("video")), res.keys())
    if res.get("video_subs", {}).get("url"):
        (OUT / "studio_with_subs.mp4").write_bytes(get(BASE + res["video_subs"]["url"]))
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(OUT / "studio_with_subs.mp4"), "-vf",
                        "select='eq(n\\,60)+eq(n\\,400)',scale=360:-2,tile=2x1", "-frames:v", "1", str(OUT / "studio_frames.png")])
    REPORT["qa"] = res.get("qa")
    REPORT["agents"] = {k: {"status": a["status"], "message": a["message"]} for k, a in p["agents"].items()}
    passed = sum(c["passed"] for c in REPORT["checks"])
    REPORT["summary"] = {"passed": passed, "total": len(REPORT["checks"])}
    (OUT / "real_studio_report.json").write_text(json.dumps(REPORT, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(REPORT["summary"]))
    sys.exit(0 if passed == len(REPORT["checks"]) else 1)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        REPORT["fatal"] = traceback.format_exc()[-1500:]
        (OUT / "real_studio_report.json").write_text(json.dumps(REPORT, ensure_ascii=False, indent=2), encoding="utf-8")
        raise
