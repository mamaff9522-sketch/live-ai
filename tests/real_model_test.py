"""End-to-end test against a running Live AI with the REAL Qwen3.5-2B model.

Usage:  python tests/real_model_test.py http://127.0.0.1:7860 reports/
Writes reports/real_model_report.json and reports/real_model_report.md (Thai).
Every check is recorded as passed or failed with the model's actual answer;
nothing is skipped silently. Exit code 1 if any check fails.
"""
from __future__ import annotations

import json
import os
import re
import signal
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:7860"
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "reports")
OUT.mkdir(parents=True, exist_ok=True)
THAI = re.compile(r"[฀-๿]")

R: dict = {"checks": [], "replies": [], "samples": []}


# --------------------------------------------------------------------------- helpers
def call(method, path, body=None, token=None, timeout=900):
    req = urllib.request.Request(BASE + path, method=method, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json", **({"Authorization": f"Bearer {token}"} if token else {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def health():
    try:
        return call("GET", "/api/health", timeout=10)[1]
    except Exception as e:
        return {"ok": False, "error": str(e)}


class Sampler:
    """Samples the model process's RAM and CPU every 0.5 s while a reply is generated."""

    def __init__(self):
        self.ram, self.cpu = [], []
        self._stop = threading.Event()

    def __enter__(self):
        self.pid = (health().get("model") or {}).get("pid")
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()
        return self

    def _run(self):
        if not self.pid:
            return
        tick = os.sysconf("SC_CLK_TCK")
        page = os.sysconf("SC_PAGE_SIZE")
        prev = None
        while not self._stop.wait(0.5):
            try:
                st = Path(f"/proc/{self.pid}/stat").read_text().rsplit(")", 1)[1].split()
                cpu_s = (int(st[11]) + int(st[12])) / tick
                rss = int(Path(f"/proc/{self.pid}/statm").read_text().split()[1]) * page / 2**20
            except OSError:
                continue
            now = time.time()
            self.ram.append(rss)
            if prev:
                self.cpu.append(100 * (cpu_s - prev[1]) / (now - prev[0]))
            prev = (now, cpu_s)

    def __exit__(self, *a):
        self._stop.set()
        self.t.join()
        R["samples"].append({"ram_mb_max": max(self.ram, default=None), "cpu_pct_avg": statistics.mean(self.cpu) if self.cpu else None})


def ask(token, text, conv=None, think=False, label=""):
    t0 = time.time()
    with Sampler():
        code, j = call("POST", "/api/chat", {"message": text, "conversation_id": conv, "think": think, "stream": False}, token)
    j.setdefault("reply", "")
    rec = {"label": label, "prompt": text, "think": think, "http": code, "reply": j.get("reply", ""), "error": j.get("error"),
           "stats": j.get("stats"), "wall_s": round(time.time() - t0, 2)}
    R["replies"].append(rec)
    print(f"\n### {label} ({rec['wall_s']}s) {text}\n{rec['reply'] or rec['error']}", flush=True)
    return code, j


def check(name, ok, detail=""):
    R["checks"].append({"name": name, "passed": bool(ok), "detail": detail[:600]})
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {('— ' + detail[:200]) if not ok else ''}", flush=True)


def thai_ratio(s):
    letters = [c for c in s if c.isalpha()]
    return sum(1 for c in letters if THAI.match(c)) / max(1, len(letters))


def code_blocks(s):
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", s, re.S)
    return blocks or ([s] if "def " in s else [])


def run_python(code, test, timeout=15):
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "t.py"
        p.write_text(code + "\n\n" + test, encoding="utf-8")
        r = subprocess.run([sys.executable, "-I", str(p)], capture_output=True, text=True, timeout=timeout, cwd=d)
        return r.returncode == 0, (r.stdout + r.stderr)[-400:]


def code_task(token, label, prompt, test):
    code, j = ask(token, prompt, label=label)
    reply = j.get("reply", "")
    blocks = code_blocks(reply)
    ok, out = False, "no code block in reply"
    for b in blocks:
        try:
            ok, out = run_python(b, test)
        except subprocess.TimeoutExpired:
            ok, out = False, "timed out"
        if ok:
            break
    check(label, code == 200 and ok, out)


# --------------------------------------------------------------------------- run
def main():
    t0 = time.time()
    print("waiting for the model…", flush=True)
    while True:
        h = health()
        if h.get("ok"):
            break
        if (h.get("model") or {}).get("state") == "failed":
            R["fatal"] = h
            break
        if time.time() - t0 > 1800:
            R["fatal"] = {"error": "model not ready after 30 min", **h}
            break
        time.sleep(3)
    R["ready_after_s"] = round(time.time() - t0, 1)
    R["health_idle"] = health()
    if R.get("fatal"):
        check("model starts", False, json.dumps(R["fatal"], ensure_ascii=False))
        return finish()
    check("model starts", True)

    a = call("POST", "/api/register", {"name": "tester-a"})[1]["token"]
    b = call("POST", "/api/register", {"name": "tester-b"})[1]["token"]

    # 1. Thai conversation
    code, j = ask(a, "สวัสดีครับ วันนี้รู้สึกเหนื่อยนิดหน่อย มีวิธีพักผ่อนสั้น ๆ ระหว่างวันแนะนำไหม", label="thai_chat")
    r = j.get("reply", "")
    check("ตอบภาษาไทยเป็นธรรมชาติ", code == 200 and len(r) > 40 and thai_ratio(r) > 0.7, f"thai={thai_ratio(r):.2f} len={len(r)}")

    # 2. knowledge
    code, j = ask(a, "เมืองหลวงของประเทศญี่ปุ่นคือเมืองอะไร ตอบสั้น ๆ", label="knowledge_1")
    check("ความรู้: เมืองหลวงญี่ปุ่น", "โตเกียว" in j.get("reply", "") or "Tokyo" in j.get("reply", ""), j.get("reply", ""))
    code, j = ask(a, "อธิบายสั้น ๆ ว่าทำไมท้องฟ้าถึงเป็นสีฟ้า", label="knowledge_2")
    r = j.get("reply", "")
    check("อธิบายความรู้ (ท้องฟ้าสีฟ้า)", thai_ratio(r) > 0.6 and any(k in r for k in ("กระเจิง", "ความยาวคลื่น", "แสง")), r)

    # 3. reasoning / problem solving
    code, j = ask(a, "มีแอปเปิล 12 ผล แบ่งให้เพื่อน 3 คนเท่า ๆ กัน แล้วแต่ละคนกินไปคนละ 1 ผล เหลือคนละกี่ผล", label="reasoning_1")
    check("คิดวิเคราะห์: โจทย์เลข (ตอบ 3)", re.search(r"(?<!\d)3(?!\d)|สาม", j.get("reply", "")) is not None, j.get("reply", ""))
    code, j = ask(a, "ร้านขายปากกาด้ามละ 15 บาท ซื้อ 4 ด้ามได้ลด 10% ต้องจ่ายกี่บาท", think=True, label="reasoning_think")
    check("คิดละเอียด (think): ส่วนลด (ตอบ 54)", "54" in j.get("reply", ""), j.get("reply", ""))

    # 4. code writing / review (the code is executed)
    code_task(a, "เขียนโค้ด: is_prime",
              "เขียนฟังก์ชัน Python ชื่อ is_prime(n) คืนค่า True ถ้า n เป็นจำนวนเฉพาะ ไม่งั้นคืน False ให้แค่โค้ด",
              "assert [is_prime(x) for x in (0,1,2,3,4,17,18,97)] == [False,False,True,True,False,True,False,True]\nprint('ok')")
    code_task(a, "เขียนโค้ด: นับคำ",
              "เขียนฟังก์ชัน Python ชื่อ word_count(text) ที่คืน dict นับจำนวนครั้งของแต่ละคำ (แยกคำด้วยช่องว่าง ไม่สนตัวพิมพ์เล็กใหญ่)",
              "assert word_count('a b A c b a') == {'a':3,'b':2,'c':1}\nprint('ok')")
    code_task(a, "ตรวจและแก้โค้ด",
              "โค้ดนี้มีบั๊ก ช่วยตรวจและส่งโค้ดที่แก้แล้ว:\n```python\ndef average(nums):\n    total = 0\n    for n in nums:\n        total += n\n    return total / len(nums) - 1\n```\nและต้องคืน 0 เมื่อลิสต์ว่าง",
              "assert average([2,4,6]) == 4\nassert average([]) == 0\nprint('ok')")

    # 5. planning
    code, j = ask(a, "ช่วยวางแผนเตรียมสอบคณิตศาสตร์ในอีก 1 สัปดาห์ แบ่งเป็นรายวัน", label="planning")
    r = j.get("reply", "")
    days = len(re.findall(r"วันที่\s*\d|วัน(?:จันทร์|อังคาร|พุธ|พฤหัส|ศุกร์|เสาร์|อาทิตย์)|^\s*(?:\d+[.)]|[-*•])", r, re.M))
    check("วางแผนงานเป็นขั้นตอน", days >= 5 and thai_ratio(r) > 0.6, f"items={days}")

    # 6. context within a conversation
    code, j = ask(a, "ผมชื่อธนากร ทำงานเป็นครูสอนฟิสิกส์", label="context_1")
    conv = j.get("conversation_id")
    code, j = ask(a, "ผมชื่ออะไร และทำอาชีพอะไร", conv, label="context_2")
    r = j.get("reply", "")
    check("จำบริบทในบทสนทนา", "ธนากร" in r and "ครู" in r, r)

    # 7. long-term memory across conversations
    ask(a, "จำไว้ว่า แมวของฉันชื่อส้มโอ", label="memory_save")
    mems = call("GET", "/api/memories", token=a)[1]
    check("บันทึกความจำ", any("ส้มโอ" in m["content"] for m in mems), json.dumps(mems, ensure_ascii=False))
    code, j = ask(a, "แมวของฉันชื่ออะไร", label="memory_recall")
    check("ระลึกความจำข้ามบทสนทนา", "ส้มโอ" in j.get("reply", ""), j.get("reply", ""))

    # 8. isolation with the real model
    code, j = ask(b, "แมวของฉันชื่ออะไร", label="isolation")
    check("ข้อมูลไม่รั่วไปผู้ใช้อื่น", code == 200 and "ส้มโอ" not in j.get("reply", ""), j.get("reply", ""))
    check("ผู้ใช้อื่นเห็นบทสนทนาไม่ได้", call("GET", f"/api/conversations/{conv}/messages", token=b)[0] == 404)

    # 9. continuous use: 10 turns in one conversation
    code, j = ask(a, "เรามาคุยเรื่องการปลูกผักสวนครัวกัน เริ่มจากผักที่ปลูกง่ายที่สุด", label="long_1")
    c2, fails = j.get("conversation_id"), 0
    for i, q in enumerate(["ต้องรดน้ำบ่อยแค่ไหน", "ใช้ดินแบบไหนดี", "ปลูกในคอนโดได้ไหม", "ต้องใช้แดดกี่ชั่วโมง",
                           "ปุ๋ยอะไรเหมาะ", "มีแมลงกวนทำยังไง", "เก็บเกี่ยวเมื่อไหร่", "สรุปที่คุยมาทั้งหมดเป็นข้อ ๆ", "ขอบคุณครับ"], 2):
        code, j = ask(a, q, c2, label=f"long_{i}")
        fails += code != 200 or not j.get("reply")
    check("ทำงานต่อเนื่อง 10 รอบไม่ล้ม", fails == 0, f"failures={fails}")

    # 10. crash recovery
    h = health()
    pid = (h.get("model") or {}).get("pid")
    restarts0 = (h.get("model") or {}).get("restarts", 0)
    if pid:
        os.kill(pid, signal.SIGKILL)
        t1 = time.time()
        code, j = ask(a, "ทดสอบหลังกู้คืน: 2 + 2 เท่ากับเท่าไร", label="after_crash")
        R["recovery_s"] = round(time.time() - t1, 1)
        h2 = health()
        check("กู้คืนเองเมื่อโมเดลหยุดทำงาน", code == 200 and "4" in j.get("reply", "") and (h2.get("model") or {}).get("restarts", 0) > restarts0,
              f"http={code} recovery_s={R['recovery_s']} reply={j.get('reply', '')[:80]}")
    else:
        check("กู้คืนเองเมื่อโมเดลหยุดทำงาน", False, "no model pid in /api/health")
    finish()


def finish():
    R["health_end"] = health()
    stats = [x["stats"] for x in R["replies"] if x.get("stats")]
    tps = [s["tokens_per_second"] for s in stats if s.get("tokens_per_second")]
    pps = [s["prompt_tokens_per_second"] for s in stats if s.get("prompt_tokens_per_second")]
    secs = [s["seconds"] for s in stats]
    ram = [s["ram_mb_max"] for s in R["samples"] if s["ram_mb_max"]]
    cpu = [s["cpu_pct_avg"] for s in R["samples"] if s["cpu_pct_avg"]]
    R["summary"] = {
        "passed": sum(c["passed"] for c in R["checks"]), "total": len(R["checks"]),
        "reply_tokens_per_s_median": round(statistics.median(tps), 2) if tps else None,
        "prompt_tokens_per_s_median": round(statistics.median(pps), 2) if pps else None,
        "reply_seconds_median": round(statistics.median(secs), 1) if secs else None,
        "reply_seconds_max": round(max(secs), 1) if secs else None,
        "model_ram_mb_max": round(max(ram)) if ram else None,
        "model_cpu_pct_avg": round(statistics.mean(cpu)) if cpu else None,
        "machine": {"cpus_used": os.environ.get("LIVEAI_THREADS"), "mem_total_mb": (R.get("health_idle") or {}).get("ram_mb", {}).get("total")},
    }
    (OUT / "real_model_report.json").write_text(json.dumps(R, ensure_ascii=False, indent=2), encoding="utf-8")
    s = R["summary"]
    md = [f"# ผลทดสอบ Live AI กับ Qwen3.5-2B (Q4_K_M) บน CPU", "",
          f"ผ่าน **{s['passed']}/{s['total']}** ข้อ", "",
          "| รายการ | ค่า |", "|---|---|",
          f"| เวลาโหลดโมเดล (รวมดาวน์โหลด) | {R.get('ready_after_s')} วินาที |",
          f"| ความเร็วสร้างคำตอบ (มัธยฐาน) | {s['reply_tokens_per_s_median']} โทเคน/วินาที |",
          f"| ความเร็วอ่านคำถาม (มัธยฐาน) | {s['prompt_tokens_per_s_median']} โทเคน/วินาที |",
          f"| เวลาตอบต่อข้อความ (มัธยฐาน / สูงสุด) | {s['reply_seconds_median']} / {s['reply_seconds_max']} วินาที |",
          f"| RAM ของโมเดลสูงสุด | {s['model_ram_mb_max']} MB |",
          f"| CPU ขณะตอบ (เฉลี่ย, 100% = 1 คอร์) | {s['model_cpu_pct_avg']}% |",
          f"| เวลากู้คืนหลังโมเดลล่ม | {R.get('recovery_s')} วินาที |", "", "## รายการทดสอบ", ""]
    md += [f"- {'✅' if c['passed'] else '❌'} {c['name']}" + ("" if c["passed"] else f" — `{c['detail'][:150]}`") for c in R["checks"]]
    md += ["", "## คำตอบจริงของโมเดล", ""]
    for x in R["replies"]:
        st = x.get("stats") or {}
        md += [f"### {x['label']} · {st.get('seconds')} วินาที · {st.get('tokens_per_second')} tok/s", f"**ถาม:** {x['prompt']}", "",
               "**ตอบ:**", "", (x["reply"] or f"(error: {x['error']})").replace("\n", "  \n"), ""]
    (OUT / "real_model_report.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps(s, ensure_ascii=False, indent=2))
    sys.exit(0 if s["passed"] == s["total"] else 1)


if __name__ == "__main__":
    main()
