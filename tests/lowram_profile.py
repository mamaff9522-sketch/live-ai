"""Measures llama-server RAM with small context sizes (for machines with little RAM, e.g. a 2 GB VPS).

usage: python tests/lowram_profile.py <llama-server> <model.gguf> <out.json>
"""
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

BIN, MODEL, OUT = sys.argv[1], sys.argv[2], Path(sys.argv[3])
PAGE = os.sysconf("SC_PAGE_SIZE")
results = []
for ctx in (2048, 4096, 8192):
    port = 8190 + ctx // 2048
    p = subprocess.Popen(["taskset", "-c", "0,1", BIN, "-m", MODEL, "-c", str(ctx), "-t", "2", "-tb", "2", "--host", "127.0.0.1",
                          "--port", str(port), "--jinja", "-np", "1", "--no-webui"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    url = f"http://127.0.0.1:{port}"
    t0 = time.time()
    while time.time() - t0 < 300:
        try:
            urllib.request.urlopen(url + "/health", timeout=3)
            break
        except Exception:
            time.sleep(1)
    load_s = round(time.time() - t0, 1)
    body = json.dumps({"messages": [{"role": "user", "content": "อธิบายสั้น ๆ ว่าฝนเกิดจากอะไร"}], "max_tokens": 200,
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    r = json.load(urllib.request.urlopen(urllib.request.Request(url + "/v1/chat/completions", body, {"Content-Type": "application/json"}), timeout=300))
    rss = int(Path(f"/proc/{p.pid}/statm").read_text().split()[1]) * PAGE / 2**20
    results.append({"ctx": ctx, "ram_mb": round(rss), "load_s": load_s,
                    "tokens_per_s": round(r.get("timings", {}).get("predicted_per_second", 0), 2),
                    "reply_start": r["choices"][0]["message"]["content"][:80]})
    print(results[-1], flush=True)
    p.terminate()
    p.wait(30)
OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
