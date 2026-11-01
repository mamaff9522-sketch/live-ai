#!/usr/bin/env python3
"""AI editor worker: runs next to a llama-server with Qwen3.5-9B (vision) — on the owner's Colab GPU — and
answers Live AI's editing jobs. Standard library only.

usage: python ai_editor_worker.py --code "https://your-live-ai#lew_..." [--llm http://127.0.0.1:8080] [--once]
       (or --server https://your-live-ai --key lew_...)

For every clip of a job it looks at the still frames (with their times) together with what is said and
when, and decides which part to keep. Live AI checks every answer again (words are never cut) before
using it, so a wrong answer can make a clip a bit longer or shorter but cannot break the story.
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
import time
import traceback
import urllib.error
import urllib.request

MOOD_EN = {"เศร้า": "sad", "โรแมนติก": "tender/romantic", "ปกติ": "neutral", "ตลก": "funny", "โกรธ": "angry", "ระทึก": "tense"}

SYSTEM = """You are an experienced film editor cutting a short Thai drama made from AI-generated video clips.
For ONE clip you get still frames with their times, what is said and when, and a first plan made by simple rules.
Decide which part of the clip to keep, like a human editor would.

Rules:
- Never cut spoken words: keep "in" at or before the speech start and "out" at or after the speech end.
- Cut away broken frames: see-through or melting characters, sudden zooms or jumps, duplicated or wrong characters,
  frozen or black frames - but only where no words are spoken.
- Cut empty time where nothing happens. Keep reactions that carry emotion (tears, a look, a smile through tears).
- Sad or tender scenes: let the moment breathe, hold about 1-2 s after the last word. Angry or tense: cut quickly.
- The first clip may open with a short calm moment; the last clip may hold longer as the ending.
- fade_in / fade_out (seconds, 0 to 0.8): use a soft fade only for sad/tender moments, scene changes, the very start
  and the very end; otherwise 0.
- "why": one short sentence in Thai explaining your choice (what you cut and why).
Answer only with the JSON object."""

SCHEMA = {
    "type": "object",
    "properties": {
        "in": {"type": "number"}, "out": {"type": "number"},
        "fade_in": {"type": "number"}, "fade_out": {"type": "number"},
        "bad_frames": {"type": "array", "items": {"type": "number"}, "maxItems": 12},
        "why": {"type": "string", "maxLength": 300},
    },
    "required": ["in", "out", "fade_in", "fade_out", "why"],
}


def http(method: str, url: str, key: str | None = None, body: dict | None = None, timeout: int = 60, raw: bool = False):
    h = {"User-Agent": "live-ai-editor-worker"}
    data = None
    if body is not None:
        data = json.dumps(body, ensure_ascii=False).encode()
        h["Content-Type"] = "application/json"
    if key:
        h["X-Worker-Key"] = key
    req = urllib.request.Request(url, data=data, headers=h, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        b = r.read()
    return b if raw else json.loads(b or b"{}")


def model_name(llm: str) -> str:
    try:
        d = http("GET", llm + "/v1/models", timeout=10)
        mid = (d.get("data") or [{}])[0].get("id", "")
        mid = mid.rsplit("/", 1)[-1].removesuffix(".gguf")          # llama-server reports the file path
        return mid[:60] or "Qwen3.5-9B"
    except Exception:
        return "Qwen3.5-9B"


def decide(llm: str, server: str, key: str, job_id: int, clip: dict) -> dict:
    content = []
    for f in clip["frames"]:
        jpg = http("GET", f"{server}/api/editor/worker/jobs/{job_id}/frames/{f['name']}", key, raw=True, timeout=60)
        content.append({"type": "text", "text": f"Frame at {f['t']:.2f} s:"})
        content.append({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(jpg).decode()}})
    speech = clip.get("speech") or []
    info = {
        "clip": f"{clip['position']} of {clip['of']}", "duration_s": clip["duration"],
        "scene_mood": MOOD_EN.get(clip.get("mood"), clip.get("mood")),
        "what_happens": clip.get("action", ""),
        "spoken_lines": [f"{l['speaker'] + ': ' if l.get('speaker') else ''}{l['text']}" for l in clip.get("lines", [])],
        "speech_start_s": speech[0][0] if speech else None, "speech_end_s": speech[-1][1] if speech else None,
        "picture_jumps_found_s": clip.get("jumps") or [],
        "rule_plan": clip.get("rule"),
    }
    content.append({"type": "text", "text": "Clip information:\n" + json.dumps(info, ensure_ascii=False, indent=1) +
                    "\n\nDecide in, out, fade_in, fade_out (seconds) and why."})
    body = {"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}],
            "stream": False, "temperature": 0.2, "max_tokens": 400, "chat_template_kwargs": {"enable_thinking": False},
            "response_format": {"type": "json_schema", "json_schema": {"name": "cut", "schema": SCHEMA}}}
    t0 = time.time()
    r = http("POST", llm + "/v1/chat/completions", body=body, timeout=900)
    text = r["choices"][0]["message"].get("content") or "{}"
    out = json.loads(text.strip().removeprefix("```json").removeprefix("```").removesuffix("```"))
    d = float(clip["duration"])
    plan = {"in": max(0.0, min(float(out["in"]), d)), "out": max(0.0, min(float(out["out"]), d)),
            "fade_in": max(0.0, min(float(out.get("fade_in", 0)), 0.8)), "fade_out": max(0.0, min(float(out.get("fade_out", 0)), 0.8)),
            "why": str(out.get("why", ""))[:300], "seconds": round(time.time() - t0, 1)}
    if speech:   # the same safety rule Live AI applies again
        plan["in"], plan["out"] = min(plan["in"], speech[0][0]), max(plan["out"], speech[-1][1])
    return plan


def run(server: str, key: str, llm: str, once: bool = False) -> None:
    server, llm = server.rstrip("/"), llm.rstrip("/")
    name = model_name(llm)
    http("POST", server + "/api/editor/worker/hello", key, {"model": name, "started": time.time()})
    print(f"connected to Live AI as the AI editor ({name}); waiting for projects …", flush=True)
    while True:
        try:
            job = http("GET", server + "/api/editor/worker/next", key, timeout=60)
        except urllib.error.HTTPError as e:
            if e.code == 401:
                sys.exit("the key was rejected: create a new one in the studio (ตั้งค่า & ส่งออก) and paste it again")
            print("server error", e.code, flush=True)
            time.sleep(10)
            continue
        except Exception as e:
            print("cannot reach Live AI:", e, flush=True)
            time.sleep(10)
            continue
        if not job.get("id"):
            continue
        jid = job["id"]
        print(f"job {jid}: project “{job.get('project')}”, {len(job['clips'])} clips", flush=True)
        plan = {}
        for clip in job["clips"]:
            try:
                plan[clip["id"]] = p = decide(llm, server, key, jid, clip)
                print(f"  {clip['position']}/{clip['of']} {clip['name']}: keep {p['in']:.2f}–{p['out']:.2f} s "
                      f"(fade {p['fade_in']}/{p['fade_out']}) {p['seconds']} s — {p['why']}", flush=True)
            except Exception as e:
                print(f"  {clip['name']}: no decision ({e}); Live AI keeps its own plan for this clip", flush=True)
                traceback.print_exc(limit=1)
        try:
            http("POST", f"{server}/api/editor/worker/jobs/{jid}/result", key, {"plan": plan, "model": name}, timeout=120)
            print(f"job {jid} done", flush=True)
        except Exception as e:
            print(f"job {jid}: could not send the answer: {e}", flush=True)
        if once:
            return


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", help="the connection code from the studio: https://your-live-ai#lew_...")
    ap.add_argument("--server")
    ap.add_argument("--key")
    ap.add_argument("--llm", default="http://127.0.0.1:8080")
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()
    if a.code:
        a.server, _, a.key = a.code.strip().partition("#")
    if not (a.server and a.key):
        sys.exit("give --code (from the studio) or --server and --key")
    run(a.server, a.key, a.llm, a.once)
