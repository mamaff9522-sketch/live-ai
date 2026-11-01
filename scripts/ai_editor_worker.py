#!/usr/bin/env python3
"""AI editor worker: runs next to a llama-server with Qwen3.5-9B (vision) — on the owner's Colab GPU — and
answers Live AI's editing jobs. Standard library only.

usage: python ai_editor_worker.py --code "https://your-live-ai#lew_..." [--llm http://127.0.0.1:8080] [--once]
       (or --server https://your-live-ai --key lew_...)

For every clip of a job it looks at the still frames (with their times) together with what is said and when,
and judges each frame: broken or not, how strongly it carries the feeling (1-5). Live AI's cutting rules turn
those judgments into cuts (broken stretches out, strong reactions kept), and words are never cut.
"""
from __future__ import annotations

import argparse
import base64
import os
import json
import sys
import time
import traceback
import urllib.error
import urllib.request

MOOD_EN = {"เศร้า": "sad", "โรแมนติก": "tender/romantic", "ปกติ": "neutral", "ตลก": "funny", "โกรธ": "angry", "ระทึก": "tense"}

SYSTEM = """You are an experienced film editor checking AI-generated video clips for a short Thai drama.
You get ONE clip as still frames (with their times), what is said and when, and the scene mood.
Judge EVERY frame, in order:
- "broken": true if the picture is faulty: a see-through, ghost-like or melting character, a sudden zoom or jump
  compared with the neighbouring frames, extra or missing limbs, a duplicated or different-looking character,
  a frozen, smeared or black frame. Otherwise false. Judge only what you can see.
- "emotion": 1-5, how strongly the frame carries the scene's feeling (5 = a powerful look, tears, a meaningful
  gesture; 1 = empty or nothing happening).
- "why": one short sentence in Thai: what is broken (if anything) and which moment is strongest.
Answer only with the JSON object."""


def schema(n: int) -> dict:
    return {
        "type": "object",
        "properties": {
            "frames": {"type": "array", "minItems": n, "maxItems": n, "items": {
                "type": "object",
                "properties": {"t": {"type": "number"}, "broken": {"type": "boolean"},
                               "emotion": {"type": "integer", "minimum": 1, "maximum": 5}},
                "required": ["t", "broken", "emotion"]}},
            "why": {"type": "string", "maxLength": 300},
        },
        "required": ["frames", "why"],
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
        "frame_times_s": [f["t"] for f in clip["frames"]],
    }
    content.append({"type": "text", "text": "Clip information:\n" + json.dumps(info, ensure_ascii=False, indent=1) +
                    "\n\nJudge every frame (same order and times) and answer with frames and why."})
    n = len(clip["frames"])
    body = {"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}],
            "stream": False, "temperature": 0.1, "max_tokens": 120 + 40 * n, "chat_template_kwargs": {"enable_thinking": False},
            "response_format": {"type": "json_schema", "json_schema": {"name": "frames", "schema": schema(n)}}}
    t0 = time.time()
    r = http("POST", llm + "/v1/chat/completions", body=body, timeout=int(os.environ.get("AIED_LLM_TIMEOUT", "900")))
    text = r["choices"][0]["message"].get("content") or "{}"
    out = json.loads(text.strip().removeprefix("```json").removeprefix("```").removesuffix("```"))
    frames = []
    for f, got in zip(clip["frames"], out.get("frames", [])):
        frames.append({"t": f["t"], "broken": bool(got.get("broken")), "emotion": max(1, min(5, int(got.get("emotion", 1))))})
    return {"frames": frames, "why": str(out.get("why", ""))[:300], "seconds": round(time.time() - t0, 1)}


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
                marks = " ".join(f"{f['t']:.1f}{'✗' if f['broken'] else ''}:{f['emotion']}" for f in p["frames"])
                print(f"  {clip['position']}/{clip['of']} {clip['name']} ({p['seconds']} s): {marks} — {p['why']}", flush=True)
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
