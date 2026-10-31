"""The model side of Live AI.

ModelFiles   - finds or downloads the GGUF file (resumable, optional SHA-256 check).
LlamaServer  - starts llama.cpp's llama-server, watches it, and restarts it when it
               dies or stops answering (with back-off), logging every event.
LLMClient    - talks to llama-server's OpenAI-compatible API, streaming tokens.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterator

from . import logs
from .config import Settings

log = logs.get("llm")


class ModelError(RuntimeError):
    pass


# --------------------------------------------------------------------------- model file
def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_model(s: Settings) -> Path:
    if s.model_path:
        p = Path(s.model_path)
        if not p.is_file():
            raise ModelError(f"model file not found: {p}")
        return p
    # sources: an explicit LIVEAI_MODEL_URL, else the first trusted Hugging Face repo that has the file
    sources: list[tuple[str, str]] = []
    errors = []
    if os.environ.get("LIVEAI_MODEL_URL"):
        sources = [(u, s.model_sha256) for u in s.model_urls]
    else:
        from . import modelsource
        try:
            found = modelsource.find()
            s.model_repo = found["repo"]
            sources = [(found["url"], s.model_sha256 or found["sha256"])]
            logs.event(log, "model source", repo=found["repo"], file=found["file"], sha256=found["sha256"])
        except Exception as e:
            errors.append(f"lookup: {e}")
            sources = [(u, s.model_sha256) for u in s.model_urls]
    for url, sha in sources:
        name = url.rsplit("/", 1)[-1].split("?")[0]
        dest = s.data_dir / "models" / name
        if dest.is_file() and dest.stat().st_size > 100_000_000:
            logs.event(log, "model ready", path=str(dest), size_mb=dest.stat().st_size // 2**20)
            return dest
        try:
            _download(url, dest)
        except Exception as e:  # try the next source
            errors.append(f"{url}: {e}")
            logs.event(log, "model source failed", logging.WARNING, url=url, error=str(e))
            continue
        if sha:
            got = sha256_of(dest)
            if got != sha.lower():
                dest.unlink()
                raise ModelError(f"SHA-256 mismatch for {name}: expected {sha}, got {got}")
        logs.event(log, "model downloaded", url=url, path=str(dest), size_mb=dest.stat().st_size // 2**20, sha256_checked=bool(sha))
        return dest
    raise ModelError("could not get the model: " + " | ".join(errors))


def ensure_mmproj(s: Settings) -> Path | None:
    """Vision projector for the model, unless turned off (LIVEAI_VISION=0) or not published."""
    if os.environ.get("LIVEAI_VISION", "1") == "0":
        return None
    if os.environ.get("LIVEAI_MMPROJ_PATH"):
        p = Path(os.environ["LIVEAI_MMPROJ_PATH"])
        return p if p.is_file() else None
    if s.model_path:      # a model file chosen by hand: only use a projector that was also given by hand
        return None
    repo = getattr(s, "model_repo", None)
    if not repo:
        from . import modelsource
        repo = modelsource.TRUSTED_REPOS[1]
    from . import modelsource
    found = modelsource.find_mmproj(repo)
    if not found:
        logs.event(log, "no vision projector found; pictures will not be analysed", logging.WARNING, repo=repo)
        return None
    dest = s.data_dir / "models" / ("mmproj-" + found["file"].rsplit("/", 1)[-1].removeprefix("mmproj-"))
    if not (dest.is_file() and dest.stat().st_size > 10_000_000):
        try:
            _download(found["url"], dest, min_size=10_000_000)
        except Exception as e:
            logs.event(log, "vision projector download failed", logging.WARNING, error=str(e))
            return None
        if found["sha256"] and sha256_of(dest) != found["sha256"]:
            dest.unlink()
            logs.event(log, "vision projector SHA-256 mismatch, not used", logging.ERROR)
            return None
    logs.event(log, "vision projector ready", file=dest.name, size_mb=dest.stat().st_size // 2**20)
    return dest


def _download(url: str, dest: Path, min_size: int = 100_000_000) -> None:
    part = dest.with_suffix(dest.suffix + ".part")
    have = part.stat().st_size if part.exists() else 0
    req = urllib.request.Request(url, headers={"User-Agent": "live-ai/1.0"})
    if have:
        req.add_header("Range", f"bytes={have}-")
    logs.event(log, "downloading model", url=url, resume_from=have)
    with urllib.request.urlopen(req, timeout=60) as r:
        if have and r.status != 206:
            have = 0  # server ignored the range: start over
        mode = "ab" if have else "wb"
        with open(part, mode) as f:
            shutil.copyfileobj(r, f, 1 << 20)
    if part.stat().st_size < min_size:
        size = part.stat().st_size
        part.unlink()
        raise ModelError(f"download too small ({size} bytes)")
    part.replace(dest)


# --------------------------------------------------------------------------- llama-server
class LlamaServer:
    """Runs llama-server as a child process and keeps it alive."""

    def __init__(self, s: Settings):
        self.s = s
        self.url = s.llm_url.rstrip("/") if s.llm_url else f"http://127.0.0.1:{s.llm_port}"
        self.external = bool(s.llm_url)
        self.proc: subprocess.Popen | None = None
        self.model_path: Path | None = None
        self.mmproj: Path | None = None
        self.state = "starting"          # starting | loading | ready | restarting | failed
        self.last_error = ""
        self.restarts = 0
        self.started_at = 0.0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._watch: threading.Thread | None = None
        self._log_fh = None

    # lifecycle -------------------------------------------------------------
    def start(self, background: bool = True) -> None:
        def run():
            try:
                if not self.external:
                    self.state = "loading"
                    self.model_path = ensure_model(self.s)
                    self.mmproj = ensure_mmproj(self.s)
                    self._spawn()
                self._wait_ready(timeout=900)
            except Exception as e:
                self.state = "failed"
                self.last_error = str(e)
                logs.event(log, "model failed to start", logging.ERROR, error=str(e))
            self._watch = threading.Thread(target=self._watchdog, name="llm-watchdog", daemon=True)
            self._watch.start()

        if background:
            threading.Thread(target=run, name="llm-start", daemon=True).start()
        else:
            run()

    def command(self) -> list[str]:
        s = self.s
        cmd = [s.llama_server_bin, "-m", str(self.model_path), "-c", str(s.ctx_size), "-t", str(s.threads),
               "-tb", str(s.threads), "--host", "127.0.0.1", "--port", str(s.llm_port), "--jinja",
               "-np", "1", "--no-webui"]
        if self.mmproj:
            cmd += ["--mmproj", str(self.mmproj)]
        return cmd

    def _spawn(self) -> None:
        with self._lock:
            if self._log_fh is None:
                self._log_fh = open(self.s.log_dir / "llama_server.log", "ab")
            cmd = self.command()
            logs.event(log, "starting llama-server", cmd=" ".join(cmd))
            self.proc = subprocess.Popen(cmd, stdout=self._log_fh, stderr=subprocess.STDOUT, start_new_session=True)
            self.started_at = time.time()
            self.state = "loading"

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        with self._lock:
            if self.proc and self.proc.poll() is None:
                try:
                    os.killpg(self.proc.pid, signal.SIGTERM)
                    self.proc.wait(timeout=20)
                except Exception:
                    self.proc.kill()

    def _healthy(self) -> bool:
        try:
            with urllib.request.urlopen(self.url + "/health", timeout=5) as r:
                return r.status == 200
        except Exception:
            return False

    def _wait_ready(self, timeout: float) -> None:
        end = time.time() + timeout
        while time.time() < end and not self._stop.is_set():
            if self.proc is not None and self.proc.poll() is not None:
                raise ModelError(f"llama-server exited with code {self.proc.returncode} (see logs/llama_server.log)")
            if self._healthy():
                self.state = "ready"
                self.last_error = ""
                logs.event(log, "model ready", load_seconds=round(time.time() - (self.started_at or time.time()), 1),
                           threads=self.s.threads, ctx=self.s.ctx_size)
                return
            time.sleep(1)
        raise ModelError("llama-server did not become ready in time")

    def _watchdog(self) -> None:
        """Check every 5 s. Restart if the process died or failed 3 health checks in a row."""
        fails, backoff = 0, 2.0
        while True:
            self._wake.wait(5)
            self._wake.clear()
            if self._stop.is_set():
                return
            if self.external:
                ok = self._healthy()
                self.state = "ready" if ok else "restarting"
                continue
            dead = self.proc is None or self.proc.poll() is not None
            if not dead and self.state == "ready":
                fails = 0 if self._healthy() else fails + 1
                if fails < 3:
                    backoff = 2.0
                    continue
            if self.state == "failed" and self.model_path is None:
                continue  # model missing; nothing to restart
            code = None if self.proc is None else self.proc.poll()
            logs.event(log, "model stopped, restarting", logging.ERROR, exit_code=code, failed_health_checks=fails)
            self.state = "restarting"
            self.restarts += 1
            with self._lock:
                if self.proc and self.proc.poll() is None:
                    try:
                        os.killpg(self.proc.pid, signal.SIGKILL)
                    except Exception:
                        pass
            if self._stop.wait(backoff):
                return
            backoff = min(backoff * 2, 60)
            try:
                self._spawn()
                self._wait_ready(timeout=600)
                fails = 0
                logs.event(log, "model recovered", restarts=self.restarts)
            except Exception as e:
                self.state = "failed"
                self.last_error = str(e)
                logs.event(log, "restart failed, will retry", logging.ERROR, error=str(e))

    def poke(self) -> None:
        """Ask the watchdog to check right now (called when a request could not reach the model)."""
        if self.proc is not None and self.proc.poll() is not None and self.state == "ready":
            self.state = "restarting"
        self._wake.set()

    def wait_ready(self, timeout: float) -> bool:
        end = time.time() + timeout
        while time.time() < end:
            if self.state == "ready" and self._healthy():
                return True
            time.sleep(0.5)
        return False

    def status(self) -> dict:
        return {"state": self.state, "restarts": self.restarts, "error": self.last_error, "vision": bool(self.mmproj),
                "model": self.model_path.name if self.model_path else None,
                "pid": self.proc.pid if self.proc and self.proc.poll() is None else None}


# --------------------------------------------------------------------------- client
class LLMError(RuntimeError):
    pass


class LLMClient:
    def __init__(self, server: LlamaServer, s: Settings):
        self.server = server
        self.s = s

    def count_tokens(self, text: str) -> int | None:
        try:
            body = json.dumps({"content": text}).encode()
            req = urllib.request.Request(self.server.url + "/tokenize", body, {"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=10) as r:
                return len(json.load(r).get("tokens", []))
        except Exception:
            return None

    def complete_json(self, messages: list[dict], schema: dict, max_tokens: int = 1200, temperature: float = 0.1) -> dict:
        """One non-streamed reply forced to match a JSON schema (llama.cpp turns it into a grammar)."""
        body = {"messages": messages, "stream": False, "temperature": temperature, "top_p": 0.9, "top_k": 20,
                "max_tokens": max_tokens, "chat_template_kwargs": {"enable_thinking": False},
                "response_format": {"type": "json_schema", "json_schema": {"name": "plan", "schema": schema}}}
        req = urllib.request.Request(self.server.url + "/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.s.request_timeout_s) as r:
                data = json.load(r)
        except urllib.error.HTTPError as e:
            raise LLMError(f"model error {e.code}: {e.read()[:300].decode('utf-8', 'replace')}") from e
        except Exception as e:
            raise LLMError(f"model unreachable: {e}") from e
        text = data["choices"][0]["message"].get("content") or ""
        text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        return json.loads(text)

    def choose_for_image(self, jpeg: bytes, question: str, options: list[str]) -> int | None:
        """Shows the model one picture and numbered descriptions; returns the 0-based choice or None."""
        if not self.server.mmproj:
            return None
        import base64
        text = question + "\n" + "\n".join(f"{i + 1}. {o}" for i, o in enumerate(options)) + \
            "\nตอบเป็น JSON {\"choice\": หมายเลข} เลือกข้อเดียวที่ตรงกับภาพที่สุด"
        body = {"messages": [{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()}},
                    {"type": "text", "text": text}]}],
                "stream": False, "temperature": 0.0, "max_tokens": 20, "chat_template_kwargs": {"enable_thinking": False},
                "response_format": {"type": "json_schema", "json_schema": {"name": "c", "schema": {
                    "type": "object", "properties": {"choice": {"type": "integer", "minimum": 1, "maximum": len(options)}},
                    "required": ["choice"]}}}}
        req = urllib.request.Request(self.server.url + "/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            data = json.load(r)
        n = int(json.loads(data["choices"][0]["message"]["content"])["choice"])
        return n - 1 if 1 <= n <= len(options) else None

    def stream_chat(self, messages: list[dict], think: bool = False, max_tokens: int | None = None,
                    code: bool = False) -> Iterator[dict]:
        """Yields {"type": "token"|"thinking", "text": ...} and finally {"type": "done", "timings": {...}}.

        Sampling follows Qwen's recommendations; code requests use a low temperature so the
        model sticks to the exact requirements. With think=True the thinking gets its own
        token budget on top of the answer's, so it can't eat the whole reply.
        """
        reply_max = max_tokens or self.s.max_reply_tokens
        body = {
            "messages": messages,
            "stream": True,
            "temperature": 0.2 if code else (0.6 if think else self.s.temperature),
            "top_p": 0.9 if code else (0.95 if think else 0.8),
            "top_k": 20,
            "max_tokens": reply_max,
            "chat_template_kwargs": {"enable_thinking": bool(think)},
            "timings_per_token": False,
        }
        if think:
            body["reasoning_budget_tokens"] = self.s.think_budget
            body["max_tokens"] = reply_max + self.s.think_budget
        req = urllib.request.Request(self.server.url + "/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        try:
            resp = urllib.request.urlopen(req, timeout=self.s.request_timeout_s)
        except urllib.error.HTTPError as e:
            detail = e.read()[:300].decode("utf-8", "replace")
            raise LLMError(f"model error {e.code}: {detail}") from e
        except Exception as e:
            raise LLMError(f"model unreachable: {e}") from e
        timings: dict = {}
        finish = None
        with resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if "timings" in obj:
                    timings = obj["timings"]
                for ch in obj.get("choices", []):
                    d = ch.get("delta") or {}
                    if d.get("reasoning_content"):
                        yield {"type": "thinking", "text": d["reasoning_content"]}
                    if d.get("content"):
                        yield {"type": "token", "text": d["content"]}
                    if ch.get("finish_reason"):
                        finish = ch["finish_reason"]
        yield {"type": "done", "timings": timings, "finish_reason": finish}
