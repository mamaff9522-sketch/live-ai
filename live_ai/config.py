"""Live AI settings.

Every value can be overridden with an environment variable (LIVEAI_*).
When a value is left empty, Live AI tunes it from the machine it runs on:
threads = physical CPU count, context size = what the free RAM allows.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str = "") -> str:
    return os.environ.get(f"LIVEAI_{name}", default).strip()


def cpu_count() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # pragma: no cover
        return os.cpu_count() or 1


def mem_total_mb() -> int:
    """Memory this process may use: the cgroup limit when there is one, else MemTotal."""
    limit = None
    for p in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            raw = Path(p).read_text().strip()
            if raw and raw != "max" and int(raw) < 1 << 50:
                limit = int(raw) // (1024 * 1024)
                break
        except (OSError, ValueError):
            pass
    total = None
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                total = int(line.split()[1]) // 1024
                break
    except OSError:
        pass
    vals = [v for v in (limit, total) if v]
    return min(vals) if vals else 2048


# Default model: Qwen3.5-2B, GGUF Q4_K_M. Sources are tried in order; the first that exists is used.
DEFAULT_MODEL_URLS = [
    "https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/resolve/main/Qwen3.5-2B-Q4_K_M.gguf",
    "https://huggingface.co/Qwen/Qwen3.5-2B-GGUF/resolve/main/Qwen3.5-2B-Q4_K_M.gguf",
]


@dataclass
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(_env("DATA_DIR", "data")).resolve())
    host: str = field(default_factory=lambda: _env("HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(_env("PORT", "7860")))

    # model / llama.cpp
    model_path: str = field(default_factory=lambda: _env("MODEL_PATH"))
    model_urls: list[str] = field(default_factory=lambda: [u for u in _env("MODEL_URL").split(",") if u] or list(DEFAULT_MODEL_URLS))
    model_sha256: str = field(default_factory=lambda: _env("MODEL_SHA256").lower())
    llama_server_bin: str = field(default_factory=lambda: _env("LLAMA_SERVER", "llama-server"))
    llm_url: str = field(default_factory=lambda: _env("LLM_URL"))  # use an already-running server instead of starting one
    llm_port: int = field(default_factory=lambda: int(_env("LLM_PORT", "8081")))
    threads: int = field(default_factory=lambda: int(_env("THREADS", "0")))
    ctx_size: int = field(default_factory=lambda: int(_env("CTX", "0")))
    max_reply_tokens: int = field(default_factory=lambda: int(_env("MAX_REPLY_TOKENS", "1024")))
    think_budget: int = field(default_factory=lambda: int(_env("THINK_BUDGET", "768")))
    temperature: float = field(default_factory=lambda: float(_env("TEMPERATURE", "0.7")))

    # service limits
    max_queue: int = field(default_factory=lambda: int(_env("MAX_QUEUE", "8")))
    request_timeout_s: int = field(default_factory=lambda: int(_env("REQUEST_TIMEOUT", "600")))
    max_message_chars: int = field(default_factory=lambda: int(_env("MAX_MESSAGE_CHARS", "8000")))
    open_registration: bool = field(default_factory=lambda: _env("OPEN_REGISTRATION", "1") == "1")
    backups_to_keep: int = field(default_factory=lambda: int(_env("BACKUPS_KEEP", "14")))

    def __post_init__(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "logs").mkdir(exist_ok=True)
        (self.data_dir / "backups").mkdir(exist_ok=True)
        (self.data_dir / "models").mkdir(exist_ok=True)
        if self.threads <= 0:
            self.threads = max(1, cpu_count())
        if self.ctx_size <= 0:
            self.ctx_size = auto_ctx(mem_total_mb())

    @property
    def db_path(self) -> Path:
        return self.data_dir / "live_ai.sqlite3"

    @property
    def log_dir(self) -> Path:
        return self.data_dir / "logs"


def auto_ctx(mem_mb: int) -> int:
    """Context window that fits the machine.

    Model weights (Q4_K_M) take ~1.3 GB; the KV cache of this model is small, so
    RAM mostly limits how much room is left for the app and the OS.
    """
    if mem_mb >= 12000:
        return 16384
    if mem_mb >= 6000:
        return 8192
    if mem_mb >= 3500:
        return 4096
    return 2048
