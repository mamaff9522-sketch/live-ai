"""Finds the Qwen3.5-2B Q4_K_M GGUF file on Hugging Face.

Trusted publishers are tried in order (the Qwen team first). For the file found,
Hugging Face's API gives its SHA-256, which the download is checked against.

    python -m live_ai.modelsource            # print what would be downloaded
"""
from __future__ import annotations

import json
import os
import re
import sys
import urllib.request

TRUSTED_REPOS = [
    "Qwen/Qwen3.5-2B-GGUF",                 # official Qwen team
    "unsloth/Qwen3.5-2B-GGUF",              # Unsloth (well-known quantizer)
    "bartowski/Qwen_Qwen3.5-2B-GGUF",
    "lmstudio-community/Qwen3.5-2B-GGUF",
]
QUANT = os.environ.get("LIVEAI_QUANT", "Q4_K_M")


def _get(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "live-ai/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def find(repos: list[str] | None = None, quant: str = QUANT) -> dict:
    """{'repo','file','url','sha256','size'} for the first trusted repo that has the file."""
    pattern = re.compile(rf"(^|[-_.]){re.escape(quant)}\.gguf$", re.I)
    tried = []
    for repo in repos or TRUSTED_REPOS:
        try:
            files = _get(f"https://huggingface.co/api/models/{repo}/tree/main?recursive=true")
        except Exception as e:
            tried.append(f"{repo}: {e}")
            continue
        for f in files:
            path = f.get("path", "")
            if pattern.search(path) and "mmproj" not in path.lower():
                lfs = f.get("lfs") or {}
                return {"repo": repo, "file": path, "size": lfs.get("size") or f.get("size"), "sha256": lfs.get("oid", ""),
                        "url": f"https://huggingface.co/{repo}/resolve/main/{path}"}
        tried.append(f"{repo}: no {quant} file")
    raise LookupError("; ".join(tried))


def find_mmproj(repo: str) -> dict | None:
    """The vision projector (lets the same Qwen3.5-2B look at pictures) from the same repo."""
    try:
        files = _get(f"https://huggingface.co/api/models/{repo}/tree/main?recursive=true")
    except Exception:
        return None
    cands = [f for f in files if "mmproj" in f.get("path", "").lower() and f["path"].endswith(".gguf")]
    def rank(p: str) -> int:
        p = p.lower()
        for i, pat in enumerate((r"[-_.]f16", r"bf16", r"q8_0", r"f32")):
            if re.search(pat, p):
                return i
        return 9
    cands.sort(key=lambda f: rank(f["path"]))
    if not cands:
        return None
    f = cands[0]
    lfs = f.get("lfs") or {}
    return {"repo": repo, "file": f["path"], "size": lfs.get("size") or f.get("size"), "sha256": lfs.get("oid", ""),
            "url": f"https://huggingface.co/{repo}/resolve/main/{f['path']}"}


if __name__ == "__main__":
    try:
        print(json.dumps(find(), indent=2))
    except LookupError as e:
        print("not found:", e)
        sys.exit(1)
