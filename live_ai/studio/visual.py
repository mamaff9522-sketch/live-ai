"""AI 4 (Visual Analyzer) helpers: keyframes, picture similarity, brightness.

Meta AI / Sala AI clips are generated with the previous clip's last frame as the next clip's
starting picture, so "last frame of A looks like first frame of B" is strong evidence that B
follows A. Pictures are compared as tiny 24x24 colour thumbnails (pure Python, no extra
packages); the same Qwen3.5-2B can also look at a frame and pick the matching shot description.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

THUMB = 24


def _frame(path: str, at: float, size: str, fmt: str) -> bytes:
    args = ["ffmpeg", "-loglevel", "error", "-nostdin"]
    if at >= 0:
        args += ["-ss", f"{at:.3f}"]
    else:   # from the end
        args += ["-sseof", f"{at:.3f}"]
    args += ["-i", path, "-frames:v", "1", "-vf", size]
    args += ["-f", "rawvideo", "-pix_fmt", "rgb24", "-"] if fmt == "raw" else ["-f", "image2pipe", "-vcodec", "mjpeg", "-q:v", "4", "-"]
    r = subprocess.run(args, capture_output=True, timeout=120)
    return r.stdout


def thumbs(path: str, duration: float) -> dict:
    """First / last frame as 24x24 RGB bytes (hex), plus mean brightness."""
    size = f"scale={THUMB}:{THUMB}:flags=area"
    first = _frame(path, 0.05, size, "raw")
    last = _frame(path, -0.25, size, "raw")
    if len(last) != THUMB * THUMB * 3:
        last = _frame(path, max(0.0, duration - 0.3), size, "raw")
    mid = _frame(path, duration / 2, size, "raw")
    def luma(b):
        return round(sum(0.299 * b[i] + 0.587 * b[i + 1] + 0.114 * b[i + 2] for i in range(0, len(b), 3)) / max(1, len(b) // 3), 1)
    return {"first": first.hex(), "last": last.hex(), "mid": mid.hex(), "luma": luma(mid) if mid else None}


def keyframe_jpeg(path: str, duration: float, width: int = 448) -> bytes:
    return _frame(path, duration / 2, f"scale={width}:-2", "jpeg")


def similarity(a_hex: str | None, b_hex: str | None) -> float:
    """1.0 = identical thumbnails, ~0 = unrelated."""
    if not a_hex or not b_hex or len(a_hex) != len(b_hex):
        return 0.0
    a, b = bytes.fromhex(a_hex), bytes.fromhex(b_hex)
    mad = sum(abs(x - y) for x, y in zip(a, b)) / len(a)
    return round(max(0.0, 1.0 - mad / 64.0), 3)     # 64 levels of average difference = unrelated
