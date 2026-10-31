"""CPU / RAM numbers from /proc (no extra packages)."""
from __future__ import annotations

import os
import time
from pathlib import Path

_PAGE = os.sysconf("SC_PAGE_SIZE")
_TICK = os.sysconf("SC_CLK_TCK")


def rss_mb(pid: int | None) -> float | None:
    if not pid:
        return None
    try:
        pages = int(Path(f"/proc/{pid}/statm").read_text().split()[1])
        return round(pages * _PAGE / 2**20, 1)
    except (OSError, ValueError, IndexError):
        return None


def cpu_seconds(pid: int | None) -> float | None:
    if not pid:
        return None
    try:
        parts = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return (int(parts[11]) + int(parts[12])) / _TICK
    except (OSError, ValueError, IndexError):
        return None


def memory_mb() -> dict:
    info = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            k, v = line.split(":", 1)
            info[k] = int(v.split()[0]) // 1024
    except OSError:
        pass
    return {"total": info.get("MemTotal"), "available": info.get("MemAvailable")}


class CpuMeter:
    """Average CPU use of a process since the previous call, in % of one core."""

    def __init__(self):
        self._last: dict[int, tuple[float, float]] = {}

    def percent(self, pid: int | None) -> float | None:
        cs = cpu_seconds(pid)
        if cs is None:
            return None
        now = time.time()
        prev = self._last.get(pid)
        self._last[pid] = (now, cs)
        if not prev or now - prev[0] < 0.2:
            return None
        return round(100 * (cs - prev[1]) / (now - prev[0]), 1)


def disk_free_mb(path: str | os.PathLike) -> int:
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize // 2**20
