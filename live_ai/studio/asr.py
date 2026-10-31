"""AI 5 (Dialogue Analyzer): speech detection + Thai speech recognition, all local.

- Speech/silence: ffmpeg silencedetect on the clip's audio.
- Recognition: whisper.cpp (MIT) with a multilingual Whisper model (default small, q5_1),
  run on the CPU; segments come back with start/end times.
- Whisper sometimes "hears" words in silence or music; clips without speech are not sent to it,
  and known filler outputs are dropped.
Set LIVEAI_ASR=fake to read transcripts from a JSON file (tests only).
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path

from .. import logs

log = logs.get("asr")

WHISPER_MODELS = {
    "base": "ggml-base-q5_1.bin",
    "small": "ggml-small-q5_1.bin",
    "medium": "ggml-medium-q5_0.bin",
    "large-v3-turbo": "ggml-large-v3-turbo-q5_0.bin",
}
WHISPER_URL = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/{}"
HALLUCINATIONS = re.compile(r"(ซับไตเติ้?ล|subtitles? by|โปรดติดตาม|ขอบคุณที่รับชม|ติดตามช่อง|thank you for watching|amara\.org|\[(?:เพลง|music|เสียงดนตรี)\])",
                            re.I)


class ASRUnavailable(RuntimeError):
    pass


def loudness(path: str) -> tuple[float, float]:
    """(mean dB, max dB) of the audio track; (-91, -91) when there is no sound at all."""
    try:
        r = subprocess.run(["ffmpeg", "-hide_banner", "-nostdin", "-i", path, "-vn", "-af", "volumedetect", "-f", "null", "-"],
                           capture_output=True, text=True, timeout=300)
    except (OSError, subprocess.TimeoutExpired):
        return -91.0, -91.0
    mean = re.search(r"mean_volume: (-?[\d.]+|-inf) dB", r.stderr)
    peak = re.search(r"max_volume: (-?[\d.]+|-inf) dB", r.stderr)
    f = lambda m: -91.0 if not m or m.group(1) == "-inf" else float(m.group(1))
    return f(mean), f(peak)


def speech_regions(path: str, duration: float, min_silence: float = 0.35) -> list[tuple[float, float]]:
    """Stretches with sound clearly above the clip's own background, e.g. [(0.4, 3.1), (4.0, 7.2)].

    The threshold follows each clip's loudness (recordings differ a lot); a clip that never gets
    louder than -50 dB is treated as silent."""
    mean, peak = loudness(path)
    if peak < -50:
        return []
    noise_db = max(-60.0, min(-20.0, peak - 26))
    try:
        r = subprocess.run(["ffmpeg", "-hide_banner", "-nostdin", "-i", path, "-vn", "-af",
                            f"silencedetect=noise={noise_db}dB:d={min_silence}", "-f", "null", "-"],
                           capture_output=True, text=True, timeout=300)
    except (OSError, subprocess.TimeoutExpired):
        return []
    starts = [float(x) for x in re.findall(r"silence_start: ([\d.]+)", r.stderr)]
    ends = [float(x) for x in re.findall(r"silence_end: ([\d.]+)", r.stderr)]
    regions, cur = [], 0.0
    for i, s in enumerate(starts):
        if s - cur > 0.15:
            regions.append((round(cur, 2), round(s, 2)))
        cur = ends[i] if i < len(ends) else duration
    if duration - cur > 0.15:
        regions.append((round(cur, 2), round(duration, 2)))
    return regions


class WhisperASR:
    name = "whisper.cpp"

    def __init__(self, data_dir: Path, threads: int):
        self.bin = os.environ.get("LIVEAI_WHISPER_BIN") or shutil.which("whisper-cli") or "whisper-cli"
        self.model_name = os.environ.get("LIVEAI_WHISPER_MODEL", "small")
        self.models = data_dir / "models" / "whisper"
        self.threads = threads
        self.lang = os.environ.get("LIVEAI_ASR_LANG", "th")

    def model_path(self) -> Path:
        custom = os.environ.get("LIVEAI_WHISPER_MODEL_PATH")
        if custom:
            return Path(custom)
        f = WHISPER_MODELS.get(self.model_name, self.model_name)
        p = self.models / f
        if not p.is_file():
            self.models.mkdir(parents=True, exist_ok=True)
            part = p.with_suffix(".part")
            logs.event(log, "downloading whisper model", file=f)
            try:
                with urllib.request.urlopen(urllib.request.Request(WHISPER_URL.format(f), headers={"User-Agent": "live-ai"}),
                                            timeout=60) as r, open(part, "wb") as out:
                    shutil.copyfileobj(r, out, 1 << 20)
            except Exception as e:
                part.unlink(missing_ok=True)
                raise ASRUnavailable(f"ดาวน์โหลดโมเดลถอดเสียงไม่ได้: {e}") from e
            part.replace(p)
        return p

    def check(self) -> None:
        if not (shutil.which(self.bin) or Path(self.bin).is_file()):
            raise ASRUnavailable("ยังไม่ได้ติดตั้งโปรแกรมถอดเสียง (whisper.cpp)")
        self.model_path()

    def transcribe(self, path: str, regions: list[tuple[float, float]], names: list[str] | None = None) -> dict:
        model = self.model_path()
        with tempfile.TemporaryDirectory(prefix="asr-") as d:
            wav = Path(d) / "a.wav"
            subprocess.run(["ffmpeg", "-loglevel", "error", "-nostdin", "-y", "-i", path, "-vn", "-ac", "1", "-ar", "16000",
                            "-c:a", "pcm_s16le", str(wav)], check=True, timeout=300)
            out = Path(d) / "out"
            cmd = [self.bin, "-m", str(model), "-l", self.lang, "-t", str(self.threads), "-oj", "-of", str(out), "-np", "-f", str(wav)]
            if names:
                cmd += ["--prompt", " ".join(names[:12])]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            if r.returncode != 0 or not out.with_suffix(".json").exists():
                raise RuntimeError(f"whisper failed: {(r.stderr or r.stdout)[-300:]}")
            data = json.loads(out.with_suffix(".json").read_text(encoding="utf-8", errors="replace"))
        segs = []
        for s in data.get("transcription", []):
            text = (s.get("text") or "").strip()
            if not text or HALLUCINATIONS.search(text) or re.fullmatch(r"[\[(（【].*[\])）】]\s*", text):
                continue                   # filler like "[ พากษ์เสียงไทย ]" is not speech
            a, b = s["offsets"]["from"] / 1000, s["offsets"]["to"] / 1000
            # drop words "heard" where the audio is actually silent
            if regions and not any(min(b, y) - max(a, x) > 0.1 for x, y in regions):
                continue
            segs.append({"start": round(a, 2), "end": round(b, 2), "text": text})
        return {"text": " ".join(s["text"] for s in segs).strip(), "segments": segs, "engine": f"whisper.cpp/{model.name}"}


class FakeASR:
    """Tests: LIVEAI_FAKE_ASR points to {"<file name>": "<transcript>" | [segments...]}"""
    name = "fake"

    def __init__(self):
        self.table = json.loads(Path(os.environ["LIVEAI_FAKE_ASR"]).read_text(encoding="utf-8"))

    def check(self) -> None:
        pass

    def transcribe_named(self, name: str, duration: float) -> dict:
        v = self.table.get(name, "")
        if isinstance(v, list):
            return {"text": " ".join(s["text"] for s in v), "segments": v, "engine": "fake"}
        segs = [{"start": 0.3, "end": round(max(0.5, duration - 0.3), 2), "text": v}] if v else []
        return {"text": v, "segments": segs, "engine": "fake"}


def make_asr(data_dir: Path, threads: int):
    if os.environ.get("LIVEAI_ASR") == "fake":
        return FakeASR()
    return WhisperASR(data_dir, threads)
