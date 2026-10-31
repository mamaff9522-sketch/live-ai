"""Long-term memory per user.

- "จำไว้ว่า ..." / "remember that ..." saves a fact for that user.
- "ลืมเรื่อง ..." / "forget ..." deletes the user's facts that contain the text.
- Before each reply the user's most relevant facts are put into the system prompt.

Relevance uses character 2/3-grams, which works for Thai (no spaces between words)
as well as English, with no extra libraries.
"""
from __future__ import annotations

import re

# Strict on purpose: "ลืมรหัสผ่านทำไงดี" (I forgot my password) must not delete memories.
REMEMBER = re.compile(r"^\s*(?:ช่วย)?(?:จำไว้(?:ด้วย)?(?:นะ)?ว่า|ช่วยจำว่า|remember that|please remember(?: that)?)\s*[:：,]?\s*(.+)$",
                      re.IGNORECASE | re.DOTALL)
FORGET = re.compile(r"^\s*(?:ช่วย)?(?:ลืมเรื่อง|ลืมที่(?:ฉัน|ผม|เรา)?(?:เคย)?บอก(?:ไว้)?ว่า|forget that|forget about)\s*[:：,]?\s*(.+)$",
                    re.IGNORECASE | re.DOTALL)
_PUNCT = re.compile(r"[\s\.,!?\"'“”‘’()\[\]{}:;，。！？]+")


def parse_command(text: str) -> tuple[str, str] | None:
    """('remember', fact) / ('forget', text) / None."""
    m = REMEMBER.match(text)
    if m and m.group(1).strip():
        return "remember", m.group(1).strip()
    m = FORGET.match(text)
    if m and m.group(1).strip():
        target = re.sub(r"\s*(?:ไป)?(?:เลย)?(?:นะ)?(?:ครับ|ค่ะ|คะ|จ้า)?\s*$", "", m.group(1).strip())
        return "forget", target or m.group(1).strip()
    return None


def _grams(text: str) -> set[str]:
    t = _PUNCT.sub(" ", text.lower()).strip()
    out: set[str] = set()
    for word in t.split():
        if len(word) <= 2:
            out.add(word)
            continue
        for n in (2, 3):
            out.update(word[i:i + n] for i in range(len(word) - n + 1))
    return out


def score(query: str, fact: str) -> float:
    q, f = _grams(query), _grams(fact)
    if not q or not f:
        return 0.0
    return len(q & f) / (len(q) ** 0.5 * len(f) ** 0.5)


def relevant(query: str, facts: list[dict], k: int = 8, always_recent: int = 3) -> list[dict]:
    """Top-k facts for this query, plus the most recent few (cheap and usually useful)."""
    if not facts:
        return []
    ranked = sorted(facts, key=lambda m: score(query, m["content"]), reverse=True)
    picked = [m for m in ranked[:k] if score(query, m["content"]) > 0.05]
    for m in facts[-always_recent:]:
        if m not in picked:
            picked.append(m)
    return picked[:k + always_recent]
