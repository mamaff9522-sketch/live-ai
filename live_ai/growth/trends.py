"""What people in Thailand are searching for today, from Google Trends' public RSS feed (free, no key).
Used only as optional inspiration for post angles and hashtags."""
from __future__ import annotations

import os
import re
import urllib.request
import xml.etree.ElementTree as ET

FEEDS = [u for u in os.environ.get("LIVEAI_TRENDS_URLS", "https://trends.google.com/trending/rss?geo=TH,"
                                   "https://trends.google.com/trends/trendingsearches/daily/rss?geo=TH").split(",") if u]
HT = "{https://trends.google.com/trending/rss}"


def _traffic(s: str) -> int:
    n = re.sub(r"[^\d]", "", s or "")
    return int(n) if n else 0


def fetch(limit: int = 20) -> list[dict]:
    for url in FEEDS:
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "LiveAI/1.0"}), timeout=20) as r:
                root = ET.fromstring(r.read())
        except Exception:
            continue
        out = []
        for item in root.iter("item"):
            title = (item.findtext("title") or "").strip()
            traffic = ""
            for child in item:
                if child.tag.endswith("approx_traffic"):
                    traffic = child.text or ""
            news = [c.text for c in item.iter() if c.tag.endswith("news_item_title") and c.text][:2]
            if title:
                out.append({"title": title, "traffic": _traffic(traffic), "news": news})
        if out:
            return out[:limit]
    return []
