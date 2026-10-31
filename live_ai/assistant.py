"""Turns one user message into a reply: memory commands, prompt building,
history trimming to fit the context window, one-at-a-time queue for the CPU."""
from __future__ import annotations

import logging
import re
import threading
import time
from typing import Iterator

from . import logs, memory
from .config import Settings
from .llm import LLMClient, LLMError, LlamaServer
from .store import Store

log = logs.get("assistant")

SYSTEM_PROMPT = """คุณคือ Live AI ผู้ช่วยที่พูดภาษาไทยเป็นหลัก
- ตอบภาษาเดียวกับผู้ใช้ ภาษาไทยให้เป็นธรรมชาติ สุภาพ กระชับ ตรงประเด็น
- ถ้าไม่แน่ใจให้บอกตรง ๆ ห้ามแต่งข้อมูล
- โจทย์ที่ต้องคิด ให้คิดเป็นขั้นตอนสั้น ๆ แล้วสรุปคำตอบสุดท้ายให้ชัด
- เมื่อเขียนโค้ด: ทำตามเงื่อนไขของโจทย์ให้ครบทุกข้อ ใส่โค้ดฉบับสมบูรณ์ในบล็อก ``` เพียงบล็อกเดียว แล้วอธิบายสั้น ๆ
- เมื่อตรวจโค้ด: ไล่ทีละบรรทัด ระบุบั๊กทุกจุด แล้วให้โค้ดที่แก้แล้วฉบับเดียวที่ถูกต้อง
- วางแผนงานเป็นข้อ ๆ ที่ทำได้จริง"""

CODE_HINT = re.compile(r"โค้ด|โค๊ด|code|ฟังก์ชัน|function|python|javascript|typescript|java\b|sql|สคริปต์|script|บั๊ก|bug|def |```", re.I)


def estimate_tokens(text: str) -> int:
    """Conservative estimate (Thai ~2 chars/token, English ~4 chars/token)."""
    thai = len(re.findall(r"[฀-๿]", text))
    return int(thai / 1.8 + (len(text) - thai) / 3.2) + 6


class Busy(Exception):
    pass


class Assistant:
    def __init__(self, s: Settings, store: Store, server: LlamaServer):
        self.s = s
        self.store = store
        self.server = server
        self.llm = LLMClient(server, s)
        self._gen = threading.Semaphore(1)       # one generation at a time: the CPU is the limit
        self._waiting = 0
        self._wlock = threading.Lock()
        self._user_busy: set[int] = set()

    # prompt ----------------------------------------------------------------
    def build_messages(self, user_id: int, conv_id: int, new_text: str, budget_scale: float = 1.0) -> list[dict]:
        facts = memory.relevant(new_text, self.store.memories(user_id))
        system = SYSTEM_PROMPT
        if facts:
            system += "\n\nข้อมูลที่ผู้ใช้คนนี้ขอให้จำไว้ (ใช้เมื่อเกี่ยวข้องเท่านั้น):\n" + "\n".join(f"- {m['content']}" for m in facts)
        system += f"\n\nวันนี้คือ {time.strftime('%Y-%m-%d')}"
        budget = int((self.s.ctx_size - self.s.max_reply_tokens - estimate_tokens(system) - estimate_tokens(new_text) - 64) * budget_scale)
        history = self.store.messages(user_id, conv_id, limit=200)
        kept: list[dict] = []
        used = 0
        for m in reversed(history):
            t = estimate_tokens(m["content"])
            if used + t > budget:
                break
            kept.append({"role": m["role"], "content": m["content"]})
            used += t
        kept.reverse()
        while kept and kept[0]["role"] != "user":   # templates want user first
            kept.pop(0)
        return [{"role": "system", "content": system}, *kept, {"role": "user", "content": new_text}]

    # one turn --------------------------------------------------------------
    def reply(self, user_id: int, conv_id: int | None, text: str, think: bool = False) -> Iterator[dict]:
        """Yields events: meta, token, thinking, done, error. Saves both messages."""
        text = text.strip()
        if not text:
            yield {"type": "error", "message": "ข้อความว่าง"}
            return
        if len(text) > self.s.max_message_chars:
            yield {"type": "error", "message": f"ข้อความยาวเกิน {self.s.max_message_chars} ตัวอักษร"}
            return
        if conv_id is None:
            conv_id = self.store.create_conversation(user_id, text.replace("\n", " ")[:40])
        else:
            self.store.get_conversation(user_id, conv_id)  # raises NotFound for someone else's id
        yield {"type": "meta", "conversation_id": conv_id}

        with self._wlock:
            if user_id in self._user_busy:
                yield {"type": "error", "message": "กำลังตอบข้อความก่อนหน้าอยู่ รอสักครู่แล้วส่งใหม่"}
                return
            if self._waiting >= self.s.max_queue:
                yield {"type": "error", "message": "ตอนนี้มีคนใช้งานมาก กรุณาลองใหม่อีกครั้ง"}
                return
            self._user_busy.add(user_id)
            self._waiting += 1
            ahead = self._waiting - 1

        started = time.time()
        reply_parts: list[str] = []
        note = ""
        timings, finish = {}, None
        try:
            cmd = memory.parse_command(text)
            if cmd and cmd[0] == "remember":
                self.store.add_memory(user_id, cmd[1])
                note = "remembered"
            elif cmd and cmd[0] == "forget":
                note = f"forgot:{self.store.delete_memories_matching(user_id, cmd[1])}"
            if ahead:
                yield {"type": "queue", "position": ahead}
            if not self._gen.acquire(timeout=self.s.request_timeout_s):
                yield {"type": "error", "message": "รอคิวนานเกินไป กรุณาลองใหม่"}
                return
            try:
                if self.server.state != "ready":
                    # wait up to 120 s for a restart in progress
                    for _ in range(120):
                        if self.server.state == "ready":
                            break
                        time.sleep(1)
                    if self.server.state != "ready":
                        yield {"type": "error", "message": "โมเดลยังไม่พร้อม (กำลังโหลดหรือกู้คืนระบบ) ลองใหม่อีกครั้งในไม่กี่นาที"}
                        return
                scales = [1.0, 0.5, 0.0]
                recovered = False
                while scales:
                    msgs = self.build_messages(user_id, conv_id, text, scales[0])
                    try:
                        for ev in self.llm.stream_chat(msgs, think=think, code=bool(CODE_HINT.search(text))):
                            if ev["type"] == "token":
                                reply_parts.append(ev["text"])
                                yield ev
                            elif ev["type"] == "thinking":
                                yield ev
                            elif ev["type"] == "done":
                                timings, finish = ev["timings"], ev["finish_reason"]
                        break
                    except LLMError as e:
                        msg = str(e).lower()
                        if reply_parts:
                            raise
                        if "context" in msg and len(scales) > 1:
                            scales.pop(0)
                            logs.event(log, "context too long, trimming history", logging.WARNING, conversation=conv_id)
                            continue
                        if "unreachable" in msg and not recovered:
                            # the model process died: let the watchdog restart it, then retry once
                            recovered = True
                            logs.event(log, "model unreachable, waiting for recovery", logging.WARNING)
                            yield {"type": "recovering"}
                            self.server.poke()
                            if self.server.wait_ready(240):
                                continue
                        raise
            finally:
                self._gen.release()
        except LLMError as e:
            logs.event(log, "generation failed", logging.ERROR, user=user_id, conversation=conv_id, error=str(e)[:300])
            yield {"type": "error", "message": "โมเดลขัดข้องชั่วคราว ระบบกำลังกู้คืนอัตโนมัติ กรุณาส่งใหม่อีกครั้ง"}
            return
        finally:
            self._user_busy.discard(user_id)
            with self._wlock:
                self._waiting -= 1

        answer = "".join(reply_parts).strip()
        if not answer:
            answer = "(ไม่มีคำตอบ)"
        self.store.add_message(user_id, conv_id, "user", text)
        self.store.add_message(user_id, conv_id, "assistant", answer)
        secs = time.time() - started
        stats = {
            "seconds": round(secs, 2),
            "prompt_tokens": timings.get("prompt_n"),
            "reply_tokens": timings.get("predicted_n"),
            "tokens_per_second": round(timings.get("predicted_per_second") or 0, 2),
            "prompt_tokens_per_second": round(timings.get("prompt_per_second") or 0, 2),
            "finish_reason": finish,
            "memory": note,
        }
        logs.event(log, "reply", user=user_id, conversation=conv_id, in_chars=len(text), out_chars=len(answer), think=think, **stats)
        yield {"type": "done", "stats": stats}
