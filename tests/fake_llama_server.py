"""Stand-in for llama-server used by the offline tests.

Accepts the same command line (only --port is used), answers /health,
/tokenize and streaming /v1/chat/completions. The reply quotes the last user
message, and every request body is appended to $FAKE_LLM_LOG so tests can
check exactly what the model was sent.
"""
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

port = int(sys.argv[sys.argv.index("--port") + 1])
LOG = os.environ.get("FAKE_LLM_LOG")


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path == "/health":
            b = b'{"status":"ok"}'
            self.send_response(200)
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/tokenize":
            b = json.dumps({"tokens": list(range(len(body["content"]) // 2))}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)
            return
        if LOG:
            with open(LOG, "a", encoding="utf-8") as f:
                f.write(json.dumps(body, ensure_ascii=False) + "\n")
        last = body["messages"][-1]["content"]
        props = (body.get("response_format") or {}).get("json_schema", {}).get("schema", {}).get("properties", {})
        if "caption" in props or "audience" in props:          # Facebook page growth (posts / plan)
            import random as _r
            words = ["แค้น", "รัก", "ทรยศ", "ความลับ", "แม่สามี", "ลูกสะใภ้", "น้ำตา", "หักหลัง", "คืนนั้น", "พินัยกรรม", "ตระกูล",
                     "บ้านใหญ่", "สร้อยคอ", "จดหมาย", "งานแต่ง", "โรงพยาบาล", "ฝนตก", "สัญญา", "ลูกชาย", "มรดก", "ทะเล", "ภาพถ่าย",
                     "เงินกู้", "แหวน", "ประตูหลัง", "เสียงร้อง", "รถคันแดง", "ไร่องุ่น", "เพื่อนสนิท", "คำโกหก"]
            story = " ".join(_r.sample(words, 12)) + f" {_r.randint(100000, 999999)}"
            if "caption" in props:
                out = {"caption": f"เธอไม่เคยรู้เลยว่า {story} ถ้าเป็นคุณจะทำอย่างไร? กดติดตามเพจเพื่อดูตอนต่อไป",
                       "lines": [f"ความลับของ {story}", "A ให้อภัย", "B เอาคืน"], "hashtags": ["ละครสั้น", "ดราม่า"]}
            else:
                out = {"audience": ["ผู้หญิงวัยทำงานที่ชอบละครดราม่า", "คนดูคลิปสั้นช่วงเย็น"],
                       "content_pillars": ["ฉากเด็ดจากละคร", "คำถามชวนคุย"],
                       "channels": ["แชร์ Reels ไป TikTok", "ปักหมุดโพสต์ที่ดีที่สุด"],
                       "this_week": ["กำลังเก็บข้อมูล", "โพสต์ Reels ช่วงสองทุ่ม"]}
            b = json.dumps({"choices": [{"message": {"role": "assistant", "content": json.dumps(out, ensure_ascii=False)}}]},
                           ensure_ascii=False).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)
            return
        if "response_format" in body:
            plan = os.environ.get("FAKE_PLAN") or json.dumps({
                "clips": [{"file": 1, "start": 0, "end": 2, "speed": 1, "mute": False},
                          {"file": 2, "start": 0, "end": -1, "speed": 1, "mute": False}],
                "aspect": "9:16", "fit": "blur",
                "texts": [{"text": "ทดสอบภาษาไทย", "start": 0, "end": -1, "position": "bottom", "size": "medium"}],
                "music_file": 3, "music_volume": 0.3, "original_volume": 1, "transition": "fade", "fade_in": 0, "fade_out": 0})
            if "เปลี่ยนเป็นแนวนอน" in last:
                prev = json.loads(last.split("แผนเดิม:\n", 1)[1].split("\n\n", 1)[0])
                prev["aspect"] = "16:9"
                plan = json.dumps(prev)
            b = json.dumps({"choices": [{"message": {"role": "assistant", "content": plan}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)
            return
        if "TOO_LONG" in last and len(body["messages"]) > 2:
            b = b'{"error":{"message":"the request exceeds the available context size"}}'
            self.send_response(400)
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self.end_headers()
        if body.get("chat_template_kwargs", {}).get("enable_thinking"):
            self.wfile.write(b'data: {"choices":[{"delta":{"reasoning_content":"thinking..."}}]}\n\n')
        for piece in ["ตอบ", ": ", last]:
            if "SLOW" in last:
                time.sleep(0.5)
            chunk = {"choices": [{"delta": {"content": piece}}]}
            self.wfile.write(f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode())
            self.wfile.flush()
        end = {"choices": [{"delta": {}, "finish_reason": "stop"}],
               "timings": {"prompt_n": 10, "predicted_n": 3, "predicted_per_second": 12.5, "prompt_per_second": 50}}
        self.wfile.write(f"data: {json.dumps(end)}\n\ndata: [DONE]\n\n".encode())
        self.close_connection = True


ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
