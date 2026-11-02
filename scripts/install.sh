#!/usr/bin/env bash
# Live AI — one-command install on a Linux server (Ubuntu/Debian, x86-64 or ARM64),
# e.g. an Oracle Cloud "Always Free" Ampere A1 machine (2 OCPU / 12 GB).
#
#   curl -fsSL https://raw.githubusercontent.com/mamaff9522-sketch/live-ai/main/scripts/install.sh | bash
#
# What it does (nothing paid, nothing removed):
#   1. installs Docker if it is missing
#   2. starts Live AI in a container that restarts by itself (data kept in the Docker volume "liveai-data")
#   3. opens a free HTTPS link through a Cloudflare quick tunnel (no account, no open ports needed)
# Run it again to update to the newest version; your users, chats and projects are kept.
# Model: unlocked Qwen3.5-2B by default; original Qwen with: LIVEAI_MODEL_VARIANT=standard bash install.sh
# Undo: docker rm -f live-ai live-ai-link   (add: docker volume rm liveai-data  — only if you want the data gone)
set -euo pipefail

REPO="mamaff9522-sketch/live-ai"
IMAGE="ghcr.io/${REPO}:latest"
PORT="${LIVEAI_PORT:-7860}"
SUDO=""; [ "$(id -u)" -eq 0 ] || SUDO="sudo"

say() { printf '\n\033[1m%s\033[0m\n' "$*"; }

if ! command -v docker >/dev/null 2>&1; then
  say "ติดตั้ง Docker (ครั้งแรกครั้งเดียว)…"
  curl -fsSL https://get.docker.com | $SUDO sh
fi
DOCKER="$SUDO docker"

# Small machines (under 6 GB RAM): add a 4 GB swap file once, so a big video job slows down instead of
# being killed. Undo: sudo swapoff /swapfile && sudo rm /swapfile (and remove its line in /etc/fstab).
RAM_MB=$(free -m | awk '/Mem:/{print $2}')
if [ "$RAM_MB" -lt 6000 ] && [ -z "$(swapon --show --noheadings 2>/dev/null)" ] && [ ! -e /swapfile ]; then
  say "เพิ่มหน่วยความจำสำรอง (swap) 4 GB สำหรับเครื่อง RAM ${RAM_MB} MB…"
  { $SUDO fallocate -l 4G /swapfile || $SUDO dd if=/dev/zero of=/swapfile bs=1M count=4096 status=none; } \
    && $SUDO chmod 600 /swapfile && $SUDO mkswap /swapfile >/dev/null && $SUDO swapon /swapfile \
    && echo '/swapfile none swap sw 0 0' | $SUDO tee -a /etc/fstab >/dev/null || echo "(ข้าม swap ได้ ไม่กระทบการทำงานหลัก)"
fi

say "ดึงโปรแกรม Live AI รุ่นล่าสุด…"
if ! $DOCKER pull "$IMAGE"; then
  say "ดึงไม่ได้ — สร้างจากซอร์สโค้ดบนเครื่องนี้แทน (ใช้เวลาประมาณ 15–30 นาที)…"
  TMP=$(mktemp -d)
  curl -fsSL "https://github.com/${REPO}/archive/refs/heads/main.tar.gz" | tar -xz -C "$TMP" --strip-components=1
  TAG=$(curl -fsSL https://api.github.com/repos/ggml-org/llama.cpp/releases/latest | sed -n 's/.*"tag_name": *"\([^"]*\)".*/\1/p' | head -1)
  $DOCKER build --build-arg LLAMA_TAG="${TAG:-master}" -t "$IMAGE" "$TMP"
  rm -rf "$TMP"
fi

say "เริ่มระบบ…"
$DOCKER rm -f live-ai >/dev/null 2>&1 || true
ADMIN_FILE="$HOME/.live-ai-admin-key"
[ -s "$ADMIN_FILE" ] || { head -c 16 /dev/urandom | od -An -tx1 | tr -d ' \n' > "$ADMIN_FILE"; chmod 600 "$ADMIN_FILE"; }
$DOCKER volume create liveai-data >/dev/null
$DOCKER run -d --name live-ai --restart unless-stopped \
  -p "127.0.0.1:${PORT}:7860" -v liveai-data:/data \
  -e LIVEAI_ADMIN_KEY="$(cat "$ADMIN_FILE")" \
  ${LIVEAI_MODEL_VARIANT:+-e LIVEAI_MODEL_VARIANT="$LIVEAI_MODEL_VARIANT"} \
  ${LIVEAI_OPEN_REGISTRATION:+-e LIVEAI_OPEN_REGISTRATION="$LIVEAI_OPEN_REGISTRATION"} \
  "$IMAGE" >/dev/null

say "รอโมเดลโหลด (ครั้งแรกต้องดาวน์โหลด ~1.3 GB อาจใช้ 2–10 นาที)…"
for i in $(seq 1 180); do
  if curl -fsS "http://127.0.0.1:${PORT}/api/health" >/dev/null 2>&1; then break; fi
  sleep 5
done
curl -fsS "http://127.0.0.1:${PORT}/api/health" || true; echo

say "เปิดลิงก์ HTTPS ฟรี…"
$DOCKER rm -f live-ai-link >/dev/null 2>&1 || true
$DOCKER run -d --name live-ai-link --restart unless-stopped --network host \
  cloudflare/cloudflared:latest tunnel --no-autoupdate --url "http://127.0.0.1:${PORT}" </dev/null >/dev/null
echo "รอลิงก์…"
URL=""
for i in $(seq 1 60); do
  URL=$(timeout 10 $DOCKER logs live-ai-link 2>&1 | grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' | head -1 || true)
  if [ -n "$URL" ]; then break; fi
  sleep 2
done

# Permanent HTTPS address (does not change on restart): https://<ip-with-dashes>.sslip.io, with a free
# Let's Encrypt certificate handled by Caddy. Only if ports 80 and 443 are free on this machine.
# Use your own domain instead with: LIVEAI_DOMAIN=ai.example.com bash install.sh
STABLE=""
PUBLIC_IP=$(curl -fsS4 -m 10 https://api.ipify.org 2>/dev/null || hostname -I | awk '{print $1}')
DOMAIN="${LIVEAI_DOMAIN:-$(echo "$PUBLIC_IP" | tr . -).sslip.io}"
if [ "${LIVEAI_HTTPS:-1}" != 1 ]; then
  echo "(ข้ามลิงก์ถาวร: LIVEAI_HTTPS=0)"
elif $DOCKER ps --format '{{.Names}}' | grep -qx live-ai-https || ! ss -ltnH 2>/dev/null | awk '{print $4}' | grep -qE ':(80|443)$'; then
  say "ตั้งลิงก์ถาวร https://${DOMAIN} …"
  $DOCKER rm -f live-ai-https >/dev/null 2>&1 || true
  $DOCKER volume create liveai-https >/dev/null
  $DOCKER run -d --name live-ai-https --restart unless-stopped --network host -v liveai-https:/data \
    caddy:2 caddy reverse-proxy --from "$DOMAIN" --to "127.0.0.1:${PORT}" </dev/null >/dev/null
  for i in $(seq 1 45); do
    if curl -fsS -m 10 --resolve "${DOMAIN}:443:127.0.0.1" "https://${DOMAIN}/api/health" >/dev/null 2>&1; then STABLE="https://${DOMAIN}"; break; fi
    sleep 4
  done
  [ -n "$STABLE" ] || echo "(ยังออกใบรับรอง HTTPS ไม่ได้ — ตรวจว่าเปิดพอร์ต 80/443 ในไฟร์วอลล์ของผู้ให้บริการ; ระหว่างนี้ใช้ลิงก์ trycloudflare ได้)"
else
  echo "(พอร์ต 80/443 มีโปรแกรมอื่นใช้อยู่ — ข้ามลิงก์ถาวร)"
fi

say "เสร็จแล้ว"
[ -n "$STABLE" ] && echo "ลิงก์ถาวร:    $STABLE   (สตูดิโอ: $STABLE/studio)"
echo "ลิงก์ใช้งาน:  ${URL:-(ยังไม่ได้ลิงก์ — ดูด้วยคำสั่ง: $DOCKER logs live-ai-link)}"
echo "สตูดิโอตัดต่อ: ${URL:+$URL/studio}"
echo "รหัสผู้ดูแล (สำรองข้อมูล): เก็บไว้ที่ $ADMIN_FILE"
echo "หมายเหตุ: ลิงก์ trycloudflare จะเปลี่ยนเมื่อเครื่องรีสตาร์ต — ดูลิงก์ใหม่ด้วย: $DOCKER logs live-ai-link 2>&1 | grep trycloudflare"
