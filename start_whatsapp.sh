#!/bin/bash
# ──────────────────────────────────────────────────────────────────
# Kisaan AI — WhatsApp Mode Launcher
#
# Starts:
#   1. FastAPI backend on :8000
#   2. ngrok tunnel → public HTTPS URL
#   3. Prints the exact webhook URL to paste in Twilio console
#
# Usage: ./start_whatsapp.sh [ngrok-authtoken]
# ──────────────────────────────────────────────────────────────────
set -e
cd "$(dirname "$0")"

# ── Kill anything already on port 8000 ───────────────────────────
lsof -ti:8000 | xargs kill -9 2>/dev/null || true

# ── Check .env has Twilio creds ───────────────────────────────────
check_env() {
  local key=$1
  grep -q "^${key}=.\+" backend/.env 2>/dev/null && return 0
  echo "⚠  Missing ${key} in backend/.env"
  return 1
}

missing=0
for var in TWILIO_ACCOUNT_SID TWILIO_AUTH_TOKEN TWILIO_WHATSAPP_NUMBER; do
  check_env "$var" || missing=1
done

if [ "$missing" -eq 1 ]; then
  echo ""
  echo "Add your Twilio credentials to backend/.env:"
  echo "  TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxx"
  echo "  TWILIO_AUTH_TOKEN=your_auth_token"
  echo "  TWILIO_WHATSAPP_NUMBER=+14155238886"
  echo ""
  echo "Get them free at: https://console.twilio.com"
  echo "Sandbox setup:    https://console.twilio.com/us1/develop/sms/try-it-out/whatsapp-learn"
  echo ""
  exit 1
fi

# ── Start backend ─────────────────────────────────────────────────
echo "Starting Kisaan AI backend..."
venv/bin/python -m uvicorn backend.main:app --port 8000 &
BACKEND_PID=$!

# Wait up to 20s for backend to be ready
echo -n "Waiting for backend"
for i in $(seq 1 20); do
  sleep 1
  echo -n "."
  if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
    echo " ready ✅"
    break
  fi
  if [ "$i" -eq 20 ]; then
    echo ""
    echo "❌ Backend did not start in 20s. Try running manually:"
    echo "   venv/bin/python -m uvicorn backend.main:app --port 8000"
    kill $BACKEND_PID 2>/dev/null
    exit 1
  fi
done

# ── Start Cloudflare tunnel ───────────────────────────────────────
echo "Opening Cloudflare tunnel..."

# cloudflared prints the public URL to stderr — capture it
TUNNEL_LOG=$(mktemp /tmp/cloudflared_kisaan.XXXXXX)
cloudflared tunnel --url http://localhost:8000 --no-autoupdate > "$TUNNEL_LOG" 2>&1 &
NGROK_PID=$!

# Poll the log until the trycloudflare.com URL appears (up to 30s)
PUBLIC_URL=""
for i in $(seq 1 30); do
  sleep 1
  PUBLIC_URL=$(grep -o 'https://[a-zA-Z0-9-]*\.trycloudflare\.com' "$TUNNEL_LOG" 2>/dev/null | head -1)
  if [ -n "$PUBLIC_URL" ]; then
    break
  fi
  if [ "$i" -eq 30 ]; then
    echo "❌ Cloudflare tunnel did not open."
    tail -20 "$TUNNEL_LOG"
    kill $BACKEND_PID 2>/dev/null
    kill $NGROK_PID 2>/dev/null
    exit 1
  fi
done

WEBHOOK_URL="${PUBLIC_URL}/api/whatsapp/webhook"

echo ""
echo "════════════════════════════════════════════════════════"
echo "  Kisaan AI — WhatsApp Live"
echo "════════════════════════════════════════════════════════"
echo ""
echo "  📋 Paste this URL in Twilio console:"
echo ""
echo "     $WEBHOOK_URL"
echo ""
echo "  Setup steps:"
echo "  1. console.twilio.com → Messaging → Try it out → WhatsApp"
echo "  2. 'When a message comes in' → paste URL above → HTTP POST → Save"
echo "  3. On phone WhatsApp → message +18665885804"
echo "     Send: join <your-sandbox-code>  (shown in Twilio console)"
echo ""
echo "  Try sending:"
echo "  • 📷 Leaf photo   → disease diagnosis"
echo "  • 🎤 Voice note   → multilingual advisory"
echo "  • 'tomato price'  → live mandi prices"
echo "  • 'PM-KISAN'      → govt scheme info"
echo "  • 'soil pH 6.2'   → soil health advisory"
echo ""
echo "  Press Ctrl+C to stop"
echo "════════════════════════════════════════════════════════"

cleanup() {
  echo ""
  echo "Shutting down..."
  kill $BACKEND_PID 2>/dev/null
  kill $NGROK_PID 2>/dev/null
  rm -f "$TUNNEL_LOG"
}
trap cleanup EXIT INT TERM
wait $BACKEND_PID
