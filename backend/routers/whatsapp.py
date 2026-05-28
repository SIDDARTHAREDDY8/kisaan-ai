"""
WhatsApp Business API webhook — production-grade implementation.

Two-phase response pattern:
  Phase 1 (< 2s): Send "Analysing..." acknowledgement via Twilio REST API,
                  return 200 OK to Twilio immediately so webhook doesn't timeout.
  Phase 2 (async): Run agent graph in FastAPI BackgroundTask,
                   send the real response when ready.

Message types handled:
  Text  → intent router → disease / market / scheme / soil / advisory
  Image → disease diagnosis pipeline
  Audio → ASR (Whisper) → translate (NLLB) → advisory → TTS (MMS) → WAV reply

Farmer memory: keyed by WhatsApp number (E.164 format).
Language: new users get a language menu; all replies are in the chosen language.
"""
import logging
import asyncio
from typing import Optional

import httpx
from fastapi import APIRouter, BackgroundTasks, Form, Request, Response
from twilio.request_validator import RequestValidator

from backend.agents.graph import run_graph
from backend.config import settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/whatsapp", tags=["whatsapp"])

TWIML_OK = '<?xml version="1.0" encoding="UTF-8"?><Response></Response>'
WA_CHAR_LIMIT = 1500

# Sentinel stored in DB when user hasn't chosen a language yet
LANG_PENDING = "pending"

# Language menu sent to new users
LANGUAGE_MENU = """🌾 *Welcome to Kisaan AI!*

I can help you with:
• Crop disease diagnosis 📷
• Mandi prices 💰
• Govt schemes 📋
• Soil health 🌱
• Farming advice 🚜

Please choose your language / कृपया भाषा चुनें:

1️⃣  English
2️⃣  हिंदी (Hindi)
3️⃣  తెలుగు (Telugu)
4️⃣  தமிழ் (Tamil)
5️⃣  मराठी (Marathi)
6️⃣  ಕನ್ನಡ (Kannada)
7️⃣  বাংলা (Bengali)

Reply with the number (1-7)"""

# Map menu choice → ISO 639-1 code
MENU_CHOICES = {
    "1": "en", "english": "en",
    "2": "hi", "hindi": "hi", "हिंदी": "hi",
    "3": "te", "telugu": "te", "తెలుగు": "te",
    "4": "ta", "tamil": "ta", "தமிழ்": "ta",
    "5": "mr", "marathi": "mr", "मराठी": "mr",
    "6": "kn", "kannada": "kn", "ಕನ್ನಡ": "kn",
    "7": "bn", "bengali": "bn", "বাংলা": "bn",
}

# Confirmation messages after language is selected
LANG_CONFIRMED = {
    "en": "✅ Great! I'll reply in English.\n\nSend me a photo of a diseased crop, a voice note, or ask anything about farming!",
    "hi": "✅ बढ़िया! मैं हिंदी में जवाब दूंगा।\n\nफसल की बीमारी की फोटो, वॉयस नोट, या खेती से जुड़ा कोई सवाल पूछें!",
    "te": "✅ చాలా బాగు! నేను తెలుగులో సమాధానం ఇస్తాను.\n\nపంట వ్యాధి ఫోటో, వాయిస్ నోట్, లేదా వ్యవసాయ సందేహాలు అడగండి!",
    "ta": "✅ சரி! நான் தமிழில் பதில் சொல்கிறேன்.\n\nபயிர் நோய் புகைப்படம், குரல் குறிப்பு, அல்லது விவசாய கேள்வி கேளுங்கள்!",
    "mr": "✅ छान! मी मराठीत उत्तर देईन.\n\nपिकाच्या रोगाचा फोटो, व्हॉइस नोट, किंवा शेतीबद्दल काहीही विचारा!",
    "kn": "✅ ಒಳ್ಳೆಯದು! ನಾನು ಕನ್ನಡದಲ್ಲಿ ಉತ್ತರಿಸುತ್ತೇನೆ.\n\nಬೆಳೆ ರೋಗದ ಫೋಟೋ, ವಾಯ್ಸ್ ನೋಟ್, ಅಥವಾ ಕೃಷಿ ಪ್ರಶ್ನೆ ಕೇಳಿ!",
    "bn": "✅ দারুণ! আমি বাংলায় উত্তর দেব।\n\nফসলের রোগের ছবি, ভয়েস নোট, বা চাষের যেকোনো প্রশ্ন করুন!",
}

ACK_MESSAGES = {
    "en": "🌿 Kisaan AI is analysing your message...",
    "hi": "🌿 किसान AI आपका संदेश विश्लेषण कर रहा है...",
    "te": "🌿 కిసాన్ AI మీ సందేశాన్ని విశ్లేషిస్తోంది...",
    "ta": "🌿 கிசான் AI உங்கள் செய்தியை பகுப்பாய்வு செய்கிறது...",
    "mr": "🌿 किसान AI तुमचा संदेश विश्लेषण करत आहे...",
    "kn": "🌿 ಕಿಸಾನ್ AI ನಿಮ್ಮ ಸಂದೇಶವನ್ನು ವಿಶ್ಲೇಷಿಸುತ್ತಿದೆ...",
    "bn": "🌿 কিসান AI আপনার বার্তা বিশ্লেষণ করছে...",
}


# ── Twilio helpers ────────────────────────────────────────────────────────────

def _twilio_configured() -> bool:
    return bool(
        settings.twilio_account_sid
        and settings.twilio_auth_token
        and settings.twilio_whatsapp_number
    )


async def _send_message(to: str, body: str) -> bool:
    if not _twilio_configured():
        logger.warning("Twilio not configured — would have sent to %s: %s", to, body[:60])
        return False

    url = f"https://api.twilio.com/2010-04-01/Accounts/{settings.twilio_account_sid}/Messages.json"
    payload = {
        "From": f"whatsapp:{settings.twilio_whatsapp_number}",
        "To": f"whatsapp:{to}",
        "Body": body,
    }
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                url, data=payload,
                auth=(settings.twilio_account_sid, settings.twilio_auth_token),
            )
            if resp.status_code in (200, 201):
                logger.info("Sent WA message to %s (%d chars)", to, len(body))
                return True
            logger.error("Twilio send failed: %s %s", resp.status_code, resp.text[:200])
            return False
    except Exception as exc:
        logger.error("Twilio send exception: %s", exc)
        return False


def _chunk_message(text: str, limit: int = WA_CHAR_LIMIT) -> list[str]:
    if len(text) <= limit:
        return [text]
    chunks = []
    while text:
        if len(text) <= limit:
            chunks.append(text)
            break
        split_at = text.rfind("\n", 0, limit)
        if split_at == -1:
            split_at = limit
        chunks.append(text[:split_at].rstrip())
        text = text[split_at:].lstrip()
    return chunks


async def _fetch_media(url: str, account_sid: str, auth_token: str) -> bytes:
    async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
        resp = await client.get(url, auth=(account_sid, auth_token))
        resp.raise_for_status()
        return resp.content


# ── Farmer session helpers ────────────────────────────────────────────────────

def _get_farmer_lang(farmer_id: str) -> str:
    """Returns language code, 'pending' if not yet chosen, 'en' on DB error."""
    try:
        from backend.db.models import FarmerSession, SyncSessionLocal
        with SyncSessionLocal() as session:
            farmer = session.query(FarmerSession).filter_by(farmer_id=farmer_id).first()
            if farmer is None:
                return LANG_PENDING  # new user
            return farmer.language or LANG_PENDING
    except Exception:
        return "en"


def _set_farmer_language(farmer_id: str, lang: str) -> None:
    try:
        from datetime import datetime
        from backend.db.models import FarmerSession, SyncSessionLocal
        with SyncSessionLocal() as session:
            farmer = session.query(FarmerSession).filter_by(farmer_id=farmer_id).first()
            if farmer:
                farmer.language = lang
                farmer.last_seen = datetime.utcnow()
            else:
                session.add(FarmerSession(farmer_id=farmer_id, language=lang))
            session.commit()
    except Exception as exc:
        logger.debug("Could not set farmer language: %s", exc)


# ── Background processor ──────────────────────────────────────────────────────

async def _process_and_reply(
    farmer_number: str,
    body: str,
    image_bytes: Optional[bytes],
    audio_bytes: Optional[bytes],
    audio_filename: str,
    farmer_lang: str,
) -> None:
    """Phase 2: run the agent graph and send the real response."""
    # ── Save image for training data collection ───────────────────────────────
    if image_bytes:
        try:
            from backend.services.data_collector import save_training_sample
            save_training_sample(
                image_bytes=image_bytes,
                farmer_id=farmer_number,
                farmer_text=Body,
                language=farmer_lang if farmer_lang not in ("en", LANG_PENDING) else "en",
            )
        except Exception as exc:
            logger.warning("Training data collection failed (non-fatal): %s", exc)

    stored_lang = farmer_lang if farmer_lang not in ("en", LANG_PENDING) else "en"
    initial_state = {
        "user_query": body,
        "image_bytes": image_bytes,
        "audio_bytes": audio_bytes,
        "audio_filename": audio_filename,
        "intent": "voice" if audio_bytes else "",
        "detected_language": stored_lang,
        "farmer_lang": stored_lang,   # hint for ASR language forcing
        "agent_trace": [],
    }

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, lambda: run_graph(initial_state, farmer_id=farmer_number)
        )
        reply = result.get("response_translated") or result.get("final_response", "")
        if not reply:
            reply = "I couldn't generate a response. Please try again."

        # Translate reply to the farmer's chosen language
        effective_lang = farmer_lang if farmer_lang not in ("en", LANG_PENDING) else \
                         result.get("detected_language", "en")

        if effective_lang != "en" and not result.get("response_translated"):
            try:
                from backend.services.translation_service import from_english
                reply = from_english(reply, effective_lang)
            except Exception as exc:
                logger.warning("Response translation failed, sending English: %s", exc)

        # Enrich the training sample with NER crop/location from routing
        if image_bytes:
            try:
                from backend.db.models import TrainingSample, SyncSessionLocal
                with SyncSessionLocal() as session:
                    sample = session.query(TrainingSample).filter_by(
                        farmer_id_hash=__import__('hashlib').sha256(farmer_number.encode()).hexdigest()
                    ).order_by(TrainingSample.id.desc()).first()
                    if sample and not sample.crop_hint:
                        sample.crop_hint = result.get("commodity") or result.get("crop") or ""
                        sample.farmer_text_en = result.get("user_query_en") or ""
                        ner_locs = result.get("ner_locations", [])
                        sample.location_hint = ner_locs[0] if ner_locs else ""
                        session.commit()
            except Exception:
                pass

        # Only auto-update language for users who already chose one (not pending/new)
        # Pending users must go through the menu; don't let auto-detection bypass it
        detected = result.get("detected_language", "en")
        if detected != "en" and farmer_lang not in ("en", LANG_PENDING):
            _set_farmer_language(farmer_number, detected)

    except Exception as exc:
        logger.exception("Agent graph failed for %s", farmer_number)
        reply = (
            "⚠️ I had trouble processing your request. Please try again, "
            "or describe your problem in text if you sent a voice note."
        )

    for chunk in _chunk_message(reply):
        await _send_message(farmer_number, chunk)


# ── Webhook endpoint ──────────────────────────────────────────────────────────

@router.get("/webhook")
async def whatsapp_webhook_verify():
    return Response(content=TWIML_OK, media_type="application/xml")


@router.post("/webhook")
async def whatsapp_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    From: str = Form(...),
    Body: str = Form(""),
    NumMedia: int = Form(0),
    MediaUrl0: str = Form(""),
    MediaContentType0: str = Form(""),
):
    farmer_number = From.replace("whatsapp:", "")
    farmer_lang = _get_farmer_lang(farmer_number)

    logger.info(
        "WA message from %s | lang=%s | body=%r | media=%d (%s)",
        farmer_number, farmer_lang, Body[:60], NumMedia, MediaContentType0,
    )

    # ── New user: show language menu ─────────────────────────────────────────
    if farmer_lang == LANG_PENDING:
        # Check if this message IS the language selection
        choice = Body.strip().lower()
        chosen = MENU_CHOICES.get(choice) or MENU_CHOICES.get(Body.strip())
        if chosen:
            _set_farmer_language(farmer_number, chosen)
            await _send_message(farmer_number, LANG_CONFIRMED[chosen])
            logger.info("Language set to %s for %s", chosen, farmer_number)
        else:
            # First message or invalid choice — show the menu
            _set_farmer_language(farmer_number, LANG_PENDING)
            await _send_message(farmer_number, LANGUAGE_MENU)
        return Response(content=TWIML_OK, media_type="application/xml")

    # ── Existing user: acknowledge and process ────────────────────────────────
    ack = ACK_MESSAGES.get(farmer_lang, ACK_MESSAGES["en"])
    await _send_message(farmer_number, ack)

    # Fetch media if present
    image_bytes: Optional[bytes] = None
    audio_bytes: Optional[bytes] = None
    audio_filename = "audio.ogg"

    if NumMedia > 0 and MediaUrl0 and _twilio_configured():
        try:
            media_bytes = await _fetch_media(
                MediaUrl0, settings.twilio_account_sid, settings.twilio_auth_token
            )
            if "image" in MediaContentType0:
                image_bytes = media_bytes
                logger.info("Received image: %d bytes", len(media_bytes))
            elif "audio" in MediaContentType0 or "ogg" in MediaContentType0:
                audio_bytes = media_bytes
                ext = MediaContentType0.split("/")[-1].split(";")[0].strip()
                audio_filename = f"audio.{ext or 'ogg'}"
                logger.info("Received audio: %d bytes (%s)", len(media_bytes), audio_filename)
        except Exception as exc:
            logger.warning("Media fetch failed: %s", exc)

    background_tasks.add_task(
        _process_and_reply,
        farmer_number=farmer_number,
        body=Body,
        image_bytes=image_bytes,
        audio_bytes=audio_bytes,
        audio_filename=audio_filename,
        farmer_lang=farmer_lang,
    )

    return Response(content=TWIML_OK, media_type="application/xml")


@router.get("/status")
async def whatsapp_status():
    configured = _twilio_configured()
    return {
        "configured": configured,
        "number": settings.twilio_whatsapp_number if configured else None,
        "message": "Ready" if configured else "Set TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_WHATSAPP_NUMBER in backend/.env",
    }
