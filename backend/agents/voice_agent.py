"""
Voice Agent — Full multilingual voice pipeline:
  1. ASR (Whisper): audio bytes → transcript + detected language
  2. Translation (NLLB-200): regional language → English
  3. Route to appropriate agent (disease/market/scheme/advisory)
  4. Translate response back to farmer's language
  5. TTS (MMS): English/regional response → WAV audio bytes

This is the core differentiator for low-literacy, low-connectivity users.
"""
import logging

from backend.agents.state import KisaanState
from backend.services.asr_service import transcribe_audio
from backend.services.translation_service import from_english, to_english
from backend.services.tts_service import synthesize

logger = logging.getLogger(__name__)


def voice_preprocess(state: KisaanState) -> KisaanState:
    """Step 1-2: Transcribe audio → translate to English."""
    audio_bytes = state.get("audio_bytes")
    audio_filename = state.get("audio_filename", "audio.wav")

    if not audio_bytes:
        return {**state, "agent_trace": ["VoicePreprocess → no audio, skipped"]}

    # Use the farmer's stored language as a hint so Whisper doesn't guess on short clips
    lang_hint = state.get("detected_language") or state.get("farmer_lang")
    lang_hint = lang_hint if lang_hint and lang_hint not in ("en", "pending") else None

    asr_result = transcribe_audio(audio_bytes, audio_filename, language_hint=lang_hint)
    detected_lang = asr_result.language or lang_hint or "en"
    transcript = asr_result.text

    # Translate to English for downstream agents if needed
    english_query = to_english(transcript, detected_lang) if detected_lang != "en" else transcript

    logger.info("ASR: lang=%s, transcript=%r", detected_lang, transcript[:80])

    return {
        **state,
        "user_query": english_query,
        "original_query": transcript,
        "detected_language": detected_lang,
        "agent_trace": [
            f"VoicePreprocess → ASR detected={detected_lang}, transcript={transcript[:60]}...",
        ],
    }


def voice_postprocess(state: KisaanState) -> KisaanState:
    """Step 4-5: Translate response back → synthesize audio."""
    response = state.get("final_response", "")
    detected_lang = state.get("detected_language", "en")

    if not response:
        return state

    # Translate back to farmer's language
    if detected_lang != "en":
        response_translated = from_english(response, detected_lang)
    else:
        response_translated = response

    # Synthesize audio
    try:
        audio_bytes = synthesize(response_translated, lang=detected_lang)
        logger.info("TTS synthesized %d bytes for lang=%s", len(audio_bytes), detected_lang)
    except Exception as exc:
        logger.warning("TTS synthesis failed: %s", exc)
        audio_bytes = None

    return {
        **state,
        "response_translated": response_translated,
        "response_audio": audio_bytes,
        "agent_trace": [
            f"VoicePostprocess → translated to {detected_lang}, TTS {'ok' if audio_bytes else 'failed'}",
        ],
    }
