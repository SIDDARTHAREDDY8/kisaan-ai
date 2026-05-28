"""
ASR Service — Whisper (openai/whisper-small via HuggingFace transformers)
HuggingFace Task: Automatic Speech Recognition

Accepts raw audio bytes (wav/mp3/ogg/webm) and returns the transcript
plus the detected language code so downstream agents can route translation.

When the farmer's language is already known (stored in DB), it's passed as a
hint so Whisper doesn't have to guess — critical for short voice notes.
"""
import logging
import os
import tempfile
from functools import lru_cache
from typing import NamedTuple, Optional

import torch
from transformers import pipeline

logger = logging.getLogger(__name__)

WHISPER_MODEL = "openai/whisper-small"

# Whisper language code → ISO 639-1 (Whisper uses its own codes for some langs)
WHISPER_TO_ISO = {
    "english": "en", "hindi": "hi", "telugu": "te", "tamil": "ta",
    "marathi": "mr", "kannada": "kn", "bengali": "bn", "urdu": "ur",
    "gujarati": "gu", "punjabi": "pa", "malayalam": "ml",
}


class ASRResult(NamedTuple):
    text: str
    language: str
    confidence: float


@lru_cache(maxsize=1)
def _load_asr():
    logger.info("Loading Whisper ASR: %s", WHISPER_MODEL)
    device = 0 if torch.cuda.is_available() else -1
    return pipeline(
        "automatic-speech-recognition",
        model=WHISPER_MODEL,
        device=device,
        return_timestamps=True,   # needed to get detected language in output
    )


def transcribe_audio(
    audio_bytes: bytes,
    filename: str = "audio.wav",
    language_hint: Optional[str] = None,
) -> ASRResult:
    """
    Transcribe audio. If language_hint (ISO 639-1 code) is provided,
    Whisper is forced to use that language — avoids mis-detection on
    short clips (< 5 seconds) which are common in WhatsApp voice notes.
    """
    suffix = "." + filename.rsplit(".", 1)[-1] if "." in filename else ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(audio_bytes)
        tmp_path = f.name

    asr = _load_asr()

    # Build generate_kwargs: force language when hint is available
    generate_kwargs: dict = {"task": "transcribe"}
    if language_hint and language_hint != "en":
        # Whisper accepts ISO 639-1 codes directly in newer versions
        generate_kwargs["language"] = language_hint
        logger.info("ASR using language hint: %s", language_hint)

    try:
        result = asr(tmp_path, generate_kwargs=generate_kwargs)
    finally:
        os.unlink(tmp_path)

    text = result.get("text", "").strip()

    # Extract detected language from chunks (only available with return_timestamps=True)
    detected_lang = language_hint or "en"
    chunks = result.get("chunks", [])
    if chunks and not language_hint:
        # Some Whisper versions put language in chunk metadata
        raw_lang = chunks[0].get("language") if isinstance(chunks[0], dict) else None
        if raw_lang:
            detected_lang = WHISPER_TO_ISO.get(raw_lang.lower(), raw_lang[:2])

    # Fallback: if no hint and language still "en" but text looks non-Latin, detect via langdetect
    if detected_lang == "en" and text and not language_hint:
        try:
            from langdetect import detect
            detected = detect(text)
            if detected != "en":
                detected_lang = detected
                logger.info("langdetect override: Whisper said 'en' but text is %s", detected_lang)
        except Exception:
            pass

    logger.info("ASR result: lang=%s, text=%r", detected_lang, text[:80])
    return ASRResult(text=text, language=detected_lang, confidence=1.0)
