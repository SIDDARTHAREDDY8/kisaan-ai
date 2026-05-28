"""
Intent Router — uses the real HuggingFace Zero-Shot Classifier (bart-large-mnli)
with NER enrichment to auto-populate commodity and location fields.
Falls back to keyword heuristics on cold start for low latency.

Non-English queries are translated to English before NER/ZSC so that
"టమోటో ధర ఎంత" correctly routes to market, not advisory.
"""
import logging

from backend.agents.state import KisaanState
from backend.services.ner_service import extract_entities
from backend.services.zero_shot_classifier import classify_intent

logger = logging.getLogger(__name__)


def _detect_and_translate(text: str) -> tuple[str, str]:
    """
    Returns (english_text, detected_lang_code).
    If already English or detection fails, returns (text, 'en').
    """
    if not text.strip():
        return text, "en"
    try:
        from langdetect import detect, LangDetectException
        lang = detect(text)
    except Exception:
        lang = "en"

    if lang == "en":
        return text, "en"

    # Translate to English for routing
    try:
        from backend.services.translation_service import to_english
        english = to_english(text, lang)
        logger.info("Translated [%s] → EN: %r → %r", lang, text[:60], english[:60])
        return english, lang
    except Exception as exc:
        logger.warning("Translation failed, routing on original text: %s", exc)
        return text, lang


def route_intent(state: KisaanState) -> KisaanState:
    query = state.get("user_query", "")
    has_image = bool(state.get("image_bytes"))
    has_audio = bool(state.get("audio_bytes"))

    # Translate non-English queries to English for NER + ZSC
    english_query, detected_lang = _detect_and_translate(query)

    # Run NER on the English version so crop/location keywords match
    ner_result = extract_entities(english_query) if english_query.strip() else None
    ner_crops = ner_result.crops if ner_result else []
    ner_locations = ner_result.locations if ner_result else []
    ner_entities = ner_result.raw_entities if ner_result else []

    # Auto-populate commodity from NER if not already set
    commodity = state.get("commodity") or (ner_crops[0].title() if ner_crops else "")
    location = state.get("location") or (ner_locations[0].title() if ner_locations else "")

    # ZSC intent classification on the English version
    if has_audio and not query.strip():
        intent, confidence = "voice", 1.0
    else:
        intent, confidence = classify_intent(english_query, has_image=has_image)

    # Persist detected language so advisory/market agents can respond in farmer's language
    farmer_lang = detected_lang if detected_lang != "en" else state.get("detected_language", "en")

    trace = (
        f"Router[ZSC] → intent={intent} ({confidence:.0%})"
        + (f", lang={detected_lang}" if detected_lang != "en" else "")
        + (f", crops={ner_crops}" if ner_crops else "")
        + (f", locs={ner_locations}" if ner_locations else "")
    )

    return {
        **state,
        "intent": intent,
        "intent_confidence": confidence,
        "commodity": commodity,
        "location": location,
        "ner_crops": ner_crops,
        "ner_locations": ner_locations,
        "ner_entities": ner_entities,
        "detected_language": farmer_lang,
        # Store English translation so downstream agents don't see the raw regional query
        "user_query_en": english_query if detected_lang != "en" else state.get("user_query", ""),
        "agent_trace": [trace],
    }


def next_node(state: KisaanState) -> str:
    return state.get("intent", "advisory")
