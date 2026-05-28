"""Shared LangGraph state schema passed between all agents."""
from typing import Annotated, Any, Optional
from typing_extensions import TypedDict


class KisaanState(TypedDict, total=False):
    # ── Input ────────────────────────────────────────────────────────────────
    image_bytes: Optional[bytes]
    audio_bytes: Optional[bytes]
    audio_filename: str
    user_query: str
    user_query_en: str           # English translation of user_query for NER/ZSC/agents
    original_query: str          # pre-translation text
    detected_language: str       # ISO 639-1 from Whisper
    commodity: str
    location: str
    soil_params: dict            # {nitrogen, phosphorus, potassium, ph, organic_carbon, moisture}
    farmer_id: str               # phone number or session identifier for memory

    # ── Routing ──────────────────────────────────────────────────────────────
    intent: str                  # disease | market | advisory | scheme | soil | voice
    intent_confidence: float

    # ── NER enrichment ───────────────────────────────────────────────────────
    ner_crops: list[str]
    ner_locations: list[str]
    ner_entities: list[dict]

    # ── Object detection ─────────────────────────────────────────────────────
    detections: list[dict]
    pest_flags: list[str]
    detection_summary: str

    # ── Vision / disease ─────────────────────────────────────────────────────
    classifier_label: str
    classifier_confidence: float
    crop: str
    condition: str
    top5: list[dict]
    follow_up_question: str        # set when classifier needs more info from farmer
    farmer_lang: str               # chosen language code, passed through pipeline

    # ── RAG / advisory ───────────────────────────────────────────────────────
    retrieved_docs: list[dict]
    treatment_advice: str
    prevention_tips: str
    severity: str

    # ── Market ───────────────────────────────────────────────────────────────
    market_data: list[dict]
    price_forecast: dict

    # ── Soil ─────────────────────────────────────────────────────────────────
    soil_result: dict

    # ── Voice pipeline ───────────────────────────────────────────────────────
    response_translated: str
    response_audio: Optional[bytes]

    # ── Cost & observability ─────────────────────────────────────────────────
    cost_tracker: Any            # SessionCost instance (not serializable, in-memory only)
    cost_summary: dict

    # ── Final ────────────────────────────────────────────────────────────────
    final_response: str
    agent_trace: Annotated[list[str], lambda a, b: a + b]
    error: Optional[str]
