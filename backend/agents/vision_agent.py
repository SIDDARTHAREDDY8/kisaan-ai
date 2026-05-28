"""
Vision Agent — runs disease classifier and decides whether to diagnose or ask a follow-up.

Confidence strategy:
  ≥ 0.65 (medium/high from Claude, or fine-tuned model) → proceed to advisory
  < 0.65 OR follow_up_question set                       → ask farmer for clarification
"""
import logging

from backend.agents.state import KisaanState
from backend.services.classifier import classify_disease

logger = logging.getLogger(__name__)

CONFIDENCE_THRESHOLD = 0.65


def vision_agent(state: KisaanState) -> KisaanState:
    image_bytes = state.get("image_bytes")
    if not image_bytes:
        return {
            **state,
            "error": "No image provided for disease analysis.",
            "agent_trace": state.get("agent_trace", []) + ["VisionAgent → skipped (no image)"],
        }

    farmer_text = state.get("user_query_en") or state.get("user_query", "")

    try:
        result = classify_disease(image_bytes, farmer_text=farmer_text)
        logger.info("Classified: %s (%.2f%%)", result.label, result.confidence * 100)

        low_confidence = result.confidence < CONFIDENCE_THRESHOLD or bool(result.follow_up_question)

        trace = (
            f"VisionAgent → disease_id={result.disease_id or 'unknown'} "
            f"conf={result.confidence:.1%}"
            + (" [needs follow-up]" if result.follow_up_question else "")
            + (" [low conf]" if low_confidence and not result.follow_up_question else "")
        )

        return {
            **state,
            "classifier_label": result.label,
            "classifier_confidence": result.confidence,
            "crop": result.crop,
            "condition": result.condition,
            "top5": result.top5,
            "follow_up_question": result.follow_up_question,
            "agent_trace": state.get("agent_trace", []) + [trace],
            "error": "low_confidence" if low_confidence else None,
        }
    except Exception as exc:
        logger.exception("Vision agent failed")
        return {
            **state,
            "error": str(exc),
            "follow_up_question": "Could you send a clearer photo of the affected leaf or fruit in natural daylight?",
            "agent_trace": state.get("agent_trace", []) + [f"VisionAgent → ERROR: {exc}"],
        }
