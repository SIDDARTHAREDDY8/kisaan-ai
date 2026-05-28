"""
Soil Agent — Tabular classification + Claude-powered amendment recommendations.
HuggingFace Task: Tabular Classification

Accepts NPK, pH, organic carbon, moisture readings and returns:
  - Soil health class + score
  - Deficiency / excess flags
  - Crop suitability list
  - Specific amendment actions with dosages
"""
import logging

import anthropic

from backend.agents.state import KisaanState
from backend.config import settings
from backend.services.cost_tracker import SessionCost, usage_from_anthropic_response
from backend.services.soil_service import classify_soil

logger = logging.getLogger(__name__)
_client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

_SOIL_SYSTEM = """You are Kisaan AI's soil health expert. Given soil test results, provide:
1. Clear explanation of what each reading means in plain farmer language
2. Priority-ordered amendment actions (most critical first)
3. Which crops to grow this season vs avoid
4. Estimated cost of recommended amendments per acre

Keep it practical — farmers need to know exactly what to buy at the agri-input shop and how much to apply."""


def soil_agent(state: KisaanState) -> KisaanState:
    soil_params = state.get("soil_params", {})
    cost_tracker: SessionCost = state.get("cost_tracker")

    if not soil_params:
        return {
            **state,
            "final_response": "Please provide soil test readings: N (kg/ha), P (kg/ha), K (kg/ha), pH, organic carbon (%), and moisture (%).",
            "agent_trace": ["SoilAgent → no params provided"],
        }

    try:
        result = classify_soil(
            nitrogen_kg_ha=float(soil_params.get("nitrogen", 280)),
            phosphorus_kg_ha=float(soil_params.get("phosphorus", 15)),
            potassium_kg_ha=float(soil_params.get("potassium", 150)),
            ph=float(soil_params.get("ph", 7.0)),
            organic_carbon=float(soil_params.get("organic_carbon", 0.6)),
            moisture_pct=float(soil_params.get("moisture", 20)),
        )
    except Exception as exc:
        logger.exception("Soil classification failed")
        return {
            **state,
            "final_response": f"Soil analysis failed: {exc}",
            "agent_trace": ["SoilAgent → classification error"],
        }

    context = (
        f"Soil health score: {result.health_score}/100 ({result.health_class})\n"
        f"Deficiencies: {', '.join(result.deficiencies) or 'None'}\n"
        f"Excesses: {', '.join(result.excesses) or 'None'}\n"
        f"Suitable crops: {', '.join(result.suitable_crops)}\n"
        f"Unsuitable crops: {', '.join(result.unsuitable_crops)}\n"
        f"Recommended amendments:\n" + "\n".join(f"  - {a}" for a in result.amendments)
    )

    message = _client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=800,
        system=[{"type": "text", "text": _SOIL_SYSTEM, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": context}],
    )

    if cost_tracker:
        cost_tracker.add(usage_from_anthropic_response("claude-haiku-4-5-20251001", message.usage))

    return {
        **state,
        "soil_result": {
            "health_class": result.health_class,
            "health_score": result.health_score,
            "deficiencies": result.deficiencies,
            "excesses": result.excesses,
            "suitable_crops": result.suitable_crops,
            "amendments": result.amendments,
        },
        "final_response": message.content[0].text,
        "agent_trace": [f"SoilAgent → score={result.health_score} ({result.health_class}), Claude advisory"],
    }
