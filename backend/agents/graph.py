"""
LangGraph DAG — full Kisaan AI agent pipeline.

             ┌─────────────────────────────────────────────────┐
             │              INTENT ROUTER                       │
             │  (ZSC bart-large-mnli + NER bert-base-NER)      │
             └──┬──────┬──────┬──────┬──────┬──────────────────┘
                │      │      │      │      │
           disease  market  scheme  soil   voice  advisory
                │      │      │      │      │      │
         ┌──────▼──┐   │      │      │      │      │
         │ Vision  │   │      │      │      │      │
         │ Agent   │   │      │      │      │      │
         │ (HF     │   │      │      │      │      │
         │ disease │   │      │      │      │      │
         │ classif)│   │      │      │      │      │
         └──┬──────┘   │      │      │      │      │
            │          │      │      │      │      │
         ┌──▼──────┐   │      │      │      │  ┌───▼──────┐
         │ Object  │   │      │      │      │  │Advisory  │
         │ Detect  │   │      │      │      │  │ Agent    │
         │ (DETR)  │   │      │      │      │  │(RAG+     │
         └──┬──────┘   │      │      │      │  │ Claude)  │
            │          │      │      │      │  └──────────┘
         ┌──▼──────┐   │   ┌──▼──┐ ┌▼────┐ │
         │Advisory │   │   │Schm │ │Soil │ │
         │ Agent   │   │   │Agent│ │Agent│ │
         └─────────┘   │   └─────┘ └─────┘ │
                    ┌───▼────┐           ┌──▼──────────┐
                    │Market  │           │Voice Pre+   │
                    │ Agent  │           │Post Process │
                    └────────┘           └─────────────┘
"""
import logging
import uuid

from langgraph.graph import END, StateGraph

from backend.agents.advisory_agent import advisory_agent
from backend.agents.market_agent import market_agent
from backend.agents.router import next_node, route_intent
from backend.agents.scheme_agent import scheme_agent
from backend.agents.soil_agent import soil_agent
from backend.agents.state import KisaanState
from backend.agents.vision_agent import vision_agent
from backend.agents.voice_agent import voice_postprocess, voice_preprocess
from backend.services.cost_tracker import SessionCost
from backend.services.object_detection import detect_objects

logger = logging.getLogger(__name__)


# ── Composite node: Vision + Object Detection + Advisory ─────────────────────

def disease_path(state: KisaanState) -> KisaanState:
    state = vision_agent(state)

    # Run object detection in parallel with advisory (pest enrichment)
    if state.get("image_bytes"):
        try:
            det = detect_objects(state["image_bytes"])
            state = {
                **state,
                "detections": det.detections,
                "pest_flags": det.pest_flags,
                "detection_summary": det.annotated_summary,
                "agent_trace": state.get("agent_trace", []) + [
                    f"ObjectDetect[DETR] → {len(det.detections)} objects, pests={det.pest_flags}"
                ],
            }
        except Exception as exc:
            logger.warning("Object detection failed (non-fatal): %s", exc)

    return advisory_agent(state)


# ── Voice path: ASR → translate → route → advisory → TTS ────────────────────

def voice_path(state: KisaanState) -> KisaanState:
    state = voice_preprocess(state)
    # Re-route based on the transcribed text
    from backend.agents.router import route_intent as re_route
    state = re_route(state)
    intent = state.get("intent", "advisory")

    if intent == "disease":
        state = disease_path(state)
    elif intent == "market":
        state = market_agent(state)
    elif intent == "scheme":
        state = scheme_agent(state)
    elif intent == "soil":
        state = soil_agent(state)
    else:
        state = advisory_agent(state)

    return voice_postprocess(state)


# ── Graph construction ────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    graph = StateGraph(KisaanState)

    graph.add_node("router", route_intent)
    graph.add_node("disease", disease_path)
    graph.add_node("market", market_agent)
    graph.add_node("scheme", scheme_agent)
    graph.add_node("soil", soil_agent)
    graph.add_node("advisory", advisory_agent)
    graph.add_node("voice", voice_path)

    graph.set_entry_point("router")
    graph.add_conditional_edges("router", next_node, {
        "disease":  "disease",
        "market":   "market",
        "scheme":   "scheme",
        "soil":     "soil",
        "voice":    "voice",
        "advisory": "advisory",
        "unknown":  "advisory",
    })

    for node in ["disease", "market", "scheme", "soil", "advisory", "voice"]:
        graph.add_edge(node, END)

    return graph.compile()


kisaan_graph = build_graph()


def run_graph(initial_state: dict, farmer_id: str = "") -> dict:
    """Entry point that injects a SessionCost tracker and optional farmer_id."""
    session_id = initial_state.get("session_id", str(uuid.uuid4()))
    cost_tracker = SessionCost(session_id=session_id)
    return kisaan_graph.invoke({
        **initial_state,
        "cost_tracker": cost_tracker,
        "farmer_id": farmer_id,
        "agent_trace": [],
    })
