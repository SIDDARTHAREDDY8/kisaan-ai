"""
Advisory Agent — RAG retrieval + Claude reasoning.

Flow:
  1. Retrieve top-3 treatment docs from pgvector
  2. Build a context-rich prompt with retrieved docs + classifier output
  3. Call Claude claude-sonnet-4-6 with prompt caching on the system prompt
  4. Return structured advice
"""
import logging

import anthropic

from backend.agents.state import KisaanState
from backend.config import settings
from backend.rag.vector_store import retrieve_treatment
from backend.services.cost_tracker import SessionCost, usage_from_anthropic_response

logger = logging.getLogger(__name__)

_client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

_SYSTEM_PROMPT = """You are Kisaan AI, an expert agricultural advisor helping smallholder farmers in India and across the world.

Your role:
- Diagnose crop diseases and explain them in simple, practical language
- Provide actionable treatment and prevention advice
- Be direct and specific — farmers need clear steps, not vague suggestions
- When uncertain, say so and ask for more information
- Keep responses concise: 3-4 paragraphs maximum
- Use metric units (kg, liters per acre) and locally available treatments when possible

Always structure your response as:
1. **Diagnosis** — what is detected and how certain you are
2. **Immediate Action** — what the farmer should do in the next 24-48 hours
3. **Treatment Plan** — specific steps with dosages if applicable
4. **Prevention** — how to avoid this next season"""


def advisory_agent(state: KisaanState) -> KisaanState:
    crop = state.get("crop", "")
    condition = state.get("condition", "")
    confidence = state.get("classifier_confidence", 0.0)
    user_query = state.get("user_query", "")
    error = state.get("error")
    cost_tracker: SessionCost = state.get("cost_tracker")

    # Low confidence path — use classifier's specific follow-up question if available
    if error == "low_confidence":
        follow_up = state.get("follow_up_question", "")
        if follow_up:
            response = follow_up
        elif condition and condition not in ("Unknown", ""):
            response = (
                f"I can see symptoms in your photo that might be **{condition}** on **{crop}** "
                f"({confidence:.0%} confidence), but I'm not certain enough to recommend treatment.\n\n"
                "Please send:\n"
                "1. A closer photo of the affected leaf or stem in natural daylight\n"
                "2. A photo showing the underside of the leaf as well"
            )
        else:
            response = (
                "I couldn't clearly identify the disease from this photo.\n\n"
                "Please send:\n"
                "1. A closer, well-lit photo of the affected plant part\n"
                "2. Which crop this is and what symptoms you first noticed"
            )
        return {
            **state,
            "final_response": response,
            "agent_trace": ["AdvisoryAgent → low_confidence, asking follow-up"],
        }

    # RAG retrieval
    docs = retrieve_treatment(condition, crop, top_k=3)
    context_chunks = "\n\n".join(
        f"### {d['disease_name']} ({d['crop']})\n"
        f"Symptoms: {d['symptoms']}\n"
        f"Cause: {d['cause']}\n"
        f"Treatment: {d['treatment']}\n"
        f"Prevention: {d['prevention']}\n"
        f"Severity: {d['severity']}"
        for d in docs
    )

    # Enrich with taxonomy treatment data if available
    disease_id = state.get("classifier_label", "")
    taxonomy_info = ""
    try:
        import json
        from pathlib import Path
        tax_path = Path(__file__).parent.parent.parent / "data" / "diseases" / "taxonomy.json"
        with open(tax_path) as f:
            diseases = json.load(f)["diseases"]
        # Match by disease_id stored in top5 or by condition name
        matched = next(
            (d for d in diseases if d["english"].lower() == condition.lower()
             or d.get("plantvillage_label", "").replace("___", " ").lower() in disease_id.lower()),
            None,
        )
        if matched:
            taxonomy_info = (
                f"\nTaxonomy data for {matched['english']} ({matched['crop']}):\n"
                f"  Symptoms: {matched.get('symptoms',{}).get('en','')}\n"
                f"  Treatment: {matched.get('treatment',{}).get('en','')}\n"
                f"  Prevention: {matched.get('prevention',{}).get('en','')}\n"
                f"  Severity: {matched.get('severity','')}"
            )
    except Exception:
        pass

    user_message = (
        f"Crop: {crop}\n"
        f"Detected condition: {condition} (confidence: {confidence:.0%})\n"
        f"Additional context from farmer: {user_query or 'none'}\n"
        f"{taxonomy_info}\n\n"
        f"Retrieved knowledge base entries:\n{context_chunks or 'No matching records found.'}"
    )

    try:
        message = _client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=[
                {
                    "type": "text",
                    "text": _SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},  # prompt caching
                }
            ],
            messages=[{"role": "user", "content": user_message}],
        )
        response_text = message.content[0].text
        severity = next((d["severity"] for d in docs if d["similarity"] > 0.7), "medium")

        if cost_tracker:
            cost_tracker.add(usage_from_anthropic_response("claude-sonnet-4-6", message.usage))

        return {
            **state,
            "retrieved_docs": docs,
            "final_response": response_text,
            "severity": severity,
            "cost_summary": cost_tracker.summary() if cost_tracker else {},
            "agent_trace": [
                f"AdvisoryAgent → retrieved {len(docs)} docs, called Claude "
                f"(in={message.usage.input_tokens}, out={message.usage.output_tokens}, "
                f"cache_read={getattr(message.usage, 'cache_read_input_tokens', 0)}) "
                f"cost=${cost_tracker.total_cost_usd:.5f}" if cost_tracker else ""
            ],
        }
    except Exception as exc:
        logger.exception("Advisory agent Claude call failed")
        return {
            **state,
            "retrieved_docs": docs,
            "final_response": f"Advisory service temporarily unavailable. Based on your image, possible issue: {condition} on {crop}.",
            "agent_trace": [f"AdvisoryAgent → ERROR: {exc}"],
        }
