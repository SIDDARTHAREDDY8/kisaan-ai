"""
Scheme Agent — RAG over Indian govt agricultural scheme knowledge base.
Answers questions about PM-KISAN, PMFBY, KCC, eNAM, PMKSY, etc.
Uses Claude for eligibility matching and plain-language explanation.
"""
import logging

import anthropic

from backend.agents.state import KisaanState
from backend.config import settings
from backend.rag.vector_store import retrieve_schemes
from backend.services.cost_tracker import SessionCost, usage_from_anthropic_response

logger = logging.getLogger(__name__)
_client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

_SCHEME_SYSTEM = """You are Kisaan AI's scheme navigator. Your job is to help smallholder farmers understand and access Indian government agricultural schemes.

Rules:
- Only recommend schemes for which the farmer appears eligible based on their query
- Explain eligibility criteria in simple language — no bureaucratic jargon
- Give concrete next steps: exactly where to go, what to bring, what to say
- Mention the scheme helpline number whenever available
- If the query is unclear, ask one clarifying question about land size or crop type
- Keep responses concise: 3-5 bullet points per scheme, then next steps"""


def scheme_agent(state: KisaanState) -> KisaanState:
    query = state.get("user_query", "")
    cost_tracker: SessionCost = state.get("cost_tracker")

    docs = retrieve_schemes(query, top_k=3)
    context = "\n\n".join(
        f"### {d['scheme_name']} ({d['category']})\n"
        f"Benefit: {d['benefit']}\n"
        f"Eligibility: {d['eligibility']}\n"
        f"Documents needed: {d['documents']}\n"
        f"How to apply: {d['how_to_apply']}\n"
        f"Contact: {d['contact']}"
        for d in docs
    )

    message = _client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=[{"type": "text", "text": _SCHEME_SYSTEM, "cache_control": {"type": "ephemeral"}}],
        messages=[{
            "role": "user",
            "content": f"Farmer query: {query}\n\nRelevant schemes from knowledge base:\n{context or 'No specific schemes found — provide general guidance.'}"
        }],
    )

    if cost_tracker:
        cost_tracker.add(usage_from_anthropic_response("claude-sonnet-4-6", message.usage))

    return {
        **state,
        "retrieved_docs": docs,
        "final_response": message.content[0].text,
        "agent_trace": [f"SchemeAgent → retrieved {len(docs)} schemes, Claude advisory"],
    }
