"""
Market Agent — fetches live mandi prices from Agmarknet API
and generates sell-window advice via Claude.
"""
import asyncio
import logging

import anthropic

from backend.agents.state import KisaanState
from backend.config import settings
from backend.services.market_service import fetch_mandi_prices

logger = logging.getLogger(__name__)
_client = anthropic.Anthropic(api_key=settings.anthropic_api_key)


def market_agent(state: KisaanState) -> KisaanState:
    commodity = state.get("commodity") or state.get("crop") or state.get("user_query_en") or state.get("user_query", "tomato")
    location = state.get("location", "")

    # Fetch prices synchronously (LangGraph nodes are sync)
    try:
        prices = asyncio.run(fetch_mandi_prices(commodity, location))
    except RuntimeError:
        # Already inside an event loop (shouldn't happen in LangGraph sync nodes)
        prices = []

    if not prices:
        return {
            **state,
            "market_data": [],
            "final_response": (
                f"I couldn't fetch live prices for **{commodity}** right now. "
                "Please check your local mandi or the eNAM portal (enam.gov.in) for current rates."
            ),
            "agent_trace": ["MarketAgent → no data from Agmarknet"],
        }

    # Format for Claude
    price_summary = "\n".join(
        f"- {p['market']}, {p['state']}: ₹{p['modal_price']}/quintal "
        f"(min ₹{p['min_price']}, max ₹{p['max_price']}) on {p['arrival_date']}"
        for p in prices[:8]
    )

    location_ctx = f" in {location}" if location else ""
    message = _client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        messages=[
            {
                "role": "user",
                "content": (
                    f"Summarize these {commodity} mandi prices for a smallholder farmer{location_ctx}. "
                    f"The first markets listed are closest to the farmer's region. "
                    f"Give a clear sell recommendation (sell now / wait / split sell) with reasoning. "
                    f"Mention which specific markets show the best prices. "
                    f"Keep it to 3-4 sentences, plain language.\n\n{price_summary}"
                ),
            }
        ],
    )
    advice = message.content[0].text

    return {
        **state,
        "market_data": prices,
        "final_response": advice,
        "agent_trace": [f"MarketAgent → fetched {len(prices)} markets, generated advice"],
    }
