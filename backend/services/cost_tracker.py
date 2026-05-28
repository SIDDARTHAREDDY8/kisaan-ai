"""
LLM Cost Tracker — logs token usage and computes per-query USD cost.
Stored in the analysis_sessions table for LLMOps dashboard queries.
"""
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Pricing as of 2026 (USD per 1M tokens)
PRICING = {
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
    "claude-haiku-4-5-20251001": {"input": 0.25, "output": 1.25},
}


@dataclass
class TokenUsage:
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    @property
    def cost_usd(self) -> float:
        rates = PRICING.get(self.model, {"input": 3.0, "output": 15.0})
        input_cost = (self.input_tokens / 1_000_000) * rates["input"]
        output_cost = (self.output_tokens / 1_000_000) * rates["output"]
        # Cache reads are ~10% of input price
        cache_cost = (self.cache_read_tokens / 1_000_000) * rates["input"] * 0.1
        return round(input_cost + output_cost + cache_cost, 6)


@dataclass
class SessionCost:
    session_id: str
    usages: list[TokenUsage] = field(default_factory=list)

    def add(self, usage: TokenUsage) -> None:
        self.usages.append(usage)
        logger.info(
            "Token usage [%s] %s: in=%d out=%d cache_read=%d → $%.5f",
            self.session_id, usage.model,
            usage.input_tokens, usage.output_tokens,
            usage.cache_read_tokens, usage.cost_usd,
        )

    @property
    def total_cost_usd(self) -> float:
        return round(sum(u.cost_usd for u in self.usages), 6)

    @property
    def total_tokens(self) -> int:
        return sum(u.input_tokens + u.output_tokens for u in self.usages)

    def summary(self) -> dict:
        return {
            "session_id": self.session_id,
            "total_cost_usd": self.total_cost_usd,
            "total_tokens": self.total_tokens,
            "by_model": [
                {"model": u.model, "cost_usd": u.cost_usd,
                 "input_tokens": u.input_tokens, "output_tokens": u.output_tokens}
                for u in self.usages
            ],
        }


def usage_from_anthropic_response(model: str, usage_obj) -> TokenUsage:
    """Parse token usage from an anthropic.types.Usage object."""
    return TokenUsage(
        model=model,
        input_tokens=getattr(usage_obj, "input_tokens", 0),
        output_tokens=getattr(usage_obj, "output_tokens", 0),
        cache_read_tokens=getattr(usage_obj, "cache_read_input_tokens", 0),
        cache_write_tokens=getattr(usage_obj, "cache_creation_input_tokens", 0),
    )
