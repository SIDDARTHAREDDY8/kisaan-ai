"""
Optional LangSmith tracing integration.
Set LANGSMITH_API_KEY in backend/.env to enable.
Without the key, all functions are no-ops — no import errors.
"""
import logging
import os

logger = logging.getLogger(__name__)

_enabled = False

try:
    if os.getenv("LANGSMITH_API_KEY"):
        from langsmith import Client
        from langchain.callbacks.tracers import LangChainTracer

        _ls_client = Client()
        _enabled = True
        logger.info("LangSmith tracing enabled.")
    else:
        logger.info("LANGSMITH_API_KEY not set — tracing disabled.")
except ImportError:
    logger.info("langsmith package not installed — tracing disabled.")


def get_tracer(project: str = "kisaan-ai"):
    """Return a LangChainTracer if enabled, else None."""
    if not _enabled:
        return None
    try:
        return LangChainTracer(project_name=project)
    except Exception:
        return None


def log_run(
    session_id: str,
    intent: str,
    latency_ms: float,
    cost_usd: float,
    agent_trace: list[str],
) -> None:
    """Log a completed agent run to LangSmith as a feedback record."""
    if not _enabled:
        return
    try:
        _ls_client.create_run(
            name=f"kisaan-{intent}",
            run_type="chain",
            inputs={"session_id": session_id, "intent": intent},
            outputs={"agent_trace": agent_trace},
            extra={
                "latency_ms": latency_ms,
                "cost_usd": cost_usd,
            },
        )
    except Exception as exc:
        logger.debug("LangSmith log_run failed (non-fatal): %s", exc)
