import time
import uuid
import logging
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from backend.agents.graph import kisaan_graph
from backend.agents.state import KisaanState

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["analyze"])

MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10 MB


class AnalyzeResponse(BaseModel):
    session_id: str
    intent: str
    crop: str
    condition: str
    confidence: float
    severity: str
    response: str
    top5: list[dict]
    retrieved_docs: list[dict]
    agent_trace: list[str]
    latency_ms: float


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    image: Optional[UploadFile] = File(None),
    query: str = Form(""),
    commodity: str = Form(""),
    location: str = Form(""),
):
    session_id = str(uuid.uuid4())
    start = time.perf_counter()

    image_bytes = None
    if image and image.filename:
        raw = await image.read()
        if len(raw) > MAX_IMAGE_SIZE:
            raise HTTPException(status_code=413, detail="Image too large (max 10 MB)")
        image_bytes = raw

    if not image_bytes and not query and not commodity:
        raise HTTPException(status_code=400, detail="Provide an image or a query.")

    initial_state: KisaanState = {
        "image_bytes": image_bytes,
        "user_query": query,
        "commodity": commodity,
        "location": location,
        "intent": "",
        "classifier_label": "",
        "classifier_confidence": 0.0,
        "crop": "",
        "condition": "",
        "top5": [],
        "retrieved_docs": [],
        "treatment_advice": "",
        "prevention_tips": "",
        "severity": "medium",
        "market_data": [],
        "final_response": "",
        "agent_trace": [],
        "error": None,
    }

    try:
        result = kisaan_graph.invoke(initial_state)
    except Exception as exc:
        logger.exception("Graph invocation failed for session %s", session_id)
        raise HTTPException(status_code=500, detail=str(exc))

    latency_ms = (time.perf_counter() - start) * 1000
    logger.info("Session %s completed in %.0fms", session_id, latency_ms)

    return AnalyzeResponse(
        session_id=session_id,
        intent=result.get("intent", ""),
        crop=result.get("crop", ""),
        condition=result.get("condition", ""),
        confidence=result.get("classifier_confidence", 0.0),
        severity=result.get("severity", "medium"),
        response=result.get("final_response", ""),
        top5=result.get("top5", []),
        retrieved_docs=result.get("retrieved_docs", []),
        agent_trace=result.get("agent_trace", []),
        latency_ms=round(latency_ms, 1),
    )
