"""
Voice endpoint — accepts audio upload, returns text response + WAV audio.
Supports multilingual voice queries via Whisper ASR + NLLB translation + MMS TTS.
"""
import time
import uuid
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel
from typing import Optional

from backend.agents.graph import run_graph

router = APIRouter(prefix="/api", tags=["voice"])

MAX_AUDIO_SIZE = 25 * 1024 * 1024  # 25 MB


class VoiceTextResponse(BaseModel):
    session_id: str
    transcript: str
    detected_language: str
    intent: str
    response: str
    response_translated: str
    agent_trace: list[str]
    latency_ms: float
    has_audio: bool


@router.post("/voice/analyze", response_model=VoiceTextResponse)
async def voice_analyze(
    audio: UploadFile = File(...),
    farmer_id: str = Form(""),
    image: Optional[UploadFile] = File(None),
):
    """Submit an audio voice note. Returns JSON with text response."""
    session_id = str(uuid.uuid4())
    start = time.perf_counter()

    raw = await audio.read()
    if len(raw) > MAX_AUDIO_SIZE:
        raise HTTPException(status_code=413, detail="Audio too large (max 25 MB)")

    image_bytes = None
    if image and image.filename:
        image_bytes = await image.read()

    initial_state = {
        "session_id": session_id,
        "audio_bytes": raw,
        "audio_filename": audio.filename or "audio.wav",
        "image_bytes": image_bytes,
        "user_query": "",
        "intent": "voice",
        "agent_trace": [],
    }

    try:
        result = run_graph(initial_state, farmer_id=farmer_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    latency_ms = (time.perf_counter() - start) * 1000
    return VoiceTextResponse(
        session_id=session_id,
        transcript=result.get("original_query", ""),
        detected_language=result.get("detected_language", "en"),
        intent=result.get("intent", ""),
        response=result.get("final_response", ""),
        response_translated=result.get("response_translated", result.get("final_response", "")),
        agent_trace=result.get("agent_trace", []),
        latency_ms=round(latency_ms, 1),
        has_audio=bool(result.get("response_audio")),
    )


@router.post("/voice/audio")
async def voice_audio(audio: UploadFile = File(...), farmer_id: str = Form("")):
    """Submit audio — returns WAV audio response directly (for playback)."""
    session_id = str(uuid.uuid4())
    raw = await audio.read()

    initial_state = {
        "session_id": session_id,
        "audio_bytes": raw,
        "audio_filename": audio.filename or "audio.wav",
        "user_query": "",
        "intent": "voice",
        "agent_trace": [],
    }

    try:
        result = run_graph(initial_state, farmer_id=farmer_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    audio_bytes = result.get("response_audio")
    if not audio_bytes:
        raise HTTPException(status_code=500, detail="TTS synthesis failed")

    return Response(content=audio_bytes, media_type="audio/wav")
