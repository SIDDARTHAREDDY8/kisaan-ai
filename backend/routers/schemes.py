from fastapi import APIRouter, Query
from pydantic import BaseModel

from backend.agents.graph import run_graph

router = APIRouter(prefix="/api", tags=["schemes"])


class SchemeResponse(BaseModel):
    response: str
    retrieved_docs: list[dict]
    agent_trace: list[str]


@router.get("/schemes", response_model=SchemeResponse)
async def query_schemes(q: str = Query(..., description="Farmer's question about govt schemes")):
    initial_state = {
        "user_query": q,
        "intent": "scheme",
        "agent_trace": [],
    }
    result = run_graph(initial_state)
    return SchemeResponse(
        response=result.get("final_response", ""),
        retrieved_docs=result.get("retrieved_docs", []),
        agent_trace=result.get("agent_trace", []),
    )
