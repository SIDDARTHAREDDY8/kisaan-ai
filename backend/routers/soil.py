from fastapi import APIRouter
from pydantic import BaseModel

from backend.agents.graph import run_graph

router = APIRouter(prefix="/api", tags=["soil"])


class SoilRequest(BaseModel):
    nitrogen: float = 280.0
    phosphorus: float = 15.0
    potassium: float = 150.0
    ph: float = 7.0
    organic_carbon: float = 0.6
    moisture: float = 20.0
    farmer_id: str = ""


class SoilResponse(BaseModel):
    health_class: str
    health_score: float
    deficiencies: list[str]
    excesses: list[str]
    suitable_crops: list[str]
    amendments: list[str]
    response: str
    agent_trace: list[str]


@router.post("/soil/analyze", response_model=SoilResponse)
async def soil_analyze(req: SoilRequest):
    initial_state = {
        "user_query": "Analyze my soil health",
        "intent": "soil",
        "soil_params": {
            "nitrogen": req.nitrogen,
            "phosphorus": req.phosphorus,
            "potassium": req.potassium,
            "ph": req.ph,
            "organic_carbon": req.organic_carbon,
            "moisture": req.moisture,
        },
        "agent_trace": [],
    }
    result = run_graph(initial_state, farmer_id=req.farmer_id)
    soil = result.get("soil_result", {})
    return SoilResponse(
        health_class=soil.get("health_class", "Unknown"),
        health_score=soil.get("health_score", 0),
        deficiencies=soil.get("deficiencies", []),
        excesses=soil.get("excesses", []),
        suitable_crops=soil.get("suitable_crops", []),
        amendments=soil.get("amendments", []),
        response=result.get("final_response", ""),
        agent_trace=result.get("agent_trace", []),
    )
