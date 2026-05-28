from fastapi import APIRouter
from pydantic import BaseModel

from backend.services.market_service import fetch_mandi_prices

router = APIRouter(prefix="/api", tags=["market"])


class MarketResponse(BaseModel):
    commodity: str
    state: str
    prices: list[dict]


@router.get("/market", response_model=MarketResponse)
async def get_market(commodity: str = "Tomato", state: str = ""):
    prices = await fetch_mandi_prices(commodity, state)
    return MarketResponse(commodity=commodity, state=state, prices=prices)
