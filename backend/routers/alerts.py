"""
Proactive Alerts endpoint.
GET /api/alerts?commodity=Tomato&state=Maharashtra
  → checks if current modal price is >15% above 7-day average → triggers alert

POST /api/alerts/register  → register a farmer for price/weather push notifications
  (stores in farmer_sessions table)

In production, a Celery beat worker would poll this on a schedule.
Here we expose it as an on-demand HTTP endpoint for demo purposes.
"""
import logging
from datetime import datetime

from fastapi import APIRouter
from pydantic import BaseModel

from backend.services.market_service import fetch_mandi_prices
from backend.services.time_series import forecast_prices

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["alerts"])


class AlertResult(BaseModel):
    commodity: str
    state: str
    alert: bool
    alert_type: str
    message: str
    sell_recommendation: str
    current_modal_price: float | None
    forecast_7d: list[float]


@router.get("/alerts", response_model=AlertResult)
async def check_price_alert(commodity: str = "Tomato", state: str = ""):
    prices = await fetch_mandi_prices(commodity, state)

    if not prices:
        return AlertResult(
            commodity=commodity, state=state, alert=False,
            alert_type="no_data", message="No market data available.",
            sell_recommendation="Check local mandi.", current_modal_price=None, forecast_7d=[],
        )

    modal_prices = [p["modal_price"] for p in prices if p.get("modal_price")]
    current = modal_prices[0] if modal_prices else None

    forecast = forecast_prices(commodity, modal_prices)

    # Spike detection: current price > 15% above mean
    if len(modal_prices) >= 2:
        avg = sum(modal_prices[1:]) / len(modal_prices[1:])
        spike = current and current > avg * 1.15
    else:
        spike = False

    if spike:
        alert_type = "price_spike"
        message = f"⚡ Price spike detected: {commodity} is ₹{current:.0f}/quintal, {((current/avg - 1)*100):.0f}% above recent average (₹{avg:.0f})."
    elif forecast.trend == "rising":
        alert_type = "rising_trend"
        message = f"📈 {commodity} prices trending upward in {state or 'your region'}. Consider waiting 3-5 days before selling."
    else:
        alert_type = "none"
        message = f"No significant price alert for {commodity} today."

    return AlertResult(
        commodity=commodity,
        state=state,
        alert=alert_type != "none",
        alert_type=alert_type,
        message=message,
        sell_recommendation=forecast.sell_recommendation,
        current_modal_price=current,
        forecast_7d=forecast.forecast_7d,
    )


class FarmerRegistration(BaseModel):
    farmer_id: str
    commodities: list[str]
    location: str
    language: str = "en"


@router.post("/alerts/register")
async def register_farmer(reg: FarmerRegistration):
    """Register farmer for proactive alerts (stores preferences in DB)."""
    from backend.db.models import FarmerSession, SyncSessionLocal
    with SyncSessionLocal() as session:
        existing = session.query(FarmerSession).filter_by(farmer_id=reg.farmer_id).first()
        if existing:
            existing.location = reg.location
            existing.language = reg.language
            existing.last_seen = datetime.utcnow()
        else:
            session.add(FarmerSession(
                farmer_id=reg.farmer_id,
                location=reg.location,
                language=reg.language,
                crop_history=reg.commodities,
            ))
        session.commit()
    return {"status": "registered", "farmer_id": reg.farmer_id}
