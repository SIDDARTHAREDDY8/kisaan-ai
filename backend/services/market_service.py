"""
Fetches commodity prices from India's Agmarknet Open API (free, no key needed).
Falls back to mock data for demo when API is unreachable.
"""
import logging
from datetime import date, timedelta

import httpx

logger = logging.getLogger(__name__)

AGMARKNET_URL = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"
AGMARKNET_API_KEY = "579b464db66ec23bdd000001cdd3946e44ce4aad38d976ea3dc2317"

# Commodity name aliases — Agmarknet uses specific spellings
COMMODITY_ALIASES = {
    "tomato": "Tomato",
    "wheat": "Wheat",
    "rice": "Rice",
    "onion": "Onion",
    "potato": "Potato",
    "maize": "Maize",
    "cotton": "Cotton(Lint)",
    "soybean": "Soyabean",
    "groundnut": "Groundnut",
    "sugarcane": "Sugarcane",
    "chilli": "Dry Chillies",
    "banana": "Banana",
    "mango": "Mango",
}

# Fallback demo data so the UI always shows something useful
def _today():
    return date.today().strftime("%d/%m/%Y")

DEMO_PRICES = {
    "tomato": [
        {"commodity": "Tomato", "market": "Bowenpally", "state": "Telangana",
         "district": "Hyderabad", "min_price": 700, "max_price": 1400, "modal_price": 1050,
         "arrival_date": _today(), "unit": "Quintal"},
        {"commodity": "Tomato", "market": "Karimnagar", "state": "Telangana",
         "district": "Karimnagar", "min_price": 600, "max_price": 1300, "modal_price": 950,
         "arrival_date": _today(), "unit": "Quintal"},
        {"commodity": "Tomato", "market": "Warangal", "state": "Telangana",
         "district": "Warangal", "min_price": 650, "max_price": 1250, "modal_price": 900,
         "arrival_date": _today(), "unit": "Quintal"},
        {"commodity": "Tomato", "market": "Azadpur", "state": "Delhi",
         "district": "Delhi", "min_price": 800, "max_price": 1400, "modal_price": 1100,
         "arrival_date": _today(), "unit": "Quintal"},
        {"commodity": "Tomato", "market": "Kolar", "state": "Karnataka",
         "district": "Kolar", "min_price": 600, "max_price": 1200, "modal_price": 900,
         "arrival_date": _today(), "unit": "Quintal"},
    ],
    "onion": [
        {"commodity": "Onion", "market": "Lasalgaon", "state": "Maharashtra",
         "district": "Nashik", "min_price": 1200, "max_price": 2000, "modal_price": 1600,
         "arrival_date": _today(), "unit": "Quintal"},
        {"commodity": "Onion", "market": "Hyderabad", "state": "Telangana",
         "district": "Hyderabad", "min_price": 1000, "max_price": 1800, "modal_price": 1400,
         "arrival_date": _today(), "unit": "Quintal"},
    ],
    "wheat": [
        {"commodity": "Wheat", "market": "Khanna", "state": "Punjab",
         "district": "Ludhiana", "min_price": 2015, "max_price": 2200, "modal_price": 2100,
         "arrival_date": _today(), "unit": "Quintal"},
    ],
    "rice": [
        {"commodity": "Rice", "market": "Nizamabad", "state": "Telangana",
         "district": "Nizamabad", "min_price": 1800, "max_price": 2400, "modal_price": 2100,
         "arrival_date": _today(), "unit": "Quintal"},
        {"commodity": "Rice", "market": "Nalgonda", "state": "Telangana",
         "district": "Nalgonda", "min_price": 1750, "max_price": 2350, "modal_price": 2050,
         "arrival_date": _today(), "unit": "Quintal"},
    ],
    "chilli": [
        {"commodity": "Dry Chillies", "market": "Khammam", "state": "Telangana",
         "district": "Khammam", "min_price": 8000, "max_price": 14000, "modal_price": 11000,
         "arrival_date": _today(), "unit": "Quintal"},
        {"commodity": "Dry Chillies", "market": "Guntur", "state": "Andhra Pradesh",
         "district": "Guntur", "min_price": 9000, "max_price": 15000, "modal_price": 12000,
         "arrival_date": _today(), "unit": "Quintal"},
    ],
    "cotton": [
        {"commodity": "Cotton(Lint)", "market": "Adilabad", "state": "Telangana",
         "district": "Adilabad", "min_price": 5800, "max_price": 6500, "modal_price": 6200,
         "arrival_date": _today(), "unit": "Quintal"},
    ],
    "maize": [
        {"commodity": "Maize", "market": "Nizamabad", "state": "Telangana",
         "district": "Nizamabad", "min_price": 1400, "max_price": 1900, "modal_price": 1650,
         "arrival_date": _today(), "unit": "Quintal"},
    ],
    "groundnut": [
        {"commodity": "Groundnut", "market": "Nalgonda", "state": "Telangana",
         "district": "Nalgonda", "min_price": 4500, "max_price": 6000, "modal_price": 5200,
         "arrival_date": _today(), "unit": "Quintal"},
    ],
}


def _filter_demo(commodity_lower: str, state: str) -> list[dict]:
    """Return demo prices, prioritising the requested state if specified."""
    all_prices = DEMO_PRICES.get(commodity_lower, [])
    if not state or not all_prices:
        return all_prices
    state_lower = state.lower()
    # Put requested-state markets first, keep others as fallback
    matching = [p for p in all_prices if state_lower in p["state"].lower()]
    others = [p for p in all_prices if state_lower not in p["state"].lower()]
    return (matching + others)[:8]


async def fetch_mandi_prices(commodity: str, state: str = "") -> list[dict]:
    # Normalise commodity name
    commodity_lower = commodity.lower().strip()
    agmarknet_name = COMMODITY_ALIASES.get(commodity_lower, commodity.title())

    params = {
        "api-key": AGMARKNET_API_KEY,
        "format": "json",
        "limit": "20",
        "filters[commodity]": agmarknet_name,
    }
    if state:
        params["filters[state]"] = state.title()

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(AGMARKNET_URL, params=params)
            resp.raise_for_status()
            data = resp.json()
            records = data.get("records", [])

            if records:
                return [
                    {
                        "commodity": r.get("commodity"),
                        "market": r.get("market"),
                        "state": r.get("state"),
                        "district": r.get("district"),
                        "min_price": _to_float(r.get("min_price")),
                        "max_price": _to_float(r.get("max_price")),
                        "modal_price": _to_float(r.get("modal_price")),
                        "arrival_date": r.get("arrival_date"),
                        "unit": "Quintal",
                    }
                    for r in records[:10]
                ]

            # API returned 0 records — use demo data
            logger.info("Agmarknet returned 0 records for %s, using demo data", agmarknet_name)
            return _filter_demo(commodity_lower, state)

    except Exception as exc:
        logger.warning("Agmarknet fetch failed for %s: %s — using demo data", agmarknet_name, exc)
        return _filter_demo(commodity_lower, state)


def _to_float(val) -> float | None:
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


async def get_weather_advisory(lat: float, lon: float, api_key: str) -> dict:
    if not api_key:
        return {"advisory": "Weather API key not configured."}
    url = "https://api.openweathermap.org/data/2.5/forecast"
    params = {"lat": lat, "lon": lon, "appid": api_key, "units": "metric", "cnt": 5}
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
            forecasts = data.get("list", [])[:3]
            summaries = [
                f"{f['weather'][0]['description']}, {f['main']['temp']}°C, rain: {f.get('rain', {}).get('3h', 0)}mm"
                for f in forecasts
            ]
            return {"forecast": summaries}
    except Exception as exc:
        logger.warning("Weather fetch failed: %s", exc)
        return {"advisory": "Weather data temporarily unavailable."}
