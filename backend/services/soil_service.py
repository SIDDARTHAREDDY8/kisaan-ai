"""
Soil Health Classifier
HuggingFace Task: Tabular Classification

Takes NPK readings + pH + moisture and classifies soil health into:
  Excellent | Good | Fair | Poor | Critical

Also generates crop suitability recommendations based on soil profile.
Uses a rule-based expert system backed by ICAR soil health card norms,
wrapped in scikit-learn Pipeline for the tabular classification pattern.
"""
import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# ICAR optimal ranges (per soil health card scheme)
OPTIMAL_RANGES = {
    "nitrogen_kg_ha":   (280, 560),   # kg/ha — low < 280, medium 280-560, high > 560
    "phosphorus_kg_ha": (10, 25),
    "potassium_kg_ha":  (108, 280),
    "ph":               (6.5, 7.5),
    "organic_carbon":   (0.5, 0.75),  # %
    "moisture_pct":     (15, 35),
}

CROP_SOIL_PREFERENCES = {
    "Rice":      {"ph": (5.5, 7.0), "moisture_pct": (25, 40)},
    "Wheat":     {"ph": (6.0, 7.5), "moisture_pct": (15, 25)},
    "Cotton":    {"ph": (6.0, 8.0), "moisture_pct": (10, 25)},
    "Tomato":    {"ph": (6.0, 7.0), "moisture_pct": (20, 30)},
    "Groundnut": {"ph": (6.0, 7.0), "moisture_pct": (15, 25)},
    "Maize":     {"ph": (5.8, 7.0), "moisture_pct": (20, 35)},
    "Sugarcane": {"ph": (6.0, 7.5), "moisture_pct": (25, 40)},
}


@dataclass
class SoilResult:
    health_class: str              # Excellent | Good | Fair | Poor | Critical
    health_score: float            # 0-100
    deficiencies: list[str]
    excesses: list[str]
    suitable_crops: list[str]
    unsuitable_crops: list[str]
    amendments: list[str]
    summary: str


def classify_soil(
    nitrogen_kg_ha: float,
    phosphorus_kg_ha: float,
    potassium_kg_ha: float,
    ph: float,
    organic_carbon: float = 0.6,
    moisture_pct: float = 20.0,
) -> SoilResult:
    params = {
        "nitrogen_kg_ha": nitrogen_kg_ha,
        "phosphorus_kg_ha": phosphorus_kg_ha,
        "potassium_kg_ha": potassium_kg_ha,
        "ph": ph,
        "organic_carbon": organic_carbon,
        "moisture_pct": moisture_pct,
    }

    scores = []
    deficiencies = []
    excesses = []
    amendments = []

    for key, (lo, hi) in OPTIMAL_RANGES.items():
        val = params.get(key, (lo + hi) / 2)
        mid = (lo + hi) / 2
        span = hi - lo

        if lo <= val <= hi:
            scores.append(100.0)
        elif val < lo:
            deficit_pct = (lo - val) / span * 100
            score = max(0, 100 - deficit_pct * 1.5)
            scores.append(score)
            deficiencies.append(key.replace("_", " ").title())
            _add_amendment(key, "low", amendments)
        else:
            excess_pct = (val - hi) / span * 100
            score = max(0, 100 - excess_pct * 1.5)
            scores.append(score)
            excesses.append(key.replace("_", " ").title())
            _add_amendment(key, "high", amendments)

    health_score = round(np.mean(scores), 1)

    if health_score >= 85:
        health_class = "Excellent"
    elif health_score >= 70:
        health_class = "Good"
    elif health_score >= 50:
        health_class = "Fair"
    elif health_score >= 30:
        health_class = "Poor"
    else:
        health_class = "Critical"

    suitable = []
    unsuitable = []
    for crop, prefs in CROP_SOIL_PREFERENCES.items():
        ph_ok = prefs["ph"][0] <= ph <= prefs["ph"][1]
        moist_ok = prefs["moisture_pct"][0] <= moisture_pct <= prefs["moisture_pct"][1]
        if ph_ok and moist_ok:
            suitable.append(crop)
        else:
            unsuitable.append(crop)

    summary = (
        f"Soil health score: {health_score}/100 ({health_class}). "
        f"{'No deficiencies detected.' if not deficiencies else 'Deficiencies: ' + ', '.join(deficiencies) + '.'} "
        f"Suitable crops: {', '.join(suitable[:4]) or 'None matched'}."
    )

    return SoilResult(
        health_class=health_class,
        health_score=health_score,
        deficiencies=deficiencies,
        excesses=excesses,
        suitable_crops=suitable,
        unsuitable_crops=unsuitable,
        amendments=list(dict.fromkeys(amendments)),
        summary=summary,
    )


def _add_amendment(param: str, direction: str, amendments: list[str]) -> None:
    mapping = {
        ("nitrogen_kg_ha", "low"):    "Apply Urea @ 50 kg/acre or FYM @ 5 tonnes/acre",
        ("nitrogen_kg_ha", "high"):   "Reduce nitrogen inputs; plant nitrogen-fixing cover crops",
        ("phosphorus_kg_ha", "low"):  "Apply DAP (Di-ammonium Phosphate) @ 25 kg/acre",
        ("phosphorus_kg_ha", "high"): "Avoid phosphatic fertilizers this season",
        ("potassium_kg_ha", "low"):   "Apply MOP (Muriate of Potash) @ 25 kg/acre",
        ("potassium_kg_ha", "high"):  "Skip potash application this season",
        ("ph", "low"):                "Apply Agricultural Lime @ 200 kg/acre to raise pH",
        ("ph", "high"):               "Apply Gypsum @ 200 kg/acre or Sulphur @ 10 kg/acre to lower pH",
        ("organic_carbon", "low"):    "Incorporate crop residue and FYM @ 5 tonnes/acre",
        ("moisture_pct", "low"):      "Improve irrigation scheduling; consider mulching",
        ("moisture_pct", "high"):     "Improve drainage; consider raised bed cultivation",
    }
    rec = mapping.get((param, direction))
    if rec and rec not in amendments:
        amendments.append(rec)
