"""
Zero-Shot Text Classifier — facebook/bart-large-mnli
HuggingFace Task: Zero-Shot Classification

Used as the high-accuracy intent router. Falls back to keyword heuristics
if the model is not yet loaded (cold start) to keep p50 latency low.
"""
import logging
from functools import lru_cache

import torch
from transformers import pipeline

logger = logging.getLogger(__name__)

ZSC_MODEL = "cross-encoder/nli-MiniLM2-L6-H768"  # 67MB — much lighter than bart-large-mnli

CANDIDATE_LABELS = [
    "crop disease diagnosis",
    "market price inquiry",
    "farming advice and agronomy",
    "government scheme and subsidy",
    "soil health analysis",
    "weather and irrigation",
]

LABEL_TO_INTENT = {
    "crop disease diagnosis": "disease",
    "market price inquiry": "market",
    "farming advice and agronomy": "advisory",
    "government scheme and subsidy": "scheme",
    "soil health analysis": "soil",
    "weather and irrigation": "advisory",
}


@lru_cache(maxsize=1)
def _load_zsc():
    logger.info("Loading zero-shot classifier: %s", ZSC_MODEL)
    device = 0 if torch.cuda.is_available() else -1
    return pipeline(
        "zero-shot-classification",
        model=ZSC_MODEL,
        device=device,
    )


def classify_intent(text: str, has_image: bool = False) -> tuple[str, float]:
    """Return (intent, confidence). Fast path if image present."""
    if has_image and not text.strip():
        return "disease", 1.0

    if not text.strip():
        return "advisory", 0.5

    # Keyword heuristics run first — they're faster and more reliable for
    # domain-specific terms like "insurance", "PM-KISAN", "mandi price".
    kw_intent, kw_conf = _keyword_fallback(text, has_image)
    if kw_conf >= 0.7:
        return kw_intent, kw_conf

    # Fall through to ZSC only when keywords give no signal
    try:
        zsc = _load_zsc()
        result = zsc(text, candidate_labels=CANDIDATE_LABELS)
        top_label = result["labels"][0]
        top_score = result["scores"][0]
        intent = LABEL_TO_INTENT.get(top_label, "advisory")
        if has_image:
            intent = "disease"
        # If ZSC confidence is low, trust keyword fallback
        if top_score < 0.45:
            return kw_intent, kw_conf
        return intent, round(top_score, 4)
    except Exception as exc:
        logger.warning("ZSC failed, using keyword fallback: %s", exc)
        return kw_intent, kw_conf


def _keyword_fallback(text: str, has_image: bool) -> tuple[str, float]:
    import re
    t = text.lower()
    # Scheme — check first: insurance/kisan/subsidy are unambiguous
    if re.search(
        r"scheme|subsidy|pm.?kisan|pmfby|kcc|enam|rkvy|pmksy|midh|"
        r"insurance|credit.?card|kisan.?credit|government|govt|"
        r"register|eligible|benefit|apply|application|document",
        t,
    ):
        return "scheme", 0.8
    if has_image or re.search(
        r"disease|spot|blight|wilt|rot|yellow|brown|fungus|pest|"
        r"aphid|insect|mold|mildew|rust|leaf|dying|infect|symptom",
        t,
    ):
        return "disease", 0.8
    if re.search(r"price|mandi|market|sell|rate|rupee|quintal|today.?rate|crop.?price", t):
        return "market", 0.8
    if re.search(r"soil|npk|nitrogen|phosphorus|potassium|\bph\b|fertility|organic.?carbon|moisture", t):
        return "soil", 0.8
    return "advisory", 0.5
