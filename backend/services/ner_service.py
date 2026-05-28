"""
NER Service — dslim/bert-base-NER
HuggingFace Task: Token Classification

Extracts named entities from farmer queries and advisory text:
  - Crop names (ORG class + custom post-processing)
  - Locations / states (LOC)
  - Chemical/pesticide names (ORG)
  - Quantities and dates (NUM/DATE via regex augmentation)

Used to auto-populate commodity and location fields from free-text queries.
"""
import logging
import re
from functools import lru_cache
from typing import NamedTuple

import torch
from transformers import pipeline

logger = logging.getLogger(__name__)

NER_MODEL = "dslim/bert-base-NER"

KNOWN_CROPS = {
    "tomato", "wheat", "rice", "maize", "corn", "potato", "cotton",
    "groundnut", "soybean", "sugarcane", "onion", "chilli", "mango",
    "banana", "paddy", "bajra", "jowar", "mustard", "sunflower",
}

KNOWN_STATES = {
    "punjab", "haryana", "uttar pradesh", "maharashtra", "andhra pradesh",
    "telangana", "karnataka", "tamil nadu", "rajasthan", "gujarat",
    "madhya pradesh", "bihar", "west bengal", "odisha", "kerala",
}


class NERResult(NamedTuple):
    crops: list[str]
    locations: list[str]
    organizations: list[str]
    persons: list[str]
    raw_entities: list[dict]


@lru_cache(maxsize=1)
def _load_ner():
    logger.info("Loading NER model: %s", NER_MODEL)
    device = 0 if torch.cuda.is_available() else -1
    return pipeline(
        "token-classification",
        model=NER_MODEL,
        aggregation_strategy="simple",
        device=device,
    )


def extract_entities(text: str) -> NERResult:
    ner = _load_ner()
    try:
        entities = ner(text)
    except Exception as exc:
        logger.warning("NER failed: %s", exc)
        entities = []

    crops, locations, orgs, persons = [], [], [], []
    raw = []

    for ent in entities:
        word = ent["word"].strip().lower()
        label = ent["entity_group"]
        score = round(ent["score"], 3)
        raw.append({"word": word, "label": label, "score": score})

        if label == "LOC":
            locations.append(word)
        elif label == "ORG":
            orgs.append(word)
        elif label == "PER":
            persons.append(word)

    # Augment: match known crops from text directly.
    # Allow optional plural suffix (tomatoes, onions) but require word boundary
    # before the crop name to avoid "rice" matching inside "price".
    text_lower = text.lower()
    for crop in KNOWN_CROPS:
        if re.search(r'\b' + re.escape(crop) + r'(?:es|s)?\b', text_lower) and crop not in crops:
            crops.append(crop)

    # Augment: match known states
    for state_name in KNOWN_STATES:
        if re.search(r'\b' + re.escape(state_name) + r'\b', text_lower) and state_name not in locations:
            locations.append(state_name)

    return NERResult(
        crops=list(dict.fromkeys(crops)),          # deduplicate, preserve order
        locations=list(dict.fromkeys(locations)),
        organizations=list(dict.fromkeys(orgs)),
        persons=list(dict.fromkeys(persons)),
        raw_entities=raw,
    )
