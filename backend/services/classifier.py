"""
Plant disease classifier.

Priority order:
  1. Fine-tuned EfficientNet-B4 (models/disease_v1/best_model.pt) when confidence ≥ FINETUNED_THRESHOLD
  2. Claude claude-sonnet-4-6 vision (reliable for any WhatsApp photo quality)
  3. MobileNetV2 (PlantVillage pretrained) as a lightweight hint to Claude

The fine-tuned model is trained on Indian crop disease data collected from WhatsApp
and PlantVillage, mapped to our regional taxonomy (data/diseases/taxonomy.json).
"""
import base64
import io
import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

import torch
from PIL import Image

from backend.config import settings

logger = logging.getLogger(__name__)

_MOBILENET_MODEL_ID = "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"
_FINETUNED_MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "disease_v1" / "best_model.pt"
_TAXONOMY_PATH = Path(__file__).parent.parent.parent / "data" / "diseases" / "taxonomy.json"
FINETUNED_THRESHOLD = 0.70  # min confidence to trust fine-tuned model over Claude


class ClassifierResult(NamedTuple):
    label: str
    confidence: float
    crop: str
    condition: str
    top5: list[dict]
    disease_id: str = ""           # taxonomy ID e.g. "tomato_early_blight"
    follow_up_question: str = ""   # non-empty when Claude needs more info


def _parse_label(raw_label: str) -> tuple[str, str]:
    """'Tomato___Early_blight' → ('Tomato', 'Early blight')"""
    parts = raw_label.replace("___", "|").replace("_", " ").split("|")
    crop = parts[0].strip() if parts else "Unknown"
    condition = parts[1].strip() if len(parts) > 1 else raw_label
    return crop, condition


@lru_cache(maxsize=1)
def _load_mobilenet():
    """Load MobileNetV2 manually since its preprocessor_config.json lacks image_processor_type."""
    from transformers import AutoModelForImageClassification
    logger.info("Loading MobileNetV2 disease classifier: %s", _MOBILENET_MODEL_ID)
    model = AutoModelForImageClassification.from_pretrained(_MOBILENET_MODEL_ID)
    model.eval()
    return model


def _run_mobilenet(image: Image.Image) -> list[dict]:
    """Run MobileNetV2 with manual torchvision preprocessing."""
    try:
        import torchvision.transforms as T
        transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        model = _load_mobilenet()
        tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            logits = model(tensor).logits
        probs = torch.softmax(logits, dim=-1)[0]
        top5_idx = probs.argsort(descending=True)[:5].tolist()
        id2label = model.config.id2label
        return [
            {"label": id2label[i], "score": round(probs[i].item(), 4)}
            for i in top5_idx
        ]
    except Exception as exc:
        logger.warning("MobileNetV2 inference failed: %s", exc)
        return []


@lru_cache(maxsize=1)
def _load_taxonomy_diseases() -> list[dict]:
    """Load disease list from taxonomy.json once and cache it."""
    try:
        with open(_TAXONOMY_PATH) as f:
            return json.load(f)["diseases"]
    except Exception:
        return []


def _build_disease_menu() -> str:
    """Format taxonomy diseases as a numbered list for the Claude prompt."""
    diseases = _load_taxonomy_diseases()
    lines = []
    for d in diseases:
        symptoms = d.get("symptoms", {}).get("en", "")[:120]
        lines.append(
            f"  ID={d['id']} | Crop={d['crop']} | Name={d['english']}\n"
            f"    Symptoms: {symptoms}"
        )
    return "\n".join(lines)


def _claude_vision_diagnosis(
    image_bytes: bytes,
    mobilenet_hint: str = "",
    farmer_text: str = "",
) -> ClassifierResult:
    """
    Use Claude vision to diagnose plant disease.
    Constrains Claude to pick from the taxonomy disease list for accuracy.
    When uncertain, generates a follow-up question instead of guessing.
    """
    import anthropic
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    b64 = base64.standard_b64encode(image_bytes).decode()
    disease_menu = _build_disease_menu()

    farmer_context = (
        f"\n\nFarmer's description: \"{farmer_text[:300]}\""
        if farmer_text and farmer_text.strip() else ""
    )
    hint_text = (
        f"\n\nAutomated pre-classifier hint: '{mobilenet_hint}' — treat as low-priority hint only."
        if mobilenet_hint else ""
    )

    prompt = f"""You are an expert plant pathologist specialising in Indian crop diseases.

KNOWN DISEASES IN THIS SYSTEM (you MUST pick from this list or say UNCERTAIN):
{disease_menu}{farmer_context}{hint_text}

TASK: Analyze the crop image and identify the disease.

RULES:
1. Pick the best matching disease ID from the list above.
2. If the image is unclear, blurry, or you are not at least 60% sure, set DISEASE_ID=UNCERTAIN.
3. If UNCERTAIN, write one short FOLLOW_UP question to ask the farmer for more info.
4. Never invent disease names outside the list.
5. If the crop looks healthy, set DISEASE_ID=healthy.

Respond in EXACTLY this format (no extra lines between fields):
CROP: <crop name visible in image>
DISEASE_ID: <id from list above, or UNCERTAIN, or healthy>
CONDITION: <disease English name or Healthy or Unknown>
CONFIDENCE: <high|medium|low>
SYMPTOMS_SEEN: <what you see in the image, 1-2 sentences>
FOLLOW_UP: <question to ask farmer if UNCERTAIN, else leave blank>
"""

    try:
        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=500,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/jpeg", "data": b64},
                    },
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        text = message.content[0].text
        lines = text.strip().split("\n")

        fields: dict = {}
        for line in lines:
            if ":" in line:
                key, _, val = line.partition(":")
                fields[key.strip()] = val.strip()

        crop = fields.get("CROP", "Unknown")
        disease_id = fields.get("DISEASE_ID", "UNCERTAIN")
        condition = fields.get("CONDITION", "Unknown")
        confidence_str = fields.get("CONFIDENCE", "low").lower()
        follow_up = fields.get("FOLLOW_UP", "").strip()

        conf_map = {"high": 0.90, "medium": 0.72, "low": 0.45}
        confidence = conf_map.get(confidence_str, 0.50)

        # Resolve disease_id to taxonomy entry for label
        diseases = _load_taxonomy_diseases()
        matched = next((d for d in diseases if d["id"] == disease_id), None)
        if matched:
            label = matched.get("plantvillage_label") or f"{crop}___{condition.replace(' ', '_')}"
            condition = matched["english"]
            crop = matched["crop"].capitalize()
        elif disease_id == "healthy":
            label = f"{crop}___Healthy"
            condition = "Healthy"
            confidence = max(confidence, 0.75)
            follow_up = ""
        else:
            # UNCERTAIN — force low confidence so follow-up is triggered
            disease_id = ""
            confidence = 0.30
            label = "Unknown___Unknown"
            condition = "Unknown"

        logger.info(
            "Claude vision: disease_id=%s crop=%s conf=%s follow_up=%r",
            disease_id, crop, confidence_str, bool(follow_up),
        )
        return ClassifierResult(
            label=label,
            confidence=confidence,
            crop=crop,
            condition=condition,
            top5=[{"label": disease_id or "unknown", "score": round(confidence, 4)}],
            disease_id=disease_id,
            follow_up_question=follow_up,
        )
    except Exception as exc:
        logger.error("Claude vision diagnosis failed: %s", exc)
        return ClassifierResult(
            label="Unknown___Unknown",
            confidence=0.0,
            crop="Unknown",
            condition="Unknown",
            top5=[],
            disease_id="",
            follow_up_question="Could you send a clearer photo of the affected leaf or fruit, preferably in natural daylight?",
        )


@lru_cache(maxsize=1)
def _load_finetuned():
    """Load fine-tuned EfficientNet-B4 + label map if checkpoint exists."""
    if not _FINETUNED_MODEL_PATH.exists():
        return None, None

    label_map_path = _FINETUNED_MODEL_PATH.parent / "label_map.json"
    if not label_map_path.exists():
        logger.warning("Fine-tuned model found but no label_map.json alongside it")
        return None, None

    try:
        from torchvision import models as tv_models
        import torch.nn as nn

        checkpoint = torch.load(_FINETUNED_MODEL_PATH, map_location="cpu")
        num_classes = checkpoint["num_classes"]
        idx2label: dict = {int(k): v for k, v in checkpoint["idx2label"].items()}

        model = tv_models.efficientnet_b4(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        if torch.backends.mps.is_available():
            model = model.to("mps")

        logger.info(
            "Loaded fine-tuned EfficientNet-B4: %d classes, val_acc=%.2f%%",
            num_classes, checkpoint.get("val_acc", 0.0),
        )
        return model, idx2label
    except Exception as exc:
        logger.warning("Failed to load fine-tuned model: %s", exc)
        return None, None


def _run_finetuned(image: Image.Image) -> tuple[str, float] | tuple[None, None]:
    """
    Run fine-tuned EfficientNet-B4. Returns (disease_id, confidence) or (None, None).
    disease_id is a taxonomy ID like 'tomato_early_blight'.
    """
    model, idx2label = _load_finetuned()
    if model is None:
        return None, None

    try:
        import torchvision.transforms as T
        device = next(model.parameters()).device

        transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(tensor)
        probs = torch.softmax(logits, dim=-1)[0]
        top_idx = probs.argmax().item()
        confidence = probs[top_idx].item()
        disease_id = idx2label[top_idx]
        logger.info("Fine-tuned model: %s (conf=%.3f)", disease_id, confidence)
        return disease_id, confidence
    except Exception as exc:
        logger.warning("Fine-tuned inference failed: %s", exc)
        return None, None


def _disease_id_to_result(disease_id: str, confidence: float) -> ClassifierResult:
    """Convert a taxonomy disease_id into a ClassifierResult using taxonomy.json."""
    diseases = _load_taxonomy_diseases()
    disease = next((d for d in diseases if d["id"] == disease_id), None)
    if disease:
        crop = disease["crop"].capitalize()
        condition = disease["english"]
        pv_label = disease.get("plantvillage_label") or f"{crop}___{condition.replace(' ', '_')}"
        return ClassifierResult(
            label=pv_label,
            confidence=confidence,
            crop=crop,
            condition=condition,
            top5=[{"label": disease_id, "score": round(confidence, 4)}],
            disease_id=disease_id,
        )

    crop, condition = _parse_label(disease_id.replace("_", " ").title().replace(" ", "___", 1))
    return ClassifierResult(
        label=disease_id,
        confidence=confidence,
        crop=crop,
        condition=condition,
        top5=[{"label": disease_id, "score": round(confidence, 4)}],
        disease_id=disease_id,
    )


def classify_disease(image_bytes: bytes, farmer_text: str = "") -> ClassifierResult:
    """
    Classify plant disease image.

    Pipeline:
      1. Fine-tuned EfficientNet-B4 (if available) — use directly if conf ≥ FINETUNED_THRESHOLD
      2. Claude vision — constrained to taxonomy disease list + farmer text for context
      3. MobileNetV2 — hint to Claude when fine-tuned model unavailable
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Step 1: try fine-tuned model
    disease_id, ft_conf = _run_finetuned(image)
    if disease_id and ft_conf and ft_conf >= FINETUNED_THRESHOLD:
        logger.info("Using fine-tuned model result (conf=%.3f ≥ %.2f)", ft_conf, FINETUNED_THRESHOLD)
        return _disease_id_to_result(disease_id, ft_conf)

    # Step 2: build hint for Claude vision
    if disease_id and ft_conf:
        hint = f"{disease_id} (model confidence {ft_conf:.0%})"
    else:
        mobilenet_results = _run_mobilenet(image)
        hint = mobilenet_results[0]["label"] if mobilenet_results else ""

    # Step 3: Claude vision as primary, grounded in taxonomy + farmer context
    return _claude_vision_diagnosis(image_bytes, mobilenet_hint=hint, farmer_text=farmer_text)
