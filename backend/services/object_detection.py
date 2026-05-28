"""
Object Detection Service — facebook/detr-resnet-50
HuggingFace Task: Object Detection

Used to detect pests, insects, and weed presence in field photos.
DETR (Detection Transformer) is end-to-end — no anchor boxes or NMS needed.

Agricultural labels of interest (COCO classes that map to farm context):
  bird, cat, dog → wildlife
  insect-adjacent COCO classes + custom thresholds for anomaly flagging
"""
import io
import logging
from functools import lru_cache
from typing import NamedTuple

import torch
from PIL import Image
from transformers import DetrImageProcessor, DetrForObjectDetection

logger = logging.getLogger(__name__)

DETR_MODEL = "facebook/detr-resnet-50"
CONFIDENCE_THRESHOLD = 0.7

# COCO labels that are agriculturally relevant
PEST_ADJACENT_LABELS = {
    "bird", "cat", "dog", "sheep", "cow", "elephant",
    "bear", "zebra", "giraffe", "horse", "insect",
}


class DetectionResult(NamedTuple):
    detections: list[dict]    # [{label, score, box}]
    pest_flags: list[str]     # labels above threshold
    annotated_summary: str    # human-readable summary


@lru_cache(maxsize=1)
def _load_detector():
    logger.info("Loading DETR object detector: %s", DETR_MODEL)
    processor = DetrImageProcessor.from_pretrained(DETR_MODEL)
    model = DetrForObjectDetection.from_pretrained(DETR_MODEL)
    model.eval()
    return processor, model


def detect_objects(image_bytes: bytes) -> DetectionResult:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    processor, model = _load_detector()

    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(
        outputs, target_sizes=target_sizes, threshold=CONFIDENCE_THRESHOLD
    )[0]

    detections = []
    for score, label_id, box in zip(
        results["scores"], results["labels"], results["boxes"]
    ):
        label = model.config.id2label[label_id.item()]
        detections.append({
            "label": label,
            "score": round(score.item(), 3),
            "box": [round(x, 1) for x in box.tolist()],
        })

    pest_flags = [d["label"] for d in detections if d["label"] in PEST_ADJACENT_LABELS]

    if not detections:
        summary = "No significant objects detected in the field photo."
    elif pest_flags:
        summary = f"Detected potential pest/wildlife presence: {', '.join(set(pest_flags))}. " \
                  f"Total detections: {len(detections)}."
    else:
        labels = [d["label"] for d in detections[:5]]
        summary = f"Detected {len(detections)} object(s): {', '.join(labels)}."

    return DetectionResult(
        detections=detections,
        pest_flags=pest_flags,
        annotated_summary=summary,
    )
