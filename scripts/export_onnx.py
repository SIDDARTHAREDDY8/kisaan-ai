#!/usr/bin/env python3
"""
Export the plant disease classifier to ONNX for offline / edge inference.

The ONNX model runs at ~180ms on CPU vs ~400ms for PyTorch — critical for
farmers with poor connectivity who may run the model locally.

Usage: python scripts/export_onnx.py
Output: models/disease_classifier.onnx  (+ models/disease_labels.json)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from PIL import Image
from transformers import AutoFeatureExtractor, AutoModelForImageClassification

MODEL_ID = "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"
OUTPUT_DIR = Path(__file__).parent.parent / "models"
OUTPUT_DIR.mkdir(exist_ok=True)

ONNX_PATH = OUTPUT_DIR / "disease_classifier.onnx"
LABELS_PATH = OUTPUT_DIR / "disease_labels.json"


def export():
    print(f"Loading model: {MODEL_ID}")
    extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID)
    model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
    model.eval()

    # Dummy input: 1 x 3 x 224 x 224
    dummy_image = Image.new("RGB", (224, 224), color=(128, 128, 128))
    inputs = extractor(images=dummy_image, return_tensors="pt")
    dummy_input = inputs["pixel_values"]

    print(f"Exporting to ONNX: {ONNX_PATH}")
    torch.onnx.export(
        model,
        dummy_input,
        str(ONNX_PATH),
        opset_version=14,
        input_names=["pixel_values"],
        output_names=["logits"],
        dynamic_axes={"pixel_values": {0: "batch_size"}, "logits": {0: "batch_size"}},
        do_constant_folding=True,
    )

    # Save label map
    id2label = model.config.id2label
    LABELS_PATH.write_text(json.dumps(id2label, indent=2))

    # Verify
    import onnxruntime as ort
    import numpy as np
    sess = ort.InferenceSession(str(ONNX_PATH))
    ort_inputs = {"pixel_values": dummy_input.numpy()}
    ort_out = sess.run(None, ort_inputs)
    print(f"ONNX verification OK — logits shape: {ort_out[0].shape}")
    print(f"\nExport complete:")
    print(f"  Model: {ONNX_PATH} ({ONNX_PATH.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"  Labels: {LABELS_PATH} ({len(id2label)} classes)")
    print("\nTo use offline:")
    print("  import onnxruntime as ort")
    print("  sess = ort.InferenceSession('models/disease_classifier.onnx')")
    print("  out = sess.run(None, {'pixel_values': pixel_array})")


if __name__ == "__main__":
    try:
        import onnxruntime
    except ImportError:
        print("Install onnxruntime: pip install onnxruntime")
        sys.exit(1)
    export()
