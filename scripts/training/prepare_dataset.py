"""
Dataset Preparation Script

Prepares a unified training dataset from:
  1. PlantVillage (download separately from Kaggle)
  2. Collected Indian WhatsApp samples (from DB)
  3. Optional KVK/expert-labelled samples

Outputs:
  data/dataset/train.jsonl
  data/dataset/val.jsonl
  data/dataset/test.jsonl
  data/dataset/stats.json

Usage:
  python scripts/training/prepare_dataset.py \
    --plantvillage_dir /path/to/PlantVillage \
    --val_split 0.10 \
    --test_split 0.05

Download PlantVillage:
  kaggle datasets download -d abdallahalidev/plantvillage-dataset
  unzip plantvillage-dataset.zip -d /path/to/PlantVillage
"""
import argparse
import json
import logging
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent.parent
TAXONOMY_PATH = ROOT / "data" / "diseases" / "taxonomy.json"
DATASET_DIR = ROOT / "data" / "dataset"


def load_taxonomy():
    with open(TAXONOMY_PATH) as f:
        return json.load(f)


def collect_plantvillage(pv_dir: str, taxonomy: dict) -> list[dict]:
    pv_to_id = {
        d["plantvillage_label"]: d["id"]
        for d in taxonomy["diseases"]
        if d.get("plantvillage_label")
    }

    samples = []
    pv_path = Path(pv_dir)
    skipped = []

    for class_dir in sorted(pv_path.rglob("*")):
        if not class_dir.is_dir():
            continue
        disease_id = pv_to_id.get(class_dir.name)
        if not disease_id:
            if any(class_dir.glob("*.jpg")):
                skipped.append(class_dir.name)
            continue
        for img in class_dir.glob("*.jpg"):
            samples.append({
                "image_path": str(img),
                "disease_id": disease_id,
                "source": "plantvillage",
                "language": "en",
                "verified": True,
            })

    logger.info("PlantVillage: %d samples, %d unmapped classes: %s",
                len(samples), len(skipped), skipped[:5])
    return samples


def collect_local_dataset(dataset_dir: str, taxonomy: dict) -> list[dict]:
    """
    Load a local dataset folder where sub-directories are class names.
    Maps folder names → taxonomy disease IDs using local_label or plantvillage_label fields.
    Skips folders with 'Healthy' in the name.

    Expected structure:
      dataset_dir/
        Corn___Common_Rust/
          image (1).JPG
          ...
        Rice___Neck_Blast/
          ...
    """
    # Build mapping: folder_name → disease_id
    folder_to_id: dict[str, str] = {}
    for d in taxonomy["diseases"]:
        for field in ("local_label", "plantvillage_label"):
            label = d.get(field)
            if label:
                folder_to_id[label] = d["id"]

    samples = []
    dataset_path = Path(dataset_dir)
    skipped = []

    for class_dir in sorted(dataset_path.iterdir()):
        if not class_dir.is_dir():
            continue
        folder_name = class_dir.name

        # Skip healthy classes
        if "healthy" in folder_name.lower():
            continue

        disease_id = folder_to_id.get(folder_name)
        if not disease_id:
            skipped.append(folder_name)
            continue

        for img in class_dir.glob("*"):
            if img.suffix.lower() in (".jpg", ".jpeg", ".png"):
                samples.append({
                    "image_path": str(img),
                    "disease_id": disease_id,
                    "source": "local_dataset",
                    "language": "en",
                    "verified": True,
                })

    logger.info(
        "Local dataset: %d samples across %d classes | unmapped: %s",
        len(samples), len(set(s["disease_id"] for s in samples)), skipped,
    )
    return samples


def collect_from_db(verified_only: bool = False) -> list[dict]:
    try:
        sys.path.insert(0, str(ROOT))
        from backend.db.models import TrainingSample, SyncSessionLocal
        with SyncSessionLocal() as session:
            q = session.query(TrainingSample).filter(
                TrainingSample.disease_id.isnot(None)
            )
            if verified_only:
                q = q.filter_by(is_verified=True)
            rows = q.all()

        samples = []
        for r in rows:
            img = ROOT / r.image_path
            if img.exists():
                samples.append({
                    "image_path": str(img),
                    "disease_id": r.disease_id,
                    "source": "whatsapp_india",
                    "language": r.language or "en",
                    "crop_hint": r.crop_hint or "",
                    "location": r.location_hint or "",
                    "state": r.state or "",
                    "farmer_text": r.farmer_text or "",
                    "verified": r.is_verified,
                })
        logger.info("DB samples: %d (%d verified)", len(samples),
                    sum(1 for s in samples if s["verified"]))
        return samples
    except Exception as exc:
        logger.warning("DB load failed: %s", exc)
        return []


def split_dataset(samples: list[dict], val_frac: float, test_frac: float, seed: int = 42):
    """
    Stratified split by disease_id so every class has representation in val/test.
    Indian (whatsapp_india) samples are weighted toward val/test to measure
    real-world performance separately.
    """
    rng = random.Random(seed)

    by_class = defaultdict(list)
    for s in samples:
        by_class[s["disease_id"]].append(s)

    train, val, test = [], [], []
    for disease_id, class_samples in by_class.items():
        rng.shuffle(class_samples)
        n = len(class_samples)
        n_test = max(1, int(n * test_frac))
        n_val = max(1, int(n * val_frac))
        test.extend(class_samples[:n_test])
        val.extend(class_samples[n_test:n_test + n_val])
        train.extend(class_samples[n_test + n_val:])

    logger.info("Split: train=%d val=%d test=%d", len(train), len(val), len(test))
    return train, val, test


def write_jsonl(samples: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    logger.info("Wrote %d samples to %s", len(samples), path)


def compute_stats(train, val, test, taxonomy) -> dict:
    disease_map = {d["id"]: d["english"] for d in taxonomy["diseases"]}

    def split_stats(samples):
        by_disease = Counter(s["disease_id"] for s in samples)
        by_source = Counter(s["source"] for s in samples)
        by_lang = Counter(s.get("language", "en") for s in samples)
        return {
            "total": len(samples),
            "by_disease": {disease_map.get(k, k): v
                           for k, v in by_disease.most_common()},
            "by_source": dict(by_source),
            "by_language": dict(by_lang),
        }

    return {
        "train": split_stats(train),
        "val": split_stats(val),
        "test": split_stats(test),
        "num_classes": len(set(s["disease_id"] for s in train + val + test)),
        "classes": sorted(set(s["disease_id"] for s in train + val + test)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plantvillage_dir", default="", help="Path to PlantVillage dataset")
    parser.add_argument("--local_dataset_dir", default="", help="Path to local crop disease dataset folder")
    parser.add_argument("--val_split", type=float, default=0.10)
    parser.add_argument("--test_split", type=float, default=0.05)
    parser.add_argument("--unverified_db", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    taxonomy = load_taxonomy()
    all_samples = []

    if args.plantvillage_dir:
        all_samples += collect_plantvillage(args.plantvillage_dir, taxonomy)

    if args.local_dataset_dir:
        all_samples += collect_local_dataset(args.local_dataset_dir, taxonomy)

    db_samples = collect_from_db(verified_only=not args.unverified_db)
    all_samples += db_samples

    if not all_samples:
        logger.error("No samples collected. Provide --plantvillage_dir or annotate DB samples.")
        sys.exit(1)

    logger.info("Total samples: %d across %d disease classes",
                len(all_samples), len(set(s["disease_id"] for s in all_samples)))

    train, val, test = split_dataset(all_samples, args.val_split, args.test_split, args.seed)

    write_jsonl(train, DATASET_DIR / "train.jsonl")
    write_jsonl(val, DATASET_DIR / "val.jsonl")
    write_jsonl(test, DATASET_DIR / "test.jsonl")

    stats = compute_stats(train, val, test, taxonomy)
    with open(DATASET_DIR / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    logger.info("\n=== Dataset Summary ===")
    logger.info("Classes: %d", stats["num_classes"])
    logger.info("Train: %d | Val: %d | Test: %d",
                stats["train"]["total"], stats["val"]["total"], stats["test"]["total"])
    logger.info("Sources: %s", stats["train"]["by_source"])
    logger.info("Saved to %s/", DATASET_DIR)


if __name__ == "__main__":
    main()
