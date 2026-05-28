"""
EfficientNet-B4 Fine-tuning for Indian Crop Disease Classification

Training data: PlantVillage + collected Indian WhatsApp samples
Model: EfficientNet-B4 pretrained on ImageNet, custom head for Indian taxonomy
Augmentation: Heavy — simulates WhatsApp JPEG compression + field conditions

Usage:
  python scripts/training/train_disease_model.py \
    --data_dir data/training_images \
    --output_dir models/disease_v1 \
    --epochs 30 \
    --batch_size 32
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms, models
from PIL import Image
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TAXONOMY_PATH = Path(__file__).parent.parent.parent / "data" / "diseases" / "taxonomy.json"


# ── Dataset ───────────────────────────────────────────────────────────────────

class IndianDiseaseDataset(Dataset):
    """
    Loads images from a flat list of (image_path, disease_id) tuples.
    disease_id comes from taxonomy.json IDs.
    """
    def __init__(self, samples: list[dict], label2idx: dict, transform=None):
        self.samples = samples
        self.label2idx = label2idx
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = sample["image_path"]
        label = sample["disease_id"]

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            # Return a blank image on corrupt file
            img = Image.new("RGB", (224, 224), (128, 128, 128))

        if self.transform:
            img = self.transform(img)

        return img, self.label2idx[label]


# ── Augmentation ──────────────────────────────────────────────────────────────

def get_train_transform():
    """
    Heavy augmentation pipeline designed for WhatsApp-quality images:
    - JPEG compression artifacts (quality 40-85)
    - Random brightness/contrast (field lighting varies widely)
    - Random blur (shaky hands, cheap phones)
    - Perspective distortion (photos taken at angles)
    """
    try:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2

        return A.Compose([
            A.RandomResizedCrop(height=224, width=224, scale=(0.6, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomRotate90(p=0.5),
            # Lighting variations (critical for field photos)
            A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.8),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=40, val_shift_limit=20, p=0.6),
            A.CLAHE(clip_limit=4.0, p=0.3),
            # Camera quality simulation
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            A.GaussNoise(var_limit=(10, 50), p=0.3),
            A.ImageCompression(quality_lower=40, quality_upper=85, p=0.5),  # WhatsApp JPEG
            A.Perspective(scale=(0.05, 0.15), p=0.4),
            # Shadow simulation (leaves under partial shade)
            A.RandomShadow(p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    except ImportError:
        logger.warning("albumentations not installed, using basic transforms")
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])


def get_val_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


# ── Model ─────────────────────────────────────────────────────────────────────

def build_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    """
    EfficientNet-B4 with custom classification head.
    Better accuracy than MobileNetV2 with acceptable inference time on CPU.
    """
    model = models.efficientnet_b4(
        weights=models.EfficientNet_B4_Weights.DEFAULT if pretrained else None
    )
    # Freeze early layers, fine-tune from block 5 onward
    for name, param in model.named_parameters():
        if "features.0" in name or "features.1" in name or "features.2" in name:
            param.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(512, num_classes),
    )
    return model


# ── Training loop ─────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for batch_idx, (imgs, labels) in enumerate(loader):
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            outputs = model(imgs)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

        if batch_idx % 20 == 0:
            logger.info("  batch %d/%d — loss=%.4f acc=%.2f%%",
                        batch_idx, len(loader), loss.item(), 100 * correct / total)

    return total_loss / len(loader), 100 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> tuple[float, float]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader), 100 * correct / total


# ── Data loading ──────────────────────────────────────────────────────────────

def load_samples_from_db(verified_only: bool = True) -> list[dict]:
    """Load labelled samples from the TrainingSample DB table."""
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from backend.db.models import TrainingSample, SyncSessionLocal

        with SyncSessionLocal() as session:
            query = session.query(TrainingSample).filter(
                TrainingSample.disease_id.isnot(None),
                TrainingSample.disease_id != "",
            )
            if verified_only:
                query = query.filter_by(is_verified=True)
            rows = query.all()

        samples = []
        for r in rows:
            img_path = Path(__file__).parent.parent.parent / r.image_path
            if img_path.exists():
                samples.append({
                    "image_path": str(img_path),
                    "disease_id": r.disease_id,
                    "split": r.split,
                    "language": r.language,
                    "source": r.source,
                })
        logger.info("Loaded %d labelled samples from DB", len(samples))
        return samples
    except Exception as exc:
        logger.warning("Could not load from DB: %s", exc)
        return []


def load_local_dataset_samples(dataset_dir: str) -> list[dict]:
    """
    Load local crop disease dataset.
    Folder names like 'Corn___Common_Rust' are mapped to taxonomy disease IDs
    using local_label / plantvillage_label fields in taxonomy.json.
    Healthy folders are skipped.
    """
    with open(TAXONOMY_PATH) as f:
        taxonomy = json.load(f)

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
        if "healthy" in folder_name.lower():
            continue
        disease_id = folder_to_id.get(folder_name)
        if not disease_id:
            skipped.append(folder_name)
            continue
        for img_file in class_dir.glob("*"):
            if img_file.suffix.lower() in (".jpg", ".jpeg", ".png"):
                samples.append({
                    "image_path": str(img_file),
                    "disease_id": disease_id,
                    "split": "train",
                    "language": "en",
                    "source": "local_dataset",
                })

    logger.info(
        "Local dataset: %d samples (%d classes) | unmapped folders: %s",
        len(samples), len(set(s["disease_id"] for s in samples)), skipped,
    )
    return samples


def load_plantvillage_samples(plantvillage_dir: str) -> list[dict]:
    """
    Load PlantVillage dataset. Expected structure:
      plantvillage_dir/
        Tomato___Early_blight/   ← PlantVillage folder names
          img001.jpg
        Rice___Blast/
          ...

    Maps PlantVillage labels to taxonomy disease IDs using plantvillage_label field.
    """
    with open(TAXONOMY_PATH) as f:
        taxonomy = json.load(f)

    pv_to_id = {
        d["plantvillage_label"]: d["id"]
        for d in taxonomy["diseases"]
        if d.get("plantvillage_label")
    }

    samples = []
    pv_path = Path(plantvillage_dir)
    if not pv_path.exists():
        logger.warning("PlantVillage dir not found: %s", plantvillage_dir)
        return []

    for class_dir in pv_path.iterdir():
        if not class_dir.is_dir():
            continue
        disease_id = pv_to_id.get(class_dir.name)
        if not disease_id:
            logger.debug("No taxonomy mapping for PlantVillage class: %s", class_dir.name)
            continue
        for img_file in class_dir.glob("*.jpg"):
            samples.append({
                "image_path": str(img_file),
                "disease_id": disease_id,
                "split": "train",
                "language": "en",
                "source": "plantvillage",
            })

    logger.info("Loaded %d PlantVillage samples (%d classes mapped)",
                len(samples), len(set(s["disease_id"] for s in samples)))
    return samples


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train Indian crop disease classifier")
    parser.add_argument("--plantvillage_dir", default="", help="Path to PlantVillage dataset root")
    parser.add_argument("--local_dataset_dir", default="", help="Path to local crop disease folder")
    parser.add_argument("--output_dir", default="models/disease_v1")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--unverified", action="store_true", help="Include unverified DB samples")
    args = parser.parse_args()

    # Load data
    all_samples = []
    if args.plantvillage_dir:
        all_samples += load_plantvillage_samples(args.plantvillage_dir)
    if args.local_dataset_dir:
        all_samples += load_local_dataset_samples(args.local_dataset_dir)
    all_samples += load_samples_from_db(verified_only=not args.unverified)

    if not all_samples:
        logger.error("No training samples found. Add --plantvillage_dir or annotate samples first.")
        sys.exit(1)

    # Build label set
    disease_ids = sorted(set(s["disease_id"] for s in all_samples))
    label2idx = {d: i for i, d in enumerate(disease_ids)}
    idx2label = {i: d for d, i in label2idx.items()}
    num_classes = len(disease_ids)
    logger.info("Classes: %d — %s", num_classes, disease_ids)

    # Train/val split
    train_samples = [s for s in all_samples if s.get("split") != "val"]
    val_samples = [s for s in all_samples if s.get("split") == "val"]
    if not val_samples:
        # Auto-split 10% as val
        rng = np.random.default_rng(42)
        val_idx = set(rng.choice(len(train_samples), len(train_samples) // 10, replace=False))
        val_samples = [train_samples[i] for i in val_idx]
        train_samples = [train_samples[i] for i in range(len(train_samples)) if i not in val_idx]

    logger.info("Train: %d | Val: %d", len(train_samples), len(val_samples))

    train_tf = get_train_transform()
    val_tf = get_val_transform()

    # Albumentations wrapper for Dataset
    class AlbuDataset(IndianDiseaseDataset):
        def __getitem__(self, idx):
            sample = self.samples[idx]
            try:
                img = np.array(Image.open(sample["image_path"]).convert("RGB"))
            except Exception:
                img = np.zeros((224, 224, 3), dtype=np.uint8)
            label = self.label2idx[sample["disease_id"]]
            if self.transform and hasattr(self.transform, "__call__"):
                try:
                    result = self.transform(image=img)
                    img_t = result["image"]
                except TypeError:
                    # torchvision transform
                    img_t = self.transform(Image.fromarray(img))
            else:
                img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            return img_t, label

    train_ds = AlbuDataset(train_samples, label2idx, train_tf)
    val_ds = IndianDiseaseDataset(val_samples, label2idx, val_tf)

    # Weighted sampler to handle class imbalance (Indian data < PlantVillage)
    class_counts = [sum(1 for s in train_samples if s["disease_id"] == d) for d in disease_ids]
    weights = [1.0 / (class_counts[label2idx[s["disease_id"]]] + 1) for s in train_samples]
    sampler = WeightedRandomSampler(weights, len(weights))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info("Device: %s", device)

    model = build_model(num_classes).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save label map
    with open(output_dir / "label_map.json", "w") as f:
        json.dump({"idx2label": idx2label, "label2idx": label2idx}, f, indent=2)
    logger.info("Label map saved to %s/label_map.json", output_dir)

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        logger.info("=== Epoch %d/%d (lr=%.6f) ===", epoch, args.epochs,
                    optimizer.param_groups[0]["lr"])
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        logger.info("Train loss=%.4f acc=%.2f%% | Val loss=%.4f acc=%.2f%%",
                    train_loss, train_acc, val_loss, val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "num_classes": num_classes,
                "label2idx": label2idx,
                "idx2label": idx2label,
            }, output_dir / "best_model.pt")
            logger.info("  ✅ New best val acc: %.2f%% — saved to %s/best_model.pt",
                        val_acc, output_dir)

    logger.info("Training complete. Best val acc: %.2f%%", best_val_acc)


if __name__ == "__main__":
    main()
