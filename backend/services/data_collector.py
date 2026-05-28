"""
Training Data Collector

Every time a farmer sends an image via WhatsApp, we save:
  - The image file to data/training_images/<date>/<uuid>.jpg
  - Metadata (farmer text, language, location, crop) to DB

This builds our proprietary Indian crop disease dataset over time.
Images are privacy-safe: farmer phone number is stored as SHA-256 hash only.
"""
import hashlib
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# Root directory for storing training images
TRAINING_IMAGES_DIR = Path(__file__).parent.parent.parent / "data" / "training_images"


def _hash_farmer_id(farmer_id: str) -> str:
    return hashlib.sha256(farmer_id.encode()).hexdigest()


def _image_path(date_str: str, sample_id: str) -> Path:
    day_dir = TRAINING_IMAGES_DIR / date_str
    day_dir.mkdir(parents=True, exist_ok=True)
    return day_dir / f"{sample_id}.jpg"


def save_training_sample(
    image_bytes: bytes,
    farmer_id: str,
    farmer_text: str = "",
    farmer_text_en: str = "",
    language: str = "en",
    crop_hint: str = "",
    location_hint: str = "",
    state: str = "",
    district: str = "",
) -> str | None:
    """
    Save image to disk and record metadata in DB.
    Returns the sample_id (UUID) on success, None on failure.
    """
    sample_id = str(uuid.uuid4())
    date_str = datetime.utcnow().strftime("%Y-%m-%d")

    try:
        # Save image file
        img_path = _image_path(date_str, sample_id)
        img_path.write_bytes(image_bytes)

        relative_path = str(img_path.relative_to(TRAINING_IMAGES_DIR.parent.parent))

        # Record in DB
        try:
            from backend.db.models import TrainingSample, SyncSessionLocal
            with SyncSessionLocal() as session:
                sample = TrainingSample(
                    image_path=relative_path,
                    farmer_text=farmer_text[:1000] if farmer_text else "",
                    farmer_text_en=farmer_text_en[:1000] if farmer_text_en else "",
                    language=language,
                    crop_hint=crop_hint[:100] if crop_hint else "",
                    location_hint=location_hint[:200] if location_hint else "",
                    state=state[:100] if state else "",
                    district=district[:100] if district else "",
                    farmer_id_hash=_hash_farmer_id(farmer_id),
                    source="whatsapp",
                )
                session.add(sample)
                session.commit()
                sample_id_db = sample.id
            logger.info(
                "Saved training sample #%s: %s (%d bytes, lang=%s, crop=%s)",
                sample_id_db, relative_path, len(image_bytes), language, crop_hint
            )
        except Exception as db_exc:
            logger.warning("DB record failed (image saved): %s", db_exc)

        return sample_id

    except Exception as exc:
        logger.error("Failed to save training sample: %s", exc)
        return None


def get_collection_stats() -> dict:
    """Return counts for the data collection dashboard."""
    try:
        from backend.db.models import TrainingSample, SyncSessionLocal
        from sqlalchemy import func
        with SyncSessionLocal() as session:
            total = session.query(func.count(TrainingSample.id)).scalar()
            verified = session.query(func.count(TrainingSample.id)).filter_by(is_verified=True).scalar()
            labelled = session.query(func.count(TrainingSample.id)).filter(
                TrainingSample.disease_id.isnot(None)
            ).scalar()
            by_lang = session.query(
                TrainingSample.language, func.count(TrainingSample.id)
            ).group_by(TrainingSample.language).all()
            by_crop = session.query(
                TrainingSample.crop_hint, func.count(TrainingSample.id)
            ).group_by(TrainingSample.crop_hint).order_by(
                func.count(TrainingSample.id).desc()
            ).limit(10).all()

        return {
            "total": total,
            "labelled": labelled,
            "verified": verified,
            "unlabelled": total - labelled,
            "by_language": dict(by_lang),
            "top_crops": dict(by_crop),
        }
    except Exception as exc:
        logger.error("Stats query failed: %s", exc)
        return {}
