from datetime import datetime
from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Column, Integer, String, Float, Text, DateTime, JSON, Boolean, create_engine
)
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

from backend.config import settings


class Base(DeclarativeBase):
    pass


class DiseaseKnowledge(Base):
    """Treatment knowledge base — embedded for RAG retrieval."""
    __tablename__ = "disease_knowledge"

    id = Column(Integer, primary_key=True)
    disease_name = Column(String(200), nullable=False, index=True)
    crop = Column(String(100), nullable=False)
    symptoms = Column(Text)
    cause = Column(String(200))
    treatment = Column(Text, nullable=False)
    prevention = Column(Text)
    severity = Column(String(20))  # low | medium | high | critical
    source = Column(String(200))
    embedding = Column(Vector(384))  # all-MiniLM-L6-v2 dim
    created_at = Column(DateTime, default=datetime.utcnow)


class SchemeKnowledge(Base):
    """Govt scheme knowledge base — embedded for RAG retrieval."""
    __tablename__ = "scheme_knowledge"

    id = Column(Integer, primary_key=True)
    scheme_name = Column(String(300), nullable=False, index=True)
    category = Column(String(100))
    benefit = Column(Text)
    eligibility = Column(Text)
    documents = Column(Text)
    how_to_apply = Column(Text)
    contact = Column(String(300))
    source = Column(String(200))
    embedding = Column(Vector(384))
    created_at = Column(DateTime, default=datetime.utcnow)


class FarmerSession(Base):
    """Persistent farmer memory — keyed by farmer_id (phone / anonymous ID)."""
    __tablename__ = "farmer_sessions"

    id = Column(Integer, primary_key=True)
    farmer_id = Column(String(100), nullable=False, index=True)
    crop_history = Column(JSON, default=list)       # list of crop names
    disease_history = Column(JSON, default=list)    # list of past diagnoses
    location = Column(String(200))
    language = Column(String(10), default="en")
    last_seen = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)


class AnalysisSession(Base):
    """Audit log of every farmer query — for eval and monitoring."""
    __tablename__ = "analysis_sessions"

    id = Column(Integer, primary_key=True)
    session_id = Column(String(36), nullable=False, unique=True)
    farmer_id = Column(String(100), index=True)
    image_filename = Column(String(300))
    user_query = Column(Text)
    intent = Column(String(50))
    classifier_label = Column(String(200))
    classifier_confidence = Column(Float)
    final_response = Column(Text)
    agent_trace = Column(JSON)
    cost_summary = Column(JSON)
    latency_ms = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)


class TrainingSample(Base):
    """
    Every image + farmer description captured via WhatsApp for model training.
    Unlabelled on arrival; annotators review via the annotation CLI to assign disease_id.
    """
    __tablename__ = "training_samples"

    id = Column(Integer, primary_key=True)
    image_path = Column(String(500), nullable=False)   # relative path under data/training_images/
    farmer_text = Column(Text)                          # raw message text (regional language)
    farmer_text_en = Column(Text)                       # auto-translated to English
    language = Column(String(10))                       # ISO 639-1 detected language
    crop_hint = Column(String(100))                     # crop extracted by NER
    location_hint = Column(String(200))                 # district/state from NER
    state = Column(String(100))                         # normalised state name
    district = Column(String(100))
    farmer_id_hash = Column(String(64))                 # SHA-256 of phone number (privacy)
    disease_id = Column(String(100))                    # FK to taxonomy disease id (after annotation)
    annotated_by = Column(String(100))                  # who labelled it
    is_verified = Column(Boolean, default=False)        # True after expert review
    split = Column(String(10), default="train")         # train | val | test
    source = Column(String(50), default="whatsapp")     # whatsapp | kvk | plantvillage
    created_at = Column(DateTime, default=datetime.utcnow)
    annotated_at = Column(DateTime)


class MarketPrice(Base):
    """Cached mandi price data."""
    __tablename__ = "market_prices"

    id = Column(Integer, primary_key=True)
    commodity = Column(String(100), nullable=False)
    market = Column(String(200))
    state = Column(String(100))
    min_price = Column(Float)
    max_price = Column(Float)
    modal_price = Column(Float)
    unit = Column(String(20), default="Quintal")
    arrival_date = Column(String(20))
    fetched_at = Column(DateTime, default=datetime.utcnow)


# Async engine for FastAPI
async_engine = create_async_engine(
    settings.database_url.replace("postgresql://", "postgresql+asyncpg://"),
    echo=False,
)
AsyncSessionLocal = async_sessionmaker(async_engine, expire_on_commit=False)

# Sync engine for scripts / ingestion
sync_engine = create_engine(settings.database_url)
SyncSessionLocal = sessionmaker(sync_engine)


async def get_db() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        yield session


async def init_db():
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
