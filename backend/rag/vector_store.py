"""pgvector-backed RAG store for disease treatments and govt schemes."""
import logging
from functools import lru_cache

from sentence_transformers import SentenceTransformer
from sqlalchemy import text
from sqlalchemy.orm import Session

from backend.config import settings
from backend.db.models import DiseaseKnowledge, SchemeKnowledge, SyncSessionLocal

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load_embedder() -> SentenceTransformer:
    logger.info("Loading embedding model: %s", settings.embedding_model)
    return SentenceTransformer(settings.embedding_model)


def embed(text_input: str) -> list[float]:
    model = _load_embedder()
    return model.encode(text_input, normalize_embeddings=True).tolist()


# ── Disease treatment retrieval ───────────────────────────────────────────────

def retrieve_treatment(disease_name: str, crop: str, top_k: int = 3) -> list[dict]:
    query = f"{crop} {disease_name} treatment symptoms prevention"
    query_vec = embed(query)

    with SyncSessionLocal() as session:
        rows = session.execute(
            text("""
                SELECT disease_name, crop, symptoms, cause, treatment,
                       prevention, severity,
                       1 - (embedding <=> CAST(:vec AS vector)) AS similarity
                FROM disease_knowledge
                ORDER BY embedding <=> CAST(:vec AS vector)
                LIMIT :k
            """),
            {"vec": str(query_vec), "k": top_k},
        ).fetchall()

    return [
        {
            "disease_name": r.disease_name,
            "crop": r.crop,
            "symptoms": r.symptoms,
            "cause": r.cause,
            "treatment": r.treatment,
            "prevention": r.prevention,
            "severity": r.severity,
            "similarity": round(r.similarity, 4),
        }
        for r in rows
    ]


def add_knowledge(session: Session, record: dict) -> None:
    text_for_embedding = (
        f"{record['disease_name']} {record['crop']} "
        f"{record.get('symptoms', '')} {record.get('treatment', '')}"
    )
    vec = embed(text_for_embedding)
    entry = DiseaseKnowledge(
        **{k: v for k, v in record.items()},
        embedding=vec,
    )
    session.add(entry)


# ── Govt scheme retrieval ─────────────────────────────────────────────────────

def retrieve_schemes(query: str, top_k: int = 3) -> list[dict]:
    query_vec = embed(query)

    with SyncSessionLocal() as session:
        rows = session.execute(
            text("""
                SELECT scheme_name, category, benefit, eligibility,
                       documents, how_to_apply, contact,
                       1 - (embedding <=> CAST(:vec AS vector)) AS similarity
                FROM scheme_knowledge
                ORDER BY embedding <=> CAST(:vec AS vector)
                LIMIT :k
            """),
            {"vec": str(query_vec), "k": top_k},
        ).fetchall()

    return [
        {
            "scheme_name": r.scheme_name,
            "category": r.category,
            "benefit": r.benefit,
            "eligibility": r.eligibility,
            "documents": r.documents,
            "how_to_apply": r.how_to_apply,
            "contact": r.contact,
            "similarity": round(r.similarity, 4),
        }
        for r in rows
    ]


def add_scheme(session: Session, record: dict) -> None:
    text_for_embedding = (
        f"{record['scheme_name']} {record.get('category', '')} "
        f"{record.get('benefit', '')} {record.get('eligibility', '')}"
    )
    vec = embed(text_for_embedding)
    entry = SchemeKnowledge(
        **{k: v for k, v in record.items()},
        embedding=vec,
    )
    session.add(entry)
