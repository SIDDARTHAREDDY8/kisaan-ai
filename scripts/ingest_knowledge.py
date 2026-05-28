#!/usr/bin/env python3
"""
Ingest plant disease treatment records into pgvector knowledge base.
Run once before starting the backend server.

Usage: python scripts/ingest_knowledge.py
"""
import json
import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.db.models import Base, DiseaseKnowledge, SyncSessionLocal, sync_engine
from backend.rag.vector_store import add_knowledge

KNOWLEDGE_FILE = Path(__file__).parent / "knowledge_base.json"


def main():
    print("Creating tables...")
    Base.metadata.create_all(sync_engine)

    records = json.loads(KNOWLEDGE_FILE.read_text())
    print(f"Ingesting {len(records)} records...")

    with SyncSessionLocal() as session:
        # Clear existing records for idempotent re-runs
        session.query(DiseaseKnowledge).delete()
        session.commit()

        for i, record in enumerate(records, 1):
            add_knowledge(session, record)
            print(f"  [{i}/{len(records)}] {record['crop']} — {record['disease_name']}")

        session.commit()

    print(f"\nDone. {len(records)} records ingested into pgvector.")


if __name__ == "__main__":
    main()
