#!/usr/bin/env python3
"""
Ingest government scheme records into pgvector scheme_knowledge table.
Run once (or after adding new schemes).

Usage: python scripts/ingest_schemes.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.db.models import Base, SchemeKnowledge, SyncSessionLocal, sync_engine
from backend.rag.vector_store import add_scheme

SCHEMES_FILE = Path(__file__).parent / "schemes_knowledge.json"


def main():
    print("Creating tables...")
    Base.metadata.create_all(sync_engine)

    records = json.loads(SCHEMES_FILE.read_text())
    print(f"Ingesting {len(records)} scheme records...")

    with SyncSessionLocal() as session:
        session.query(SchemeKnowledge).delete()
        session.commit()
        for i, record in enumerate(records, 1):
            add_scheme(session, record)
            print(f"  [{i}/{len(records)}] {record['scheme_name']}")
        session.commit()

    print(f"\nDone. {len(records)} schemes ingested.")


if __name__ == "__main__":
    main()
