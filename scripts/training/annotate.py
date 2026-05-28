"""
Annotation CLI — Review and label collected WhatsApp training samples

Usage:
  python scripts/training/annotate.py           # interactive mode
  python scripts/training/annotate.py --stats   # show collection stats
  python scripts/training/annotate.py --export  # export labelled samples

Controls during annotation:
  Enter disease number → label the sample
  s → skip
  q → quit
  ? → show disease list
  open → open image in default viewer
"""
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
TAXONOMY_PATH = ROOT / "data" / "diseases" / "taxonomy.json"
sys.path.insert(0, str(ROOT))


def load_taxonomy():
    with open(TAXONOMY_PATH) as f:
        return json.load(f)


def print_disease_menu(diseases: list[dict], crop_filter: str = ""):
    filtered = [d for d in diseases if not crop_filter or d["crop"] == crop_filter]
    print(f"\n{'─'*60}")
    print(f"  Disease classes ({len(filtered)}):")
    for i, d in enumerate(filtered, 1):
        en_name = d["english"]
        te_name = d["regional_names"].get("te", [""])[0]
        hi_name = d["regional_names"].get("hi", [""])[0]
        print(f"  {i:3}. [{d['crop']:12}] {en_name:30} | {hi_name}")
    print(f"{'─'*60}")
    return filtered


def annotate_interactive():
    taxonomy = load_taxonomy()
    diseases = taxonomy["diseases"]

    from backend.db.models import TrainingSample, SyncSessionLocal
    from sqlalchemy import func

    while True:
        with SyncSessionLocal() as session:
            sample = session.query(TrainingSample).filter(
                TrainingSample.disease_id.is_(None),
                TrainingSample.image_path.isnot(None),
            ).order_by(func.random()).first()

            if not sample:
                print("\n✅ All samples labelled! No more unlabelled images.")
                break

            sample_id = sample.id
            img_path = ROOT / sample.image_path
            farmer_text = sample.farmer_text or "(no text)"
            language = sample.language or "?"
            crop_hint = sample.crop_hint or "?"
            location = sample.location_hint or "?"

        print(f"\n{'═'*60}")
        print(f"  Sample #{sample_id}")
        print(f"  Image:    {img_path.name}")
        print(f"  Language: {language} | Crop hint: {crop_hint} | Location: {location}")
        print(f"  Farmer said: \"{farmer_text[:120]}\"")
        print(f"{'═'*60}")

        # Filter disease list to crop_hint if we have one
        filtered = print_disease_menu(diseases, crop_filter=crop_hint if crop_hint != "?" else "")
        if not filtered:
            filtered = print_disease_menu(diseases)

        print("\n  Enter number, 's' skip, 'q' quit, '?' all diseases, 'open' view image:")
        choice = input("  > ").strip().lower()

        if choice == "q":
            print("Exiting annotation tool.")
            break
        elif choice == "s":
            print("  Skipped.")
            continue
        elif choice == "open":
            os.system(f"open '{img_path}'" if sys.platform == "darwin" else f"xdg-open '{img_path}'")
            continue
        elif choice == "?":
            print_disease_menu(diseases)
            continue
        elif choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(filtered):
                selected = filtered[idx]
                annotator = os.environ.get("ANNOTATOR", "admin")

                with SyncSessionLocal() as session:
                    s = session.query(TrainingSample).get(sample_id)
                    if s:
                        s.disease_id = selected["id"]
                        s.annotated_by = annotator
                        s.annotated_at = datetime.utcnow()
                        session.commit()

                print(f"  ✅ Labelled as: {selected['english']} ({selected['id']})")
            else:
                print("  Invalid number. Try again.")
        else:
            print("  Unknown command.")


def show_stats():
    from backend.services.data_collector import get_collection_stats
    stats = get_collection_stats()
    print(f"\n{'═'*50}")
    print("  Training Data Collection Stats")
    print(f"{'═'*50}")
    print(f"  Total images:     {stats.get('total', 0)}")
    print(f"  Labelled:         {stats.get('labelled', 0)}")
    print(f"  Verified:         {stats.get('verified', 0)}")
    print(f"  Unlabelled:       {stats.get('unlabelled', 0)}")
    print(f"\n  By Language:")
    for lang, count in sorted(stats.get('by_language', {}).items(), key=lambda x: -x[1]):
        print(f"    {lang:6}: {count}")
    print(f"\n  Top Crops:")
    for crop, count in list(stats.get('top_crops', {}).items())[:8]:
        print(f"    {(crop or 'unknown'):15}: {count}")
    print(f"{'═'*50}\n")


def export_labelled(output_file: str = "data/labelled_export.jsonl"):
    from backend.db.models import TrainingSample, SyncSessionLocal
    out_path = ROOT / output_file
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with SyncSessionLocal() as session:
        samples = session.query(TrainingSample).filter(
            TrainingSample.disease_id.isnot(None)
        ).all()

    count = 0
    with open(out_path, "w") as f:
        for s in samples:
            img_path = ROOT / s.image_path
            if not img_path.exists():
                continue
            f.write(json.dumps({
                "id": s.id,
                "image_path": str(img_path),
                "disease_id": s.disease_id,
                "language": s.language,
                "crop_hint": s.crop_hint,
                "location": s.location_hint,
                "state": s.state,
                "farmer_text": s.farmer_text,
                "farmer_text_en": s.farmer_text_en,
                "is_verified": s.is_verified,
                "source": s.source,
                "annotated_by": s.annotated_by,
            }) + "\n")
            count += 1

    print(f"✅ Exported {count} labelled samples to {out_path}")


def verify_mode():
    """Mark samples as expert-verified after agronomist review."""
    from backend.db.models import TrainingSample, SyncSessionLocal
    from sqlalchemy import func

    while True:
        with SyncSessionLocal() as session:
            sample = session.query(TrainingSample).filter(
                TrainingSample.disease_id.isnot(None),
                TrainingSample.is_verified.is_(False),
            ).order_by(func.random()).first()

            if not sample:
                print("\n✅ All labelled samples are verified!")
                break

            sample_id = sample.id
            img_path = ROOT / sample.image_path
            disease_id = sample.disease_id
            annotated_by = sample.annotated_by or "unknown"

        taxonomy = load_taxonomy()
        disease = next((d for d in taxonomy["diseases"] if d["id"] == disease_id), None)
        disease_name = disease["english"] if disease else disease_id

        print(f"\n{'═'*60}")
        print(f"  Sample #{sample_id} — labelled by: {annotated_by}")
        print(f"  Diagnosis: {disease_name} ({disease_id})")
        print(f"  Image: {img_path.name}")
        print(f"\n  y = confirm correct | n = wrong (unlabel) | o = open | q = quit")

        choice = input("  > ").strip().lower()
        if choice == "q":
            break
        elif choice == "o":
            os.system(f"open '{img_path}'" if sys.platform == "darwin" else f"xdg-open '{img_path}'")
        elif choice == "y":
            with SyncSessionLocal() as session:
                s = session.query(TrainingSample).get(sample_id)
                if s:
                    s.is_verified = True
                    session.commit()
            print("  ✅ Verified")
        elif choice == "n":
            with SyncSessionLocal() as session:
                s = session.query(TrainingSample).get(sample_id)
                if s:
                    s.disease_id = None
                    s.annotated_by = None
                    s.annotated_at = None
                    s.is_verified = False
                    session.commit()
            print("  ↩ Returned to unlabelled queue")


def main():
    parser = argparse.ArgumentParser(description="Kisaan AI — Annotation Tool")
    parser.add_argument("--stats", action="store_true", help="Show collection stats")
    parser.add_argument("--export", action="store_true", help="Export labelled samples to JSONL")
    parser.add_argument("--verify", action="store_true", help="Expert verification mode")
    parser.add_argument("--output", default="data/labelled_export.jsonl")
    args = parser.parse_args()

    if args.stats:
        show_stats()
    elif args.export:
        export_labelled(args.output)
    elif args.verify:
        verify_mode()
    else:
        annotate_interactive()


if __name__ == "__main__":
    main()
