from pathlib import Path
from rag_funding_engine.pipeline.ingest_schedule import ingest_schedule


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    pdf_path = root / "data" / "raw" / "ACC1520-Med-pract-nurse-pract-and-nurses-costs-v2.pdf"
    out_dir = root / "data" / "processed"
    schedule_id = "acc1520-medical"

    result = ingest_schedule(
        pdf_path=pdf_path,
        schedule_id=schedule_id,
        out_dir=out_dir,
    )
    print("Ingestion complete")
    print(f"Schedule ID: {result.schedule_id}")
    print(f"DB: {result.db_path}")
    print(f"Parsed codes: {result.code_count}")
    print(f"Indexed chunks: {result.indexed_chunks}")
    print(f"Manifest: {result.manifest_path}")
    print(f"Profile type: {result.profile.schedule_type}")


if __name__ == "__main__":
    main()
