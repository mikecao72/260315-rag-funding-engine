from pathlib import Path
from rag_funding_engine.pipeline.ingest_acc1520 import ingest_schedule


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    pdf_path = root / "data" / "raw" / "ACC1520-Med-pract-nurse-pract-and-nurses-costs-v2.pdf"
    out_dir = root / "data" / "processed"

    result = ingest_schedule(pdf_path=pdf_path, out_dir=out_dir)
    print("Ingestion complete")
    print(f"DB: {result.db_path}")
    print(f"Parsed codes: {result.code_count}")
    print(f"Indexed chunks: {result.indexed_chunks}")
    print(f"Manifest: {result.manifest_path}")


if __name__ == "__main__":
    main()
