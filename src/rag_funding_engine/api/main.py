from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

from rag_funding_engine.pipeline.ingest_acc1520 import ingest_schedule
from rag_funding_engine.pipeline.recommend import recommend_codes


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PDF = ROOT / "data" / "raw" / "ACC1520-Med-pract-nurse-pract-and-nurses-costs-v2.pdf"
DEFAULT_OUT = ROOT / "data" / "processed"
DEFAULT_DB = DEFAULT_OUT / "acc1520.sqlite3"


app = FastAPI(title="RAG Funding Engine", version="0.2.0")


class RecommendRequest(BaseModel):
    consult_text: str | None = None
    consult_template: dict | None = None
    top_n: int = 5


@app.get("/health")
def health() -> dict:
    return {"ok": True, "service": "rag-funding-engine"}


@app.post("/ingest")
def ingest() -> dict:
    result = ingest_schedule(pdf_path=DEFAULT_PDF, out_dir=DEFAULT_OUT)
    return {
        "status": "ok",
        "db_path": result.db_path,
        "code_count": result.code_count,
        "indexed_chunks": result.indexed_chunks,
        "manifest": result.manifest_path,
    }


@app.post("/recommend")
def recommend(payload: RecommendRequest) -> dict:
    if not DEFAULT_DB.exists():
        ingest_schedule(pdf_path=DEFAULT_PDF, out_dir=DEFAULT_OUT)

    return recommend_codes(
        db_path=DEFAULT_DB,
        consult_text=payload.consult_text,
        consult_template=payload.consult_template,
        top_n=payload.top_n,
    )
