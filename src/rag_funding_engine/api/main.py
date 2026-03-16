from pathlib import Path
from typing import Any

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[3] / ".env")

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from rag_funding_engine.pipeline.ingest_schedule import ingest_schedule, ScheduleProfile
from rag_funding_engine.pipeline.recommend import recommend_codes


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PDF = ROOT / "data" / "raw" / "ACC1520-Med-pract-nurse-pract-and-nurses-costs-v2.pdf"
DEFAULT_OUT = ROOT / "data" / "processed"

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

app = FastAPI(title="RAG Funding Engine", version="0.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class RecommendRequest(BaseModel):
    consult_text: str | None = None
    consult_template: dict | None = None
    schedule_id: str = "acc1520-medical"
    top_n: int = 5
    gst_mode: str = "excl"  # "excl" | "incl" - controls which fee is returned as primary


@app.get("/health")
def health() -> dict:
    return {"ok": True, "service": "rag-funding-engine"}


@app.post("/ingest")
def ingest(
    pdf: UploadFile = File(...),
    schedule_id: str = Form(...),
    llm_model: str = Form("gpt-4o"),
) -> dict:
    """
    Ingest any schedule document.
    User provides schedule_id (e.g., "acc1520-rural").
    LLM dynamically profiles the document.
    """
    # Save uploaded file temporarily
    temp_path = DEFAULT_OUT / "_temp" / pdf.filename
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    
    with temp_path.open("wb") as f:
        content = pdf.file.read()
        f.write(content)
    
    try:
        result = ingest_schedule(
            pdf_path=temp_path,
            schedule_id=schedule_id,
            out_dir=DEFAULT_OUT,
            llm_model=llm_model,
        )
        
        return {
            "status": "ok",
            "schedule_id": result.schedule_id,
            "db_path": result.db_path,
            "code_count": result.code_count,
            "indexed_chunks": result.indexed_chunks,
            "manifest": result.manifest_path,
            "profile": {
                "schedule_type": result.profile.schedule_type,
                "description": result.profile.description,
                "key_dimensions": result.profile.key_dimensions,
                "location_matters": result.profile.location_matters,
                "mileage_reimbursable": result.profile.mileage_reimbursable,
                "after_hours_premium": result.profile.after_hours_premium,
            },
        }
    finally:
        # Clean up temp file
        if temp_path.exists():
            temp_path.unlink()


@app.post("/recommend")
def recommend(payload: RecommendRequest) -> dict:
    """
    Recommend billing codes for a consultation.
    Loads schedule's profile → generates queries → searches → returns codes.
    """
    return recommend_codes(
        schedule_id=payload.schedule_id,
        consult_text=payload.consult_text,
        consult_template=payload.consult_template,
        base_dir=DEFAULT_OUT,
        top_n=payload.top_n,
        gst_mode=payload.gst_mode,
    )


@app.get("/schedules")
def list_schedules() -> dict:
    """List all ingested schedules."""
    schedules = []
    
    if DEFAULT_OUT.exists():
        for item in DEFAULT_OUT.iterdir():
            if item.is_dir() and not item.name.startswith("_"):
                manifest_path = item / "manifest.json"
                if manifest_path.exists():
                    import json
                    try:
                        manifest = json.loads(manifest_path.read_text())
                        schedules.append({
                            "schedule_id": manifest.get("schedule_id"),
                            "description": manifest.get("profile", {}).get("description"),
                            "schedule_type": manifest.get("profile", {}).get("schedule_type"),
                            "codes_parsed": manifest.get("codes_parsed"),
                            "chunks_indexed": manifest.get("chunks_indexed"),
                        })
                    except Exception:
                        pass
    
    return {"schedules": schedules}


@app.get("/schedules/{schedule_id}")
def get_schedule(schedule_id: str) -> dict:
    """Get details of a specific schedule."""
    import json
    
    manifest_path = DEFAULT_OUT / schedule_id / "manifest.json"
    if not manifest_path.exists():
        return {"error": f"Schedule not found: {schedule_id}"}
    
    try:
        manifest = json.loads(manifest_path.read_text())
        return {
            "schedule_id": manifest.get("schedule_id"),
            "source": manifest.get("source"),
            "source_hash": manifest.get("source_hash"),
            "pages": manifest.get("pages"),
            "codes_parsed": manifest.get("codes_parsed"),
            "chunks_indexed": manifest.get("chunks_indexed"),
            "profile": manifest.get("profile"),
        }
    except Exception as e:
        return {"error": f"Failed to load schedule: {e}"}


@app.get("/")
def index():
    """Serve the frontend."""
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
