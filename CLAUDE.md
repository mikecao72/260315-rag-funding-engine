# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RAG Funding Engine — a Retrieval-Augmented Generation system that transforms unstructured clinical consultation notes into ranked, explainable medical billing code recommendations with fee estimates. Built as a generic multi-schedule platform: any fee schedule PDF can be ingested via LLM-based dynamic profiling, not just ACC1520.

## Common Commands

```bash
# Install
pip install -r requirements.txt

# Run API server
PYTHONPATH=src uvicorn rag_funding_engine.api.main:app --host 0.0.0.0 --port 8010

# Run all tests
pytest tests/

# Run a single test file
pytest tests/test_gst_mode.py

# Run a single test
pytest tests/test_gst_mode.py::TestGstMode::test_default_gst_mode_is_excl

# Ingest the default ACC1520 schedule
python scripts/ingest_schedule.py

# Smoke test (requires ingested DB)
python scripts/eval_smoke.py

# Regression tests (requires ingested DB)
python scripts/run_regression.py
```

## Architecture

### Two-Pipeline Design

**Ingest Pipeline** (`pipeline/ingest_schedule.py`): PDF → text extraction (pypdf) → LLM profiling → regex code parsing → SQLite DB + embeddings + manifest.json

**Query Pipeline** (`pipeline/recommend.py`): Consult text → fact extraction (LLM/heuristic) → keyword scoring + heuristic boosts → de-duplication → context-sensitive gates → optional LLM adjudication → pricing rules → evidence-backed recommendations

### Multi-Schedule Isolation

Each ingested schedule gets its own directory under `data/processed/{schedule_id}/` containing an isolated SQLite database, manifest.json (with ScheduleProfile), and embeddings. The schedule_id routes all queries to the correct data.

### Key Source Modules

- `src/rag_funding_engine/api/main.py` — FastAPI app (port 8010): `/health`, `/ingest`, `/recommend`, `/schedules`, `/schedules/{id}`
- `src/rag_funding_engine/pipeline/ingest_schedule.py` — PDF ingestion, LLM-based ScheduleProfile generation, code row parsing, chunking, embedding
- `src/rag_funding_engine/pipeline/recommend.py` — Full recommendation pipeline with scoring, heuristic boosts, pricing rules (50% for additional treatments), and per-code reasoning
- `src/rag_funding_engine/pipeline/semantic.py` — OpenAI embeddings (`text-embedding-3-small`, 1536-d) with deterministic hash fallback (128-d) for offline dev
- `src/rag_funding_engine/pipeline/constraints.py` — Prototype constraint engine (duplicate family filtering, consult-type gating)
- `src/rag_funding_engine/db/models.py` — SQLAlchemy ORM: `ScheduleVersion`, `ScheduleCode`, `BillingRule` tables

### Data Flow

```
data/raw/*.pdf  →  ingest  →  data/processed/{schedule_id}/
                                 ├── {schedule_id}.sqlite3
                                 ├── manifest.json (includes ScheduleProfile)
                                 └── embeddings/
```

### ScheduleProfile

LLM-generated metadata stored in manifest.json. Contains `schedule_type`, `key_dimensions`, `unique_rules`, `query_patterns`, and flags like `location_matters`, `mileage_reimbursable`, `after_hours_premium`. This profile drives query generation and constraint application for each schedule.

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `OPENAI_API_KEY` | — | Required for embeddings and LLM features |
| `LLM_MODEL` | `gpt-4.1-mini` | Consult fact extraction model |
| `FINAL_REASONING_MODEL` | `gpt-5.4` | Final recommendation adjudication model |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |

## Domain-Specific Logic

- **GST mode**: `gst_mode="excl"` (default) or `"incl"` controls which fee is the primary `fee` field; both `fee_excl_gst` and `fee_incl_gst` are always returned
- **Multi-treatment pricing**: First treatment at full rate, additional same-visit treatments at 50%
- **Context-sensitive gates**: Specific routing for distal radius/ulna fractures, tibia/fibula fractures, epistaxis with nasal packing, multi-site abrasions
- **Negation detection**: Heuristic-based (e.g., "no fracture" suppresses fracture codes)
- **Code row regex**: Pattern `^([A-Z]{2,6}\d{0,3})\s+(.+)$` with fee extraction `\b\d{1,4}\.\d{2}\b`

## Testing

Tests use pytest with monkeypatching. `test_gst_mode.py` creates mock SQLite databases inline. `regression_cases.json` defines expected outputs for 3 clinical scenarios (expected codes, top code, total fee within tolerance). Smoke/regression scripts exit 0/1/2 (pass/fail/missing DB).
