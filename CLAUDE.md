# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RAG Funding Engine — a Retrieval-Augmented Generation system that transforms unstructured clinical consultation notes into ranked, explainable medical billing code recommendations with fee estimates. Built as a generic multi-schedule platform: any fee schedule PDF can be ingested via LLM-based dynamic profiling, not just ACC1520.

## Common Commands

```bash
# Install
pip install -r requirements.txt

# Run API server (works on Windows, Mac, Linux — no PYTHONPATH needed)
python run.py

# Run on a custom port
python run.py --port 9000

# Run with auto-reload during development
python run.py --reload

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

**Ingest Pipeline** (`pipeline/ingest_schedule.py`): PDF → text extraction (pypdf) → LLM profiling → LLM schedule guidance generation → regex code parsing → SQLite DB + embeddings + manifest.json

**Query Pipeline** (`pipeline/recommend.py`): Consult text → fact extraction (LLM) → LLM code selection (wide net) → reasoning model adjudication (narrowing) → pricing rules → evidence-backed recommendations

### Query Pipeline — Detailed Workflow

When a user submits a job via `POST /recommend` with a transcription or SOAP note, the following stages execute in sequence inside `recommend_codes()`:

#### Stage 1: Input Reception & Profile Loading
- The API receives `consult_text` (free-text transcription/SOAP note) and/or `consult_template` (structured key-value facts), along with `schedule_id`, `top_n`, `gst_mode`, and `min_confidence`.
- The schedule's `manifest.json` is loaded to retrieve both the `ScheduleProfile` and the `schedule_guidance` (document scope, code groupings, selection rules). These shape the LLM's code selection behaviour.
- The schedule's SQLite database is opened and all `schedule_codes` rows (code, description, fee_excl_gst, fee_incl_gst, page) are fetched. Header artifacts (`ACC`, `Code`) are filtered out.

#### Stage 2: Consult Fact Extraction (Input Condensation)
- If a `consult_template` is provided, it is used directly as structured facts (`mode: "template"`).
- Otherwise the raw consult text is sent to the LLM (`LLM_MODEL`, default `gpt-4.1-mini`) to extract billing-relevant facts as strict JSON. The extraction keys are driven by the profile's `key_dimensions` (e.g., `injury_type`, `body_site`, `procedure`, `severity`).
- The consult text and any template values are concatenated into a single `query_text` string.

#### Stage 3: LLM Code Selection (Wide Net)
- The full `query_text`, extracted consult facts, complete `schedule_codes` table (code + description, no fee amounts), and `schedule_guidance` are sent to the LLM (`LLM_MODEL`, default `gpt-4.1-mini`).
- The LLM receives the schedule guidance (document scope, code groupings, selection rules) as context for how to interpret the schedule.
- The LLM returns a list of `{code, confidence, reason}` for every code it considers potentially relevant.
- Confidence is 0.0–1.0: `1.0` = definitely applies, `0.7` = very likely, `0.4` = possibly, `0.2` = unlikely but worth considering.
- Results are filtered by `min_confidence` threshold (default `0.1` — wide net) and capped at `top_n` (default `10`).
- **No fallback**: `OPENAI_API_KEY` is required. Without it the pipeline returns an error.

#### Stage 4: Reasoning Model Adjudication (Narrowing)
- The wide-net candidates, consultation note, and extracted facts are sent to the `FINAL_REASONING_MODEL` (default `gpt-5.4`).
- The reasoning model acts as a clinical billing adjudicator: it returns `{code, keep, reason}` decisions per candidate.
- Rules enforced: respect negations, prefer clinically-supported items, avoid same-family duplicates, keep exactly one consultation code, cap at 8 final codes.
- If the reasoning model call fails, the wide-net candidates pass through unchanged.

#### Stage 5: Pricing Rules
- Consultation codes (GP*, GN*, NN*, NC*, NU*, PM*) priced at 100%.
- Treatment codes sorted by fee descending — highest-fee treatment at 100%, additional treatments at 50% if multiple injury sites are detected.
- Each rec gets `pricing_multiplier`, `line_total_excl_gst`, and `quantity` fields.

#### Stage 6: Evidence & Reasoning Assembly
- **Per-code reasoning**: combines the LLM selector's reason, the adjudicator's reason, and the pricing rule into a single `reason` string.
- **Semantic evidence chunks** (`_fetch_semantic_chunks`): the query text is embedded via OpenAI `text-embedding-3-small` and compared via cosine similarity against pre-indexed `policy_chunk_embeddings` in the SQLite DB. Top 3 most similar chunks are returned as supporting evidence.

#### Stage 7: Response Assembly
The final JSON response includes:
| Field | Description |
|-------|-------------|
| `query` | The combined query text used for matching |
| `consult_facts` | Extracted facts with mode indicator (llm/template) |
| `schedule_id` | Which schedule was queried |
| `profile` | The schedule's LLM-generated profile metadata |
| `schedule_guidance` | The schedule's LLM-generated usage instructions |
| `gst_mode` | Whether primary fee is excl or incl GST |
| `min_confidence` | The confidence threshold used for code selection |
| `recommendations[]` | Ranked list with: `code`, `description`, `fee`, `fee_gst`, `fee_excl_gst`, `fee_incl_gst`, `page`, `confidence`, `selector_reason`, `adjudicator_reason`, `pricing_multiplier`, `line_total_excl_gst`, `quantity`, `reason` |
| `estimated_total_excl_gst` | Sum of all `line_total_excl_gst` values |
| `estimated_total_incl_gst` | Sum of all incl-GST line totals |
| `evidence_chunks[]` | Top 3 supporting policy text snippets with page and similarity score |
| `notes` | System-level annotations |
| `job_id` | Unique job identifier (YYMMDD-HHMMSS format) |
| `job_log` | Filesystem path to the full pipeline audit log |

### Multi-Schedule Isolation

Each ingested schedule gets its own directory under `data/processed/{schedule_id}/` containing an isolated SQLite database, manifest.json (with ScheduleProfile), and embeddings. The schedule_id routes all queries to the correct data.

### Key Source Modules

- `run.py` — Server launcher (`python run.py`). Adds `src/` to path automatically, works on Windows/Mac/Linux without `PYTHONPATH`
- `src/rag_funding_engine/api/main.py` — FastAPI app (port 8010): `/health`, `/ingest`, `/recommend`, `/schedules`, `/schedules/{id}`, `/db/{id}/tables`, `/db/{id}/tables/{name}`, `/jobs`, `/jobs/{id}`, `/jobs/{id}/log`, `/jobs-ui`
- `src/rag_funding_engine/pipeline/ingest_schedule.py` — PDF ingestion, LLM-based ScheduleProfile + schedule_guidance generation, code row parsing, chunking, embedding
- `src/rag_funding_engine/pipeline/recommend.py` — LLM-based code selection (wide net) → reasoning model adjudication (narrowing) → pricing rules → evidence assembly. Every step is logged via `JobLogger`.
- `src/rag_funding_engine/pipeline/job_logger.py` — Per-job logging. Creates `logs/YYMMDD/YYMMDD-HHMMSS/` with `job.log` (human-readable pipeline trace) and `job.json` (full input + output for frontend review).
- `src/rag_funding_engine/pipeline/semantic.py` — OpenAI embeddings (`text-embedding-3-small`, 1536-d) with deterministic hash fallback (128-d) for offline dev
- `src/rag_funding_engine/pipeline/constraints.py` — Prototype constraint engine (may be deprecated — LLM handles dedup/gating now)
- `src/rag_funding_engine/db/models.py` — SQLAlchemy ORM: `ScheduleVersion`, `ScheduleCode`, `BillingRule` tables

### Data Flow

```
data/raw/*.pdf  →  ingest  →  data/processed/{schedule_id}/
                                 ├── {schedule_id}.sqlite3
                                 ├── manifest.json (ScheduleProfile + schedule_guidance)
                                 └── embeddings/
```

### ScheduleProfile

LLM-generated metadata stored in manifest.json. Contains `schedule_type`, `key_dimensions`, `unique_rules`, `query_patterns`, and flags like `location_matters`, `mileage_reimbursable`, `after_hours_premium`. This profile drives consult fact extraction.

### Schedule Guidance

LLM-generated instructions stored in manifest.json under `schedule_guidance`. Generated during ingestion by analyzing the parsed codes and document text. Contains:
- `document_scope`: What the schedule covers, its environment, and what is out of scope
- `code_groupings`: Array of `{codes, category, description}` grouping related codes (e.g., nurse consultation codes, fracture codes)
- `selection_rules`: Array of rules the code selector LLM must follow (e.g., "always select one consultation code", "select wound code based on length")

### Job Logging

Every recommendation run creates a job folder under `logs/YYMMDD/YYMMDD-HHMMSS/` containing:
- `job.log` — Human-readable step-by-step trace: input, profile/guidance loaded, all DB codes, LLM prompts and raw responses, selected/dropped codes, pricing, evidence chunks, final output
- `job.json` — Machine-readable snapshot of full input parameters and API response, used by the Jobs review page

The `job_id` and `job_log` path are returned in the API response for traceability.

### Frontend Pages

- **`/`** (index.html) — Main app with tabs: Recommend, Schedules, Ingest, DB Inspector. Links to Jobs page.
- **`/jobs-ui`** (jobs.html) — Standalone job review page:
  - **Job list** — Filterable/sortable table of all past jobs (by date, schedule, description, code count, total)
  - **Tab 1 (Input/Output)** — Replays the recommendation: consult text on the left, results table on the right (read-only, matching the Recommend layout)
  - **Tab 2 (Audit Log)** — Full `job.log` rendered as preformatted text

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `OPENAI_API_KEY` | — | Required for embeddings and LLM features |
| `LLM_MODEL` | `gpt-4.1-mini` | Consult fact extraction + code selection model |
| `FINAL_REASONING_MODEL` | `gpt-5.4` | Final recommendation adjudication model |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |

## Domain-Specific Logic

- **GST mode**: `gst_mode="excl"` (default) or `"incl"` controls which fee is the primary `fee` field; both `fee_excl_gst` and `fee_incl_gst` are always returned
- **Multi-treatment pricing**: First treatment at full rate, additional same-visit treatments at 50%
- **Confidence threshold**: `min_confidence` (default `0.1`) controls how wide a net the LLM code selector casts. Lower = more candidates for review. The reasoning model then narrows.
- **LLM-driven code selection**: All clinical logic (negation detection, code routing, mutual exclusivity) is handled by the LLM with schedule_guidance context, not heuristic rules
- **Code row regex**: Pattern `^([A-Z]{2,6}\d{0,3})\s+(.+)$` with fee extraction `\b\d{1,4}\.\d{2}\b`

## Testing

Tests use pytest with monkeypatching. `test_gst_mode.py` creates mock SQLite databases inline. `regression_cases.json` defines expected outputs for 3 clinical scenarios (expected codes, top code, total fee within tolerance). Smoke/regression scripts exit 0/1/2 (pass/fail/missing DB).
