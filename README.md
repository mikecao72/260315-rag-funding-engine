# RAG Funding Engine

A RAG-driven (Retrieval-Augmented Generation) system for medical billing code recommendation. This engine transforms unstructured clinical consultation notes into ranked, explainable funding code recommendations with fee estimates.

**Now with generic schedule support** - ingest ANY fee schedule document, not just ACC1520!

---

## Table of Contents

1. [Overview](#overview)
2. [Generic Schedule Ingestion](#generic-schedule-ingestion)
3. [Data Model](#data-model)
4. [Ingestion Pipeline](#ingestion-pipeline)
5. [Query Pipeline](#query-pipeline)
6. [API](#api)
7. [Configuration](#configuration)
8. [Running It](#running-it)
9. [Testing](#testing)

---

## Overview

### What Problem Does This Solve?

Medical fee schedules are complex documents containing hundreds of billing codes, fees, and rules. Clinicians face the challenge of:
- **Mapping** unstructured clinical notes to the correct billing codes
- **Navigating** complex rules (eligibility, age bands, procedure combinations, location modifiers)
- **Optimizing** revenue while staying compliant
- **Documenting** rationale for audit purposes

The RAG Funding Engine solves this by:
1. **Ingesting** ANY fee schedule PDF into structured relational data + vector embeddings
2. **Dynamically profiling** the schedule using LLM analysis
3. **Accepting** free-text clinical consultation notes
4. **Retrieving** candidate codes via keyword matching + semantic search
5. **Filtering** through constraint rules and heuristics
6. **Ranking** with an optional LLM adjudication layer
7. **Explaining** every recommendation with quoted evidence from the input

### Example Use Case

**Input:**
```
17-year-old male high school student. Dependent child of a Community Services Card holder.
Presented to urgent care clinic. Seen by nurse and GP together.
Multiple abrasions: left forearm 9 cm x 5 cm, right knee 7 cm x 6 cm, right lateral hip/flank 8 cm x 4 cm.
Embedded dirt/gravel. No deep laceration requiring suturing. No fracture suspected clinically.
Active anterior epistaxis from right nostril. Persistent epistaxis treated with nasal packing.
```

**Output:**
- `GNCD` — Dependent CSC nurse+GP consultation (full fee)
- `MB5` — Significant multi-site abrasion treatment (full fee)
- `MM8` — Epistaxis with nasal packing (full fee)
- `MM5` — Foreign body removal from abrasions (50% — additional treatment rule)

---

## Generic Schedule Ingestion

The engine now supports **any** fee schedule document through dynamic profiling.

### ScheduleProfile

When you ingest a schedule, the LLM analyzes it and generates a `ScheduleProfile`:

```python
@dataclass
class ScheduleProfile:
    schedule_type: str           # "urgent_care", "rural", "gp", "specialist", etc.
    description: str             # Human-readable description
    key_dimensions: list[str]    # Factors that affect billing
    unique_rules: list[str]      # Schedule-specific billing rules
    query_patterns: list[str]    # Template patterns for searching
    example_queries: list[str]   # Example natural language queries
    location_matters: bool       # Whether location affects pricing
    mileage_reimbursable: bool   # Whether travel is billable
    after_hours_premium: bool    # Whether after-hours premiums apply
```

### Example Profiles

**ACC1520 Medical Schedule:**
```json
{
  "schedule_type": "urgent_care",
  "description": "ACC urgent care schedule for medical practitioners and nurses",
  "key_dimensions": ["injury_type", "body_site", "procedure", "severity", "age_band"],
  "unique_rules": ["CSC eligibility affects consultation codes", "Multi-treatment pricing applies"],
  "query_patterns": ["{procedure} {body_site}", "{injury_type} treatment"],
  "example_queries": ["laceration hand", "distal radius fracture"],
  "location_matters": false,
  "mileage_reimbursable": false,
  "after_hours_premium": false
}
```

**Rural Practitioner Schedule:**
```json
{
  "schedule_type": "rural",
  "description": "Rural practitioner schedule with travel modifiers",
  "key_dimensions": ["location", "travel_time", "mileage", "after_hours"],
  "unique_rules": ["Travel >20km bills separately", "Rural loading applies"],
  "query_patterns": ["{specialty} {location}", "travel {distance}"],
  "example_queries": ["GP rural clinic", "travel 45km"],
  "location_matters": true,
  "mileage_reimbursable": true,
  "after_hours_premium": true
}
```

---

## Data Model

### Per-Schedule Storage Structure

Each schedule gets its own directory:

```
data/processed/{schedule_id}/
├── {schedule_id}.sqlite3       # Database with codes, chunks, embeddings
├── manifest.json               # Includes ScheduleProfile
└── embeddings/                 # Optional external vector store
```

### SQLite Relational Database

The system uses SQLite with the following generic schema:

#### `schedule_versions`
Tracks ingested PDF versions for auditability.

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER PK | Auto-increment ID |
| `schedule_id` | TEXT UNIQUE | Schedule identifier (e.g., "acc1520-medical") |
| `version` | TEXT | Version string |
| `source_hash` | TEXT | SHA256 hash of source PDF |
| `source_path` | TEXT | Path to original PDF |

#### `schedule_codes`
The core billing code table (generic, works for any schedule).

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER PK | Auto-increment ID |
| `schedule_id` | TEXT | Schedule identifier |
| `code` | TEXT | Billing code (e.g., "GNCD", "MW2", "MF9") |
| `description` | TEXT | Full code description from schedule |
| `fee_excl_gst` | REAL | Fee excluding GST |
| `fee_incl_gst` | REAL | Fee including GST |
| `page` | INTEGER | Source page number in PDF |

**Unique constraint:** `(schedule_id, code)`

#### `policy_chunks`
Text chunks extracted from the PDF for semantic search.

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER PK | Auto-increment ID |
| `schedule_id` | TEXT | Schedule identifier |
| `page` | INTEGER | Source page number |
| `chunk_text` | TEXT | Chunk of extracted text (~120 words) |

#### `policy_chunk_embeddings`
Vector embeddings for semantic similarity search.

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER PK | Auto-increment ID |
| `schedule_id` | TEXT | Schedule identifier |
| `chunk_id` | INTEGER | Foreign key to policy_chunks |
| `embedding_json` | TEXT | JSON-serialized embedding vector |

### Vector Store / Embeddings

The system uses **OpenAI `text-embedding-3-small`** (1536 dimensions) for semantic search.

**Fallback behavior:** If `OPENAI_API_KEY` is not available, the system falls back to deterministic pseudo-embeddings (128-dimensional hash-based vectors) for offline development.

**Similarity metric:** Cosine similarity

```python
from rag_funding_engine.pipeline.semantic import cosine_similarity

similarity = cosine_similarity(query_embedding, chunk_embedding)
```

---

## Ingestion Pipeline

### How Any Schedule PDF Gets Processed

**Entry point:** `rag_funding_engine.pipeline.ingest_schedule.ingest_schedule()`

**Steps:**

1. **PDF Text Extraction**
   ```python
   from pypdf import PdfReader
   reader = PdfReader(pdf_path)
   pages = [{"page": i, "text": page.extract_text()} for i, page in enumerate(reader.pages, 1)]
   ```

2. **LLM Analysis → ScheduleProfile**
   ```python
   profile = _analyze_schedule_with_llm(full_text, model=llm_model)
   ```
   The LLM analyzes the document and generates a profile describing:
   - Schedule type (urgent_care, rural, gp, etc.)
   - Key dimensions that affect billing
   - Unique rules
   - Query patterns

3. **Code Row Parsing**
   - Pattern matches lines like: `GNCD GP Consultation Dependent CSC 14–17 years 37.50 43.13`
   - Regex: `^([A-Z]{2,6}\d{0,3})\s+(.+)$`
   - Handles multi-line descriptions
   - Extracts fees using: `\b\d{1,4}\.\d{2}\b`

4. **Deduplication**
   - Keeps last occurrence if duplicate codes exist

5. **Database Population**
   - Creates/updates `schedule_versions` record
   - Bulk inserts into `schedule_codes`
   - Chunks text into ~120-word segments for `policy_chunks`

6. **Embedding Generation**
   - Generates embeddings for all chunks
   - Stores as JSON in `policy_chunk_embeddings`

7. **Artifact Generation**
   - `{schedule_id}_text_pages.json` — Raw extracted text per page
   - `{schedule_id}_codes.json` — Parsed code rows
   - `manifest.json` — Processing metadata + ScheduleProfile

### Ingesting a New Schedule

```python
from pathlib import Path
from rag_funding_engine.pipeline.ingest_schedule import ingest_schedule

result = ingest_schedule(
    pdf_path=Path("/path/to/schedule.pdf"),
    schedule_id="my-schedule-2024",  # Unique identifier
    out_dir=Path("data/processed"),
    llm_model="gpt-4o",
)

print(f"Ingested {result.code_count} codes")
print(f"Profile: {result.profile.schedule_type}")
```

---

## Query Pipeline

This is the core recommendation engine. Flow: **Input → Decomposition → Relational Lookup → Semantic Search → Constraint Filtering → Context Assembly → LLM Reasoning → Output**

### Key Difference: Profile-Guided Queries

The pipeline now uses the schedule's `ScheduleProfile` to:
- Extract relevant facts using `key_dimensions`
- Generate queries using `query_patterns`
- Apply schedule-specific constraints

### Example: Using Different Schedules

```python
from rag_funding_engine.pipeline.recommend import recommend_codes

# Query ACC1520 urgent care schedule
result = recommend_codes(
    schedule_id="acc1520-medical",
    consult_text="6 cm laceration left hand...",
    consult_template=None,
    top_n=5,
)

# Query rural practitioner schedule
result = recommend_codes(
    schedule_id="rural-gp-2024",
    consult_text="GP consultation, patient 45km from clinic...",
    consult_template=None,
    top_n=5,
)
```

---

## API

### Endpoints

#### `GET /health`
Health check endpoint.

**Response:**
```json
{
    "ok": true,
    "service": "rag-funding-engine"
}
```

#### `POST /ingest`
Ingest any schedule document.

**Request:**
- `pdf`: UploadFile (PDF document)
- `schedule_id`: str (unique identifier, e.g., "acc1520-rural")
- `llm_model`: str (optional, default: "gpt-4o")

**Response:**
```json
{
    "status": "ok",
    "schedule_id": "acc1520-rural",
    "db_path": "data/processed/acc1520-rural/acc1520-rural.sqlite3",
    "code_count": 73,
    "indexed_chunks": 45,
    "manifest": "data/processed/acc1520-rural/manifest.json",
    "profile": {
        "schedule_type": "rural",
        "description": "Rural practitioner schedule with travel modifiers",
        "key_dimensions": ["location", "travel_time", "mileage"],
        "location_matters": true,
        "mileage_reimbursable": true,
        "after_hours_premium": false
    }
}
```

#### `POST /recommend`
Main recommendation endpoint.

**Request:**
```json
{
    "consult_text": "47-year-old female... distal radius and ulna fracture...",
    "consult_template": null,
    "schedule_id": "acc1520-medical",
    "top_n": 5
}
```

**Response:**
```json
{
    "query": "full query text...",
    "consult_facts": {
        "mode": "llm",
        "facts": {...}
    },
    "schedule_id": "acc1520-medical",
    "profile": {
        "schedule_type": "urgent_care",
        "key_dimensions": ["injury_type", "body_site", "procedure"]
    },
    "recommendations": [...],
    "estimated_total_excl_gst": 140.63,
    "estimated_total_incl_gst": 161.72,
    "evidence_chunks": [...],
    "constraint_checks": {...}
}
```

#### `GET /schedules`
List all ingested schedules.

**Response:**
```json
{
    "schedules": [
        {
            "schedule_id": "acc1520-medical",
            "description": "ACC urgent care schedule",
            "schedule_type": "urgent_care",
            "codes_parsed": 73,
            "chunks_indexed": 45
        },
        {
            "schedule_id": "rural-gp-2024",
            "description": "Rural practitioner schedule",
            "schedule_type": "rural",
            "codes_parsed": 56,
            "chunks_indexed": 38
        }
    ]
}
```

#### `GET /schedules/{schedule_id}`
Get details of a specific schedule.

**Response:**
```json
{
    "schedule_id": "acc1520-medical",
    "source": "/path/to/ACC1520.pdf",
    "source_hash": "abc123...",
    "pages": 25,
    "codes_parsed": 73,
    "chunks_indexed": 45,
    "profile": {
        "schedule_type": "urgent_care",
        "description": "ACC urgent care schedule",
        "key_dimensions": ["injury_type", "body_site", "procedure"],
        "unique_rules": ["CSC eligibility affects consultation codes"],
        "query_patterns": ["{procedure} {body_site}"],
        "example_queries": ["laceration hand"],
        "location_matters": false,
        "mileage_reimbursable": false,
        "after_hours_premium": false
    }
}
```

### Example API Calls

**Ingest a new schedule:**
```bash
curl -X POST http://localhost:8010/ingest \
  -F "pdf=@/path/to/rural-schedule.pdf" \
  -F "schedule_id=rural-gp-2024"
```

**Get recommendations:**
```bash
curl -X POST http://localhost:8010/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "consult_text": "32-year-old male. 6 cm laceration left hand.",
    "schedule_id": "acc1520-medical",
    "top_n": 3
  }'
```

**List schedules:**
```bash
curl http://localhost:8010/schedules
```

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | Required for LLM fact extraction, embeddings, and final adjudication |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model |
| `LLM_MODEL` | `gpt-4.1-mini` | Model for consult fact extraction |
| `FINAL_REASONING_MODEL` | `gpt-5.4` | Model for final recommendation adjudication |

### File Paths

| Path | Description |
|------|-------------|
| `data/raw/` | Source PDFs |
| `data/processed/{schedule_id}/` | Per-schedule processed data |
| `data/processed/{schedule_id}/{schedule_id}.sqlite3` | SQLite database |
| `data/processed/{schedule_id}/manifest.json` | Ingestion metadata + profile |

---

## Running It

### Setup

1. **Create virtual environment:**
   ```bash
   cd "/home/pi/clawd/projects/rag-funding-engine"
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set environment variables:**
   ```bash
   export OPENAI_API_KEY="sk-..."
   export FINAL_REASONING_MODEL="gpt-5.4"
   ```

### Run Backend

```bash
cd "/home/pi/clawd/projects/rag-funding-engine"
source .venv/bin/activate
PYTHONPATH=src uvicorn rag_funding_engine.api.main:app --host 0.0.0.0 --port 8010
```

The API will be available at `http://localhost:8010`.

### Run Frontend (Mission Control)

```bash
cd /home/pi/clawd/mission-control
npm run dev -- --hostname 0.0.0.0 --port 3000
```

Access demo at: `http://localhost:3000/projects/rag-funding-engine/demo`

---

## Testing

### Regression Tests

Test cases are defined in `tests/regression_cases.json`:

```json
[
  {
    "id": "hand_laceration_6cm_foreign_body",
    "input": {
      "consult_text": "32-year-old male construction worker...",
      "schedule_id": "acc1520-medical",
      "top_n": 3
    },
    "expected": {
      "must_include_codes": ["GP1", "MW2", "MM5"],
      "top_code": "MW2",
      "expected_total_excl_gst": 140.63,
      "total_tolerance": 0.02
    }
  }
]
```

### Running Tests

```bash
cd "/home/pi/clawd/projects/rag-funding-engine"
source .venv/bin/activate
pytest tests/
```

### Adding New Test Cases

Add to `tests/regression_cases.json`:
```json
{
  "id": "unique_case_id",
  "input": {
    "consult_text": "Clinical description...",
    "schedule_id": "acc1520-medical",
    "top_n": 5
  },
  "expected": {
    "must_include_codes": ["CODE1", "CODE2"],
    "top_code": "CODE1",
    "expected_total_excl_gst": 123.45,
    "total_tolerance": 0.02
  }
}
```

---

## Architecture Notes

### Multi-Schedule Support

The engine now supports multiple schedules side-by-side:

```
data/processed/
├── acc1520-medical/
│   ├── acc1520-medical.sqlite3
│   ├── manifest.json
│   └── acc1520-medical_codes.json
├── acc1520-rural/
│   ├── acc1520-rural.sqlite3
│   ├── manifest.json
│   └── acc1520-rural_codes.json
└── gp-2024/
    ├── gp-2024.sqlite3
    ├── manifest.json
    └── gp-2024_codes.json
```

Each schedule has:
- Its own SQLite database
- Its own ScheduleProfile
- Its own embeddings
- Independent versioning

### Dynamic Profiling

The LLM analysis step extracts schedule-specific characteristics:

1. **Document text** is extracted from PDF
2. **LLM analyzes** the first 8000 characters
3. **Profile is generated** with schedule type, dimensions, rules
4. **Profile guides** query generation and constraint application

This allows the same codebase to handle:
- Urgent care schedules
- Rural practitioner schedules with travel modifiers
- GP consultation schedules
- Specialist schedules
- Hospital schedules

---

## Known Limitations

1. **Constraint engine depth:** Current constraints are prototype-level, not a full rule graph
2. **Pricing rule granularity:** Same-visit pricing logic is simplified
3. **Negation NLP:** Improved but still heuristic; formal NLP would strengthen this
4. **Schedule detection:** Profile generation depends on LLM; may need refinement for unusual schedules

---

## License

Proprietary — developed for internal use.
