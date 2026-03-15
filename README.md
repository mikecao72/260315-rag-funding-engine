# ACC RAG Funding Engine

A RAG-driven (Retrieval-Augmented Generation) system for ACC1520 medical billing code recommendation. This engine transforms unstructured clinical consultation notes into ranked, explainable ACC funding code recommendations with fee estimates.

---

## Table of Contents

1. [Overview](#overview)
2. [Data Model](#data-model)
3. [Ingestion Pipeline](#ingestion-pipeline)
4. [Query Pipeline](#query-pipeline)
5. [API](#api)
6. [Configuration](#configuration)
7. [Running It](#running-it)
8. [Testing](#testing)

---

## Overview

### What Problem Does This Solve?

New Zealand's Accident Compensation Corporation (ACC) publishes the **ACC1520 schedule** — a complex document containing hundreds of billing codes, fees, and rules for medical practitioners and nurses treating accident-related injuries. 

Clinicians face the challenge of:
- **Mapping** unstructured clinical notes to the correct billing codes
- **Navigating** complex rules (CSC eligibility, age bands, procedure combinations)
- **Optimizing** revenue while staying compliant
- **Documenting** rationale for audit purposes

The ACC RAG Funding Engine solves this by:
1. **Ingesting** the ACC1520 PDF into structured relational data + vector embeddings
2. **Accepting** free-text clinical consultation notes
3. **Retrieving** candidate codes via keyword matching + semantic search
4. **Filtering** through constraint rules and heuristics
5. **Ranking** with an optional LLM adjudication layer
6. **Explaining** every recommendation with quoted evidence from the input

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

## Data Model

### SQLite Relational Database

The system uses SQLite (`data/processed/acc1520.sqlite3`) with the following schema:

#### `schedule_versions`
Tracks ingested PDF versions for auditability.

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER PK | Auto-increment ID |
| `version` | TEXT UNIQUE | Schedule version identifier (e.g., "ACC1520-v2") |
| `source_hash` | TEXT | SHA256 hash of source PDF |
| `source_path` | TEXT | Path to original PDF |

#### `acc_codes`
The core billing code table extracted from the schedule.

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER PK | Auto-increment ID |
| `schedule_version` | TEXT | Foreign key to schedule_versions |
| `code` | TEXT | Billing code (e.g., "GNCD", "MW2", "MF9") |
| `description` | TEXT | Full code description from schedule |
| `fee_excl_gst` | REAL | Fee excluding GST |
| `fee_incl_gst` | REAL | Fee including GST |
| `page` | INTEGER | Source page number in PDF |

**Unique constraint:** `(schedule_version, code)`

**Example rows:**
```sql
SELECT code, description, fee_excl_gst, fee_incl_gst, page
FROM acc_codes
WHERE schedule_version = 'ACC1520-v2'
LIMIT 3;
```

| code | description | fee_excl_gst | fee_incl_gst | page |
|------|-------------|--------------|--------------|------|
| GNCD | GP Consultation Dependent CSC 14–17 years | 37.50 | 43.13 | 8 |
| MW2 | Laceration 2–7 cm | 68.75 | 79.06 | 12 |
| MF9 | Distal radius/ulna without reduction | 124.44 | 143.11 | 15 |

#### `policy_chunks`
Text chunks extracted from the PDF for semantic search.

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER PK | Auto-increment ID |
| `schedule_version` | TEXT | Version identifier |
| `page` | INTEGER | Source page number |
| `chunk_text` | TEXT | Chunk of extracted text (~120 words) |

#### `policy_chunk_embeddings`
Vector embeddings for semantic similarity search.

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER PK | Auto-increment ID |
| `schedule_version` | TEXT | Version identifier |
| `chunk_id` | INTEGER | Foreign key to policy_chunks |
| `embedding_json` | TEXT | JSON-serialized embedding vector |

**Unique constraint:** `(schedule_version, chunk_id)`

### Vector Store / Embeddings

The system uses **OpenAI `text-embedding-3-small`** (1536 dimensions) for semantic search. Embeddings are stored as JSON in SQLite and computed at ingestion time.

**Fallback behavior:** If `OPENAI_API_KEY` is not available, the system falls back to deterministic pseudo-embeddings (128-dimensional hash-based vectors) for offline development.

**Similarity metric:** Cosine similarity

```python
from rag_funding_engine.pipeline.semantic import cosine_similarity

similarity = cosine_similarity(query_embedding, chunk_embedding)
```

---

## Ingestion Pipeline

### How the ACC1520 PDF Gets Processed

**Entry point:** `rag_funding_engine.pipeline.ingest_acc1520.ingest_schedule()`

**Steps:**

1. **PDF Text Extraction**
   ```python
   from pypdf import PdfReader
   reader = PdfReader(pdf_path)
   pages = [{"page": i, "text": page.extract_text()} for i, page in enumerate(reader.pages, 1)]
   ```

2. **Code Row Parsing**
   - Pattern matches lines like: `GNCD GP Consultation Dependent CSC 14–17 years 37.50 43.13`
   - Regex: `^([A-Z]{2,6}\d{0,3})\s+(.+)$`
   - Handles multi-line descriptions by continuing until next code or price line
   - Extracts fees using: `\b\d{1,4}\.\d{2}\b`

3. **Deduplication**
   - Keeps last occurrence if duplicate codes exist
   - Uses dict: `{row["code"]: row for row in code_rows}`

4. **Database Population**
   - Creates/updates `schedule_versions` record
   - Bulk inserts into `acc_codes`
   - Chunks text into ~120-word segments for `policy_chunks`

5. **Embedding Generation**
   ```python
   from rag_funding_engine.pipeline.semantic import index_policy_chunks
   indexed_count = index_policy_chunks(db_path, schedule_version="ACC1520-v2")
   ```
   - Queries all chunks without embeddings
   - Calls OpenAI Embeddings API in batches
   - Stores as JSON in `policy_chunk_embeddings`

6. **Artifact Generation**
   - `acc1520_text_pages.json` — Raw extracted text per page
   - `acc1520_codes.json` — Parsed code rows
   - `ingest_manifest.json` — Processing metadata

### What Comes Out

After ingestion, you have:
- **~73-200 billing codes** (depending on PDF version)
- **Policy chunks** for semantic search
- **Embeddings** for similarity retrieval
- **Audit trail** via source hash and manifest

---

## Query Pipeline

This is the core recommendation engine. Flow: **Input → Decomposition → Relational Lookup → Semantic Search → Constraint Filtering → Context Assembly → LLM Reasoning → Output**

### Step 1: Input

**Accepts:**
- `consult_text`: Free-text clinical notes (required)
- `consult_template`: Optional structured dict for future form-based input

**Example input:**
```python
{
    "consult_text": "32-year-old male construction worker. 6 cm laceration across dorsum of left hand...",
    "consult_template": None,
    "top_n": 5
}
```

The system concatenates both sources:
```python
query_text = f"{consult_text or ''} {template_text}".strip()
```

### Step 2: Query Decomposition

**Fact Extraction:**

If `OPENAI_API_KEY` is available:
```python
prompt = (
    "Extract consult billing-relevant facts as strict compact JSON with keys: "
    "injury_type, body_site, procedure, severity, anaesthesia, consult_type, age_band. "
    "If unknown use null."
)

response = client.chat.completions.create(
    model=os.getenv("LLM_MODEL", "gpt-4.1-mini"),
    temperature=0,
    response_format={"type": "json_object"},
    messages=[
        {"role": "system", "content": prompt},
        {"role": "user", "content": text},
    ],
)
```

**Fallback:** Returns `{"mode": "heuristic", "facts": {"summary": text}}`

**Tokenization:**
```python
TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9\-]{1,}")
query_tokens = {t.lower() for t in TOKEN_RE.findall(query_text) if len(t) > 2}
```

### Step 3: Relational DB Lookup

**SQL Query:**
```sql
SELECT code, description, fee_excl_gst, fee_incl_gst, page
FROM acc_codes
WHERE schedule_version = ?
```

**Returns:** All codes for the specified schedule version (typically 73+ rows).

**Python-side filtering:**
- Drops pseudo-rows (`ACC`, `Code`)
- Drops CSC variants if input explicitly states "no Community Services Card"
- Drops amputation codes (`MW4*`) unless positive amputation context detected

### Step 4: Semantic/Vector Search

**Embedding the query:**
```python
from rag_funding_engine.pipeline.semantic import embed_texts
q_emb = embed_texts([query_text])[0]  # 1536-dim OpenAI or 128-dim fallback
```

**Retrieving chunks:**
```sql
SELECT p.page, p.chunk_text, e.embedding_json
FROM policy_chunks p
JOIN policy_chunk_embeddings e ON e.chunk_id = p.id
WHERE p.schedule_version = ? AND e.schedule_version = ?
```

**Scoring:**
```python
scored = []
for row in rows:
    emb = json.loads(row["embedding_json"])
    score = cosine_similarity(q_emb, emb)
    scored.append((score, row["page"], row["chunk_text"][:300]))

scored.sort(key=lambda x: x[0], reverse=True)
top_chunks = scored[:3]  # Top 3 semantic matches
```

**Fallback:** If no embeddings exist, falls back to keyword similarity on raw chunks.

### Step 5: Constraint Filtering

**Hard Gates (in `recommend.py`):**

1. **Score threshold:** Drop if `combined_score <= 0.05`
2. **Prefix deduplication:** Keep highest-scoring code per alpha prefix family
   ```python
   prefix = re.sub(r"\d+$", "", c.code)  # "GNCD" → "GN"
   if prefix in seen_prefixes and c.score < 0.95:
       continue
   ```
3. **Context gates:**
   - Distal radius/ulna fracture → only allow `GP1`, `MF9`, `MF10`, `GP14`, `GPN`
   - Distal tibia/fibula fracture → only allow `GP1`, `MF15`, `MF14`, `MF16`, `MW1`, `MW2`, `MW3`

**Constraint Engine (`constraints.py`):**
```python
from rag_funding_engine.pipeline.constraints import apply_basic_constraints

result = apply_basic_constraints(recommendations, consult_facts)
# Returns: RuleResult(allowed=True, reasons=[...])
```

Current constraint rules:
1. Remove duplicate family candidates (same alpha prefix)
2. Filter GP-prefixed codes for nurse consult types

### Step 6: Context Assembly

**Candidate payload for LLM adjudication:**
```python
payload = {
    "query_text": query_text,
    "consult_facts": consult_facts,
    "candidates": [
        {
            "code": r.get("code"),
            "description": r.get("description"),
            "fee_excl_gst": r.get("fee_excl_gst"),
            "match_score": r.get("match_score"),
        }
        for r in recs
    ],
}
```

**Evidence chunks:**
```python
evidence_chunks = [
    {"page": p, "score": round(s, 4), "snippet": t}
    for s, p, t in top_semantic_matches
]
```

### Step 7: LLM Reasoning

**Final Adjudication Stage (`_final_reasoning_review`):**

```python
prompt = (
    "You are the final clinical billing adjudicator for ACC1520 prototype outputs. "
    "Given consult input, extracted facts, and candidate codes, return STRICT JSON with key 'final_recommendations'. "
    "Each item must include: code, keep (boolean), reason (short). "
    "Rules: respect explicit negations in input (e.g. 'no fracture', 'no amputation'); "
    "prefer specific clinically-supported items; avoid mutually incompatible/same-family duplicates unless clearly justified; "
    "retain at most 6 final codes. Do not invent codes not present in candidates."
)

response = client.chat.completions.create(
    model=os.getenv("FINAL_REASONING_MODEL", "gpt-5.4"),
    temperature=0,
    response_format={"type": "json_object"},
    messages=[
        {"role": "system", "content": prompt},
        {"role": "user", "content": json.dumps(payload)},
    ],
)
```

**Fallback:** Returns original heuristic-ranked recommendations on any error.

**Primary Consult Selection:**
```python
recs = _select_primary_consult(recs)
# Sorts consultation codes by score+fee, keeps highest as primary
```

### Step 8: Output

**Final response structure:**
```python
{
    "query": "full query text...",
    "consult_facts": {
        "mode": "llm",  # or "heuristic"
        "facts": {...}  # extracted structured facts
    },
    "schedule_version": "ACC1520-v2",
    "recommendations": [
        {
            "code": "MW2",
            "description": "Laceration 2–7 cm",
            "fee_excl_gst": 68.75,
            "fee_incl_gst": 79.06,
            "page": 12,
            "match_score": 2.66,
            "pricing_multiplier": 1.0,
            "line_total_excl_gst": 68.75,
            "quantity": 1,
            "reason": "MW2 selected because laceration dimensions align with 2–7 cm range..."
        }
    ],
    "estimated_total_excl_gst": 140.63,
    "estimated_total_incl_gst": 161.72,
    "evidence_chunks": [
        {"page": 12, "score": 0.8234, "snippet": "Laceration repair..."}
    ],
    "constraint_checks": {
        "allowed": True,
        "reasons": ["Removed duplicate family candidate: MW1"]
    },
    "notes": [
        "Prototype model with optional OpenAI fact extraction + semantic evidence retrieval.",
        "Hard ACC rule constraints still require explicit implementation for production use."
    ]
}
```

**Reason format per code:**
```
{code} selected because {reasons}. Quoted input: "{snippet}" | "{snippet}". 
Priced at {full|50%} rate for this visit. 
Relevant schedule descriptor: {description}
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
Trigger PDF ingestion pipeline.

**Response:**
```json
{
    "status": "ok",
    "db_path": "/home/pi/clawd/projects/RAG Funding Engine/data/processed/acc1520.sqlite3",
    "code_count": 73,
    "indexed_chunks": 45,
    "manifest": "/home/pi/clawd/projects/RAG Funding Engine/data/processed/ingest_manifest.json"
}
```

#### `POST /recommend`
Main recommendation endpoint.

**Request:**
```json
{
    "consult_text": "47-year-old female... distal radius and ulna fracture...",
    "consult_template": null,
    "top_n": 5
}
```

**Response:** See [Step 8: Output](#step-8-output) for full schema.

### Example API Call

```bash
curl -X POST http://localhost:8010/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "consult_text": "32-year-old male. 6 cm laceration left hand. Embedded metal fragment removed.",
    "top_n": 3
  }'
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
| `SCHEDULE_VERSION` | `ACC1520-v2` | Default schedule version to query |
| `DATABASE_URL` | — | Postgres URL (future use; currently uses SQLite) |

### File Paths

| Path | Description |
|------|-------------|
| `data/raw/ACC1520-Med-pract-nurse-pract-and-nurses-costs-v2.pdf` | Source PDF |
| `data/processed/acc1520.sqlite3` | SQLite database |
| `data/processed/ingest_manifest.json` | Ingestion metadata |
| `data/processed/acc1520_codes.json` | Parsed codes JSON |

---

## Running It

### Setup

1. **Create virtual environment:**
   ```bash
   cd "/home/pi/clawd/projects/RAG Funding Engine"
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
cd "/home/pi/clawd/projects/RAG Funding Engine"
source .venv/bin/activate
PYTHONPATH=src uvicorn rag_funding_engine.api.main:app --host 0.0.0.0 --port 8010
```

The API will be available at `http://localhost:8010`.

### Run Ingestion

Ingestion runs automatically on first `/recommend` call if no database exists, or manually:

```bash
curl -X POST http://localhost:8010/ingest
```

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

**Manual validation via API:**
```bash
curl -X POST http://localhost:8010/recommend \
  -H "Content-Type: application/json" \
  -d '@tests/regression_cases.json[0].input'
```

**Smoke test coverage:**
The regression cases cover:
- Hand laceration with foreign body removal (MW2 + MM5 combo)
- Distal radius/ulna fracture without reduction (MF9 routing)
- Distal tibia/fibula fracture with reduction + scalp laceration (MF15 + MW1 combo)

### What the Tests Validate

1. **Code inclusion:** Expected codes appear in recommendations
2. **Ranking:** Primary code matches expected top code
3. **Pricing:** Total fee estimate within tolerance
4. **Constraint compliance:** No mutually incompatible codes returned

### Adding New Test Cases

Add to `tests/regression_cases.json`:
```json
{
  "id": "unique_case_id",
  "input": {
    "consult_text": "Clinical description...",
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

### Heuristic Scoring Formula

```python
combined_score = keyword_similarity(query_tokens, desc_tokens) + heuristic_boost(query_text, code, description)
```

**Keyword similarity:** Jaccard index of token sets
```python
intersection / max(len(a), len(b))
```

**Heuristic boosts** include:
- Age-band detection (regex: `\b(\d{1,3})\s*[-–—]?\s*year\s*[-–—]?\s*old\b`)
- CSC eligibility parsing
- Nurse+GP joint consult detection
- Laceration length extraction (`\b(\d+(?:\.\d+)?)\s*cm\b`)
- Negation handling for fractures/amputations
- Multi-site abrasion detection
- Epistaxis with packing detection

### Pricing Rules (Prototype)

```python
# Consultation codes: full value
# Non-consult treatments: 100% for highest-cost, 50% for additional treatments
# (if multiple injury sites detected)
```

---

## Known Limitations

1. **Constraint engine depth:** Current constraints are prototype-level, not a full ACC rule graph
2. **Pricing rule granularity:** Same-visit pricing logic is simplified
3. **Negation NLP:** Improved but still heuristic; formal NLP would strengthen this
4. **Audit mode UI:** Could add model confidence and "kept/dropped by final model" badges

---

## License

Proprietary — developed for internal use.
