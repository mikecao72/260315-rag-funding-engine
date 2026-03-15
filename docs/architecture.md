# Architecture - RAG Funding Engine (ACC1520)

## System Design

1. **Ingestion Layer**
   - Source: ACC1520 PDF versions
   - Outputs:
     - structured tables (codes, prices, limits)
     - sectioned policy text chunks
     - source/version manifest

2. **Knowledge Layer**
   - Relational DB (Postgres):
     - `schedule_versions`
     - `acc_codes`
     - `billing_rules`
     - (future) `code_relationships`, `constraints`
   - Vector index:
     - embeddings for narrative policy/guidance text
     - metadata: page, section, version

3. **Consult Understanding Layer**
   - Input: transcript and/or structured consult template
   - Extracts normalized facts used for retrieval and constraints

4. **Candidate Retrieval Layer**
   - SQL retrieval of directly-matching codes
   - RAG retrieval of contextual policy/rule snippets

5. **Constraint Engine (Hard Rules)**
   - Enforces exclusions, prerequisites, frequency caps, etc.
   - Discards non-compliant code bundles

6. **Revenue Optimizer**
   - Scores valid bundles by reimbursable value
   - Returns top-N with confidence and rationale

7. **Audit/Explainability**
   - For each recommendation:
     - applied rules
     - failed alternatives
     - citations to schedule page/section/version

## Delivery Phases

### Phase 1 (now)
- Scaffold service + ingestion + schema
- Parse text and build artifact manifest

### Phase 2
- Table extraction and structured DB load
- Chunking + embeddings index

### Phase 3
- Constraint engine + deterministic ranking
- `/recommend` endpoint with evidence payload

### Phase 4
- Optimizer upgrade (ILP/CP-SAT)
- Evaluation harness vs manually coded consult set
