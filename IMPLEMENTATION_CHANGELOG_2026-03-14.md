# RAG Funding Engine — Implementation Changelog (2026-03-14)

This document summarizes the recent backend + frontend changes made for the ACC1520 billing recommendation demo, mapped to commit IDs so you can review on GitHub.

## Scope

These changes span:
- **Backend**: `projects/RAG Funding Engine/src/rag_funding_engine/pipeline/recommend.py`
- **Frontend (Mission Control demo UI)**: `mission-control/src/app/projects/rag-funding-engine/demo/page.tsx`

---

## Tech stack in this solution

### Backend
- **Python 3**
- **FastAPI** (`/health`, `/ingest`, `/recommend`)
- **Uvicorn** server
- **SQLite** schedule datastore (`acc_codes`, policy chunks)
- **Heuristic scoring layer** (keyword + rule boosts/penalties)
- **Optional LLM extraction** via OpenAI for consult facts
- **Optional final LLM adjudication** stage (configurable, default `gpt-5.4`)

### Frontend
- **Next.js** (Mission Control app)
- Demo page at: `/projects/rag-funding-engine/demo`
- API route proxy: `/api/rag-funding/recommend` → backend `/recommend`

---

## Pipeline (current flow)

1. User enters consult text in demo input panel.
2. Frontend calls `POST /api/rag-funding/recommend`.
3. API route forwards to backend `POST http://127.0.0.1:8010/recommend`.
4. Backend:
   - extracts consult facts (LLM if available, otherwise heuristic)
   - builds candidate codes from schedule DB
   - applies heuristic boosts/penalties + context gates
   - **(new)** runs final LLM adjudication step for keep/drop review
   - applies consultation selection + basic constraints
   - applies pricing multipliers (prototype same-visit rule)
   - emits recommendations with explainability reasons
5. Frontend renders results table with expandable reasoning per code.

---

## Commit-by-commit summary

### `ef10cd6` — Improve RAG recommend heuristics for CSC teen abrasion+epistaxis cases
- Added stronger rule boosts for:
  - `GNCD` when context indicates dependent CSC + age 14–17 + nurse+GP consult.
  - `MB5` for significant multi-site abrasions (>4 cm² patterns).
  - `MM8` for epistaxis with nasal packing.
- Added penalties for less appropriate nearby alternatives in that scenario.
- Added primary consultation down-selection helper to reduce duplicate consult variants.
- Tightened low-score filtering threshold.

### `bd53d35` — Add per-code recommendation reasons in backend and demo UI
- Backend now appends `reason` text per recommended code.
- Reason explains why code was selected and how pricing multiplier was applied.
- Frontend demo table updated to display reasoning output.

### `788fe90` — Make billing-code reasoning expandable in demo table
- UI changed from always-visible reason text to **click-to-expand per code**.
- Keeps table compact while preserving transparent rationale.

### `791d556` — Add quoted input evidence in reasons and tighten amputation code guardrails
- Explainability upgraded to include **quoted snippets directly from input**.
- Added stronger MW4 (amputation) guardrail requiring positive amputation context.

### `2b6f9cd` — Handle negated fracture context in scoring and reasoning
- Added negation-aware fracture detection.
- Prevents fracture-family boosts when text says things like `"no fracture suspected"`.
- Reasoning text now reflects positive vs negated fracture context.

### `f72004a` — Add final LLM adjudication stage for recommendation output
- Added `_final_reasoning_review(...)` final-stage model pass before final output.
- Model is configurable with env var:
  - `FINAL_REASONING_MODEL` (default: `gpt-5.4`)
- Inputs to final step: query text + extracted facts + candidate codes.
- Outputs: keep/drop decisions + short model rationale.
- Safe fallback: if unavailable/error, system falls back to heuristic shortlist.

---

## Explainability behavior (current)

For each suggested code, reasoning now includes:
- why it was selected,
- quotes from input text that triggered the decision (when available),
- pricing status (full vs 50% in same-visit prototype rule),
- linked schedule descriptor.

---

## Known limits / next hardening priorities

1. **Rule engine depth**
   - Current constraints are still a prototype, not a full ACC rule graph.
2. **Pricing rule granularity**
   - Same-visit pricing logic is simplified and should be expanded with explicit official combinations/precedence.
3. **Negation + context NLP**
   - Improved, but still heuristic in many branches; should be formalized across all major code families.
4. **Audit mode UI**
   - Could add model confidence and "kept/dropped by final model" badges.

---

## File map for review

- Backend main logic:
  - `projects/RAG Funding Engine/src/rag_funding_engine/pipeline/recommend.py`
- Backend API entrypoint:
  - `projects/RAG Funding Engine/src/rag_funding_engine/api/main.py`
- Frontend demo page:
  - `mission-control/src/app/projects/rag-funding-engine/demo/page.tsx`
- Frontend API proxy route:
  - `mission-control/src/app/api/rag-funding/recommend/route.ts`

---

If useful, next step can be a formal `ARCHITECTURE.md` + sequence diagram for the full ingest/recommend/adjudication flow.
