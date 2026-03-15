# RAG Funding Engine — Handoff Notes

_Last updated: 2026-03-14 (NZ)_

## Where to read first next session

1. `docs/ARCHITECTURE_V2.md` — **Generic multi-schedule platform architecture**
2. `IMPLEMENTATION_CHANGELOG_2026-03-14.md`
3. `DETERMINISTIC_QUERY_WALKTHROUGH.md`
4. `src/rag_funding_engine/pipeline/recommend.py`
5. `mission-control/src/app/projects/rag-funding-engine/demo/page.tsx`

## Current architecture (working)

- Frontend demo: `mission-control` page `/projects/rag-funding-engine/demo`
- Frontend API proxy: `/api/rag-funding/recommend`
- Backend API: `rag_funding_engine.api.main` on port `8010`
- Core pipeline: `recommend.py`

Flow:
1. Textarea input
2. Deterministic shortlist from `acc_codes` (SQLite)
3. Heuristic boosts/penalties + context gates
4. Final LLM adjudication stage (configurable model; default `gpt-5.4`)
5. Pricing/constraints/reasons
6. UI output with expandable reasoning

## Data stores

SQLite DB: `data/processed/acc1520.sqlite3`

Tables:
- `acc_codes` (deterministic code source)
- `policy_chunks` (PDF chunks)
- `policy_chunk_embeddings` (vector store in same SQLite DB)

## Key behavior implemented

- Strong scenario routing for CSC-dependent teen + nurse/GP + multi-site abrasions + epistaxis.
- Negation handling for fracture context (`no fracture` should suppress MF fracture family boosts).
- Amputation guardrail (`MW4`) requires positive amputation evidence.
- Per-code reasons include quoted input evidence where available.
- Frontend supports click-to-expand reasoning per code.

## Important env/model knobs

- `OPENAI_API_KEY` (enables OpenAI extraction/embedding/review)
- `FINAL_REASONING_MODEL` (default: `gpt-5.4`)
- `LLM_MODEL` for consult fact extraction (default from code)
- `EMBEDDING_MODEL` for chunk embeddings (default from code)

## Run commands

Backend:
```bash
cd "/home/pi/clawd/projects/RAG Funding Engine"
source .venv/bin/activate
PYTHONPATH=src uvicorn rag_funding_engine.api.main:app --host 0.0.0.0 --port 8010
```

Frontend:
```bash
cd /home/pi/clawd/mission-control
npm run dev -- --hostname 0.0.0.0 --port 3000
```

## Known next priorities

1. Upgrade deterministic prefilter to SQL FTS / better lexical retrieval.
2. Expand official ACC multi-treatment/same-injury suppression rules.
3. Add UI audit badges (heuristic-kept vs model-kept).
4. Add more adversarial regression test cases.
