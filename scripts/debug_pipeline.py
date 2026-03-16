"""Debug script to trace full recommendation pipeline."""
import json, os, re, sqlite3
from pathlib import Path

# Ensure env is loaded
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from rag_funding_engine.pipeline.recommend import (
    _tokens, _keyword_similarity, _heuristic_boost, _has_no_csc,
    _is_csc_variant, _has_positive_amputation, _has_positive_fracture,
    Candidate, _extract_consult_facts, _load_profile,
    _select_primary_consult, _final_reasoning_review,
    _apply_pricing_rules, _is_consultation_code
)
from rag_funding_engine.pipeline.semantic import cosine_similarity, embed_texts

schedule_id = "acc1520-medical"
profile = _load_profile(schedule_id)
db_path = f"data/processed/{schedule_id}/{schedule_id}.sqlite3"
conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row
rows = conn.execute(
    "SELECT code, description, fee_excl_gst, fee_incl_gst, page FROM schedule_codes WHERE schedule_id = ?",
    (schedule_id,),
).fetchall()

query_text = (
    "SUBJECTIVE: Fell off a bike today at approximately 08:30, landing on the left side. "
    "Was wearing a helmet at the time. Reports scalp pain but denies loss of consciousness. "
    "Reports left shoulder pain with reduced range of movement and a palpable step over the left shoulder. "
    "Pain over the ribs, an open wound over the fourth left knuckle, and an abrasion on the left knee. "
    "Fully weight-bearing. "
    "OBJECTIVE: Left shoulder: Open wound, palpable step over the acromioclavicular (AC) joint, intense pain, "
    "and reduced range of movement. Left fourth metacarpophalangeal (MCP) joint: Open wound present. "
    "Left knee: Abrasion on the lateral knee joint area. "
    "Investigations: Chest X-ray shows no pneumothorax. Left shoulder X-ray reveals acromioclavicular (AC) "
    "joint disruption, classified as Rockwood Type V. Hand X-ray shows no fracture. Wounds are clean and dry. "
    "ASSESSMENT: Contusion of the left side with multiple injuries. "
    "Acromioclavicular (AC) joint disruption (Rockwood Type V). "
    "Open wounds on the left shoulder and left fourth MCP joint. Abrasion on the left knee. "
    "PLAN: Administered tetanus vaccine. Discuss X-ray findings with orthopaedic team. Wound care and follow-up."
)

facts = _extract_consult_facts(query_text, None, profile)
search_queries = facts.get("search_queries", [])
print("=== SEARCH QUERIES ===")
for sq in search_queries:
    print(f"  {sq}")

# Build code list
code_descs, code_rows = [], []
for r in rows:
    c = r["code"]
    if c in {"ACC", "Code"}:
        continue
    if _has_no_csc(query_text) and _is_csc_variant(c, r["description"]):
        continue
    if c.startswith("MW4") and not _has_positive_amputation(query_text):
        continue
    code_descs.append(f"{c} {r['description']}")
    code_rows.append(r)

all_texts = search_queries + code_descs
all_embs = embed_texts(all_texts)
qembs = all_embs[: len(search_queries)]
dembs = all_embs[len(search_queries) :]

candidates = []
for i, r in enumerate(code_rows):
    code, desc = r["code"], r["description"]
    dt = _tokens(desc)
    best_kw = max((_keyword_similarity(_tokens(sq), dt) for sq in search_queries), default=0.0)
    best_sem = max((cosine_similarity(qe, dembs[i]) for qe in qembs), default=0.0)
    boost = _heuristic_boost(query_text, code, desc, profile)
    score = best_kw * 0.4 + best_sem * 0.6 + boost
    if score > 0.05:
        candidates.append(
            Candidate(
                code=code, description=desc,
                fee_excl_gst=r["fee_excl_gst"], fee_incl_gst=r["fee_incl_gst"],
                page=r["page"], score=score,
            )
        )

candidates.sort(key=lambda c: (c.score, c.fee_excl_gst or 0), reverse=True)
print(f"\n=== ALL CANDIDATES ({len(candidates)}) ===")
for c in candidates[:20]:
    print(f"  {c.code:8s} score={c.score:.3f} fee={c.fee_excl_gst} {c.description[:70]}")

# Dedup
selected, seen = [], set()
for c in candidates:
    prefix = re.sub(r"\d+$", "", c.code)
    if prefix in seen and c.score < 0.95:
        continue
    selected.append(c)
    seen.add(prefix)
    if len(selected) >= 8:
        break

print(f"\n=== AFTER DEDUP ({len(selected)}) ===")
recs = []
for c in selected:
    recs.append({
        "code": c.code, "description": c.description,
        "fee_excl_gst": c.fee_excl_gst, "fee_incl_gst": c.fee_incl_gst,
        "page": c.page, "match_score": round(c.score, 4),
    })
    print(f"  {c.code:8s} score={c.score:.3f} {c.description[:70]}")

print("\n=== SENDING TO FINAL LLM REVIEW ===")
reviewed = _final_reasoning_review(query_text, facts, recs, profile)
print(f"\n=== AFTER LLM REVIEW ({len(reviewed)}) ===")
for r in reviewed:
    print(f"  {r['code']:8s} score={r['match_score']} reason: {r.get('llm_reason','')[:100]}")

conn.close()
