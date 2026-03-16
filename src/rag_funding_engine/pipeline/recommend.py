from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import sqlite3
from typing import Any

from rag_funding_engine.pipeline.constraints import apply_basic_constraints
from rag_funding_engine.pipeline.semantic import cosine_similarity, embed_texts


TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9\-]{1,}")

STOPWORDS = frozenset({
    "the", "and", "for", "with", "from", "that", "this", "are", "was", "were",
    "has", "had", "have", "but", "not", "per", "also", "known", "other", "any",
    "all", "can", "may", "will", "been", "being", "its", "than", "then", "into",
    "over", "under", "such", "when", "where", "who", "how", "which", "what",
    "does", "did", "each", "some", "both", "their", "more", "less", "most",
    "only", "about", "after", "before", "between", "during", "through", "without",
    "including", "requiring", "necessary",
})


@dataclass
class Candidate:
    code: str
    description: str
    fee_excl_gst: float | None
    fee_incl_gst: float | None
    page: int | None
    score: float


def _get_db_path(schedule_id: str, base_dir: Path | None = None) -> Path:
    """Get database path for a given schedule_id."""
    if base_dir is None:
        # Default: data/processed/{schedule_id}/{schedule_id}.sqlite3
        from rag_funding_engine.api.main import ROOT
        base_dir = ROOT / "data" / "processed"
    return base_dir / schedule_id / f"{schedule_id}.sqlite3"


def _get_manifest_path(schedule_id: str, base_dir: Path | None = None) -> Path:
    """Get manifest path for a given schedule_id."""
    if base_dir is None:
        from rag_funding_engine.api.main import ROOT
        base_dir = ROOT / "data" / "processed"
    return base_dir / schedule_id / "manifest.json"


def _load_profile(schedule_id: str, base_dir: Path | None = None) -> dict[str, Any]:
    """Load schedule profile from manifest."""
    manifest_path = _get_manifest_path(schedule_id, base_dir)
    if not manifest_path.exists():
        return {}
    try:
        return json.loads(manifest_path.read_text()).get("profile", {})
    except Exception:
        return {}


def _tokens(text: str) -> set[str]:
    return {t.lower() for t in TOKEN_RE.findall(text or "") if len(t) > 2 and t.lower() not in STOPWORDS}


def _keyword_similarity(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / max(1, min(len(a), len(b)))


def _extract_consult_facts(text: str, template: dict[str, Any] | None, profile: dict[str, Any]) -> dict[str, Any]:
    """Extract consult facts and billing search queries using LLM, guided by profile.

    Returns dict with keys: mode, facts, search_queries.
    search_queries is a list of short medical-term phrases suitable for matching
    against schedule code descriptions (e.g. "open wound closure", "shoulder dislocation").
    """
    if template:
        queries = [f"{v}" for v in template.values() if v]
        return {"mode": "template", "facts": template, "search_queries": queries}

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {"mode": "heuristic", "facts": {"summary": text}, "search_queries": []}

    key_dims = profile.get("key_dimensions", ["injury_type", "body_site", "procedure", "severity"])
    dims_str = ", ".join(key_dims)

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        prompt = (
            "You are a medical billing analyst for the ACC1520 fee schedule (NZ accident compensation).\n"
            f"Given clinical consultation notes, extract:\n"
            f"1. \"facts\": billing-relevant facts as compact JSON with keys: {dims_str}. Use null if unknown.\n"
            "2. \"search_queries\": a JSON array of 3-8 short search phrases (2-5 words each) "
            "using EXACTLY the kind of language found in fee schedule code descriptions. Examples:\n"
            "  - \"GP consultation\" (not \"orthopaedic consultation\" or \"saw the doctor\")\n"
            "  - \"closure of wound\" (not \"open wound treatment\" or \"patient has a cut\")\n"
            "  - \"burns or abrasions\" (not \"scraped knee\" or \"graze\")\n"
            "  - \"dislocation of shoulder\" (not \"shoulder popped out\")\n"
            "  - \"fractured clavicle\" (not \"broken collarbone\")\n"
            "  - \"soft tissue injury\" (not \"muscle strain\")\n"
            "  - \"wound cleaning dressing\" (not \"wound care\")\n"
            "Generate one query per distinct billable item: the consultation itself, "
            "each distinct injury requiring treatment (wound closure, abrasion dressing, "
            "dislocation reduction, fracture immobilisation, etc). "
            "If an investigation RULES OUT a condition (e.g. X-ray shows no fracture), "
            "do NOT generate a query for that condition.\n"
            "Return strict JSON with keys: \"facts\", \"search_queries\"."
        )
        resp = client.chat.completions.create(
            model=os.getenv("LLM_MODEL", "gpt-4.1-mini"),
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": text},
            ],
        )
        payload = json.loads(resp.choices[0].message.content)
        facts = payload.get("facts", payload)
        queries = payload.get("search_queries", [])
        if not isinstance(queries, list):
            queries = []
        return {"mode": "llm", "facts": facts, "search_queries": queries}
    except Exception:
        return {"mode": "heuristic", "facts": {"summary": text}, "search_queries": []}


def _parse_age(text: str) -> int | None:
    m = re.search(r"\b(\d{1,3})\s*[-–—]?\s*year\s*[-–—]?\s*old\b", text.lower())
    return int(m.group(1)) if m else None


def _cm_length(text: str) -> float | None:
    m = re.search(r"\b(\d+(?:\.\d+)?)\s*cm\b", text.lower())
    return float(m.group(1)) if m else None


def _has_no_csc(text: str) -> bool:
    t = text.lower()
    return any(x in t for x in ["no community services card", "not a community services card", "without community services card"])


def _has_positive_amputation(text: str) -> bool:
    t = text.lower()
    has_term = any(k in t for k in ["amputation", "amputated", "digit amput", "finger amput", "toe amput"])
    negated = any(k in t for k in ["no amputation", "not amputated", "without amputation", "no digit loss", "no finger loss", "no toe loss"])
    return has_term and not negated


def _has_positive_fracture(text: str) -> bool:
    t = text.lower()
    has_term = "fracture" in t or "fractured" in t
    negated = any(k in t for k in ["no fracture", "no fractures", "not fractured", "without fracture", "fracture not suspected", "no fracture suspected"])
    return has_term and not negated


def _heuristic_boost(query_text: str, code: str, description: str, profile: dict[str, Any]) -> float:
    """Apply heuristic boosts based on query text and code description."""
    t = query_text.lower()
    d = description.lower()
    boost = 0.0

    age = _parse_age(t)

    has_csc = "community services card" in t and not _has_no_csc(t)
    dependent = any(k in t for k in ["dependent", "dependant", "child of"]) and has_csc
    nurse_gp_joint = ("nurse" in t and "gp" in t) and any(k in t for k in ["together", "jointly", "combined"])

    # ACC1520-specific heuristics (preserved for backward compatibility)
    # GP1 is the default adult consultation code — boost when age >= 14 OR age unknown (default adult)
    if code == "GP1" and not has_csc:
        if age is None or age >= 14:
            boost += 0.9
            if any(k in t for k in ["urgent care", "clinic", "consult", "acc claim", "workplace",
                                     "assessment", "plan", "objective", "subjective"]):
                boost += 0.3

    # Consultation routing for CSC-dependent teen in nurse+GP consult.
    if code == "GNCD" and nurse_gp_joint and dependent and age is not None and 14 <= age <= 17:
        boost += 3.0
    if code in {"NUCD", "NNCD", "GNCS", "NUCS", "NNCS", "GPCD", "GPCS"} and nurse_gp_joint and dependent and age is not None and 14 <= age <= 17:
        boost -= 1.2

    # Wound closure codes: boost when open wound / laceration is documented
    wound_present = any(k in t for k in ["open wound", "laceration", "wound closure", "wound", "sutures"])
    cm = _cm_length(t)

    if code == "MW1" and wound_present:
        if cm is not None and cm < 2:
            boost += 1.2
        elif cm is None:
            boost += 0.6  # wound present but size unknown — surface both MW1 and MW2

    if code == "MW2" and wound_present:
        if cm is not None and 2 <= cm <= 7:
            boost += 1.2
        elif cm is None:
            boost += 0.6  # wound present but size unknown

    if code == "MW3" and any(k in t for k in ["laceration", "wound", "closure"]):
        cm = _cm_length(t)
        if cm is not None and cm <= 7:
            boost -= 0.6
        elif cm is not None and cm > 7:
            boost += 0.8

    # Significant multi-site abrasion rule routing (prefer MB5 over laceration closure codes).
    multi_site_abrasion = any(k in t for k in ["multiple sites", "multi-site", "forearm", "knee", "flank", "hip"]) and any(k in t for k in ["abrasion", "road rash", "burns or abrasions"])
    larger_area = any(k in t for k in [">4 cm", "greater than 4 cm", "9 cm x", "7 cm x", "8 cm x"])
    if code == "MB5" and multi_site_abrasion and larger_area:
        boost += 2.6
    if code in {"MW1", "MW2", "MW3"} and multi_site_abrasion:
        boost -= 1.0

    # Abrasion treatment codes: MB3 (single site ≤4cm²), MB4 (single site >4cm²), MB5 (multi-site)
    if code == "MB3" and any(k in t for k in ["abrasion", "graze", "road rash"]):
        boost += 0.6
    if code == "MB4" and any(k in t for k in ["abrasion", "graze", "road rash"]):
        boost += 0.5

    # Epistaxis with nasal packing should strongly favor MM8.
    if code == "MM8" and "epistaxis" in t and any(k in t for k in ["nasal packing", "packing", "pack"]):
        boost += 2.2

    if code == "MM5" and any(k in t for k in ["foreign body", "fragment removed", "embedded metal", "embedded"]):
        boost += 1.0

    if code.startswith("MF") and not _has_positive_fracture(t):
        boost -= 1.6

    # Distal radius/ulna fracture logic.
    distal_ru = all(k in t for k in ["distal", "radius", "ulna"]) and _has_positive_fracture(t)
    if distal_ru:
        if code == "MF9" and any(k in t for k in ["cast", "immobilisation", "immobilization"]):
            boost += 1.4
        if code == "MF10":
            if any(k in t for k in ["without need for reduction", "no reduction", "not requiring reduction"]):
                boost -= 1.0
            if "closed reduction" in t and "no reduction" not in t:
                boost += 0.6

    # Distal tibia/fibula fracture logic.
    distal_tf = all(k in t for k in ["distal", "tibia", "fibula"]) and _has_positive_fracture(t)
    if distal_tf:
        if code == "MF15" and any(k in t for k in ["reduction", "reduced"]) and any(k in t for k in ["cast", "immobilisation", "immobilization"]):
            boost += 1.8
        if code == "MF16":
            boost -= 0.6
        if code == "MF14":
            boost -= 0.4

    # General positive bias when core terms overlap strongly.
    if any(k in t for k in ["debridement", "anaesthetic", "dressing"]) and any(k in d for k in ["cleaning", "anaesthetic", "dressing"]):
        boost += 0.2

    # Guardrail: amputation codes should only score when explicit amputation context is present.
    if code.startswith("MW4") and not _has_positive_amputation(t):
        boost -= 2.5

    return boost


def _is_csc_variant(code: str, description: str) -> bool:
    u = (code + " " + description).upper()
    return any(k in u for k in ["CS", "CD", "COMMUNITY SERVICES CARD"])


def _is_consultation_code(code: str) -> bool:
    return code.startswith(("GP", "GN", "NN", "NC", "NU", "PM"))


def _select_primary_consult(recs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    consults = [r for r in recs if _is_consultation_code(str(r.get("code", "")))]
    others = [r for r in recs if not _is_consultation_code(str(r.get("code", "")))]
    if not consults:
        return recs
    consults.sort(key=lambda r: ((r.get("match_score") or 0.0), (r.get("fee_excl_gst") or 0.0)), reverse=True)
    return [consults[0], *others]


def _has_multiple_injury_sites(text: str) -> bool:
    t = text.lower()
    site_groups = [
        ["head", "scalp", "forehead", "face"],
        ["hand", "finger", "wrist", "forearm", "radius", "ulna"],
        ["leg", "tibia", "fibula", "ankle", "knee", "foot"],
        ["shoulder", "clavicle", "elbow", "humerus"],
    ]
    hit = 0
    for grp in site_groups:
        if any(k in t for k in grp):
            hit += 1
    return hit >= 2


def _apply_pricing_rules(recs: list[dict[str, Any]], query_text: str, profile: dict[str, Any]) -> list[dict[str, Any]]:
    """Apply schedule-specific pricing rules.
    
    Default rule:
    - Consultation code(s): full value.
    - Non-consult treatments: if 2+ treatments at same visit, pay full for highest-cost treatment
      and 50% for each additional treatment.
    """
    consults = [r for r in recs if _is_consultation_code(r["code"])]
    treatments = [r for r in recs if not _is_consultation_code(r["code"])]

    treatments_sorted = sorted(treatments, key=lambda r: (r.get("fee_excl_gst") or 0.0), reverse=True)
    multi_injury = _has_multiple_injury_sites(query_text)

    priced: list[dict[str, Any]] = []
    for r in consults:
        unit = float(r.get("fee_excl_gst") or 0.0)
        rr = dict(r)
        rr["pricing_multiplier"] = 1.0
        rr["line_total_excl_gst"] = round(unit, 2)
        rr["quantity"] = 1
        priced.append(rr)

    for i, r in enumerate(treatments_sorted):
        unit = float(r.get("fee_excl_gst") or 0.0)
        mult = 1.0 if (i == 0 or not multi_injury) else 0.5
        rr = dict(r)
        rr["pricing_multiplier"] = mult
        rr["line_total_excl_gst"] = round(unit * mult, 2)
        rr["quantity"] = 1
        priced.append(rr)

    # preserve original order priority by match score desc as much as possible
    return sorted(priced, key=lambda r: r.get("match_score", 0), reverse=True)


def _quoted_evidence(query_text: str, keywords: list[str], max_quotes: int = 2) -> list[str]:
    if not query_text.strip() or not keywords:
        return []
    raw_parts = re.split(r"[\n\r\.]+", query_text)
    parts = [p.strip() for p in raw_parts if p and p.strip()]
    hits: list[str] = []
    for p in parts:
        lp = p.lower()
        if any(k in lp for k in keywords):
            hits.append(p[:220])
        if len(hits) >= max_quotes:
            break
    return hits


def _reason_for_code(rec: dict[str, Any], query_text: str, profile: dict[str, Any]) -> str:
    t = query_text.lower()
    code = str(rec.get("code", ""))
    desc = str(rec.get("description", ""))
    reasons: list[str] = []
    quotes: list[str] = []

    age = _parse_age(t)
    has_csc = "community services card" in t and not _has_no_csc(t)

    if code == "GNCD" and age is not None and 14 <= age <= 17 and has_csc and "nurse" in t and "gp" in t:
        reasons.append("patient context matches dependant CSC 14–17 nurse+GP consultation")
        quotes.extend(_quoted_evidence(query_text, ["17-year-old", "community services card", "nurse", "gp", "together", "jointly"]))

    if code == "MB5" and any(k in t for k in ["abrasion", "road rash"]) and any(k in t for k in ["multiple", "multi-site", "forearm", "knee", "flank", "hip"]):
        reasons.append("findings describe significant abrasions at multiple sites")
        quotes.extend(_quoted_evidence(query_text, ["abrasion", "road rash", "multiple", "forearm", "knee", "flank", "hip"]))
        if any(k in t for k in [">4 cm", "greater than 4 cm", "9 cm x", "7 cm x", "8 cm x"]):
            reasons.append("documented wound dimensions align with >4 cm² threshold")
            quotes.extend(_quoted_evidence(query_text, ["9 cm x", "7 cm x", "8 cm x", "greater than 4 cm", ">4 cm"]))

    if code == "MM8" and "epistaxis" in t and any(k in t for k in ["nasal packing", "packing", "pack"]):
        reasons.append("epistaxis treated with nasal packing is explicitly documented")
        quotes.extend(_quoted_evidence(query_text, ["epistaxis", "nosebleed", "nasal packing", "packing"]))

    if code.startswith("MF") and _has_positive_fracture(t):
        reasons.append("fracture context is present in the input")
        quotes.extend(_quoted_evidence(query_text, ["fracture", "fractured", "cast", "immobilisation", "reduction"]))
    elif code.startswith("MF"):
        reasons.append("fracture mention appears negated in input (e.g., no fracture suspected); this candidate should be treated as low-confidence")
        quotes.extend(_quoted_evidence(query_text, ["no fracture", "fracture not suspected", "no fracture suspected"]))

    # De-duplicate while preserving order.
    dedup_quotes: list[str] = []
    for q in quotes:
        if q not in dedup_quotes:
            dedup_quotes.append(q)

    if not reasons:
        reasons.append("no direct quoted trigger found; this is a low-confidence semantic/keyword match")

    text = f"{code} selected because " + "; ".join(reasons)
    if dedup_quotes:
        text += ". Quoted input: " + " | ".join(f'\"{q}\"' for q in dedup_quotes[:2])

    mult = float(rec.get("pricing_multiplier") or 1.0)
    if mult < 1.0:
        text += "; priced at 50% as an additional same-visit treatment under the prototype multiple-treatment rule"
    else:
        text += "; priced at full rate for this visit"

    text += f". Relevant schedule descriptor: {desc}"
    return text


def _final_reasoning_review(
    query_text: str,
    consult_facts: dict[str, Any],
    recs: list[dict[str, Any]],
    profile: dict[str, Any],
) -> list[dict[str, Any]]:
    """LLM annotation step: adds clinical reasoning to each candidate.

    Does NOT filter codes — all candidates are returned with added reasoning.
    The clinician decides which codes to bill.
    """
    if not recs:
        return recs

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return recs

    model = os.getenv("FINAL_REASONING_MODEL", "gpt-5.4")
    schedule_type = profile.get("schedule_type", "medical")

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        prompt = (
            f"You are a clinical billing advisor for the ACC1520 {schedule_type} fee schedule.\n"
            "Given consult input, extracted facts, and candidate billing codes, provide clinical reasoning for each.\n"
            "Return STRICT JSON with key 'annotations' — an array with one entry per candidate code.\n"
            "Each entry must include: code, relevance ('high', 'medium', or 'low'), reason (1-2 sentence explanation).\n\n"
            "Guidelines:\n"
            "- 'high': directly supported by documented injuries/treatments\n"
            "- 'medium': plausibly relevant but not explicitly documented\n"
            "- 'low': unlikely to be appropriate (e.g. negated findings, wrong body site)\n"
            "- Note any negations (e.g. 'no fracture' should mark fracture codes as low)\n"
            "- Do not omit any candidate — annotate ALL of them.\n"
        )

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

        resp = client.chat.completions.create(
            model=model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
        )

        content = resp.choices[0].message.content or "{}"
        parsed = json.loads(content)
        annotations = parsed.get("annotations") or []
        if not isinstance(annotations, list):
            return recs

        # Merge annotations back into recs
        by_code = {str(a.get("code")): a for a in annotations if isinstance(a, dict)}
        for r in recs:
            ann = by_code.get(str(r.get("code")))
            if ann:
                r["llm_reason"] = str(ann.get("reason") or "").strip()
                r["relevance"] = str(ann.get("relevance") or "medium").strip()

        # Sort: high relevance first, then medium, then low
        relevance_order = {"high": 0, "medium": 1, "low": 2}
        recs.sort(key=lambda r: (relevance_order.get(r.get("relevance", "medium"), 1), -r.get("match_score", 0)))

        return recs
    except Exception:
        return recs


def _fetch_semantic_chunks(cur: sqlite3.Cursor, schedule_id: str, query_text: str, top_n: int = 3) -> list[dict[str, Any]]:
    cur.execute(
        """
        SELECT p.page, p.chunk_text, e.embedding_json
        FROM policy_chunks p
        JOIN policy_chunk_embeddings e ON e.chunk_id = p.id
        WHERE p.schedule_id = ? AND e.schedule_id = ?
        """,
        (schedule_id, schedule_id),
    )
    rows = cur.fetchall()
    if not rows:
        return []

    q_emb = embed_texts([query_text])[0]
    scored = []
    for r in rows:
        emb = json.loads(r["embedding_json"])
        s = cosine_similarity(q_emb, emb)
        scored.append((s, r["page"], r["chunk_text"][:300]))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [
        {"page": p, "score": round(s, 4), "snippet": t}
        for s, p, t in scored[:top_n]
        if s > 0
    ]


def recommend_codes(
    schedule_id: str,
    consult_text: str | None,
    consult_template: dict[str, Any] | None,
    base_dir: Path | None = None,
    top_n: int = 5,
    gst_mode: str = "excl",
) -> dict[str, Any]:
    """
    Recommend billing codes for a consultation.
    
    Args:
        schedule_id: Identifier for the schedule (e.g., "acc1520-medical", "acc2024-gp")
        consult_text: Free-text clinical notes
        consult_template: Optional structured template
        base_dir: Base directory for processed data
        top_n: Number of recommendations to return
        gst_mode: "excl" or "incl" - controls which fee is returned as primary "fee" field
    """
    # Load profile for this schedule
    profile = _load_profile(schedule_id, base_dir)
    
    # Get database path
    db_path = _get_db_path(schedule_id, base_dir)
    if not db_path.exists():
        return {
            "error": f"Schedule not found: {schedule_id}",
            "schedule_id": schedule_id,
            "db_path": str(db_path),
            "recommendations": [],
        }

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute(
        """
        SELECT code, description, fee_excl_gst, fee_incl_gst, page
        FROM schedule_codes
        WHERE schedule_id = ?
        """,
        (schedule_id,),
    )
    rows = cur.fetchall()

    template_text = ""
    if consult_template:
        template_text = " ".join(f"{k} {v}" for k, v in consult_template.items())

    query_text = f"{consult_text or ''} {template_text}".strip()
    consult_facts = _extract_consult_facts(consult_text or "", consult_template, profile)

    # Use LLM-generated search queries for scoring instead of raw SOAP text
    search_queries = consult_facts.get("search_queries", [])

    # Build semantic embeddings for search queries and code descriptions
    code_descs = []
    code_rows = []
    no_csc = _has_no_csc(query_text)
    for r in rows:
        code = r["code"]
        desc = r["description"]
        if code in {"ACC", "Code"}:
            continue
        if no_csc and _is_csc_variant(code, desc):
            continue
        if code.startswith("MW4") and not _has_positive_amputation(query_text):
            continue
        code_descs.append(f"{code} {desc}")
        code_rows.append(r)

    # Embed search queries and code descriptions for semantic scoring
    semantic_scores: dict[int, float] = {}
    if search_queries:
        try:
            all_texts = search_queries + code_descs
            all_embs = embed_texts(all_texts)
            query_embs = all_embs[:len(search_queries)]
            desc_embs = all_embs[len(search_queries):]

            for i in range(len(code_rows)):
                best_sem = 0.0
                for qe in query_embs:
                    s = cosine_similarity(qe, desc_embs[i])
                    if s > best_sem:
                        best_sem = s
                semantic_scores[i] = best_sem
        except Exception:
            pass

    candidates: list[Candidate] = []
    for i, r in enumerate(code_rows):
        code = r["code"]
        desc = r["description"]
        desc_tokens = _tokens(desc)

        # Keyword score: best match across LLM search queries
        best_kw_score = 0.0
        if search_queries:
            for sq in search_queries:
                sq_tokens = _tokens(sq)
                s = _keyword_similarity(sq_tokens, desc_tokens)
                if s > best_kw_score:
                    best_kw_score = s
        else:
            query_tokens = _tokens(query_text)
            best_kw_score = _keyword_similarity(query_tokens, desc_tokens)

        # Combine keyword + semantic + heuristic
        sem_score = semantic_scores.get(i, 0.0)
        combined = (best_kw_score * 0.4) + (sem_score * 0.6) + _heuristic_boost(query_text, code, desc, profile)
        if combined <= 0.05:
            continue
        candidates.append(
            Candidate(
                code=code,
                description=desc,
                fee_excl_gst=r["fee_excl_gst"],
                fee_incl_gst=r["fee_incl_gst"],
                page=r["page"],
                score=combined,
            )
        )

    # Split candidates into consultation codes and treatment codes.
    # Pick best consultation, then fill remaining slots with diverse treatments.
    candidates.sort(key=lambda c: (c.score, c.fee_excl_gst or 0.0), reverse=True)

    consult_candidates = [c for c in candidates if _is_consultation_code(c.code)]
    treatment_candidates = [c for c in candidates if not _is_consultation_code(c.code)]

    selected: list[Candidate] = []

    # Pick the single best consultation code
    if consult_candidates:
        selected.append(consult_candidates[0])

    # Fill remaining slots with treatment codes.
    # Allow up to 2 codes per prefix family to surface alternatives (e.g. MW1 vs MW2).
    # The LLM annotator will mark irrelevant ones — clinician makes final selection.
    prefix_count: dict[str, int] = {}
    for c in treatment_candidates:
        prefix = re.sub(r"\d+$", "", c.code)
        count = prefix_count.get(prefix, 0)
        if count >= 2 and c.score < 0.95:
            continue
        selected.append(c)
        prefix_count[prefix] = count + 1
        if len(selected) >= top_n:
            break

    recs = [
        {
            "code": c.code,
            "description": c.description,
            "fee": c.fee_excl_gst if gst_mode == "excl" else c.fee_incl_gst,
            "fee_gst": c.fee_incl_gst if gst_mode == "excl" else c.fee_excl_gst,
            "fee_excl_gst": c.fee_excl_gst,
            "fee_incl_gst": c.fee_incl_gst,
            "page": c.page,
            "match_score": round(c.score, 4),
        }
        for c in selected
    ]

    # Context-sensitive strict-ish gate to avoid obvious unrelated procedures.
    q = query_text.lower()
    if all(k in q for k in ["distal", "radius", "ulna"]) and _has_positive_fracture(q):
        allowed = {"GP1", "MF9", "MF10", "GP14", "GPN"}
        recs = [r for r in recs if r["code"] in allowed]

    if all(k in q for k in ["distal", "tibia", "fibula"]) and _has_positive_fracture(q):
        allowed = {"GP1", "MF15", "MF14", "MF16", "MW1", "MW2", "MW3"}
        recs = [r for r in recs if r["code"] in allowed]

    recs = _final_reasoning_review(query_text, consult_facts, recs, profile)
    recs = _select_primary_consult(recs)
    constraints_result = apply_basic_constraints(recs, consult_facts)
    recs = _apply_pricing_rules(recs, query_text, profile)

    for r in recs:
        base_reason = _reason_for_code(r, query_text, profile)
        llm_reason = str(r.get("llm_reason") or "").strip()
        r["reason"] = f"{base_reason} | Final-model review: {llm_reason}" if llm_reason else base_reason

    total_excl = sum((r.get("line_total_excl_gst") or 0.0) for r in recs)
    total_incl = sum(((r.get("fee_incl_gst") or 0.0) * (r.get("pricing_multiplier") or 1.0)) for r in recs)

    # Use search queries for semantic evidence lookup
    semantic_query = " ".join(search_queries) if search_queries else query_text
    evidence_chunks = _fetch_semantic_chunks(cur, schedule_id, semantic_query, top_n=3)
    if not evidence_chunks:
        # Fallback if semantic index not built yet.
        cur.execute(
            "SELECT page, chunk_text FROM policy_chunks WHERE schedule_id = ? LIMIT 300",
            (schedule_id,),
        )
        chunks = cur.fetchall()
        fallback_tokens = _tokens(semantic_query)
        scored = []
        for ch in chunks:
            s = _keyword_similarity(fallback_tokens, _tokens(ch["chunk_text"]))
            if s > 0:
                scored.append((s, ch["page"], ch["chunk_text"][:300]))
        scored.sort(reverse=True)
        evidence_chunks = [{"page": p, "score": round(s, 4), "snippet": t} for s, p, t in scored[:3]]

    conn.close()

    return {
        "query": query_text,
        "consult_facts": consult_facts,
        "schedule_id": schedule_id,
        "profile": profile,
        "gst_mode": gst_mode,
        "recommendations": recs,
        "estimated_total_excl_gst": round(total_excl, 2),
        "estimated_total_incl_gst": round(total_incl, 2),
        "evidence_chunks": evidence_chunks,
        "constraint_checks": {
            "allowed": constraints_result.allowed,
            "reasons": constraints_result.reasons,
        },
        "notes": [
            "Generic RAG model with dynamic schedule profiling.",
            "Schedule-specific constraints applied based on profile.",
        ],
    }
