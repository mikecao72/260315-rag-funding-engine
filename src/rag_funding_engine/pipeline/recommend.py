from __future__ import annotations

import json
import os
from pathlib import Path
import sqlite3
from typing import Any

from rag_funding_engine.pipeline.semantic import cosine_similarity, embed_texts


# ---------------------------------------------------------------------------
# Helpers: paths, loading
# ---------------------------------------------------------------------------

def _get_db_path(schedule_id: str, base_dir: Path | None = None) -> Path:
    if base_dir is None:
        from rag_funding_engine.api.main import ROOT
        base_dir = ROOT / "data" / "processed"
    return base_dir / schedule_id / f"{schedule_id}.sqlite3"


def _get_manifest_path(schedule_id: str, base_dir: Path | None = None) -> Path:
    if base_dir is None:
        from rag_funding_engine.api.main import ROOT
        base_dir = ROOT / "data" / "processed"
    return base_dir / schedule_id / "manifest.json"


def _load_manifest(schedule_id: str, base_dir: Path | None = None) -> dict[str, Any]:
    manifest_path = _get_manifest_path(schedule_id, base_dir)
    if not manifest_path.exists():
        return {}
    try:
        return json.loads(manifest_path.read_text())
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Step 2: Consult fact extraction (kept from original)
# ---------------------------------------------------------------------------

def _extract_consult_facts(text: str, template: dict[str, Any] | None, profile: dict[str, Any]) -> dict[str, Any]:
    """Extract consult facts using LLM, guided by profile."""
    if template:
        return {"mode": "template", "facts": template}

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {"mode": "no_api_key", "facts": {"summary": text}}

    key_dims = profile.get("key_dimensions", ["injury_type", "body_site", "procedure", "severity"])
    dims_str = ", ".join(key_dims)

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        prompt = (
            f"Extract consult billing-relevant facts as strict compact JSON with keys: {dims_str}. "
            "If unknown use null."
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
        return {"mode": "llm", "facts": payload}
    except Exception:
        return {"mode": "llm_error", "facts": {"summary": text}}


# ---------------------------------------------------------------------------
# Step 3 (NEW): LLM-based code selection
# ---------------------------------------------------------------------------

def _build_codes_table(rows: list[dict[str, Any]]) -> str:
    """Format schedule codes as a text table for the LLM (no fee amounts)."""
    lines = ["Code     | Description", "-------- | -----------"]
    for r in rows:
        code = r["code"]
        desc = r["description"]
        lines.append(f"{code:<8s} | {desc}")
    return "\n".join(lines)


def _build_guidance_text(guidance: dict[str, Any]) -> str:
    """Format schedule_guidance into readable instructions for the LLM."""
    if not guidance:
        return "No schedule-specific guidance available."

    parts: list[str] = []

    scope = guidance.get("document_scope", "")
    if scope:
        parts.append(f"DOCUMENT SCOPE:\n{scope}")

    groupings = guidance.get("code_groupings", [])
    if groupings:
        parts.append("CODE GROUPINGS:")
        for g in groupings:
            codes = g.get("codes", "")
            cat = g.get("category", "")
            desc = g.get("description", "")
            parts.append(f"  - {cat} [{codes}]: {desc}")

    rules = guidance.get("selection_rules", [])
    if rules:
        parts.append("SELECTION RULES:")
        for i, rule in enumerate(rules, 1):
            parts.append(f"  {i}. {rule}")

    return "\n\n".join(parts)


LLM_SELECT_PROMPT = """You are a medical billing code selector. Given a clinical consultation note, select ALL potentially relevant billing codes from the schedule below.

{guidance_text}

AVAILABLE CODES:
{codes_table}

Cast a WIDE net. It is much better to include a borderline-relevant code than to miss one. A human reviewer and a reasoning model will narrow the list later.

For EVERY code you select, provide:
- code: The code identifier (must match exactly from the list above)
- confidence: A number from 0.0 to 1.0
    1.0 = definitely applies
    0.7 = very likely applies
    0.4 = possibly applies
    0.2 = unlikely but worth considering
- reason: A brief explanation of why this code may be relevant to the consultation

You MUST select at least one consultation code if any are available in the schedule.
Do NOT invent codes that are not in the AVAILABLE CODES list.

Respond with STRICT JSON: {{"selected_codes": [...]}}"""


def _llm_select_codes(
    query_text: str,
    consult_facts: dict[str, Any],
    code_rows: list[dict[str, Any]],
    profile: dict[str, Any],
    guidance: dict[str, Any],
    min_confidence: float = 0.1,
) -> list[dict[str, Any]]:
    """Use LLM to select relevant codes from the full schedule.

    Returns list of {code, confidence, reason} dicts.
    Raises RuntimeError if LLM call fails (no fallback).
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is required for code selection. "
            "Set it in your environment or .env file."
        )

    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    model = os.getenv("LLM_MODEL", "gpt-4.1-mini")

    codes_table = _build_codes_table(code_rows)
    guidance_text = _build_guidance_text(guidance)

    system_prompt = LLM_SELECT_PROMPT.format(
        guidance_text=guidance_text,
        codes_table=codes_table,
    )

    user_payload = {
        "consultation_note": query_text,
        "extracted_facts": consult_facts.get("facts", {}),
    }

    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
    )

    content = resp.choices[0].message.content or "{}"
    parsed = json.loads(content)
    selected = parsed.get("selected_codes", [])

    if not isinstance(selected, list):
        raise RuntimeError(f"LLM returned unexpected format: {type(selected)}")

    # Validate codes exist in schedule and apply confidence threshold
    valid_codes = {r["code"] for r in code_rows}
    filtered: list[dict[str, Any]] = []
    for item in selected:
        if not isinstance(item, dict):
            continue
        code = str(item.get("code", "")).strip()
        confidence = float(item.get("confidence", 0))
        reason = str(item.get("reason", "")).strip()
        if code not in valid_codes:
            continue
        if confidence < min_confidence:
            continue
        filtered.append({
            "code": code,
            "confidence": confidence,
            "reason": reason,
        })

    # Sort by confidence descending
    filtered.sort(key=lambda x: x["confidence"], reverse=True)
    return filtered


# ---------------------------------------------------------------------------
# Final reasoning review (narrowing step - reworked)
# ---------------------------------------------------------------------------

def _final_reasoning_review(
    query_text: str,
    consult_facts: dict[str, Any],
    recs: list[dict[str, Any]],
    profile: dict[str, Any],
    guidance: dict[str, Any],
) -> list[dict[str, Any]]:
    """Reasoning-model adjudication to narrow the wide-net LLM selections."""
    if not recs:
        return recs

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return recs

    model = os.getenv("FINAL_REASONING_MODEL", "gpt-5.4")
    schedule_type = profile.get("schedule_type", "medical")
    guidance_text = _build_guidance_text(guidance)

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        prompt = (
            f"You are the final clinical billing adjudicator for a {schedule_type} fee schedule.\n\n"
            f"SCHEDULE GUIDANCE:\n{guidance_text}\n\n"
            "Given the consultation input, extracted facts, and a wide-net list of candidate codes "
            "(each with a selection reason and confidence from the initial selector), "
            "narrow the list to the most appropriate codes.\n\n"
            "Return STRICT JSON with key 'final_recommendations'. "
            "Each item must include: code, keep (boolean), reason (short explanation).\n\n"
            "Rules:\n"
            "- Respect explicit negations in the input (e.g. 'no fracture', 'no amputation')\n"
            "- Prefer specific, clinically-supported codes over generic ones\n"
            "- Avoid mutually incompatible or same-family duplicates unless clearly justified\n"
            "- Keep exactly ONE consultation code (the most appropriate for the provider/patient)\n"
            "- Retain at most 8 final codes\n"
            "- Do NOT invent codes not present in the candidates"
        )

        payload = {
            "consultation_note": query_text,
            "extracted_facts": consult_facts,
            "candidates": [
                {
                    "code": r.get("code"),
                    "description": r.get("description"),
                    "confidence": r.get("confidence"),
                    "selector_reason": r.get("selector_reason"),
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
        final = parsed.get("final_recommendations") or []
        if not isinstance(final, list):
            return recs

        by_code = {str(r.get("code")): r for r in recs}
        reviewed: list[dict[str, Any]] = []
        for item in final:
            if not isinstance(item, dict):
                continue
            code = str(item.get("code") or "")
            keep = bool(item.get("keep"))
            if not keep or code not in by_code:
                continue
            rr = dict(by_code[code])
            llm_reason = str(item.get("reason") or "").strip()
            if llm_reason:
                rr["adjudicator_reason"] = llm_reason
            reviewed.append(rr)
            if len(reviewed) >= 8:
                break

        return reviewed or recs
    except Exception:
        return recs


# ---------------------------------------------------------------------------
# Pricing rules (kept from original)
# ---------------------------------------------------------------------------

def _is_consultation_code(code: str) -> bool:
    return code.startswith(("GP", "GN", "NN", "NC", "NU", "PM"))


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

    - Consultation code(s): full value.
    - Non-consult treatments: highest-cost at full rate,
      additional same-visit treatments at 50% if multiple injury sites.
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

    return sorted(priced, key=lambda r: r.get("confidence", 0), reverse=True)


# ---------------------------------------------------------------------------
# Semantic evidence (kept from original)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def recommend_codes(
    schedule_id: str,
    consult_text: str | None,
    consult_template: dict[str, Any] | None,
    base_dir: Path | None = None,
    top_n: int = 10,
    gst_mode: str = "excl",
    min_confidence: float = 0.1,
) -> dict[str, Any]:
    """
    Recommend billing codes for a consultation.

    Pipeline:
      1. Load profile + schedule_guidance from manifest
      2. Extract consult facts via LLM
      3. LLM selects candidate codes (wide net)
      4. Reasoning model narrows candidates
      5. Apply pricing rules
      6. Fetch semantic evidence chunks
      7. Assemble response
    """
    # 1. Load manifest (profile + guidance)
    manifest = _load_manifest(schedule_id, base_dir)
    profile = manifest.get("profile", {})
    guidance = manifest.get("schedule_guidance", {})

    # Validate DB exists
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

    # Load all codes from DB
    cur.execute(
        "SELECT code, description, fee_excl_gst, fee_incl_gst, page FROM schedule_codes WHERE schedule_id = ?",
        (schedule_id,),
    )
    all_rows = [dict(r) for r in cur.fetchall()]

    # Filter out header artifacts
    code_rows = [r for r in all_rows if r["code"] not in {"ACC", "Code"}]

    # Build query text
    template_text = ""
    if consult_template:
        template_text = " ".join(f"{k} {v}" for k, v in consult_template.items())
    query_text = f"{consult_text or ''} {template_text}".strip()

    if not query_text:
        conn.close()
        return {
            "error": "No consultation text provided",
            "schedule_id": schedule_id,
            "recommendations": [],
        }

    # 2. Extract consult facts
    consult_facts = _extract_consult_facts(consult_text or "", consult_template, profile)

    # 3. LLM code selection (wide net)
    try:
        llm_selections = _llm_select_codes(
            query_text=query_text,
            consult_facts=consult_facts,
            code_rows=code_rows,
            profile=profile,
            guidance=guidance,
            min_confidence=min_confidence,
        )
    except RuntimeError as e:
        conn.close()
        return {
            "error": str(e),
            "schedule_id": schedule_id,
            "recommendations": [],
        }

    # Map LLM selections back to full DB rows (with fees)
    code_lookup = {r["code"]: r for r in code_rows}
    recs: list[dict[str, Any]] = []
    for sel in llm_selections:
        code = sel["code"]
        db_row = code_lookup.get(code)
        if not db_row:
            continue
        recs.append({
            "code": code,
            "description": db_row["description"],
            "fee": db_row["fee_excl_gst"] if gst_mode == "excl" else db_row["fee_incl_gst"],
            "fee_gst": db_row["fee_incl_gst"] if gst_mode == "excl" else db_row["fee_excl_gst"],
            "fee_excl_gst": db_row["fee_excl_gst"],
            "fee_incl_gst": db_row["fee_incl_gst"],
            "page": db_row["page"],
            "confidence": sel["confidence"],
            "selector_reason": sel["reason"],
        })

    # Cap to top_n before sending to reasoning model
    recs = recs[:top_n]

    # 4. Final reasoning review (narrowing)
    recs = _final_reasoning_review(query_text, consult_facts, recs, profile, guidance)

    # 5. Apply pricing rules
    recs = _apply_pricing_rules(recs, query_text, profile)

    # Build final reason string per code
    for r in recs:
        parts = []
        selector_reason = r.get("selector_reason", "")
        if selector_reason:
            parts.append(selector_reason)
        adjudicator_reason = r.get("adjudicator_reason", "")
        if adjudicator_reason:
            parts.append(f"Adjudicator: {adjudicator_reason}")
        mult = float(r.get("pricing_multiplier", 1.0))
        if mult < 1.0:
            parts.append("Priced at 50% as additional same-visit treatment")
        r["reason"] = " | ".join(parts) if parts else "Selected by LLM code selector"

    total_excl = sum((r.get("line_total_excl_gst") or 0.0) for r in recs)
    total_incl = sum(((r.get("fee_incl_gst") or 0.0) * (r.get("pricing_multiplier") or 1.0)) for r in recs)

    # 6. Semantic evidence chunks
    evidence_chunks = _fetch_semantic_chunks(cur, schedule_id, query_text, top_n=3)

    conn.close()

    # 7. Assemble response
    return {
        "query": query_text,
        "consult_facts": consult_facts,
        "schedule_id": schedule_id,
        "profile": profile,
        "schedule_guidance": guidance,
        "gst_mode": gst_mode,
        "min_confidence": min_confidence,
        "recommendations": recs,
        "estimated_total_excl_gst": round(total_excl, 2),
        "estimated_total_incl_gst": round(total_incl, 2),
        "evidence_chunks": evidence_chunks,
        "notes": [
            "Codes selected by LLM with full schedule context.",
            "Narrowed by reasoning model adjudication.",
            "Pricing rules applied (50% for additional same-visit treatments).",
        ],
    }
