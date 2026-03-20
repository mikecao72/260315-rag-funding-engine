from __future__ import annotations

import json
import os
from pathlib import Path
import sqlite3
from typing import Any

from rag_funding_engine.pipeline.job_logger import JobLogger
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
# Step 2: Consult fact extraction
# ---------------------------------------------------------------------------

def _extract_consult_facts(
    text: str,
    template: dict[str, Any] | None,
    profile: dict[str, Any],
    logger: JobLogger | None = None,
) -> dict[str, Any]:
    """Extract consult facts using LLM, guided by profile."""
    if template:
        result = {"mode": "template", "facts": template}
        if logger:
            logger.log("Mode: template (structured input provided)")
            logger.log_json("Facts", result["facts"])
        return result

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        result = {"mode": "no_api_key", "facts": {"summary": text}}
        if logger:
            logger.log("Mode: no_api_key (OPENAI_API_KEY not set)")
        return result

    key_dims = profile.get("key_dimensions", ["injury_type", "body_site", "procedure", "severity"])
    dims_str = ", ".join(key_dims)
    model = os.getenv("LLM_MODEL", "gpt-4.1-mini")

    prompt = (
        f"Extract consult billing-relevant facts as strict compact JSON with keys: {dims_str}. "
        "If unknown use null."
    )

    if logger:
        logger.log(f"Model: {model}")
        logger.log(f"Extraction dimensions: {dims_str}")
        logger.log_prompt("System prompt", prompt)

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": text},
            ],
        )
        raw_content = resp.choices[0].message.content
        payload = json.loads(raw_content)

        if logger:
            logger.log("Mode: llm (success)")
            logger.log_json("Raw LLM response", raw_content)
            logger.log_json("Parsed facts", payload)

        return {"mode": "llm", "facts": payload}
    except Exception as e:
        if logger:
            logger.log(f"Mode: llm_error ({e})")
        return {"mode": "llm_error", "facts": {"summary": text}}


# ---------------------------------------------------------------------------
# Step 3: LLM-based code selection
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
        if isinstance(scope, dict):
            scope_text = json.dumps(scope, indent=2, ensure_ascii=False)
        else:
            scope_text = str(scope)
        parts.append(f"DOCUMENT SCOPE:\n{scope_text}")

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
    logger: JobLogger | None = None,
) -> list[dict[str, Any]]:
    """Use LLM to select relevant codes from the full schedule."""
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

    if logger:
        logger.log(f"Model: {model}")
        logger.log(f"Min confidence threshold: {min_confidence}")
        logger.log(f"Codes in schedule: {len(code_rows)}")
        logger.log_prompt("System prompt", system_prompt)
        logger.log_json("User payload", user_payload)

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

    if logger:
        logger.log_json("Raw LLM response", parsed)

    if not isinstance(selected, list):
        raise RuntimeError(f"LLM returned unexpected format: {type(selected)}")

    # Validate codes exist in schedule and apply confidence threshold
    valid_codes = {r["code"] for r in code_rows}
    filtered: list[dict[str, Any]] = []
    dropped_invalid: list[str] = []
    dropped_confidence: list[dict[str, Any]] = []

    for item in selected:
        if not isinstance(item, dict):
            continue
        code = str(item.get("code", "")).strip()
        confidence = float(item.get("confidence", 0))
        reason = str(item.get("reason", "")).strip()
        if code not in valid_codes:
            dropped_invalid.append(code)
            continue
        if confidence < min_confidence:
            dropped_confidence.append({"code": code, "confidence": confidence, "reason": reason})
            continue
        filtered.append({
            "code": code,
            "confidence": confidence,
            "reason": reason,
        })

    filtered.sort(key=lambda x: x["confidence"], reverse=True)

    if logger:
        logger.log(f"LLM returned {len(selected)} codes total")
        if dropped_invalid:
            logger.log(f"Dropped (invalid code): {dropped_invalid}")
        if dropped_confidence:
            logger.log(f"Dropped (below min_confidence {min_confidence}):")
            for d in dropped_confidence:
                logger.log(f"  {d['code']} ({d['confidence']}) - {d['reason']}")
        logger.log(f"Retained after filtering: {len(filtered)} codes")
        logger.log("")
        logger.log_table(
            "Selected codes",
            filtered,
            ["code", "confidence", "reason"],
        )

    return filtered


# ---------------------------------------------------------------------------
# Final reasoning review (narrowing step)
# ---------------------------------------------------------------------------

def _final_reasoning_review(
    query_text: str,
    consult_facts: dict[str, Any],
    recs: list[dict[str, Any]],
    profile: dict[str, Any],
    guidance: dict[str, Any],
    logger: JobLogger | None = None,
) -> list[dict[str, Any]]:
    """Reasoning-model adjudication to narrow the wide-net LLM selections."""
    if not recs:
        if logger:
            logger.log("No candidates to adjudicate — skipping.")
        return recs

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        if logger:
            logger.log("No API key — skipping adjudication, passing all candidates through.")
        return recs

    model = os.getenv("FINAL_REASONING_MODEL", "gpt-5.4")
    schedule_type = profile.get("schedule_type", "medical")
    guidance_text = _build_guidance_text(guidance)

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

    if logger:
        logger.log(f"Model: {model}")
        logger.log(f"Candidates sent: {len(recs)}")
        logger.log_prompt("System prompt", prompt)
        logger.log_json("User payload", payload)

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
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

        if logger:
            logger.log_json("Raw LLM response", parsed)

        if not isinstance(final, list):
            if logger:
                logger.log("Response was not a list — passing all candidates through.")
            return recs

        by_code = {str(r.get("code")): r for r in recs}
        reviewed: list[dict[str, Any]] = []
        dropped: list[dict[str, Any]] = []

        for item in final:
            if not isinstance(item, dict):
                continue
            code = str(item.get("code") or "")
            keep = bool(item.get("keep"))
            reason = str(item.get("reason") or "").strip()
            if not keep:
                dropped.append({"code": code, "reason": reason})
                continue
            if code not in by_code:
                continue
            rr = dict(by_code[code])
            if reason:
                rr["adjudicator_reason"] = reason
            reviewed.append(rr)
            if len(reviewed) >= 8:
                break

        if logger:
            logger.log(f"Adjudicator kept {len(reviewed)} codes, dropped {len(dropped)}")
            if dropped:
                logger.log("Dropped codes:")
                for d in dropped:
                    logger.log(f"  {d['code']} - {d['reason']}")
            logger.log("")
            logger.log_table(
                "Final codes after adjudication",
                [{"code": r["code"], "confidence": r.get("confidence", ""), "adjudicator_reason": r.get("adjudicator_reason", "")} for r in reviewed],
                ["code", "confidence", "adjudicator_reason"],
            )

        return reviewed or recs
    except Exception as e:
        if logger:
            logger.log(f"Adjudication failed ({e}) — passing all candidates through.")
        return recs


# ---------------------------------------------------------------------------
# Pricing rules
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


def _apply_pricing_rules(
    recs: list[dict[str, Any]],
    query_text: str,
    profile: dict[str, Any],
    logger: JobLogger | None = None,
) -> list[dict[str, Any]]:
    """Apply schedule-specific pricing rules."""
    consults = [r for r in recs if _is_consultation_code(r["code"])]
    treatments = [r for r in recs if not _is_consultation_code(r["code"])]
    treatments_sorted = sorted(treatments, key=lambda r: (r.get("fee_excl_gst") or 0.0), reverse=True)
    multi_injury = _has_multiple_injury_sites(query_text)

    if logger:
        logger.log(f"Consultation codes: {len(consults)}")
        logger.log(f"Treatment codes: {len(treatments)}")
        logger.log(f"Multiple injury sites detected: {multi_injury}")
        logger.log("")

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

    result = sorted(priced, key=lambda r: r.get("confidence", 0), reverse=True)

    if logger:
        logger.log_table(
            "Priced codes",
            [{"code": r["code"], "fee_excl_gst": r.get("fee_excl_gst", ""), "multiplier": r["pricing_multiplier"], "line_total": r["line_total_excl_gst"]} for r in result],
            ["code", "fee_excl_gst", "multiplier", "line_total"],
        )

    return result


# ---------------------------------------------------------------------------
# Semantic evidence
# ---------------------------------------------------------------------------

def _fetch_semantic_chunks(
    cur: sqlite3.Cursor,
    schedule_id: str,
    query_text: str,
    top_n: int = 3,
    logger: JobLogger | None = None,
) -> list[dict[str, Any]]:
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

    if logger:
        logger.log(f"Total embedded chunks in DB: {len(rows)}")

    if not rows:
        if logger:
            logger.log("No embedded chunks found — skipping semantic evidence.")
        return []

    q_emb = embed_texts([query_text])[0]
    scored = []
    for r in rows:
        emb = json.loads(r["embedding_json"])
        s = cosine_similarity(q_emb, emb)
        scored.append((s, r["page"], r["chunk_text"][:300]))

    scored.sort(key=lambda x: x[0], reverse=True)
    results = [
        {"page": p, "score": round(s, 4), "snippet": t}
        for s, p, t in scored[:top_n]
        if s > 0
    ]

    if logger:
        logger.log(f"Top {top_n} semantic matches:")
        for chunk in results:
            logger.log(f"  Page {chunk['page']} (score {chunk['score']}): {chunk['snippet'][:120]}...")
        logger.log("")

    return results


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
    # Create job logger
    logger = JobLogger()

    # ── STEP 1: INPUT & CONFIG ──
    logger.step(1, "INPUT & CONFIGURATION")
    logger.log(f"Schedule ID: {schedule_id}")
    logger.log(f"GST Mode: {gst_mode}")
    logger.log(f"Top N: {top_n}")
    logger.log(f"Min Confidence: {min_confidence}")
    logger.log("")
    if consult_text:
        logger.log("Consult Text:")
        for line in (consult_text or "").split("\n"):
            logger.log(f"  {line}")
        logger.log("")
    if consult_template:
        logger.log_json("Consult Template", consult_template)

    # Load manifest
    manifest = _load_manifest(schedule_id, base_dir)
    profile = manifest.get("profile", {})
    guidance = manifest.get("schedule_guidance", {})

    # ── STEP 2: PROFILE & GUIDANCE ──
    logger.step(2, "PROFILE & GUIDANCE LOADED")
    logger.log_json("Profile", profile)
    logger.log_json("Schedule Guidance", guidance)

    # Validate DB exists
    db_path = _get_db_path(schedule_id, base_dir)
    if not db_path.exists():
        logger.log(f"ERROR: Database not found at {db_path}")
        logger.finish()
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
    code_rows = [r for r in all_rows if r["code"] not in {"ACC", "Code"}]

    # ── STEP 3: SCHEDULE CODES FROM DB ──
    logger.step(3, "SCHEDULE CODES LOADED FROM DB")
    logger.log(f"Total rows in DB: {len(all_rows)}")
    logger.log(f"After filtering header artifacts: {len(code_rows)} codes")
    logger.log("")
    logger.log_table(
        "All schedule codes",
        [{"code": r["code"], "description": r["description"][:80], "fee_excl": r.get("fee_excl_gst", "")} for r in code_rows],
        ["code", "description", "fee_excl"],
    )

    # Build query text
    template_text = ""
    if consult_template:
        template_text = " ".join(f"{k} {v}" for k, v in consult_template.items())
    query_text = f"{consult_text or ''} {template_text}".strip()

    if not query_text:
        logger.log("ERROR: No consultation text provided")
        logger.finish()
        conn.close()
        return {
            "error": "No consultation text provided",
            "schedule_id": schedule_id,
            "recommendations": [],
        }

    logger.log(f"Combined query_text ({len(query_text)} chars):")
    logger.log(f"  {query_text[:500]}")
    logger.log("")

    # ── STEP 4: CONSULT FACT EXTRACTION ──
    logger.step(4, "CONSULT FACT EXTRACTION")
    consult_facts = _extract_consult_facts(consult_text or "", consult_template, profile, logger=logger)

    # ── STEP 5: LLM CODE SELECTION (WIDE NET) ──
    logger.step(5, "LLM CODE SELECTION (WIDE NET)")
    try:
        llm_selections = _llm_select_codes(
            query_text=query_text,
            consult_facts=consult_facts,
            code_rows=code_rows,
            profile=profile,
            guidance=guidance,
            min_confidence=min_confidence,
            logger=logger,
        )
    except RuntimeError as e:
        logger.log(f"ERROR: {e}")
        logger.finish()
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
    logger.log(f"Mapped to DB rows with fees: {len(recs)} codes (capped at top_n={top_n})")
    logger.log("")

    # ── STEP 6: REASONING MODEL ADJUDICATION ──
    logger.step(6, "REASONING MODEL ADJUDICATION (NARROWING)")
    recs = _final_reasoning_review(query_text, consult_facts, recs, profile, guidance, logger=logger)

    # ── STEP 7: PRICING RULES ──
    logger.step(7, "PRICING RULES")
    recs = _apply_pricing_rules(recs, query_text, profile, logger=logger)

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

    logger.log(f"Estimated total (excl GST): ${total_excl:.2f}")
    logger.log(f"Estimated total (incl GST): ${total_incl:.2f}")
    logger.log("")

    # ── STEP 8: SEMANTIC EVIDENCE ──
    logger.step(8, "SEMANTIC EVIDENCE CHUNKS")
    evidence_chunks = _fetch_semantic_chunks(cur, schedule_id, query_text, top_n=3, logger=logger)

    conn.close()

    # ── STEP 9: FINAL OUTPUT ──
    logger.step(9, "FINAL OUTPUT")

    response = {
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
        "job_id": logger.job_id,
        "job_log": str(logger.log_path),
    }

    logger.log(f"Recommendations: {len(recs)} codes")
    logger.log_table(
        "Final recommendation table",
        [
            {
                "code": r["code"],
                "description": r["description"][:60],
                "fee_excl": r.get("fee_excl_gst", ""),
                "multiplier": r.get("pricing_multiplier", ""),
                "line_total": r.get("line_total_excl_gst", ""),
                "confidence": r.get("confidence", ""),
            }
            for r in recs
        ],
        ["code", "description", "fee_excl", "multiplier", "line_total", "confidence"],
    )
    logger.log_json("Full JSON response", response, max_length=5000)

    # Save job data for frontend review
    input_data = {
        "consult_text": consult_text,
        "consult_template": consult_template,
        "schedule_id": schedule_id,
        "top_n": top_n,
        "gst_mode": gst_mode,
        "min_confidence": min_confidence,
    }
    logger.save_job_data(input_data, response)
    logger.finish()

    return response
