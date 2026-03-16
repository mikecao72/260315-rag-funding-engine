#!/usr/bin/env python3
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from rag_funding_engine.pipeline.recommend import recommend_codes  # noqa: E402

SCHEDULE_ID = "acc1520-medical"
CODE_RE = re.compile(r"^[A-Za-z0-9\-]{2,}$")

CASES = [
    {
        "name": "minor_wrist_followup",
        "consult_text": "Follow-up consult after minor wrist fracture. Re-assessment and pain management advice. No surgery.",
    },
    {
        "name": "complex_knee_initial",
        "consult_text": "Initial consult for complex knee injury with instability and significant swelling after sports incident.",
    },
    {
        "name": "post_op_shoulder_review",
        "consult_text": "Post-operative shoulder review. Wound check, progress review, and treatment plan adjustment.",
    },
    {
        "name": "spine_persistent_pain",
        "consult_text": "Persistent lower-back pain after workplace injury requiring specialist reassessment and management planning.",
    },
    {
        "name": "hand_laceration_followup",
        "consult_text": "Follow-up for hand laceration healing progress with function review and return-to-work advice.",
    },
]


def sanity_checks(result: dict) -> list[str]:
    errors: list[str] = []
    recs = result.get("recommendations", [])

    if not recs:
        errors.append("no recommendations returned")
        return errors

    top = recs[0]
    top_code = str(top.get("code", ""))
    if not CODE_RE.match(top_code):
        errors.append(f"top code format invalid: {top_code!r}")

    scores = [float(r.get("match_score", 0.0) or 0.0) for r in recs]
    if any(scores[i] < scores[i + 1] for i in range(len(scores) - 1)):
        errors.append("match_score order is not descending")

    total_incl = float(result.get("estimated_total_incl_gst", 0.0) or 0.0)
    if total_incl <= 0:
        errors.append("estimated_total_incl_gst is not positive")

    evidence = result.get("evidence_chunks", [])
    if len(evidence) == 0:
        errors.append("no evidence chunks returned")

    return errors


def main() -> int:
    from rag_funding_engine.pipeline.recommend import _get_db_path
    db_path = _get_db_path(SCHEDULE_ID, ROOT / "data" / "processed")
    
    if not db_path.exists():
        print(f"ERROR: DB not found at {db_path}")
        print("Run ingest first (e.g. POST /ingest) then re-run this smoke eval.")
        return 2

    print("RAG Funding Engine smoke eval")
    print(f"Schedule: {SCHEDULE_ID}")
    print(f"DB: {db_path}")
    print(f"Cases: {len(CASES)}")
    print("-" * 72)

    failures = 0
    for idx, case in enumerate(CASES, start=1):
        result = recommend_codes(
            schedule_id=SCHEDULE_ID,
            consult_text=case["consult_text"],
            consult_template=None,
            top_n=5,
        )
        recs = result.get("recommendations", [])
        top = recs[0] if recs else {}
        code = top.get("code", "<none>")
        score = top.get("match_score", 0.0)
        total = result.get("estimated_total_incl_gst", 0.0)

        print(f"[{idx}] {case['name']}")
        print(f"    top_code={code}  score={score}  total_incl_gst={total}")

        errors = sanity_checks(result)
        if errors:
            failures += 1
            print("    sanity=FAIL")
            for err in errors:
                print(f"      - {err}")
        else:
            print("    sanity=PASS")

    print("-" * 72)
    if failures:
        print(f"RESULT: FAIL ({failures}/{len(CASES)} cases failed sanity checks)")
        return 1

    print("RESULT: PASS (all smoke cases passed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
