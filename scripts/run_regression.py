from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from rag_funding_engine.pipeline.recommend import recommend_codes


def approx_equal(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    cases_path = root / "tests" / "regression_cases.json"

    cases = json.loads(cases_path.read_text())
    failures = 0

    for case in cases:
        cid = case["id"]
        consult_text = case["input"]["consult_text"]
        schedule_id = case["input"].get("schedule_id", "acc1520-medical")
        top_n = case["input"].get("top_n", 5)
        exp = case["expected"]

        res = recommend_codes(
            schedule_id=schedule_id,
            consult_text=consult_text,
            consult_template=None,
            top_n=top_n,
        )
        
        # Handle error case (schedule not found)
        if "error" in res:
            failures += 1
            print(f"[FAIL] {cid}")
            print(f"  - {res['error']}")
            continue
        
        recs = res["recommendations"]
        codes = [r["code"] for r in recs]
        total = float(res["estimated_total_excl_gst"])

        errors = []

        for c in exp.get("must_include_codes", []):
            if c not in codes:
                errors.append(f"missing expected code: {c}")

        top_code = exp.get("top_code")
        if top_code and (not codes or codes[0] != top_code):
            errors.append(f"top_code expected {top_code}, got {codes[0] if codes else 'none'}")

        expected_total = exp.get("expected_total_excl_gst")
        if expected_total is not None:
            tol = float(exp.get("total_tolerance", 0.01))
            if not approx_equal(total, float(expected_total), tol):
                errors.append(f"total mismatch expected {expected_total}, got {total}")

        if errors:
            failures += 1
            print(f"[FAIL] {cid}")
            for e in errors:
                print(f"  - {e}")
            print(f"  output codes={codes} total={total}")
        else:
            print(f"[PASS] {cid} codes={codes} total={total}")

    print(f"\nSummary: {len(cases)-failures} passed / {len(cases)} total")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
