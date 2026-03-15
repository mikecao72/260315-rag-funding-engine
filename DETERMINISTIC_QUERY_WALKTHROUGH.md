# Deterministic Query Walkthrough (Textarea → SQL → Ranked Candidates)

Generated: 2026-03-14T10:46:17.450501

## 1) Input from textarea

```text
17-year-old male high school student. Dependent child of a Community Services Card holder.
Presented to urgent care clinic. Seen by nurse and GP together.
Multiple abrasions: left forearm 9 cm x 5 cm, right knee 7 cm x 6 cm, right lateral hip/flank 8 cm x 4 cm.
Embedded dirt/gravel. No deep laceration requiring suturing. No fracture suspected clinically.
Active anterior epistaxis from right nostril. Persistent epistaxis treated with nasal packing.
```

## 2) SQL issued by the pipeline

```sql
SELECT code, description, fee_excl_gst, fee_incl_gst, page
FROM acc_codes
WHERE schedule_version = ?
```

Bind params: `schedule_version = "ACC1520-v2"`

Rows returned from DB (before Python scoring): **73**

## 3) Deterministic Python scoring inputs

- `query_text` length: 451 chars
- `query_tokens` count: 48
- sample `query_tokens` (first 40 sorted): `['abrasions', 'active', 'and', 'anterior', 'card', 'care', 'child', 'clinic', 'clinically', 'community', 'deep', 'dependent', 'dirt', 'embedded', 'epistaxis', 'flank', 'forearm', 'fracture', 'from', 'gravel', 'high', 'hip', 'holder', 'knee', 'laceration', 'lateral', 'left', 'male', 'multiple', 'nasal', 'nostril', 'nurse', 'packing', 'persistent', 'presented', 'requiring', 'right', 'school', 'seen', 'services']`

## 4) Candidate scoring formula

For each code row:

`combined_score = keyword_similarity(query_tokens, desc_tokens) + heuristic_boost(query_text, code, description)`

Filter applied:

- Drop if `combined_score <= 0.05`
- Drop `ACC`/`Code` pseudo rows
- Optional CSC exclusion if input explicitly says no CSC

Candidates surviving score filter: **8**

## 5) Top scored candidates before family dedupe

| code | keyword_score | heuristic_boost | combined_score |
|---|---:|---:|---:|
| GNCD | 0.1042 | 3.0 | 3.1042 |
| MB5 | 0.0625 | 2.6 | 2.6625 |
| MM8 | 0.0833 | 2.2 | 2.2833 |
| MM5 | 0.0833 | 1.0 | 1.0833 |
| NCCD | 0.1042 | 0.0 | 0.1042 |
| NCCS | 0.1042 | 0.0 | 0.1042 |
| PMCD | 0.0833 | 0.0 | 0.0833 |
| PMCS | 0.0833 | 0.0 | 0.0833 |

## 6) Prefix-family dedupe + top_n(8) selection

Rule: if same alpha prefix already seen and score < 0.95, suppress duplicate family item.

| selected_order | code | combined_score |
|---:|---|---:|
| 1 | GNCD | 3.1042 |
| 2 | MB5 | 2.6625 |
| 3 | MM8 | 2.2833 |
| 4 | MM5 | 1.0833 |
| 5 | NCCD | 0.1042 |
| 6 | NCCS | 0.1042 |
| 7 | PMCD | 0.0833 |
| 8 | PMCS | 0.0833 |

## 7) Primary consult reduction (deterministic)

Rule: keep highest scoring consultation family code as primary consult.

Before:

`GNCD, MB5, MM8, MM5, NCCD, NCCS, PMCD, PMCS`

After:

`GNCD, MB5, MM8, MM5`

## 8) Hand-off to later stages

The deterministic shortlist above is then passed to:
1. Final LLM adjudication step (if enabled)
2. Constraint checks
3. Pricing multiplier logic
4. Explainability reason generation
