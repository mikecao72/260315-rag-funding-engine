# Stage 3 Walkthrough: Keyword Scoring Against the Code Database

This document traces a concrete example through the tokenization and keyword
matching logic in `recommend.py`, so you can see exactly how raw clinical text
becomes match scores against schedule codes.

---

## 1. Starting Point — Raw Input

A user submits a SOAP note via the API:

```
"18-year-old male presents to urgent care clinic following a workplace
accident. Sustained a laceration to the left forearm, approximately 3 cm,
cleaned and closed with 4 sutures. No fracture suspected on X-ray.
ACC claim lodged."
```

If a `consult_template` was also provided (e.g. `{"injury_type": "laceration",
"body_site": "forearm"}`), its values are appended. The combined string becomes
`query_text`:

```
query_text = "18-year-old male presents to urgent care clinic following a
workplace accident. Sustained a laceration to the left forearm, approximately
3 cm, cleaned and closed with 4 sutures. No fracture suspected on X-ray.
ACC claim lodged. injury_type laceration body_site forearm"
```

---

## 2. Tokenization — How `_tokens()` Works

The regex pattern is:

```python
TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9\-]{1,}")
```

**Rules:**
- Must start with a letter (a-z, A-Z)
- Followed by 1+ letters, digits, or hyphens
- After regex extraction, tokens shorter than 3 chars are discarded
- All tokens are lowercased
- Result is a **set** (no duplicates)

### Tokenizing the query_text

```
Raw text:  "18-year-old male presents to urgent care clinic following a
            workplace accident. Sustained a laceration to the left forearm,
            approximately 3 cm, cleaned and closed with 4 sutures. No
            fracture suspected on X-ray. ACC claim lodged. injury_type
            laceration body_site forearm"

Regex matches (raw):
  "year-old", "male", "presents", "to", "urgent", "care", "clinic",
  "following", "a", "workplace", "accident", "Sustained", "a",
  "laceration", "to", "the", "left", "forearm", "approximately", "cm",
  "cleaned", "and", "closed", "with", "sutures", "No", "fracture",
  "suspected", "on", "X-ray", "ACC", "claim", "lodged", "injury_type",
  "laceration", "body_site", "forearm"

After discarding len <= 2 ("to", "a", "cm", "No", "on"):
  "year-old", "male", "presents", "urgent", "care", "clinic",
  "following", "workplace", "accident", "Sustained", "laceration",
  "the", "left", "forearm", "approximately", "cleaned", "and",
  "closed", "with", "sutures", "fracture", "suspected", "X-ray",
  "ACC", "claim", "lodged", "injury_type", "body_site"

After lowercasing + dedup (it's a set):
```

```
query_tokens = {
    "year-old", "male", "presents", "urgent", "care", "clinic",
    "following", "workplace", "accident", "sustained", "laceration",
    "the", "left", "forearm", "approximately", "cleaned", "and",
    "closed", "with", "sutures", "fracture", "suspected", "x-ray",
    "acc", "claim", "lodged", "injury_type", "body_site"
}

Count: 27 tokens
```

---

## 3. Tokenizing Schedule Code Descriptions

Every row in `schedule_codes` gets the same treatment. Here are several real
examples from the ACC1520 database:

### Code MW1 — "Treatment of laceration, wound closure (less than 2cm long)"

```
Regex matches → "Treatment", "of", "laceration", "wound", "closure",
                "less", "than", "2cm", "long"

After len > 2 filter + lowercase:
desc_tokens = {"treatment", "laceration", "wound", "closure", "less", "than", "2cm", "long"}
```

### Code MW2 — "Treatment of laceration, wound closure (2-7cm): suturing, skin adhesive or similar"

```
desc_tokens = {"treatment", "laceration", "wound", "closure", "2-7cm",
               "suturing", "skin", "adhesive", "similar"}
```

### Code MF9 — "Fractured distal radius and ulna: cast immobilisation not requiring reduction"

```
desc_tokens = {"fractured", "distal", "radius", "and", "ulna", "cast",
               "immobilisation", "not", "requiring", "reduction"}
```

### Code GP1 — "GP consultation - if the client is 14 years old or over (also known as CON)"

```
desc_tokens = {"consultation", "the", "client", "years", "old", "over",
               "also", "known", "con"}
```

### Code MB5 — "Significant burns or abrasions (not including fractures) at multiple sites (greater than 4cm2): necessary wound cleaning, preparation, and dressing"

```
desc_tokens = {"significant", "burns", "abrasions", "not", "including",
               "fractures", "multiple", "sites", "greater", "than", "4cm",
               "necessary", "wound", "cleaning", "preparation", "and",
               "dressing"}
```

---

## 4. Scoring — How `_keyword_similarity()` Works

```python
def _keyword_similarity(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / max(1, len(a))
```

Where:
- `a` = query_tokens (from the consult text) — 27 tokens in our example
- `b` = desc_tokens (from a single schedule code's description)
- `a & b` = the set intersection (tokens that appear in BOTH)
- Score = count of shared tokens / count of query tokens

**This means:** "what fraction of the words in my clinical note also appear in
this code's description?" Higher = more of the query is covered by this code.

### Worked Examples

#### MW1 vs query_tokens

```
query_tokens (27):  {year-old, male, presents, urgent, care, clinic,
                     following, workplace, accident, sustained, laceration,
                     the, left, forearm, approximately, cleaned, closed,
                     with, sutures, fracture, suspected, x-ray, acc,
                     claim, lodged, injury_type, body_site}

desc_tokens (8):    {treatment, laceration, wound, closure, less, than,
                     2cm, long}

intersection:       {laceration}                          → 1 match

score = 1 / 27 = 0.0370
```

#### MW2 vs query_tokens

```
desc_tokens (9):    {treatment, laceration, wound, closure, 2-7cm,
                     suturing, skin, adhesive, similar}

intersection:       {laceration}                          → 1 match

score = 1 / 27 = 0.0370
```

#### MF9 vs query_tokens

```
desc_tokens (10):   {fractured, distal, radius, and, ulna, cast,
                     immobilisation, not, requiring, reduction}

intersection:       {and}                                 → 1 match

score = 1 / 27 = 0.0370
```

Note: "fracture" (query) != "fractured" (description) — no stemming is
applied. This is a limitation of pure keyword matching; the heuristic boost
layer (Stage 4) compensates for this.

#### GP1 vs query_tokens

```
desc_tokens (9):    {consultation, the, client, years, old, over,
                     also, known, con}

intersection:       {the}                                 → 1 match

score = 1 / 27 = 0.0370
```

#### MB5 vs query_tokens

```
desc_tokens (17):   {significant, burns, abrasions, not, including,
                     fractures, multiple, sites, greater, than, 4cm,
                     necessary, wound, cleaning, preparation, and,
                     dressing}

intersection:       {and, cleaned→NO (cleaned != cleaning)}
                    Actually: {and}                       → 1 match

score = 1 / 27 = 0.0370
```

---

## 5. Why Most Raw Keyword Scores Look Similar (and That's OK)

You'll notice many codes score identically at the keyword stage. This is
intentional — keyword matching is a **coarse first pass** designed to
cheaply eliminate clearly irrelevant codes (score near zero).

The real differentiation happens in **Stage 4 (Heuristic Boosts)** which
adds/subtracts from these base scores using clinical pattern matching:

```
Code    Keyword Score    Heuristic Boost    Final Score
----    -------------    ---------------    -----------
GP1     0.0370           +0.90 (age>=14,    1.2370
                          urgent care,
                          ACC claim)

MW1     0.0370           +1.20 (laceration  1.2370
                          + cm < 2)
                          BUT 3cm → no
                          boost applies
                          → +0.00           0.0370

MW2     0.0370           +0.50 (laceration  0.5370
                          + sutures, cm
                          in 2-7 range →
                          +1.20)            1.2370

MF9     0.0370           -1.60 (fracture    -1.5630 → ELIMINATED
                          negated: "No                 (score <= 0.05)
                          fracture
                          suspected")

MB5     0.0370           +0.00 (no multi-   0.0370 → ELIMINATED
                          site abrasion               (score <= 0.05,
                          context)                     borderline)
```

**The keyword score gets you into the game. The heuristic boost picks the
winner.**

---

## 6. Pre-Filters (Applied Before Scoring Even Starts)

Before any code reaches the tokenization/scoring step, three filters run:

### Filter 1: Header artifact removal
```
Code "ACC" with description "Information sheet This information sheet..."
→ SKIPPED (code in {"ACC", "Code"})
```

### Filter 2: CSC exclusion
If the input contains "no community services card":
```
Codes NCCS, NCCD, NUCS, NUCD, GNCS, GNCD, GPCS, GPCD, NNCS, NNCD, PMCS
→ ALL SKIPPED
```
Our example text does NOT contain this phrase, so CSC codes remain as
candidates.

### Filter 3: Amputation guard
```
Codes MW4, MW4A, MW4B, etc.
→ SKIPPED unless text contains positive amputation context
   ("amputation", "amputated", etc. WITHOUT negation)

Our text has no amputation mention → MW4* codes SKIPPED
```

---

## 7. Visual Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                     RAW CONSULT INPUT                           │
│  "18-year-old male presents to urgent care clinic following     │
│   a workplace accident. Sustained a laceration to the left     │
│   forearm, approximately 3 cm, cleaned and closed with 4       │
│   sutures. No fracture suspected on X-ray. ACC claim lodged."  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    TOKENIZE (regex + lowercase + dedup)         │
│                                                                 │
│  {year-old, male, presents, urgent, care, clinic, following,   │
│   workplace, accident, sustained, laceration, the, left,       │
│   forearm, approximately, cleaned, closed, with, sutures,      │
│   fracture, suspected, x-ray, acc, claim, lodged, ...}         │
│                                                                 │
│  → 27 unique tokens                                            │
└────────────────────────────┬────────────────────────────────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
┌──────────────────┐ ┌──────────────┐ ┌──────────────────┐
│ schedule_codes   │ │ schedule_    │ │ schedule_codes   │
│ row: MW2         │ │ codes row:   │ │ row: MF9         │
│                  │ │ GP1          │ │                  │
│ desc_tokens:     │ │              │ │ desc_tokens:     │
│ {treatment,      │ │ desc_tokens: │ │ {fractured,      │
│  laceration,     │ │ {consult-    │ │  distal, radius, │
│  wound, closure, │ │  ation, the, │ │  and, ulna,      │
│  2-7cm, sutur-   │ │  client,     │ │  cast, immobil-  │
│  ing, skin,      │ │  years, old, │ │  isation, not,   │
│  adhesive,       │ │  over, also, │ │  requiring,      │
│  similar}        │ │  known, con} │ │  reduction}      │
│                  │ │              │ │                  │
│ ∩ = {laceration} │ │ ∩ = {the}    │ │ ∩ = {and}        │
│ score = 1/27     │ │ score = 1/27 │ │ score = 1/27     │
│ = 0.037          │ │ = 0.037      │ │ = 0.037          │
│                  │ │              │ │                  │
│ + heuristic:     │ │ + heuristic: │ │ + heuristic:     │
│ +1.20 (laceratn  │ │ +0.90 (age   │ │ -1.60 (fracture  │
│  + 3cm in 2-7)   │ │  >=14, no    │ │  negated!)       │
│                  │ │  CSC, urgent │ │                  │
│ FINAL: 1.237     │ │  care, ACC)  │ │ FINAL: -1.563    │
│ ✓ KEPT           │ │              │ │ ✗ ELIMINATED     │
│                  │ │ FINAL: 1.237 │ │ (score <= 0.05)  │
└──────────────────┘ │ ✓ KEPT       │ └──────────────────┘
                     └──────────────┘
```

---

## 8. Key Takeaways

| Aspect | Detail |
|--------|--------|
| **Tokenizer** | Simple regex, no stemming, no stop-word removal. "fracture" and "fractured" are different tokens. |
| **Score formula** | `shared_tokens / query_token_count` — measures what % of your input matches a code description. |
| **Score range** | 0.0 to 1.0 at the keyword stage. Most real scores are low (0.03–0.15) because clinical notes contain many words that aren't in code descriptions. |
| **Why it works** | Keyword scoring is just the first filter. The heuristic boost layer (Stage 4) adds clinical intelligence, and the optional LLM adjudication (Stage 7) adds reasoning. |
| **Limitation** | No stemming means "laceration" matches but "lacerations" wouldn't match "laceration". No synonym expansion ("cut" won't match "laceration"). These are accepted trade-offs for speed and predictability. |
