# Expert Validation Toolkit (AcSO)

This kit lets you collect **real human-expert evidence** that AcSO's recommendations
are sensible — the single most valuable addition you can make before submitting.
It turns the validation into a 1–2 hour task. **Nothing here fabricates data; you
collect the ratings from real experts.**

## What it produces

- `expert_rating_sheet.csv` — a blinded sheet of (student interests, candidate
  supervisor expertise) pairs for experts to rate (1 = suitable, 0 = not).
- `rating_key.csv` — the hidden key (which pairs were AcSO's top-k, and their rank).
- `analyze_ratings.py` — computes expert precision@k, top-k vs distractor rate,
  inter-rater agreement (Cohen's/Fleiss' kappa), and Spearman rank correlation.

## Recommended protocol (small but publishable)

1. **Experts:** recruit **2–3** senior faculty / programme coordinators in your
   department. Two is enough to report Cohen's kappa; three lets you report
   Fleiss' kappa and a majority consensus.
2. **Use real profiles (important).** Edit `build_rating_sheet.py` so the cohort
   reflects your reality:
   - Replace the synthetic supervisors with your **real faculty**: map each
     faculty member's expertise to the AcSO topic IRIs (see `ontology/acso.ttl`).
   - Use **10–20 representative student interest profiles** (real or realistic for
     your programme).
   Keep the same data fields; only the values change.
3. **Generate the sheet:** `python build_rating_sheet.py`
   (15 students × (5 AcSO + 2 distractor) ≈ 105 pairs by default — about 20–30
   minutes per expert).
4. **Rate independently.** Give each expert a copy of `expert_rating_sheet.csv`
   with only their own column. They must **not** see `rating_key.csv` or each
   other's ratings. They mark 1 (suitable supervisor for this student) or 0 (not).
5. **Merge** the rater columns back into one `expert_rating_sheet.csv`
   (columns `expert_1`, `expert_2`, ...).
6. **Analyze:** `python analyze_ratings.py`
7. **Report** the printed numbers in the manuscript's *Expert Validation* table
   (the subsection and a placeholder table are already in the .tex).

## Pipeline dry-run (no real data yet)

```bash
python build_rating_sheet.py
python analyze_ratings.py --mock     # random ratings -> ONLY tests the scripts
```
The `--mock` numbers are random and are **not** evidence; they exist solely to
confirm the scripts run before you collect real ratings.

## What to expect / how to read it

- **Expert precision@k** is the headline real-world number: of the supervisors
  AcSO recommended, what fraction did experts judge genuinely suitable.
- **Top-k vs distractor suitable rate** shows AcSO's picks are rated suitable far
  more often than random controls (a sanity check on construct validity).
- **kappa** reports how much the experts agreed (context for the precision number).
- **Spearman** checks whether AcSO's ranking order tracks expert preference.

Report all four honestly, including kappa even if agreement is moderate — reviewers
value the transparency.

---

## Real-faculty pilot (UNIMUS) — ready to use

`build_rating_sheet_real.py` builds the sheet with the **five real faculty** of
the Master's of Informatics (UNIMUS) as candidate supervisors, their expertise
mapped to AcSO topics:

| Faculty | AcSO topics |
|---|---|
| Prof. Dr. Edy Winarno | Artificial Intelligence, Network Security, Computer Vision |
| Dr. Muhammad Munsarif | Artificial Intelligence, Computer Vision |
| Dr. Dhendra Marutho | Artificial Intelligence, Natural Language Processing |
| Dr. Ahmad Ilham | Artificial Intelligence, Data Mining |
| Asdani Kindarto, PhD | Software Engineering, Requirements Engineering* |

\* approximate mapping of E-Gov / IT governance.

It uses 12 representative student interest profiles (spanning the faculty areas;
not claims about specific students). For each student all five faculty are scored
by AcSO's semantic match; the top-2 form the "recommended" set, the rest are
controls. Output = 60 pairs (`expert_rating_sheet.csv`).

```bash
python build_rating_sheet_real.py    # regenerate the real-faculty sheet
# 2-3 faculty fill expert_1 / expert_2 / expert_3 with 1 (suitable) / 0 (not)
python analyze_ratings.py            # prints precision@2, control rate, Fleiss kappa, Spearman
```

Interpretation note: `acso_rank` 1 = best, so a **negative** Spearman rho between
rank and rating means the system agrees with experts (better-ranked = higher
rating). Report it with that sign explained, or report rho against the score.

The headline number to put in the manuscript (Table `tab:expert`) is the
**Expert precision@2** — the fraction of AcSO's two recommended supervisors that
the faculty consensus judged genuinely suitable.
