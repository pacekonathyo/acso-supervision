"""
Analyze a COMPLETED expert-rating sheet to validate AcSO against human judgement.

Reads expert_rating_sheet.csv (rater columns filled with 1 = suitable, 0 = not)
and rating_key.csv, then computes:

  * Expert precision@k : of AcSO's top-k recommendations, the fraction the expert
    consensus (majority vote) judged suitable -- the headline real-world number.
  * Top-k vs distractor suitability : sanity check that AcSO's picks are rated
    suitable far more often than random distractors.
  * Inter-rater agreement : Cohen's kappa (2 raters) or Fleiss' kappa (>2).
  * Rank correlation : Spearman between AcSO rank and mean expert rating.

Run:  python analyze_ratings.py
(For a dry run before you have real ratings, use:  python analyze_ratings.py --mock)
"""
import os, csv, sys
import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))


def load():
    sheet = list(csv.DictReader(open(os.path.join(HERE, "expert_rating_sheet.csv"))))
    key = {r["pair_code"]: r for r in csv.DictReader(open(os.path.join(HERE, "rating_key.csv")))}
    raters = [c for c in sheet[0].keys()
              if c not in ("pair_code", "student_interests", "candidate_supervisor_expertise")]
    return sheet, key, raters


def cohen_kappa(a, b):
    a, b = np.array(a), np.array(b)
    po = np.mean(a == b)
    cats = sorted(set(a) | set(b))
    pe = sum((np.mean(a == c) * np.mean(b == c)) for c in cats)
    return (po - pe) / (1 - pe) if pe != 1 else 1.0


def fleiss_kappa(matrix):           # matrix: items x categories counts
    matrix = np.array(matrix, float)
    n = matrix.sum(1)[0]
    p = matrix.sum(0) / matrix.sum()
    P = (np.square(matrix).sum(1) - n) / (n * (n - 1))
    Pbar = P.mean(); Pe = np.square(p).sum()
    return (Pbar - Pe) / (1 - Pe) if Pe != 1 else 1.0


def mock_fill(sheet, key, raters):  # ONLY for testing the pipeline; NOT evidence
    rng = np.random.default_rng(0)
    for row in sheet:
        base = 0.85 if key[row["pair_code"]]["acso_topk"] == "1" else 0.20
        for r in raters:
            row[r] = str(int(rng.random() < base + rng.normal(0, 0.05)))


def main():
    sheet, key, raters = load()
    if "--mock" in sys.argv:
        mock_fill(sheet, key, raters)
        print("** MOCK ratings (random, for pipeline testing only -- NOT real evidence) **\n")

    # parse ratings
    R = {}
    for row in sheet:
        vals = []
        for r in raters:
            v = row[r].strip()
            if v == "":
                continue
            vals.append(int(float(v) >= (3 if max(_safe_int(row[x]) for x in raters) > 1 else 1)))
        R[row["pair_code"]] = vals
    consensus = {pc: (1 if sum(v) > len(v) / 2 else 0) for pc, v in R.items() if v}

    # expert precision@k over AcSO top-k, per student
    by_student = {}
    for pc, k in key.items():
        if k["acso_topk"] == "1" and pc in consensus:
            by_student.setdefault(k["student_id"], []).append(consensus[pc])
    prec = np.mean([np.mean(v) for v in by_student.values()]) if by_student else float("nan")

    # top-k vs distractor suitability
    topk = [consensus[pc] for pc, k in key.items() if k["acso_topk"] == "1" and pc in consensus]
    dist = [consensus[pc] for pc, k in key.items() if k["acso_topk"] == "0" and pc in consensus]

    # inter-rater agreement
    common = [pc for pc in R if len(R[pc]) == len(raters) and len(raters) >= 2]
    if len(raters) == 2 and common:
        agree = cohen_kappa([R[pc][0] for pc in common], [R[pc][1] for pc in common])
        agree_name = "Cohen's kappa"
    elif len(raters) > 2 and common:
        mat = [[sum(1 for x in R[pc] if x == c) for c in (0, 1)] for pc in common]
        agree = fleiss_kappa(mat); agree_name = "Fleiss' kappa"
    else:
        agree, agree_name = float("nan"), "kappa"

    # rank correlation (AcSO rank vs mean expert rating)
    ranks, means = [], []
    for pc, k in key.items():
        if k["acso_rank"] not in ("", None) and pc in R and R[pc]:
            ranks.append(int(k["acso_rank"])); means.append(np.mean(R[pc]))
    rho, pval = spearmanr(ranks, means) if len(ranks) > 2 else (float("nan"), float("nan"))

    print("=== Expert validation results ===")
    print(f"Raters: {len(raters)}  |  rated pairs: {len(consensus)}  "
          f"|  students: {len(by_student)}")
    print(f"Expert precision@k (AcSO top-k judged suitable) : {prec:.3f}")
    print(f"AcSO top-k suitable rate   : {np.mean(topk):.3f}  (n={len(topk)})")
    print(f"Distractor suitable rate   : {np.mean(dist):.3f}  (n={len(dist)})")
    print(f"Inter-rater agreement ({agree_name}) : {agree:.3f}")
    print(f"Spearman(AcSO rank, mean expert rating) : rho={rho:.3f}, p={pval:.3g}")
    print("\nFill these into Table (Expert Validation) in the manuscript.")


def _safe_int(x):
    try: return int(float(x))
    except Exception: return 0


if __name__ == "__main__":
    main()
