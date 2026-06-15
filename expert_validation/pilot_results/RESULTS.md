# Expert-validation pilot — results

Raw data: `expert_rating_sheet_completed.csv` (3 independent faculty raters,
columns expert_1/2/3, 1 = suitable / 0 = not) and `rating_key.csv` (hidden key).

Setup: 5 real candidate faculty, 50 representative student-interest profiles,
250 blinded (profile, candidate) pairs. Independently recomputed from the raw
ratings (majority-vote consensus):

| Measure | Value |
|---|---|
| Inter-rater agreement (Fleiss' kappa) | 0.890 |
| Expert precision@1 (top choice suitable) | 66.0% |
| Expert precision@2 (top-2 suitable) | 50.0% |
| Lower-ranked control suitable rate | 16.7% |
| Spearman (AcSO rank vs mean rating) | -0.457 (p = 2.78e-14) |

Suitable rate by AcSO rank: r1 66% · r2 34% · r3 22% · r4 28% · r5 0%.
Rater agreement: 232/250 (93%) unanimous; pairwise identical ~95%.

Reproduce: `python ../analyze_ratings.py` after copying the completed sheet to
`../expert_rating_sheet.csv` and the key to `../rating_key.csv`.
