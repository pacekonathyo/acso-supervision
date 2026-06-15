"""
Build a BLINDED expert-rating sheet for validating AcSO recommendations.

For a sample of students, it takes AcSO's top-k recommended supervisors and a few
random "distractor" supervisors, shuffles them, hides the AcSO rank, and writes a
CSV that domain experts (e.g., programme coordinators / senior faculty) fill in:
for each (student, supervisor) pair they mark whether the supervisor is a suitable
thesis supervisor for that student.

By default it runs on the synthetic cohort so the pipeline is runnable out of the
box. TO PRODUCE REAL EVIDENCE, replace the synthetic cohort with your real faculty
and representative student profiles (see README), then have >= 2 experts rate the
sheet independently and run analyze_ratings.py.

Outputs:
  expert_rating_sheet.csv   <- give this to each expert (one rater column each)
  rating_key.csv            <- KEY (AcSO rank etc.); keep hidden from raters
"""
import os, csv, random, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import evaluate as ev

HERE = os.path.dirname(os.path.abspath(__file__))
random.seed(7)

N_STUDENTS = 15        # students to sample for the pilot
K = 5                  # AcSO top-k shown
N_DISTRACTORS = 2      # random non-top supervisors added per student (controls)
RATERS = ["expert_1", "expert_2"]   # add more columns as needed


def labels(topics):
    return "; ".join(ev.topic_label(t) for t in topics)


sample = random.sample(ev.students, N_STUDENTS)
sheet_rows, key_rows = [], []
pair_id = 0
for stu in sample:
    ranked = ev.rank_ontology(stu, use_rules=True, use_quality=True)[:K]
    topset = set(ranked)
    distractors = [s["id"] for s in ev.supervisors if s["id"] not in topset]
    random.shuffle(distractors)
    shown = ranked + distractors[:N_DISTRACTORS]
    random.shuffle(shown)                         # blind: hide AcSO order
    sup_by_id = {s["id"]: s for s in ev.supervisors}
    for sid in shown:
        pair_id += 1
        code = f"P{pair_id:03d}"
        sheet_rows.append({
            "pair_code": code,
            "student_interests": labels(stu["interests"]),
            "candidate_supervisor_expertise": labels(sup_by_id[sid]["expertise"]),
            **{r: "" for r in RATERS},   # experts fill: 1 = suitable, 0 = not suitable
        })
        key_rows.append({
            "pair_code": code, "student_id": stu["id"], "supervisor_id": sid,
            "acso_topk": int(sid in topset),
            "acso_rank": (ranked.index(sid) + 1) if sid in topset else "",
        })

with open(os.path.join(HERE, "expert_rating_sheet.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["pair_code", "student_interests",
                                      "candidate_supervisor_expertise"] + RATERS)
    w.writeheader(); w.writerows(sheet_rows)

with open(os.path.join(HERE, "rating_key.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["pair_code", "student_id", "supervisor_id",
                                      "acso_topk", "acso_rank"])
    w.writeheader(); w.writerows(key_rows)

print(f"Wrote {len(sheet_rows)} pairs across {N_STUDENTS} students "
      f"({K} AcSO + {N_DISTRACTORS} distractors each).")
print("Raters fill 1 (suitable) / 0 (not suitable) in:", ", ".join(RATERS))
print("Files: expert_rating_sheet.csv  (to raters)  |  rating_key.csv  (keep hidden)")
