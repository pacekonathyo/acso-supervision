"""
Build a BLINDED expert-rating sheet using the FIVE REAL faculty of the
Master's of Informatics, UNIMUS, as candidate supervisors.

Candidate pool (expertise mapped to AcSO topic IRIs from the faculty page):
  - Prof. Dr. Edy Winarno        : AI, Network Security (Kriptografi), Computer Vision
  - Dr. Muhammad Munsarif        : AI, Computer Vision
  - Dr. Dhendra Marutho          : AI, NLP
  - Dr. Ahmad Ilham              : AI, Data Mining
  - Asdani Kindarto, PhD         : Software Engineering, Requirements Engineering
                                   (approx. mapping of E-Gov / IT governance)

Student profiles below are REPRESENTATIVE research interests spanning the
faculty's areas (NOT claims about specific real students). For each student,
all five faculty are scored by AcSO's semantic match (Wu-Palmer over the topic
taxonomy); the two highest are the "recommended" set, the rest are controls.
The order shown to raters is shuffled and the AcSO rank is hidden.

NOTE: capacity (R1) and quality re-ranking are omitted here because real
load/completion figures were not collected; this pilot validates the SEMANTIC
suitability of recommendations, which is what experts can judge directly.

Output (overwrites the demo files):
  expert_rating_sheet.csv   -> to each rater (fill expert_1/expert_2/expert_3)
  rating_key.csv            -> hidden key (AcSO rank); do not show raters
"""
import os, csv, random, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import evaluate as ev

HERE = os.path.dirname(os.path.abspath(__file__))
A = ev.ACSO
random.seed(11)

RECOMMENDED_K = 2     # top-2 of the 5 faculty are AcSO's recommended set
RATERS = ["expert_1", "expert_2", "expert_3"]

faculty = [
    ("fac1", "Prof. Dr. Edy Winarno",     ["ArtificialIntelligence", "NetworkSecurity", "ComputerVision"]),
    ("fac2", "Dr. Muhammad Munsarif",     ["ArtificialIntelligence", "ComputerVision"]),
    ("fac3", "Dr. Dhendra Marutho",       ["ArtificialIntelligence", "NaturalLanguageProcessing"]),
    ("fac4", "Dr. Ahmad Ilham",           ["ArtificialIntelligence", "DataMining"]),
    ("fac5", "Asdani Kindarto, PhD",      ["SoftwareEngineering", "RequirementsEngineering"]),
]

students = [
    ("s01", ["NaturalLanguageProcessing", "InformationRetrieval"]),
    ("s02", ["ComputerVision", "DeepLearning"]),
    ("s03", ["DataMining", "Databases"]),
    ("s04", ["NetworkSecurity", "WirelessNetworks"]),
    ("s05", ["SoftwareTesting", "RequirementsEngineering"]),
    ("s06", ["MachineLearning", "ReinforcementLearning"]),
    ("s07", ["SemanticWeb", "Ontologies"]),
    ("s08", ["DeepLearning", "ComputerVision"]),
    ("s09", ["NaturalLanguageProcessing", "MachineLearning"]),
    ("s10", ["RecommenderSystems", "DataMining"]),
    ("s11", ["ComputerVision", "ArtificialIntelligence"]),
    ("s12", ["NaturalLanguageProcessing", "DataMining"]),
]

def iri(code): return A + code
def labels(codes): return "; ".join(ev.topic_label(iri(c)) for c in codes)

def score(stu_codes, fac_codes):
    return max(ev.wu_palmer(iri(si), iri(se)) for si in stu_codes for se in fac_codes)

sheet_rows, key_rows = [], []
pair_id = 0
for sid, interests in students:
    scored = [(fid, name, fcodes, round(score(interests, fcodes), 4))
              for (fid, name, fcodes) in faculty]
    scored.sort(key=lambda x: x[3], reverse=True)            # AcSO ranking
    ranked_ids = [fid for fid, _, _, _ in scored]
    shown = list(scored); random.shuffle(shown)              # blind order
    for (fid, name, fcodes, sc) in shown:
        pair_id += 1
        code = f"P{pair_id:03d}"
        rank = ranked_ids.index(fid) + 1
        sheet_rows.append({
            "pair_code": code,
            "student_interests": labels(interests),
            "candidate_supervisor_expertise": f"{name} \u2014 {labels(fcodes)}",
            **{r: "" for r in RATERS},
        })
        key_rows.append({
            "pair_code": code, "student_id": sid, "supervisor_id": fid,
            "supervisor_name": name, "acso_score": sc,
            "acso_topk": int(rank <= RECOMMENDED_K), "acso_rank": rank,
        })

with open(os.path.join(HERE, "expert_rating_sheet.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["pair_code", "student_interests",
                                      "candidate_supervisor_expertise"] + RATERS)
    w.writeheader(); w.writerows(sheet_rows)

with open(os.path.join(HERE, "rating_key.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["pair_code", "student_id", "supervisor_id",
                                      "supervisor_name", "acso_score",
                                      "acso_topk", "acso_rank"])
    w.writeheader(); w.writerows(key_rows)

print(f"Wrote {len(sheet_rows)} pairs = {len(students)} students x {len(faculty)} real faculty.")
print(f"Recommended set = AcSO top-{RECOMMENDED_K} per student; the rest are controls.")
print("Raters fill 1 (suitable) / 0 (not suitable) in:", ", ".join(RATERS))
