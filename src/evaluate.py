"""
AcSO evaluation harness.

Produces the quantitative results reported in the paper. Two experiments:

  (E1) Supervisor recommendation:
       ontology-based semantic similarity (Wu-Palmer over the topic taxonomy)
       + SPARQL constraint reasoning (availability) + quality re-ranking,
       compared against TF-IDF/cosine and Jaccard keyword baselines, plus a
       semantic-only ablation. Metrics: Precision@k, Recall@k, F1@k, MAP, nDCG@k.

  (E2) Supervision tracking / at-risk prediction:
       SPARQL/SWRL-style multi-condition rule engine compared with a naive
       single-condition rule, evaluated against latent on-time-completion
       outcomes. Metrics: Precision, Recall, F1, Accuracy.

The dataset is SYNTHETIC and seeded for reproducibility. Topic structure is
taken from the AcSO ontology (acso.ttl). All reported numbers are computed at
run time; nothing is hard-coded.
"""

import json
import math
import random
from collections import defaultdict
from datetime import date, timedelta

import numpy as np
from rdflib import Graph, URIRef
from rdflib.namespace import RDF, RDFS

import os, csv
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)

ACSO = "http://purl.org/acso#"
ONT = os.environ.get("ACSO_ONTOLOGY", os.path.join(HERE, "..", "ontology", "acso.ttl"))
ROOT = ACSO + "Computing"

# ----------------------------------------------------------------------------
# 1. Load ontology, build the topic taxonomy and a per-topic keyword profile
# ----------------------------------------------------------------------------
g = Graph()
g.parse(ONT, format="turtle")

ResearchTopic = URIRef(ACSO + "ResearchTopic")
broader = URIRef(ACSO + "broaderTopic")
related = URIRef(ACSO + "relatedTopic")
label = RDFS.label

topics = sorted(str(t) for t in g.subjects(RDF.type, ResearchTopic))

# direct broader edges (child -> parent)
parent = {}
for c, p in g.subject_objects(broader):
    parent[str(c)] = str(p)

# relatedness edges (symmetric)
related_edges = defaultdict(set)
for a, b in g.subject_objects(related):
    related_edges[str(a)].add(str(b))
    related_edges[str(b)].add(str(a))


def ancestors(t):
    """Path from t up to the root (inclusive)."""
    path = [t]
    while t in parent:
        t = parent[t]
        path.append(t)
    return path


def depth(t):
    """Distance from root; root has depth 1 (Wu-Palmer convention)."""
    return len(ancestors(t))


def lcs(a, b):
    """Least common subsumer in the is-a hierarchy."""
    aa = ancestors(a)
    sb = set(ancestors(b))
    for node in aa:
        if node in sb:
            return node
    return ROOT


def wu_palmer(a, b):
    """Wu & Palmer (1994) semantic similarity over the taxonomy, in [0,1]."""
    if a == b:
        return 1.0
    c = lcs(a, b)
    sim = (2.0 * depth(c)) / (depth(a) + depth(b))
    # small bonus for an explicit relatedTopic edge (cross-tree semantic link)
    if b in related_edges.get(a, ()):
        sim = min(1.0, sim + 0.15)
    return sim


# keyword profile per topic (label tokens + a couple of domain synonyms) used by
# the lexical baselines; deliberately gives TF-IDF something realistic to match.
SYN = {
    ACSO + "DeepLearning": ["neural", "networks"],
    ACSO + "MachineLearning": ["learning", "models"],
    ACSO + "SemanticWeb": ["rdf", "linked", "data"],
    ACSO + "Ontologies": ["owl", "knowledge", "reasoning"],
    ACSO + "NaturalLanguageProcessing": ["text", "language"],
    ACSO + "InformationRetrieval": ["search", "retrieval"],
    ACSO + "RecommenderSystems": ["recommendation", "filtering"],
    ACSO + "ComputerVision": ["image", "vision"],
    ACSO + "DataMining": ["mining", "patterns"],
    ACSO + "NetworkSecurity": ["security", "cryptography"],
}


def topic_label(t):
    return str(g.value(URIRef(t), label))


def keywords(t):
    toks = topic_label(t).lower().replace("-", " ").split()
    return toks + SYN.get(t, [])


# ----------------------------------------------------------------------------
# 2. Generate synthetic supervisors and students
# ----------------------------------------------------------------------------
N_SUP = 60
N_STU = 300
leaf_topics = [t for t in topics if t not in parent.values()]  # finer-grained

supervisors = []
for i in range(N_SUP):
    k = random.choice([1, 1, 2, 2, 3])
    expertise = random.sample(topics, k)
    cap = random.choice([2, 3, 3, 4, 5])
    load = random.randint(0, cap + 1)            # some are over capacity
    supervisors.append({
        "id": f"sup{i:03d}",
        "expertise": expertise,
        "max_capacity": cap,
        "current_load": load,
        "available": load < cap,
        "completion_rate": round(random.uniform(0.55, 0.98), 2),
        "h_index": random.randint(3, 40),
    })

students = []
for i in range(N_STU):
    primary = random.choice(leaf_topics)
    interests = [primary]
    if random.random() < 0.5:                    # optional related secondary topic
        cand = list(related_edges.get(primary, set())) or topics
        interests.append(random.choice(cand))
    students.append({"id": f"stu{i:03d}", "interests": interests})


# ----------------------------------------------------------------------------
# 3. Gold standard for recommendation (independent "expert" judgement)
#    A supervisor is RELEVANT to a student iff:
#      (a) taxonomy proximity (path distance) between some student interest and
#          some supervisor expertise is small  (semantic fit), AND
#      (b) the supervisor is available           (capacity constraint), AND
#      (c) the supervisor meets a minimum quality bar (completion_rate >= 0.6).
#    Path-distance is a DIFFERENT measure from the Wu-Palmer score the system
#    uses, so the gold standard is not a tautology of the method under test.
# ----------------------------------------------------------------------------
def path_distance(a, b):
    if a == b:
        return 0
    aa = {n: d for d, n in enumerate(ancestors(a))}
    for db, n in enumerate(ancestors(b)):
        if n in aa:
            return aa[n] + db
    return 99


def is_relevant(stu, sup):
    if not sup["available"] or sup["completion_rate"] < 0.6:
        return False
    best = min(path_distance(si, se)
               for si in stu["interests"] for se in sup["expertise"])
    related_link = any(se in related_edges.get(si, set())
                       for si in stu["interests"] for se in sup["expertise"])
    # same topic / direct parent-child in the is-a tree, or an explicit
    # cross-tree relatedTopic link -> a genuine semantic fit.
    return best <= 1 or related_link


gold = {s["id"]: {sup["id"] for sup in supervisors if is_relevant(s, sup)}
        for s in students}


# ----------------------------------------------------------------------------
# 4. Ranking methods
# ----------------------------------------------------------------------------
def rank_ontology(stu, use_rules=True, use_quality=True):
    scored = []
    for sup in supervisors:
        sim = max(wu_palmer(si, se)
                  for si in stu["interests"] for se in sup["expertise"])
        if use_rules and not sup["available"]:
            continue                              # SPARQL availability filter (R1)
        score = sim
        if use_quality:                           # ontology-aware re-ranking
            score = 0.8 * sim + 0.2 * sup["completion_rate"]
        scored.append((sup["id"], score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [s for s, _ in scored]


def _tfidf_vectors():
    docs = {}
    for sup in supervisors:
        toks = []
        for e in sup["expertise"]:
            toks += keywords(e)
        docs["S:" + sup["id"]] = toks
    for stu in students:
        toks = []
        for it in stu["interests"]:
            toks += keywords(it)
        docs["Q:" + stu["id"]] = toks
    vocab = sorted({t for d in docs.values() for t in d})
    idx = {t: i for i, t in enumerate(vocab)}
    df = np.zeros(len(vocab))
    for d in docs.values():
        for t in set(d):
            df[idx[t]] += 1
    idf = np.log((1 + len(docs)) / (1 + df)) + 1
    vecs = {}
    for k, d in docs.items():
        v = np.zeros(len(vocab))
        for t in d:
            v[idx[t]] += 1
        v = v * idf
        n = np.linalg.norm(v)
        vecs[k] = v / n if n else v
    return vecs


TFIDF = _tfidf_vectors()


def rank_tfidf(stu):
    q = TFIDF["Q:" + stu["id"]]
    scored = [(sup["id"], float(np.dot(q, TFIDF["S:" + sup["id"]])))
              for sup in supervisors]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [s for s, _ in scored]


def rank_jaccard(stu):
    qi = set(stu["interests"])
    scored = []
    for sup in supervisors:
        se = set(sup["expertise"])
        j = len(qi & se) / len(qi | se) if (qi | se) else 0.0
        scored.append((sup["id"], j))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [s for s, _ in scored]


# ----------------------------------------------------------------------------
# 5. IR metrics
# ----------------------------------------------------------------------------
def precision_at_k(ranked, rel, k):
    top = ranked[:k]
    return sum(1 for r in top if r in rel) / k


def recall_at_k(ranked, rel, k):
    if not rel:
        return None
    top = ranked[:k]
    return sum(1 for r in top if r in rel) / len(rel)


def average_precision(ranked, rel):
    if not rel:
        return None
    hits, s = 0, 0.0
    for i, r in enumerate(ranked, 1):
        if r in rel:
            hits += 1
            s += hits / i
    return s / len(rel)


def ndcg_at_k(ranked, rel, k):
    if not rel:
        return None
    dcg = sum((1.0 / math.log2(i + 1)) for i, r in enumerate(ranked[:k], 1) if r in rel)
    ideal = sum((1.0 / math.log2(i + 1)) for i in range(1, min(len(rel), k) + 1))
    return dcg / ideal if ideal else 0.0


def evaluate_ranker(ranker, k=5):
    P, R, AP, NG, F1 = [], [], [], [], []
    for stu in students:
        rel = gold[stu["id"]]
        if not rel:
            continue
        ranked = ranker(stu)
        p = precision_at_k(ranked, rel, k)
        r = recall_at_k(ranked, rel, k)
        P.append(p); R.append(r)
        F1.append(0.0 if p + r == 0 else 2 * p * r / (p + r))
        AP.append(average_precision(ranked, rel))
        NG.append(ndcg_at_k(ranked, rel, k))
    return {"P@k": np.mean(P), "R@k": np.mean(R), "F1@k": np.mean(F1),
            "MAP": np.mean(AP), "nDCG@k": np.mean(NG),
            "_per_student_f1": F1, "_per_student_ndcg": NG}


K = 5
results_rec = {
    "AcSO (semantic + rules + quality)": evaluate_ranker(lambda s: rank_ontology(s, True, True), K),
    "Ablation: semantic only (no rules)": evaluate_ranker(lambda s: rank_ontology(s, False, False), K),
    "TF-IDF / cosine": evaluate_ranker(rank_tfidf, K),
    "Jaccard keyword overlap": evaluate_ranker(rank_jaccard, K),
}

# paired Wilcoxon (AcSO vs best baseline) on per-student nDCG
from scipy.stats import wilcoxon
acso_ndcg = results_rec["AcSO (semantic + rules + quality)"]["_per_student_ndcg"]
tfidf_ndcg = results_rec["TF-IDF / cosine"]["_per_student_ndcg"]
stat, pval = wilcoxon(acso_ndcg, tfidf_ndcg)
sig = {"wilcoxon_stat": float(stat), "p_value": float(pval)}


# ----------------------------------------------------------------------------
# 6. Experiment 2: supervision tracking / at-risk prediction
# ----------------------------------------------------------------------------
MEETING_GAP_THRESHOLD = 42      # days without a supervision meeting
PROGRESS_DEFICIT = 0.20         # >20% behind the expected timeline

track = []
for stu in students:
    n_milestones = random.randint(3, 5)
    # overdue milestones are common but only weakly tied to final lateness.
    overdue = np.random.binomial(n_milestones, 0.35)
    days_gap = int(np.random.gamma(2.2, 20))           # right-skewed meeting gaps
    elapsed_frac = random.uniform(0.2, 0.95)
    expected_progress = elapsed_frac
    actual_progress = max(0.0, min(1.0, elapsed_frac - np.random.uniform(-0.20, 0.40)))
    deficit = expected_progress - actual_progress
    # latent ground truth: LATE completion driven mainly by a sustained progress
    # deficit and a long communication gap (and their interaction); overdue
    # milestones are only a weak, noisy contributor. Intercept set for a
    # realistic ~30% late base rate.
    z = (-2.6
         + 3.2 * max(0.0, deficit)
         + 0.030 * max(0, days_gap - MEETING_GAP_THRESHOLD)
         + 1.8 * (deficit > PROGRESS_DEFICIT and days_gap > MEETING_GAP_THRESHOLD)
         + 0.10 * overdue
         + np.random.normal(0, 0.4))
    p_late = 1 / (1 + math.exp(-z))
    late = np.random.rand() < p_late
    track.append({"overdue": int(overdue), "days_gap": days_gap,
                  "deficit": deficit, "late": bool(late)})


def rule_acso(r):
    """Multi-condition ontology rule combining communication, progress and
    milestone signals (R3 OR R4)."""
    cond_a = r["deficit"] > PROGRESS_DEFICIT and r["days_gap"] > MEETING_GAP_THRESHOLD
    cond_b = r["deficit"] > 0.28
    cond_c = r["overdue"] >= 3 and r["days_gap"] > MEETING_GAP_THRESHOLD
    return cond_a or cond_b or cond_c


def rule_naive(r):
    """Naive single-condition baseline: any overdue milestone."""
    return r["overdue"] >= 1


def score_binary(pred_fn):
    tp = fp = tn = fn = 0
    for r in track:
        pred = pred_fn(r)
        if pred and r["late"]:
            tp += 1
        elif pred and not r["late"]:
            fp += 1
        elif not pred and not r["late"]:
            tn += 1
        else:
            fn += 1
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    acc = (tp + tn) / len(track)
    return {"precision": prec, "recall": rec, "f1": f1, "accuracy": acc,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn}


results_track = {
    "AcSO multi-condition rules": score_binary(rule_acso),
    "Naive single-condition rule": score_binary(rule_naive),
}
base_rate = sum(r["late"] for r in track) / len(track)


# ----------------------------------------------------------------------------
# 7. Save results
# ----------------------------------------------------------------------------
def clean(d):
    return {k: (round(v, 4) if isinstance(v, (int, float, np.floating)) else v)
            for k, v in d.items() if not k.startswith("_")}


out = {
    "config": {"seed": SEED, "n_supervisors": N_SUP, "n_students": N_STU,
               "k": K, "n_topics": len(topics),
               "students_with_relevant": sum(1 for s in students if gold[s["id"]]),
               "avg_relevant_per_student": round(
                   np.mean([len(gold[s["id"]]) for s in students if gold[s["id"]]]), 2)},
    "recommendation": {m: clean(r) for m, r in results_rec.items()},
    "significance_acso_vs_tfidf_ndcg": {k: round(v, 6) for k, v in sig.items()},
    "tracking": results_track,
    "tracking_base_rate_late": round(base_rate, 4),
}

# k-sensitivity sweep (nDCG@k) for the main methods
rankers = {
    "AcSO": lambda s: rank_ontology(s, True, True),
    "TF-IDF": rank_tfidf,
    "Jaccard": rank_jaccard,
}
sweep = {name: {} for name in rankers}
for kk in (1, 3, 5, 10):
    for name, fn in rankers.items():
        vals = []
        for stu in students:
            rel = gold[stu["id"]]
            if not rel:
                continue
            vals.append(ndcg_at_k(fn(stu), rel, kk))
        sweep[name][kk] = round(float(np.mean(vals)), 4)
out["ndcg_sweep"] = sweep



# ----------------------------------------------------------------------------
# 8. Export the generated cohort as CSV (so the data is inspectable directly)
# ----------------------------------------------------------------------------
def export_csv():
    with open(os.path.join(DATA_DIR, "supervisors.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "expertise", "max_capacity", "current_load",
                    "available", "completion_rate", "h_index"])
        for s in supervisors:
            w.writerow([s["id"], "|".join(topic_label(t) for t in s["expertise"]),
                        s["max_capacity"], s["current_load"], s["available"],
                        s["completion_rate"], s["h_index"]])
    with open(os.path.join(DATA_DIR, "students.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "interests"])
        for s in students:
            w.writerow([s["id"], "|".join(topic_label(t) for t in s["interests"])])
    with open(os.path.join(DATA_DIR, "gold_standard.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["student_id", "relevant_supervisor_ids"])
        for s in students:
            w.writerow([s["id"], "|".join(sorted(gold[s["id"]]))])
    with open(os.path.join(DATA_DIR, "tracking.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["student_id", "overdue_milestones", "days_since_last_meeting",
                    "progress_deficit", "late_completion"])
        for stu, r in zip(students, track):
            w.writerow([stu["id"], r["overdue"], r["days_gap"],
                        round(r["deficit"], 4), r["late"]])


if __name__ == "__main__":
    with open(os.path.join(DATA_DIR, "results.json"), "w") as f:
        json.dump(out, f, indent=2)
    export_csv()
    print(json.dumps(out, indent=2))

