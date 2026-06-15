"""
Graph-embedding baseline for the recommendation experiment.

Builds node embeddings of the research-topic graph (broaderTopic + relatedTopic
edges) via Laplacian Eigenmaps, represents each person as the mean embedding of
their topics, and ranks supervisors by cosine similarity. This is a genuine,
deterministic semantic baseline beyond lexical TF-IDF/Jaccard, computed on the
SAME seeded cohort as the main harness (evaluate.py).

Run:  python graph_embedding_baseline.py
Importing this module exposes EMB (metrics + nDCG sweep) for figure scripts.
"""
import numpy as np
from scipy.stats import wilcoxon
import evaluate as ev   # importing runs the harness and exposes its objects

idx = {t: i for i, t in enumerate(ev.topics)}
n = len(ev.topics)

# Undirected adjacency over the topic graph: is-a (broader) + related links.
A = np.zeros((n, n))
for child, par in ev.parent.items():
    if child in idx and par in idx:
        A[idx[child], idx[par]] = A[idx[par], idx[child]] = 1.0
for a, bs in ev.related_edges.items():
    for b in bs:
        if a in idx and b in idx:
            A[idx[a], idx[b]] = A[idx[b], idx[a]] = 1.0

# Normalized Laplacian L = I - D^-1/2 A D^-1/2 ; Laplacian-Eigenmaps embedding.
deg = A.sum(1); deg[deg == 0] = 1e-9
Dinv = np.diag(1.0 / np.sqrt(deg))
L = np.eye(n) - Dinv @ A @ Dinv
_, vecs = np.linalg.eigh(L)
DIM = 8
emb = vecs[:, 1:DIM + 1]                                   # skip trivial eigenvector
emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)


def _vec(topic_list):
    v = np.mean([emb[idx[t]] for t in topic_list if t in idx], axis=0)
    nrm = np.linalg.norm(v)
    return v / nrm if nrm else v


_sup_vec = {s["id"]: _vec(s["expertise"]) for s in ev.supervisors}


def rank_embedding(stu):
    q = _vec(stu["interests"])
    scored = [(s["id"], float(np.dot(q, _sup_vec[s["id"]]))) for s in ev.supervisors]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [sid for sid, _ in scored]


_res = ev.evaluate_ranker(rank_embedding, 5)
_sweep = {kk: round(float(np.mean(
            [ev.ndcg_at_k(rank_embedding(s), ev.gold[s["id"]], kk)
             for s in ev.students if ev.gold[s["id"]]])), 4)
          for kk in (1, 3, 5, 10)}

EMB = {"P@5": round(_res["P@k"], 4), "R@5": round(_res["R@k"], 4),
       "F1@5": round(_res["F1@k"], 4), "MAP": round(_res["MAP"], 4),
       "nDCG@5": round(_res["nDCG@k"], 4), "ndcg_sweep": _sweep}

if __name__ == "__main__":
    acso = ev.results_rec["AcSO (semantic + rules + quality)"]
    _, p_ndcg = wilcoxon(acso["_per_student_ndcg"], _res["_per_student_ndcg"])
    _, p_f1 = wilcoxon(acso["_per_student_f1"], _res["_per_student_f1"])
    print("=== Graph-embedding (Laplacian Eigenmaps) baseline ===")
    for k in ("P@5", "R@5", "F1@5", "MAP", "nDCG@5"):
        print(f"{k:7s} {EMB[k]:.4f}")
    print("nDCG sweep:", EMB["ndcg_sweep"])
    print(f"Wilcoxon AcSO vs embedding: nDCG p={p_ndcg:.2e}, F1 p={p_f1:.2e}")
