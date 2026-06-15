"""
Regenerate the result figures from the computed numbers.

Produces (in ../figures):
  fig_rec_results.png  – grouped bars: P@5, MAP, nDCG@5 for all methods
  fig_ndcg_sweep.png   – nDCG@k vs cut-off k
  fig_tracking.png     – tracking metrics + false-alarm comparison

The architecture and ontology-schema diagrams (fig_architecture.png,
fig_ontology.png) are conceptual diagrams shipped as static assets.

Run:  python make_figures.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import evaluate as ev
import graph_embedding_baseline as gemb   # provides EMB

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
os.makedirs(FIG_DIR, exist_ok=True)
plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans"})
TEAL, GOLD, GREY, SLATE, RISK, INK = "#1c6e69", "#b8893b", "#9aa6ad", "#46606e", "#b14a2c", "#16242e"

R = ev.results_rec
acso = R["AcSO (semantic + rules + quality)"]
abl = R["Ablation: semantic only (no rules)"]
tf = R["TF-IDF / cosine"]
ja = R["Jaccard keyword overlap"]

# ---- Figure: grouped bars ----
methods = ["TF-IDF", "Jaccard", "Graph-emb.", "Semantic-only", "AcSO (full)"]
colors = [GREY, SLATE, "#7a9aa6", "#caa45a", TEAL]
P = [tf["P@k"], ja["P@k"], gemb.EMB["P@5"], abl["P@k"], acso["P@k"]]
MAP = [tf["MAP"], ja["MAP"], gemb.EMB["MAP"], abl["MAP"], acso["MAP"]]
ND = [tf["nDCG@k"], ja["nDCG@k"], gemb.EMB["nDCG@5"], abl["nDCG@k"], acso["nDCG@k"]]
fig, ax = plt.subplots(figsize=(8.2, 4.6), dpi=300)
x = np.arange(3); w = 0.15
data = np.array([P, MAP, ND]).T
for i, m in enumerate(methods):
    ax.bar(x + (i - 2) * w, data[i], w, label=m, color=colors[i], edgecolor="white", linewidth=0.6)
    for j, v in enumerate(data[i]):
        ax.text(x[j] + (i - 2) * w, v + 0.012, f"{v:.2f}", ha="center", va="bottom", fontsize=8, color=INK)
ax.set_xticks(x); ax.set_xticklabels(["P@5", "MAP", "nDCG@5"]); ax.set_ylim(0, 1.08); ax.set_ylabel("Score")
ax.legend(ncol=3, fontsize=9, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.16))
ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", alpha=0.25)
fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "fig_rec_results.png"), bbox_inches="tight")

# ---- Figure: nDCG@k sweep ----
sw = ev.sweep  # AcSO/TF-IDF/Jaccard from the harness
ks = [1, 3, 5, 10]
series = {"AcSO (full)": [sw["AcSO"][k] for k in ks],
          "Graph-emb.": [gemb.EMB["ndcg_sweep"][k] for k in ks],
          "TF-IDF": [sw["TF-IDF"][k] for k in ks],
          "Jaccard": [sw["Jaccard"][k] for k in ks]}
cmap = {"AcSO (full)": TEAL, "Graph-emb.": "#7a9aa6", "TF-IDF": GREY, "Jaccard": SLATE}
mk = {"AcSO (full)": "o", "Graph-emb.": "^", "TF-IDF": "s", "Jaccard": "D"}
fig, ax = plt.subplots(figsize=(7.4, 4.4), dpi=300)
for name, ys in series.items():
    ax.plot(ks, ys, marker=mk[name], color=cmap[name], linewidth=2, markersize=7, label=name)
ax.set_xticks(ks); ax.set_xlabel("Cut-off k"); ax.set_ylabel("nDCG@k"); ax.set_ylim(0.5, 1.03)
ax.legend(frameon=False, fontsize=10); ax.spines[["top", "right"]].set_visible(False); ax.grid(alpha=0.25)
fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "fig_ndcg_sweep.png"), bbox_inches="tight")

# ---- Figure: tracking ----
T = ev.results_track
a = T["AcSO multi-condition rules"]; nv = T["Naive single-condition rule"]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 4.0), dpi=300, gridspec_kw={"width_ratios": [2, 1]})
mets = ["Precision", "Recall", "F1", "Accuracy"]
av = [a["precision"], a["recall"], a["f1"], a["accuracy"]]
nvv = [nv["precision"], nv["recall"], nv["f1"], nv["accuracy"]]
x = np.arange(len(mets)); w = 0.38
ax1.bar(x - w / 2, av, w, label="AcSO multi-condition", color=INK)
ax1.bar(x + w / 2, nvv, w, label="Naive single-condition", color=GOLD)
for xi, v in zip(x - w / 2, av): ax1.text(xi, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
for xi, v in zip(x + w / 2, nvv): ax1.text(xi, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
ax1.set_xticks(x); ax1.set_xticklabels(mets); ax1.set_ylim(0, 1.05); ax1.set_ylabel("Score")
ax1.legend(frameon=False, fontsize=9); ax1.spines[["top", "right"]].set_visible(False); ax1.grid(axis="y", alpha=0.25)
ax2.bar(["AcSO", "Naive"], [a["fp"], nv["fp"]], color=[INK, GOLD], width=0.6)
for xi, v in enumerate([a["fp"], nv["fp"]]): ax2.text(xi, v + 2, str(v), ha="center", fontsize=10, fontweight="bold")
ax2.set_ylabel("False alarms (FP)"); ax2.set_title("Wasted advisor alerts", fontsize=11)
ax2.spines[["top", "right"]].set_visible(False); ax2.grid(axis="y", alpha=0.25)
fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "fig_tracking.png"), bbox_inches="tight")

print("saved fig_rec_results.png, fig_ndcg_sweep.png, fig_tracking.png")
