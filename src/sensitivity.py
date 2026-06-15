"""
Sensitivity and robustness analysis (Section "Sensitivity and Robustness").

  (a) alpha/beta sweep for the recommender quality weighting,
  (b) contact-gap threshold sweep for the tracking rules,
  (c) noise-robustness of the recommender (corrupt a fraction of observed
      topic labels, average over 5 seeds).

Run:  python sensitivity.py        (prints tables, saves fig_sensitivity.png)
"""
import os
import numpy as np
import random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import evaluate as ev

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
os.makedirs(FIG_DIR, exist_ok=True)


# ---------- (a) alpha/beta sweep (recommendation) ----------
def rank_alpha(stu, alpha):
    scored = []
    for sup in ev.supervisors:
        if not sup["available"]:
            continue
        sim = max(ev.wu_palmer(si, se)
                  for si in stu["interests"] for se in sup["expertise"])
        scored.append((sup["id"], alpha * sim + (1 - alpha) * sup["completion_rate"]))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [s for s, _ in scored]


def alpha_sweep(alphas=(0.6, 0.7, 0.8, 0.9, 1.0)):
    rows = {}
    for a in alphas:
        r = ev.evaluate_ranker(lambda s, aa=a: rank_alpha(s, aa), 5)
        rows[a] = (round(r["P@k"], 3), round(r["nDCG@k"], 3))
    return rows


# ---------- (b) contact-gap threshold sweep (tracking) ----------
def rule_gap(r, gap):
    return ((r["deficit"] > 0.20 and r["days_gap"] > gap)
            or r["deficit"] > 0.28
            or (r["overdue"] >= 3 and r["days_gap"] > gap))


def gap_sweep(gaps=(28, 35, 42, 49, 56)):
    rows = {}
    for g in gaps:
        tp = fp = tn = fn = 0
        for r in ev.track:
            p = rule_gap(r, g)
            if p and r["late"]: tp += 1
            elif p and not r["late"]: fp += 1
            elif not p and not r["late"]: tn += 1
            else: fn += 1
        prec = tp / (tp + fp) if tp + fp else 0
        rec = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
        acc = (tp + tn) / len(ev.track)
        rows[g] = (round(prec, 3), round(rec, 3), round(f1, 3), round(acc, 3), fp)
    return rows


# ---------- (c) noise robustness (recommendation) ----------
def _perturb(frac, seed):
    rng = random.Random(seed)
    sups = [dict(s, expertise=[rng.choice(ev.topics) if rng.random() < frac else t
                               for t in s["expertise"]]) for s in ev.supervisors]
    pint = {st["id"]: [rng.choice(ev.topics) if rng.random() < frac else t
                       for t in st["interests"]] for st in ev.students}
    return sups, pint


def noise_robustness(fracs=(0.0, 0.1, 0.2, 0.3), seeds=(1, 2, 3, 4, 5)):
    rows = {}
    for frac in fracs:
        per_seed = []
        for sd in seeds:
            sups, pint = _perturb(frac, sd)
            vals = []
            for stu in ev.students:
                rel = ev.gold[stu["id"]]
                if not rel:
                    continue
                interests = pint[stu["id"]]
                scored = []
                for sup in sups:
                    if not sup["available"]:
                        continue
                    sim = max(ev.wu_palmer(si, se)
                              for si in interests for se in sup["expertise"])
                    scored.append((sup["id"], 0.8 * sim + 0.2 * sup["completion_rate"]))
                scored.sort(key=lambda x: x[1], reverse=True)
                vals.append(ev.ndcg_at_k([s for s, _ in scored], rel, 5))
            per_seed.append(np.mean(vals))
        rows[frac] = (round(float(np.mean(per_seed)), 3), round(float(np.std(per_seed)), 3))
    return rows


def make_figure(asweep, nrows):
    TEAL, GREY, RISK = "#1c6e69", "#9aa6ad", "#b14a2c"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.7), dpi=300)
    alphas = list(asweep); nd = [asweep[a][1] for a in alphas]
    ax1.plot(alphas, nd, marker="o", color=TEAL, lw=2, ms=7, label="AcSO (full)")
    ax1.axhline(0.657, color=GREY, ls="--", lw=1.5, label="best baseline")
    ax1.set_xlabel(r"quality weight $\alpha$  ($\beta=1-\alpha$)"); ax1.set_ylabel("nDCG@5")
    ax1.set_ylim(0.5, 1.02); ax1.set_xticks(alphas); ax1.legend(frameon=False, fontsize=9, loc="lower left")
    ax1.spines[["top", "right"]].set_visible(False); ax1.grid(alpha=0.25); ax1.set_title("(a) Weight sensitivity")
    fr = list(nrows); mean = [nrows[f][0] for f in fr]; err = [nrows[f][1] for f in fr]
    ax2.errorbar(fr, mean, yerr=err, marker="s", color=RISK, lw=2, ms=7, capsize=4, label="AcSO (full)")
    ax2.axhline(0.657, color=GREY, ls="--", lw=1.5, label="best baseline")
    ax2.set_xlabel("fraction of corrupted topic labels"); ax2.set_ylabel("nDCG@5")
    ax2.set_ylim(0.5, 1.02); ax2.set_xticks(fr); ax2.legend(frameon=False, fontsize=9, loc="lower left")
    ax2.spines[["top", "right"]].set_visible(False); ax2.grid(alpha=0.25); ax2.set_title("(b) Noise robustness")
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "fig_sensitivity.png"), bbox_inches="tight")


if __name__ == "__main__":
    asweep = alpha_sweep(); gsweep = gap_sweep(); nrows = noise_robustness()
    print("=== alpha/beta sensitivity (recommendation) ===")
    print("alpha  P@5    nDCG@5")
    for a, (p, nd) in asweep.items():
        print(f"{a:.1f}   {p:.3f}  {nd:.3f}")
    print("\n=== contact-gap threshold sensitivity (tracking) ===")
    print("gap  Prec   Rec    F1     Acc    FP")
    for g, (p, r, f, a, fp) in gsweep.items():
        print(f"{g:>3}  {p:.3f}  {r:.3f}  {f:.3f}  {a:.3f}  {fp}")
    print("\n=== noise robustness (recommendation, nDCG@5) ===")
    print("noise  nDCG@5  (std)")
    for fr, (m, s) in nrows.items():
        print(f"{fr:.1f}   {m:.3f}   ({s:.3f})")
    make_figure(asweep, nrows)
    print("\nsaved figures/fig_sensitivity.png")
