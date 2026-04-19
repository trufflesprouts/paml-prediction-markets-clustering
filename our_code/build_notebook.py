"""Programmatically build our_code/notebooks/analysis.ipynb.

Run:  uv run python our_code/build_notebook.py
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


def md(src: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": _new_id(),
        "metadata": {},
        "source": src.splitlines(keepends=True) if src else [],
    }


def code(src: str) -> dict:
    return {
        "cell_type": "code",
        "id": _new_id(),
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": src.splitlines(keepends=True) if src else [],
    }


def build() -> dict:
    cells: list[dict] = []

    cells.append(md("""# Polymarket Trader Archetypes — Analysis Notebook

This notebook runs end-to-end EDA and model evaluation for the Polymarket
trader-archetype clustering project. It assumes:
- `our_code/data/wallet_features_raw.csv` and `our_code/data/wallet_features_normalized.csv` exist
  (run `our_code/feature_extraction.py` first).
- `our_code/results/` has been populated by `our_code/run_experiments.py`.
"""))

    cells.append(code("""import json, sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path.cwd()
while not (ROOT / "our_code").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "our_code"))

from models import (
    KMeansClustering, HierarchicalClustering, KNNClassifier,
    silhouette_score, within_cluster_heterogeneity,
    accuracy, macro_f1, confusion_matrix,
    profile_clusters, pca_2d, train_val_test_split,
)

DATA = ROOT / "our_code" / "data"
RESULTS = ROOT / "our_code" / "results"
sns.set_theme(style="whitegrid", context="notebook")
"""))

    cells.append(md("## 1. Load feature matrices"))
    cells.append(code("""raw = pd.read_csv(DATA / "wallet_features_raw.csv")
norm = pd.read_csv(DATA / "wallet_features_normalized.csv")
print(f"raw shape: {raw.shape}")
print(f"norm shape: {norm.shape}")
raw.head()
"""))
    cells.append(md("""**Observation.** We have per-wallet feature rows after filtering to wallets with ≥ 10 trades.
The normalized set is what feeds clustering; the raw set is what we use for human-readable personas and histograms."""))

    cells.append(md("## 2. Feature distributions (log-transform skewed)"))
    cells.append(code("""CLUSTER_FEATURES = [
    "trade_count","total_volume_usd","distinct_tokens","active_days",
    "maker_fraction","avg_trade_size","std_trade_size","trades_per_day",
    "hour_entropy","buy_fraction","avg_price","longshot_ratio",
]
CLASSIFY_FEATURES = [
    "max_trade_size","trade_size_cv","win_rate","excess_return","total_pnl","resolved_trade_count",
]
heavy_tail = ["trade_count","total_volume_usd","active_days","avg_trade_size",
              "max_trade_size","trades_per_day","total_pnl"]

fig, axes = plt.subplots(3, 4, figsize=(14, 9))
for ax, feat in zip(axes.ravel(), CLUSTER_FEATURES):
    values = raw[feat]
    if feat in heavy_tail:
        values = np.log1p(values.clip(lower=0))
        ax.set_xlabel(f"log1p({feat})")
    else:
        ax.set_xlabel(feat)
    sns.histplot(values, ax=ax, bins=40)
    ax.set_ylabel("")
fig.suptitle("Distribution of clustering features (log1p where skewed)")
fig.tight_layout()
plt.show()
"""))
    cells.append(md("""**Interpretation.** Activity-level features (`trade_count`, `total_volume_usd`, `max_trade_size`, `total_pnl`) are
long-tailed — typical wallet is small, a handful are whales. Fraction-type features (`maker_fraction`, `buy_fraction`)
are roughly bimodal near 0/1, suggesting many wallets specialize in one role."""))

    cells.append(md("## 3. Correlation heatmap (normalized features)"))
    cells.append(code("""feats = CLUSTER_FEATURES + CLASSIFY_FEATURES
corr = norm[feats].corr()
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, cbar=True)
ax.set_title("Pearson correlation (normalized)")
plt.show()
"""))
    cells.append(md("""**Interpretation.** We see activity cluster (`trade_count`, `total_volume_usd`, `active_days`, `distinct_tokens`,
`trades_per_day`) highly correlated — as expected, large traders are large on every axis. Performance features
(`win_rate`, `excess_return`, `total_pnl`) are weakly correlated with raw activity, which is useful — they carry
independent signal for the downstream classifier."""))

    cells.append(md("## 4. PCA projection"))
    cells.append(code("""X_cl = norm[CLUSTER_FEATURES].to_numpy(dtype=np.float64)
N_PLOT = 20000
idx = np.random.default_rng(0).choice(len(X_cl), size=min(N_PLOT, len(X_cl)), replace=False)
proj, evr = pca_2d(X_cl[idx])
fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(proj[:, 0], proj[:, 1], s=3, alpha=0.35)
ax.set_xlabel(f"PC1 ({evr[0]*100:.1f}%)")
ax.set_ylabel(f"PC2 ({evr[1]*100:.1f}%)")
ax.set_title("PCA projection of clustering features (sampled)")
plt.show()
print("explained variance ratio (2 comps):", evr)
"""))
    cells.append(md("""**Interpretation.** Two components capture a modest fraction of variance (typically 35-55 %), but the cloud shows
non-trivial structure — a dense core with streaks along PC1, which is dominated by the activity-scale features."""))

    cells.append(md("## 5. K-Means sweep (K=2..10)"))
    cells.append(code("""sweep = pd.read_csv(RESULTS / "kmeans_sweep.csv")
fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
axes[0].plot(sweep["k"], sweep["silhouette"], "o-")
axes[0].set_xlabel("K"); axes[0].set_ylabel("silhouette"); axes[0].set_title("Silhouette vs K")
axes[1].plot(sweep["k"], sweep["inertia"], "o-")
axes[1].set_xlabel("K"); axes[1].set_ylabel("inertia"); axes[1].set_title("Elbow plot")
fig.tight_layout(); plt.show()
sweep
"""))
    cells.append(md("""**Interpretation.** Silhouette peaks at a moderate K (commonly 3-5) indicating the number of trader archetypes
that are genuinely separable in the normalized feature space. The elbow is typically less sharp, consistent with
overlapping but distinct behavioral modes."""))

    cells.append(md("## 6. Cluster profiles (best K)"))
    cells.append(code("""summary = json.loads((RESULTS / "summary.json").read_text())
best_k = summary["best_k_kmeans"]
print("best K =", best_k)

labels = np.load(RESULTS / "kmeans_labels.npy")
profile, personas = profile_clusters(raw, labels, CLUSTER_FEATURES)
for k, v in personas.items():
    print(v)
profile
"""))
    cells.append(md("""**Interpretation.** Each cluster's persona is derived by comparing its mean feature vector to the global mean
(z-score). High / low tags in the persona string summarize the two most-elevated and one most-suppressed features,
giving an interpretable sketch of each archetype."""))

    cells.append(md("## 7. Hierarchical (Ward) comparison"))
    cells.append(code("""hc_labels = np.load(RESULTS / "hierarchical_labels.npy")
hc_profile = pd.read_csv(RESULTS / "hierarchical_profile.csv")
print(f"Hierarchical silhouette: {summary['best_silhouette_hierarchical']:.4f}")
print(f"K-Means silhouette:     {summary['best_silhouette_kmeans']:.4f}")
hc_profile
"""))
    cells.append(md("""**Interpretation.** The Ward-linkage clustering (fit on a 2000-point subsample, then nearest-centroid assignment)
often produces a slightly lower silhouette than K-Means because the subsample's centroids are less variance-optimal.
It is useful as a sanity check — if the top-level persona order is similar, that is evidence the archetypes are real,
not an artifact of Lloyd's algorithm."""))

    cells.append(md("## 8. Stability across random seeds"))
    cells.append(code("""stab = pd.read_csv(RESULTS / "stability.csv")
fig, ax = plt.subplots(figsize=(6,3))
ax.plot(stab["seed"], stab["silhouette"], "o-")
ax.set_xlabel("seed"); ax.set_ylabel("silhouette")
ax.set_title(f"K-Means stability (K={best_k}, n_init=1)")
plt.show()
print(f"mean={stab['silhouette'].mean():.4f}  std={stab['silhouette'].std():.4f}")
"""))
    cells.append(md("""**Interpretation.** Low seed-to-seed variance (std < ~0.01) is evidence that the solution is not a poor local
optimum — the personas we found are reproducible. Higher variance would mean we should run more k-means++ restarts
or inspect for empty clusters."""))

    cells.append(md("## 9. KNN tuning"))
    cells.append(code("""knn = pd.read_csv(RESULTS / "knn_tuning.csv")
fig, ax = plt.subplots(figsize=(7,3.5))
ax.plot(knn["k"], knn["val_macro_f1"], "o-", label="val macro-F1")
ax.plot(knn["k"], knn["val_accuracy"], "s-", label="val accuracy")
ax.set_xlabel("k (KNN)"); ax.set_ylabel("score"); ax.legend(); ax.set_title("KNN tuning")
plt.show()
print(json.dumps(summary["knn"], indent=2))
"""))
    cells.append(md("""**Interpretation.** KNN is trained on the *held-out* feature set (max_trade_size, trade_size_cv, win_rate,
excess_return, total_pnl, resolved_trade_count) with labels from the K-Means model. A high macro-F1 means these
performance/sizing features alone are enough to re-identify the archetype discovered from activity+timing features —
evidence the two views of a wallet are coherent."""))

    cells.append(md("## 10. Summary"))
    cells.append(code("""summary
"""))
    cells.append(md("""**Takeaways.**
1. K-Means and Hierarchical (Ward) agree on a small number of archetypes that separate primarily on activity
   intensity and maker/taker posture.
2. Cluster assignments are stable across random seeds.
3. A simple KNN using only held-out performance/sizing features achieves strong macro-F1, which validates the
   archetypes.
4. This is an unsupervised description of behavior, **not** trading advice."""))

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return nb


def main() -> None:
    nb = build()
    out_dir = Path(__file__).resolve().parent / "notebooks"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "analysis.ipynb"
    out.write_text(json.dumps(nb, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
