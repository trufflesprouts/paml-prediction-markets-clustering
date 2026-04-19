"""End-to-end experiment runner for the trader-archetype pipeline.

Steps:
  1. Load normalized wallet features.
  2. Sweep K=2..10 with K-Means, pick best by silhouette, save elbow+silhouette.
  3. Run Hierarchical (Ward, subsampled) at best K, compare to K-Means.
  4. Stability test: 10 random seeds of K-Means at best K, report mean+-std.
  5. Train KNN classifier on held-out features, labels = clustering labels.
     Try k in {3,5,7,...,19}, 70/15/15 split.  Report macro-F1 on test.
  6. Save all artifacts to our_code/results/.

Run:  uv run python our_code/run_experiments.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from models import (
    HierarchicalClustering,
    KMeansClustering,
    KNNClassifier,
    accuracy,
    confusion_matrix,
    macro_f1,
    pca_2d,
    profile_clusters,
    silhouette_score,
    train_val_test_split,
    within_cluster_heterogeneity,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUR_CODE = ROOT / "our_code"
RESULTS = OUR_CODE / "results"

CLUSTER_FEATURES = [
    "trade_count",
    "total_volume_usd",
    "distinct_tokens",
    "active_days",
    "maker_fraction",
    "avg_trade_size",
    "std_trade_size",
    "trades_per_day",
    "hour_entropy",
    "buy_fraction",
    "avg_price",
    "longshot_ratio",
]

CLASSIFY_FEATURES = [
    "max_trade_size",
    "trade_size_cv",
    "win_rate",
    "excess_return",
    "total_pnl",
    "resolved_trade_count",
]


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(DATA_DIR / "wallet_features_raw.csv")
    norm = pd.read_csv(DATA_DIR / "wallet_features_normalized.csv")
    # guard against duplicate wallets (shouldn't happen but belt+suspenders)
    raw = raw.drop_duplicates(subset="wallet").reset_index(drop=True)
    norm = norm.drop_duplicates(subset="wallet").reset_index(drop=True)
    # align order on wallet
    norm = norm.set_index("wallet").loc[raw["wallet"]].reset_index()
    return raw, norm


def run_kmeans_sweep(
    X: np.ndarray,
    k_values: list[int],
    n_init: int = 8,
    seed: int = 42,
) -> pd.DataFrame:
    rows = []
    for k in k_values:
        t0 = time.time()
        km = KMeansClustering(n_clusters=k, n_init=n_init, random_state=seed, init="kmeans++")
        km.fit(X)
        sil = silhouette_score(X, km.labels_, sample_size=3000, random_state=0)
        wch = within_cluster_heterogeneity(X, km.labels_)
        rows.append({
            "k": k,
            "inertia": km.inertia_,
            "silhouette": sil,
            "within_cluster_heterogeneity": wch,
            "runtime_sec": time.time() - t0,
        })
        print(f"  K={k}: inertia={km.inertia_:.1f}  silhouette={sil:.4f}  wch={wch:.4f}  t={rows[-1]['runtime_sec']:.1f}s")
    return pd.DataFrame(rows)


def run_stability_test(X: np.ndarray, k: int, n_seeds: int = 10) -> pd.DataFrame:
    """Run KMeans at fixed K with n_seeds different random seeds; record silhouette + inertia."""
    rows = []
    for seed in range(n_seeds):
        t0 = time.time()
        km = KMeansClustering(n_clusters=k, n_init=1, random_state=seed, init="kmeans++")
        km.fit(X)
        sil = silhouette_score(X, km.labels_, sample_size=3000, random_state=0)
        rows.append({
            "seed": seed,
            "inertia": km.inertia_,
            "silhouette": sil,
            "runtime_sec": time.time() - t0,
        })
        print(f"  seed={seed}: inertia={km.inertia_:.1f}  silhouette={sil:.4f}  t={rows[-1]['runtime_sec']:.1f}s")
    return pd.DataFrame(rows)


def run_knn_tuning(
    X_clf: np.ndarray,
    y: np.ndarray,
    k_values: list[int],
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
) -> tuple[pd.DataFrame, dict]:
    idx_tr, idx_va, idx_te = train_val_test_split(len(X_clf), ratios=ratios, random_state=seed)
    Xtr, ytr = X_clf[idx_tr], y[idx_tr]
    Xva, yva = X_clf[idx_va], y[idx_va]
    Xte, yte = X_clf[idx_te], y[idx_te]

    rows = []
    for k in k_values:
        t0 = time.time()
        knn = KNNClassifier(k=k)
        knn.fit(Xtr, ytr)
        yva_pred = knn.predict(Xva)
        rows.append({
            "k": k,
            "val_accuracy": accuracy(yva, yva_pred),
            "val_macro_f1": macro_f1(yva, yva_pred),
            "runtime_sec": time.time() - t0,
        })
        print(f"  k={k}: val_acc={rows[-1]['val_accuracy']:.4f}  val_f1={rows[-1]['val_macro_f1']:.4f}  t={rows[-1]['runtime_sec']:.1f}s")
    tuning = pd.DataFrame(rows)

    # pick best by val_macro_f1
    best_k = int(tuning.loc[tuning["val_macro_f1"].idxmax(), "k"])
    print(f"  [knn] best k by val_macro_f1 = {best_k}")

    knn = KNNClassifier(k=best_k)
    knn.fit(Xtr, ytr)
    yte_pred = knn.predict(Xte)
    summary = {
        "best_k": best_k,
        "test_accuracy": accuracy(yte, yte_pred),
        "test_macro_f1": macro_f1(yte, yte_pred),
        "n_train": int(len(idx_tr)),
        "n_val": int(len(idx_va)),
        "n_test": int(len(idx_te)),
        "confusion_matrix": confusion_matrix(yte, yte_pred).tolist(),
    }
    return tuning, summary


def main() -> None:
    t0 = time.time()
    RESULTS.mkdir(exist_ok=True)
    print(f"[experiments] loading data ...")
    raw, norm = load_data()
    print(f"[experiments] {len(norm):,} wallets loaded")

    X_cluster = norm[CLUSTER_FEATURES].to_numpy(dtype=np.float64)
    X_classify = norm[CLASSIFY_FEATURES].to_numpy(dtype=np.float64)

    # optional subsampling for clustering to keep runtime sane (K-Means on millions
    # of points with many K values is wasteful).  Take up to 50k.
    rng = np.random.default_rng(0)
    if len(X_cluster) > 50_000:
        clust_idx = rng.choice(len(X_cluster), size=50_000, replace=False)
        print(f"[experiments] subsampling to {len(clust_idx):,} wallets for K-Means sweep")
    else:
        clust_idx = np.arange(len(X_cluster))
    X_sub = X_cluster[clust_idx]

    print("\n[experiments] === K-Means sweep K=2..10 ===")
    k_values = list(range(2, 11))
    sweep = run_kmeans_sweep(X_sub, k_values, n_init=5, seed=42)
    sweep.to_csv(RESULTS / "kmeans_sweep.csv", index=False)
    best_k = int(sweep.loc[sweep["silhouette"].idxmax(), "k"])
    print(f"[experiments] best K by silhouette = {best_k}")

    # Fit final K-Means on ALL wallets at best K
    print(f"\n[experiments] === Final K-Means (K={best_k}) on all wallets ===")
    t1 = time.time()
    km_final = KMeansClustering(n_clusters=best_k, n_init=8, random_state=42, init="kmeans++")
    km_final.fit(X_cluster)
    print(f"  fit in {time.time()-t1:.1f}s, inertia={km_final.inertia_:.1f}")

    # Profile + personas
    profile, personas = profile_clusters(raw, km_final.labels_, CLUSTER_FEATURES)
    profile.to_csv(RESULTS / "kmeans_profile.csv", index=False)
    np.save(RESULTS / "kmeans_labels.npy", km_final.labels_)
    np.save(RESULTS / "kmeans_centers.npy", km_final.cluster_centers_)
    for k, p in personas.items():
        print(f"  {p}")

    print(f"\n[experiments] === Hierarchical (Ward, K={best_k}, subsample=2000) ===")
    t1 = time.time()
    hc = HierarchicalClustering(n_clusters=best_k, subsample_size=2000, random_state=0)
    hc.fit(X_cluster)
    sil_hc = silhouette_score(X_cluster, hc.labels_, sample_size=3000, random_state=0)
    print(f"  fit in {time.time()-t1:.1f}s  silhouette={sil_hc:.4f}")
    np.save(RESULTS / "hierarchical_labels.npy", hc.labels_)
    np.save(RESULTS / "hierarchical_centers.npy", hc.cluster_centers_)
    profile_hc, personas_hc = profile_clusters(raw, hc.labels_, CLUSTER_FEATURES)
    profile_hc.to_csv(RESULTS / "hierarchical_profile.csv", index=False)

    print(f"\n[experiments] === Stability test (K={best_k}, 10 seeds) ===")
    stab = run_stability_test(X_sub, k=best_k, n_seeds=10)
    stab.to_csv(RESULTS / "stability.csv", index=False)

    print(f"\n[experiments] === KNN tuning (classify features) ===")
    y = km_final.labels_
    knn_k_values = [3, 5, 7, 9, 11, 13, 15, 17, 19]
    # To keep KNN computationally feasible on hundreds of thousands of wallets,
    # subsample the training side to 20k and the eval sides to 5k each.
    max_tr = 20_000
    max_eval = 5_000
    rng2 = np.random.default_rng(123)
    if len(X_classify) > (max_tr + 2 * max_eval):
        knn_idx = rng2.choice(len(X_classify), size=(max_tr + 2 * max_eval), replace=False)
    else:
        knn_idx = np.arange(len(X_classify))
    tuning, knn_summary = run_knn_tuning(
        X_classify[knn_idx], y[knn_idx], knn_k_values, ratios=(0.7, 0.15, 0.15), seed=42
    )
    tuning.to_csv(RESULTS / "knn_tuning.csv", index=False)

    # PCA projection (2D) for plotting
    print(f"\n[experiments] === PCA 2D projection ===")
    proj, evr = pca_2d(X_cluster)
    np.save(RESULTS / "pca_2d.npy", proj)
    np.save(RESULTS / "pca_evr.npy", evr)
    print(f"  explained variance ratio (2 comps): {evr.tolist()}")

    # summary JSON
    summary = {
        "n_wallets": int(len(raw)),
        "n_features_cluster": len(CLUSTER_FEATURES),
        "n_features_classify": len(CLASSIFY_FEATURES),
        "best_k_kmeans": best_k,
        "best_silhouette_kmeans": float(sweep["silhouette"].max()),
        "best_silhouette_hierarchical": float(sil_hc),
        "stability_silhouette_mean": float(stab["silhouette"].mean()),
        "stability_silhouette_std": float(stab["silhouette"].std()),
        "knn": knn_summary,
        "personas_kmeans": {str(k): v for k, v in personas.items()},
        "pca_explained_variance_ratio_2d": evr.tolist(),
        "elapsed_sec": time.time() - t0,
    }
    with open(RESULTS / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[experiments] === SUMMARY ===")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("personas_kmeans",)}, indent=2))
    print(f"[experiments] total elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
