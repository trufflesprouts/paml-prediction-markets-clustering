"""V2 clustering experiment: refined feature set.

Changes vs V1:
  REMOVE from clustering: maker_fraction, active_days, hour_entropy
  ADD to clustering: directional_conviction = |buy_fraction - 0.5| * 2
                     size_variability      = std_trade_size / avg_trade_size
                     (price_std skipped — would require re-extracting per-trade
                      prices; the raw wallet CSV only carries avg_price.)

Runs full K=2..10 sweep + explicit K=5, saves:
  our_code/results/kmeans_sweep_v2.csv
  our_code/results/kmeans_v2_labels.npy
  our_code/results/kmeans_v2_k5_labels.npy
  our_code/results/cluster_profiles_v2.csv
  our_code/results/pca_v2_best.png
  our_code/results/pca_v2_k5.png
  our_code/results/summary_v2.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from models import (  # noqa: E402
    KMeansClustering,
    pca_2d,
    profile_clusters,
    silhouette_score,
    within_cluster_heterogeneity,
)

ROOT = HERE.parent
DATA = ROOT / "data"
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)


CLUSTER_FEATURES_V2 = [
    "trade_count",
    "total_volume_usd",
    "distinct_tokens",
    "avg_trade_size",
    "std_trade_size",
    "trades_per_day",
    "buy_fraction",
    "avg_price",
    "longshot_ratio",
    "directional_conviction",
    "size_variability",
]

# log-transform heavy-tailed features (skip [0,1] fractions)
LOG_FEATURES_V2 = {
    "trade_count",
    "total_volume_usd",
    "distinct_tokens",
    "avg_trade_size",
    "std_trade_size",
    "trades_per_day",
    "size_variability",
}
CLIP_STD = 5.0


def log1p_signed(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


def build_v2_features(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df["directional_conviction"] = (df["buy_fraction"] - 0.5).abs() * 2.0
    # size_variability already computed in the CSV as `trade_size_cv`
    df["size_variability"] = df["trade_size_cv"]
    return df


def normalize_v2(df: pd.DataFrame) -> pd.DataFrame:
    out = df[CLUSTER_FEATURES_V2].copy().astype(np.float64)
    for col in CLUSTER_FEATURES_V2:
        if col in LOG_FEATURES_V2:
            out[col] = log1p_signed(out[col].to_numpy())
        mu = out[col].mean()
        sd = out[col].std(ddof=0)
        out[col] = (out[col] - mu) / sd if sd > 0 else 0.0
        out[col] = out[col].clip(-CLIP_STD, CLIP_STD)
    return out


def pca_scatter(X: np.ndarray, labels: np.ndarray, title: str, out: Path) -> tuple[float, float]:
    proj, evr = pca_2d(X)
    fig, ax = plt.subplots(figsize=(7, 6))
    palette = sns.color_palette("husl", int(labels.max()) + 1)
    for c in sorted(np.unique(labels)):
        mask = labels == c
        ax.scatter(proj[mask, 0], proj[mask, 1], s=4, alpha=0.4,
                   color=palette[int(c)], label=f"C{c} (n={int(mask.sum()):,})")
    ax.set_xlabel(f"PC1 ({evr[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({evr[1]*100:.1f}%)")
    ax.set_title(title)
    ax.legend(loc="best", markerscale=3, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return float(evr[0]), float(evr[1])


def run_sweep(X: np.ndarray, k_values: list[int], seed: int = 42, n_init: int = 5) -> pd.DataFrame:
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
        print(f"  K={k}: inertia={km.inertia_:.1f}  silhouette={sil:.4f}  wch={wch:.4f}  t={rows[-1]['runtime_sec']:.1f}s",
              flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    t0 = time.time()

    raw = pd.read_csv(DATA / "wallet_features_raw.csv")
    df = build_v2_features(raw)
    norm_v2 = normalize_v2(df)
    print(f"[v2] {len(norm_v2):,} wallets x {len(CLUSTER_FEATURES_V2)} features", flush=True)
    print(f"[v2] features: {CLUSTER_FEATURES_V2}", flush=True)

    X_all = norm_v2.to_numpy(dtype=np.float64)

    # Subsample for the sweep (same strategy as V1)
    rng = np.random.default_rng(0)
    SUB_N = 50_000
    if len(X_all) > SUB_N:
        idx = rng.choice(len(X_all), size=SUB_N, replace=False)
    else:
        idx = np.arange(len(X_all))
    X_sub = X_all[idx]
    raw_sub = df.iloc[idx].reset_index(drop=True)

    print("\n[v2] === K-Means sweep K=2..10 ===", flush=True)
    sweep = run_sweep(X_sub, list(range(2, 11)), seed=42, n_init=5)
    sweep.to_csv(RESULTS / "kmeans_sweep_v2.csv", index=False)
    best_k = int(sweep.loc[sweep["silhouette"].idxmax(), "k"])
    best_sil = float(sweep["silhouette"].max())
    print(f"[v2] best K by silhouette = {best_k} (silhouette={best_sil:.4f})", flush=True)

    # Fit at best K on all wallets
    print(f"\n[v2] === Fitting K={best_k} on all {len(X_all):,} wallets ===", flush=True)
    km_best = KMeansClustering(n_clusters=best_k, n_init=8, random_state=42, init="kmeans++").fit(X_all)
    np.save(RESULTS / "kmeans_v2_labels.npy", km_best.labels_)
    sil_best_all = silhouette_score(X_all, km_best.labels_, sample_size=3000, random_state=0)
    print(f"  inertia={km_best.inertia_:.1f}  silhouette(full)={sil_best_all:.4f}", flush=True)

    prof_best, personas_best = profile_clusters(df, km_best.labels_, CLUSTER_FEATURES_V2)

    # Fit K=5 explicitly on all wallets
    print(f"\n[v2] === Fitting K=5 on all {len(X_all):,} wallets ===", flush=True)
    km_k5 = KMeansClustering(n_clusters=5, n_init=8, random_state=42, init="kmeans++").fit(X_all)
    np.save(RESULTS / "kmeans_v2_k5_labels.npy", km_k5.labels_)
    sil_k5_all = silhouette_score(X_all, km_k5.labels_, sample_size=3000, random_state=0)
    print(f"  inertia={km_k5.inertia_:.1f}  silhouette(full)={sil_k5_all:.4f}", flush=True)

    prof_k5, personas_k5 = profile_clusters(df, km_k5.labels_, CLUSTER_FEATURES_V2)

    # Combined profile CSV (tag with K)
    prof_best_tagged = prof_best.assign(model=f"V2_bestK{best_k}")
    prof_k5_tagged = prof_k5.assign(model="V2_K5")
    pd.concat([prof_best_tagged, prof_k5_tagged], ignore_index=True).to_csv(
        RESULTS / "cluster_profiles_v2.csv", index=False
    )

    # PCA visualizations — compute on a 20k sample so plots render quickly
    plot_rng = np.random.default_rng(7)
    N_PLOT = min(20_000, len(X_all))
    plot_idx = plot_rng.choice(len(X_all), size=N_PLOT, replace=False)
    evr_best = pca_scatter(X_all[plot_idx], km_best.labels_[plot_idx],
                           f"V2 — best K={best_k} (silhouette={sil_best_all:.3f})",
                           RESULTS / "pca_v2_best.png")
    evr_k5 = pca_scatter(X_all[plot_idx], km_k5.labels_[plot_idx],
                         f"V2 — K=5 (silhouette={sil_k5_all:.3f})",
                         RESULTS / "pca_v2_k5.png")

    # Personas
    print("\n[v2] Personas — best K={}:".format(best_k), flush=True)
    for k, v in personas_best.items():
        print(f"  {v}", flush=True)
    print("\n[v2] Personas — K=5:", flush=True)
    for k, v in personas_k5.items():
        print(f"  {v}", flush=True)

    # V1 summary for comparison
    with (RESULTS / "summary.json").open() as f:
        v1_summary = json.load(f)

    # Compute V1 K=5 silhouette if not present (it isn't in the original summary;
    # read sweep to check)
    v1_sweep = pd.read_csv(RESULTS / "kmeans_sweep.csv")
    v1_k5_row = v1_sweep[v1_sweep["k"] == 5]
    v1_k5_sil = float(v1_k5_row["silhouette"].iloc[0]) if len(v1_k5_row) else float("nan")

    # Cluster-size summaries
    def _sizes(labels: np.ndarray) -> list[int]:
        return [int(x) for x in np.bincount(labels)]
    sizes_best = _sizes(km_best.labels_)
    sizes_k5 = _sizes(km_k5.labels_)

    comparison = {
        "v1": {
            "best_k": v1_summary["best_k_kmeans"],
            "best_silhouette_subsample": v1_summary["best_silhouette_kmeans"],
            "k5_silhouette_subsample": v1_k5_sil,
            "n_features": v1_summary["n_features_cluster"],
        },
        "v2": {
            "best_k": best_k,
            "best_silhouette_subsample": best_sil,
            "best_silhouette_fullfit": sil_best_all,
            "k5_silhouette_fullfit": sil_k5_all,
            "n_features": len(CLUSTER_FEATURES_V2),
            "features": CLUSTER_FEATURES_V2,
            "cluster_sizes_best": sizes_best,
            "cluster_sizes_k5": sizes_k5,
            "pca_evr_best": list(evr_best),
            "pca_evr_k5": list(evr_k5),
        },
        "personas_best": {str(k): v for k, v in personas_best.items()},
        "personas_k5": {str(k): v for k, v in personas_k5.items()},
        "elapsed_sec": time.time() - t0,
    }
    with (RESULTS / "summary_v2.json").open("w") as f:
        json.dump(comparison, f, indent=2)

    # Side-by-side table
    print("\n" + "=" * 60, flush=True)
    print("V1 vs V2 side-by-side", flush=True)
    print("=" * 60, flush=True)
    print(f"{'Metric':<32} {'V1':<20} {'V2':<20}", flush=True)
    print(f"{'-' * 32:<32} {'-' * 20:<20} {'-' * 20:<20}", flush=True)
    print(f"{'Best K':<32} {v1_summary['best_k_kmeans']:<20} {best_k:<20}", flush=True)
    print(f"{'Silhouette (best K)':<32} {v1_summary['best_silhouette_kmeans']:<20.4f} {sil_best_all:<20.4f}", flush=True)
    print(f"{'Silhouette @ K=5':<32} {v1_k5_sil:<20.4f} {sil_k5_all:<20.4f}", flush=True)
    print(f"{'# features':<32} {v1_summary['n_features_cluster']:<20} {len(CLUSTER_FEATURES_V2):<20}", flush=True)
    v1_sizes = v1_summary.get("cluster_sizes_best", "n/a")
    print(f"{'Cluster sizes (best K)':<32} {str(v1_sizes):<20} {sizes_best}", flush=True)
    print(f"{'Cluster sizes (K=5)':<32} {'n/a':<20} {sizes_k5}", flush=True)
    print(f"{'total elapsed':<32} {'-':<20} {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
