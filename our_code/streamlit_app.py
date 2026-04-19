"""Streamlit app for the Polymarket trader-archetype pipeline (V2 defaults).

Three pages:
  1. Explore Wallets — filters, histograms, PCA scatter.
  2. Cluster Wallets — V2 K=5 archetypes by default (loaded from disk);
     can also re-fit KMeans/Hierarchical live for comparison.
  3. Classify a Wallet — KNN prediction for a selected wallet, using V2
     classification features including maker_fraction.

Launch:  uv run streamlit run our_code/streamlit_app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from models import (  # noqa: E402
    HierarchicalClustering,
    KMeansClustering,
    KNNClassifier,
    pca_2d,
    profile_clusters,
    silhouette_score,
    train_val_test_split,
)

ROOT = _HERE.parent
DATA_DIR = ROOT / "our_code" / "data"
RESULTS = _HERE / "results"


# V2 feature split
CLUSTER_FEATURES_V2 = [
    "trade_count", "total_volume_usd", "distinct_tokens",
    "avg_trade_size", "std_trade_size", "trades_per_day",
    "buy_fraction", "avg_price", "longshot_ratio",
    "directional_conviction", "size_variability",
]
CLASSIFY_FEATURES_V2 = [
    "max_trade_size", "trade_size_cv", "win_rate",
    "excess_return", "total_pnl", "resolved_trade_count",
    "maker_fraction",
]
LOG_CLUSTER = {
    "trade_count", "total_volume_usd", "distinct_tokens",
    "avg_trade_size", "std_trade_size", "trades_per_day",
    "size_variability",
}
LOG_CLASSIFY = {"max_trade_size", "resolved_trade_count", "total_pnl"}
CLIP_STD = 5.0


def _log1p_signed(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


def _zscale(values: np.ndarray) -> np.ndarray:
    mu, sd = values.mean(), values.std(ddof=0)
    return (values - mu) / sd if sd > 0 else np.zeros_like(values)


@st.cache_data(show_spinner=True)
def load_raw() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "wallet_features_raw.csv")
    # V2 engineered features
    df["directional_conviction"] = (df["buy_fraction"] - 0.5).abs() * 2.0
    df["size_variability"] = df["trade_size_cv"]
    return df


@st.cache_data(show_spinner=True)
def build_v2_cluster_matrix(raw: pd.DataFrame) -> np.ndarray:
    cols = []
    for col in CLUSTER_FEATURES_V2:
        v = raw[col].to_numpy(dtype=np.float64)
        if col in LOG_CLUSTER:
            v = _log1p_signed(v)
        v = np.clip(_zscale(v), -CLIP_STD, CLIP_STD)
        cols.append(v)
    return np.column_stack(cols)


@st.cache_data(show_spinner=True)
def build_v2_classify_matrix(raw: pd.DataFrame) -> np.ndarray:
    cols = []
    for col in CLASSIFY_FEATURES_V2:
        v = raw[col].to_numpy(dtype=np.float64)
        if col in LOG_CLASSIFY:
            v = _log1p_signed(v)
        v = np.clip(_zscale(v), -CLIP_STD, CLIP_STD)
        cols.append(v)
    return np.column_stack(cols)


@st.cache_data(show_spinner=False)
def load_v2_labels() -> np.ndarray | None:
    p = RESULTS / "kmeans_v2_labels.npy"
    if p.exists():
        return np.load(p)
    return None


@st.cache_data(show_spinner=False)
def load_v2_profile() -> pd.DataFrame | None:
    p = RESULTS / "cluster_profiles_v2.csv"
    if p.exists():
        return pd.read_csv(p)
    return None


@st.cache_data(show_spinner=False)
def load_sweep_v2() -> pd.DataFrame | None:
    p = RESULTS / "kmeans_sweep_v2.csv"
    if p.exists():
        return pd.read_csv(p)
    return None


@st.cache_data(show_spinner=False)
def load_pca_sample(_norm_v2: np.ndarray, n_sample: int = 20000, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = _norm_v2
    if len(X) > n_sample:
        idx = np.random.default_rng(seed).choice(len(X), size=n_sample, replace=False)
    else:
        idx = np.arange(len(X))
    proj, evr = pca_2d(X[idx])
    return proj, evr, idx


@st.cache_resource(show_spinner=True)
def cached_kmeans(k: int, seed: int, X_key: int) -> KMeansClustering:  # X_key via caller
    raise NotImplementedError  # replaced below to accept ndarray


def _km_fit(k: int, seed: int, X: np.ndarray) -> KMeansClustering:
    km = KMeansClustering(n_clusters=k, n_init=3, random_state=seed, init="kmeans++")
    km.fit(X)
    return km


def _hc_fit(k: int, seed: int, X: np.ndarray) -> HierarchicalClustering:
    hc = HierarchicalClustering(n_clusters=k, subsample_size=2000, random_state=seed)
    hc.fit(X)
    return hc


def _sidebar_disclaimer() -> None:
    st.sidebar.markdown("---\n**Disclaimer**: This application does not provide trading advice.")


# ----------------------------------------------------------------------- pages

def page_explore(raw: pd.DataFrame, norm_v2: np.ndarray) -> None:
    st.header("Explore Wallets")
    st.markdown(
        "Inspect the wallet-level feature set before clustering. "
        "Filter by trade count and volume, view feature histograms, "
        "and project all wallets into 2-D via PCA using the V2 clustering features."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        min_trades = st.number_input("Min trades", min_value=10, value=10, step=10)
    with c2:
        min_vol = st.number_input("Min total volume (USD)", min_value=0, value=0, step=100)
    with c3:
        max_rows = st.number_input("Max rows to show", min_value=10, value=200, step=50)

    mask = (raw["trade_count"] >= min_trades) & (raw["total_volume_usd"] >= min_vol)
    filtered = raw[mask].copy()
    st.write(f"**{len(filtered):,}** wallets match.")
    st.dataframe(filtered.head(int(max_rows)), use_container_width=True)

    st.subheader("Summary stats")
    display_cols = CLUSTER_FEATURES_V2 + CLASSIFY_FEATURES_V2
    st.dataframe(filtered[display_cols].describe().T)

    st.subheader("Feature histograms")
    feature = st.selectbox("Pick a feature", display_cols)
    log_scale = st.checkbox("log-y axis", value=True)
    fig, ax = plt.subplots(figsize=(7, 3))
    sns.histplot(filtered[feature], bins=60, ax=ax)
    if log_scale:
        ax.set_yscale("log")
    ax.set_title(f"Distribution of {feature}")
    st.pyplot(fig)

    st.subheader("PCA (2D) scatter — V2 features")
    proj, evr, idx = load_pca_sample(norm_v2)
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    ax2.scatter(proj[:, 0], proj[:, 1], s=3, alpha=0.3)
    ax2.set_xlabel(f"PC1 ({evr[0]*100:.1f}%)")
    ax2.set_ylabel(f"PC2 ({evr[1]*100:.1f}%)")
    ax2.set_title(f"PCA projection ({len(idx):,} wallets sampled)")
    st.pyplot(fig2)


def page_cluster(raw: pd.DataFrame, norm_v2: np.ndarray) -> None:
    st.header("Cluster Wallets — V2 archetypes")
    st.markdown(
        "Default view shows the pre-computed V2 K-Means K=5 archetypes. "
        "You can also re-fit K-Means or Hierarchical live at a different K for comparison."
    )

    v2_labels = load_v2_labels()
    v2_profile = load_v2_profile()
    sweep_v2 = load_sweep_v2()

    mode = st.radio(
        "Cluster source",
        ["Load V2 K=5 (default)", "Re-fit live"],
        horizontal=True,
    )

    if mode == "Load V2 K=5 (default)" and v2_labels is not None:
        labels_display = v2_labels
        X_display = norm_v2
        raw_display = raw
        st.info(f"Loaded {len(np.unique(v2_labels))} V2 archetypes from disk "
                f"({RESULTS / 'kmeans_v2_labels.npy'}).")
    else:
        algo = st.radio("Algorithm", ["K-Means", "Hierarchical (Ward)"], horizontal=True)
        k = st.slider("K", min_value=2, max_value=10, value=5)
        seed = int(st.number_input("Random seed", value=42, step=1))
        n_sub = st.slider(
            "Wallets to cluster (sub-sample)",
            min_value=2000, max_value=min(50000, len(norm_v2)),
            value=min(20000, len(norm_v2)), step=1000,
        )
        rng = np.random.default_rng(seed)
        sidx = rng.choice(len(norm_v2), size=int(n_sub), replace=False)
        X_display = norm_v2[sidx]
        raw_display = raw.iloc[sidx].reset_index(drop=True)
        if algo == "K-Means":
            model = _km_fit(int(k), seed, X_display)
        else:
            model = _hc_fit(int(k), seed, X_display)
        labels_display = model.labels_

    sil = silhouette_score(
        X_display, labels_display, sample_size=min(3000, len(X_display)), random_state=0
    )
    st.metric("Silhouette (sample)", f"{sil:.4f}")

    # PCA scatter colored by cluster
    proj, evr = pca_2d(X_display)
    K = int(labels_display.max()) + 1
    fig, ax = plt.subplots(figsize=(6, 5))
    palette = sns.color_palette("husl", K)
    for c in range(K):
        mask = labels_display == c
        ax.scatter(proj[mask, 0], proj[mask, 1], s=5, alpha=0.5,
                   color=palette[c], label=f"C{c} (n={int(mask.sum()):,})")
    ax.set_xlabel(f"PC1 ({evr[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({evr[1]*100:.1f}%)")
    ax.set_title(f"V2 clusters in PCA space  —  K={K}")
    ax.legend(loc="best", markerscale=3, fontsize=8)
    st.pyplot(fig)

    st.subheader("Archetype personas")
    if mode == "Load V2 K=5 (default)" and v2_profile is not None:
        v2_best = v2_profile[v2_profile["model"].str.startswith("V2_bestK")]
        for _, row in v2_best.iterrows():
            st.markdown(f"- {row['persona']}")
        st.dataframe(v2_best.drop(columns=["model"]), use_container_width=True)
    else:
        profile, personas = profile_clusters(raw_display, labels_display, CLUSTER_FEATURES_V2)
        for c in sorted(personas):
            st.markdown(f"- {personas[c]}")
        st.dataframe(profile, use_container_width=True)

    if sweep_v2 is not None:
        st.subheader("K-Means V2 sweep")
        fig3, ax3 = plt.subplots(1, 2, figsize=(10, 3))
        ax3[0].plot(sweep_v2["k"], sweep_v2["silhouette"], "o-")
        ax3[0].set_xlabel("K")
        ax3[0].set_ylabel("silhouette")
        ax3[0].set_title("Silhouette vs K (V2)")
        ax3[0].axvline(int(sweep_v2.loc[sweep_v2.silhouette.idxmax(), "k"]),
                       color="red", ls="--", alpha=0.5)
        ax3[1].plot(sweep_v2["k"], sweep_v2["inertia"], "o-")
        ax3[1].set_xlabel("K")
        ax3[1].set_ylabel("inertia")
        ax3[1].set_title("Elbow plot (V2)")
        st.pyplot(fig3)


def page_classify(raw: pd.DataFrame, norm_v2: np.ndarray, classify_v2: np.ndarray) -> None:
    st.header("Classify a Wallet — V2")
    st.markdown(
        "Uses V2 K=5 archetype labels as targets and the V2 held-out feature set "
        "(includes `maker_fraction`) as KNN inputs."
    )

    v2_labels = load_v2_labels()
    if v2_labels is None:
        st.error("V2 labels not found. Run `our_code/run_experiments_v2.py` first.")
        return

    c1, c2 = st.columns(2)
    with c1:
        knn_k = st.select_slider("KNN k", options=[3, 5, 7, 9, 11, 13, 15, 17, 19], value=5)
    with c2:
        seed = int(st.number_input("Seed", value=42, step=1))

    rng = np.random.default_rng(seed)
    N = min(30_000, len(raw))
    sidx = rng.choice(len(raw), size=N, replace=False)
    Xc = classify_v2[sidx]
    y = v2_labels[sidx]
    raw_sub = raw.iloc[sidx].reset_index(drop=True)

    tr, va, te = train_val_test_split(N, ratios=(0.7, 0.15, 0.15), random_state=seed)
    knn = KNNClassifier(k=int(knn_k)).fit(Xc[tr], y[tr])

    # pick a wallet from the test set
    st.subheader("Pick a wallet (from test set)")
    wallet_list = raw_sub.iloc[te]["wallet"].tolist()[:500]
    pick = st.selectbox("Wallet address", options=wallet_list)

    row_abs = int(raw_sub.index[raw_sub["wallet"] == pick][0])
    x = Xc[row_abs].reshape(1, -1)
    prob = knn.predict_proba(x)[0]
    pred = int(np.argmax(prob))
    conf = float(prob[pred])
    true_label = int(y[row_abs])

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Predicted archetype", f"C{pred}")
    col_b.metric("Confidence", f"{conf:.2%}")
    col_c.metric("Reference cluster", f"C{true_label}")

    # saved persona for predicted cluster
    v2_profile = load_v2_profile()
    if v2_profile is not None:
        v2_best = v2_profile[v2_profile["model"].str.startswith("V2_bestK")]
        persona_row = v2_best[v2_best["cluster"] == pred]
        if len(persona_row):
            st.markdown(f"**Persona:** {persona_row.iloc[0]['persona']}")

    # feature comparison
    st.subheader("Feature comparison: wallet vs cluster mean (raw)")
    display_cols = CLUSTER_FEATURES_V2 + CLASSIFY_FEATURES_V2
    cluster_mean = (
        raw_sub.assign(__c=y)
        .groupby("__c")[display_cols]
        .mean()
        .loc[pred]
    )
    wallet_row = raw_sub.iloc[row_abs][display_cols]
    comp = pd.DataFrame({"wallet": wallet_row, "cluster_mean": cluster_mean})
    st.dataframe(comp, use_container_width=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    x_pos = np.arange(len(comp))
    w = 0.4
    ax.bar(x_pos - w / 2, comp["wallet"].values, w, label="wallet")
    ax.bar(x_pos + w / 2, comp["cluster_mean"].values, w, label="cluster mean")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(comp.index, rotation=45, ha="right")
    ax.set_yscale("symlog")
    ax.legend()
    ax.set_title("Raw features (symlog y)")
    fig.tight_layout()
    st.pyplot(fig)


def main() -> None:
    st.set_page_config(page_title="Polymarket Trader Archetypes (V2)", layout="wide")
    st.title("Polymarket Trader Archetypes — V2")

    try:
        raw = load_raw()
    except FileNotFoundError:
        st.error("Feature files not found. Run `uv run python our_code/feature_extraction.py` first.")
        return

    norm_v2 = build_v2_cluster_matrix(raw)
    classify_v2 = build_v2_classify_matrix(raw)

    page = st.sidebar.radio(
        "Page", ["Explore Wallets", "Cluster Wallets", "Classify a Wallet"]
    )
    _sidebar_disclaimer()

    if page == "Explore Wallets":
        page_explore(raw, norm_v2)
    elif page == "Cluster Wallets":
        page_cluster(raw, norm_v2)
    else:
        page_classify(raw, norm_v2, classify_v2)


if __name__ == "__main__":
    main()
