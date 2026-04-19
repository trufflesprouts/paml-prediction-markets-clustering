"""Quick diagnostics requested during the midpoint check-in."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUR = ROOT / "our_code"
SS = OUR / "screenshots"
SS.mkdir(exist_ok=True)

CLUSTER_FEATURES = [
    "trade_count", "total_volume_usd", "distinct_tokens", "active_days",
    "maker_fraction", "avg_trade_size", "std_trade_size", "trades_per_day",
    "hour_entropy", "buy_fraction", "avg_price", "longshot_ratio",
]

raw = pd.read_csv(DATA / "wallet_features_raw.csv")
norm = pd.read_csv(DATA / "wallet_features_normalized.csv")
labels = np.load(OUR / "results" / "kmeans_labels.npy")

# 1. Correlation heatmap of clustering features (use normalized since log+z has
#    already straightened skew)
sns.set_theme(style="whitegrid", context="notebook")
fig, ax = plt.subplots(figsize=(10, 8))
corr = norm[CLUSTER_FEATURES].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax,
            square=True, cbar_kws={"shrink": 0.7})
ax.set_title("Correlation matrix — 12 clustering features (normalized)")
fig.tight_layout()
fig.savefig(SS / "corr_heatmap.png", dpi=130)
plt.close(fig)
print(f"[1/4] saved {SS / 'corr_heatmap.png'}")

# 2. Distribution of maker_fraction
fig, ax = plt.subplots(figsize=(8, 4))
sns.histplot(raw["maker_fraction"], bins=60, ax=ax)
ax.set_xlabel("maker_fraction")
ax.set_title(f"maker_fraction distribution (n={len(raw):,})")
ax.axvline(0.5, color="red", linestyle="--", alpha=0.5, label="0.5")
ax.legend()
fig.tight_layout()
fig.savefig(SS / "maker_fraction_hist.png", dpi=130)
plt.close(fig)
# numeric summary for context
frac_zero = (raw["maker_fraction"] < 0.05).mean()
frac_one = (raw["maker_fraction"] > 0.95).mean()
frac_mid = ((raw["maker_fraction"] >= 0.3) & (raw["maker_fraction"] <= 0.7)).mean()
print(
    f"[2/4] saved {SS / 'maker_fraction_hist.png'}  "
    f"| <0.05: {frac_zero:.2%}  >0.95: {frac_one:.2%}  in [0.3,0.7]: {frac_mid:.2%}"
)

# 3. Top 100 wallets by total_volume_usd
top100 = raw.sort_values("total_volume_usd", ascending=False).head(100).copy()
top100.to_csv(SS / "top100_by_volume.csv", index=False)
print(f"[3/4] saved {SS / 'top100_by_volume.csv'}  (top vol = ${top100['total_volume_usd'].iloc[0]:,.0f})")

# 4. Cluster sizes × volume quartile
raw_with_labels = raw.copy()
raw_with_labels["cluster"] = labels
raw_with_labels["vol_q"] = pd.qcut(raw_with_labels["total_volume_usd"], q=4,
                                    labels=["Q1 (low)", "Q2", "Q3", "Q4 (high)"])
xt = pd.crosstab(raw_with_labels["cluster"], raw_with_labels["vol_q"])
xt_pct = xt.div(xt.sum(axis=1), axis=0) * 100
print(f"\n[4/4] cluster × volume-quartile")
print("\nCounts:")
print(xt.to_string())
print("\nRow %:")
print(xt_pct.round(1).to_string())
print("\nColumn % (how each quartile distributes across clusters):")
col_pct = xt.div(xt.sum(axis=0), axis=1) * 100
print(col_pct.round(1).to_string())
