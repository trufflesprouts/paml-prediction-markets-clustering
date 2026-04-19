"""V2 KNN classification.

Targets: V2 K-Means K=5 labels (from kmeans_v2_labels.npy).
Features (held out from V2 clustering):
    max_trade_size, trade_size_cv, win_rate, excess_return, total_pnl,
    resolved_trade_count, maker_fraction   (maker_fraction moved here in V2)

Same split logic as V1: 70/15/15 at wallet level, seed=42.
Tune k in {3,5,7,...,19}; pick best by val_macro_f1; report test metrics.
Saves:
    our_code/results/knn_v2_tuning.csv
    our_code/results/knn_v2_metrics.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from models import (  # noqa: E402
    KNNClassifier,
    accuracy,
    confusion_matrix,
    macro_f1,
    train_val_test_split,
)

ROOT = HERE.parent
DATA = ROOT / "our_code" / "data"
RESULTS = HERE / "results"

CLASSIFY_FEATURES_V2 = [
    "max_trade_size",
    "trade_size_cv",
    "win_rate",
    "excess_return",
    "total_pnl",
    "resolved_trade_count",
    "maker_fraction",
]

LOG_FEATURES = {
    "max_trade_size", "resolved_trade_count", "total_pnl",
}
CLIP_STD = 5.0


def log1p_signed(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    out = df[CLASSIFY_FEATURES_V2].copy().astype(np.float64)
    for col in CLASSIFY_FEATURES_V2:
        if col in LOG_FEATURES:
            out[col] = log1p_signed(out[col].to_numpy())
        mu = out[col].mean()
        sd = out[col].std(ddof=0)
        out[col] = (out[col] - mu) / sd if sd > 0 else 0.0
        out[col] = out[col].clip(-CLIP_STD, CLIP_STD)
    return out


def main() -> None:
    raw = pd.read_csv(DATA / "wallet_features_raw.csv")
    y_all = np.load(RESULTS / "kmeans_v2_labels.npy")
    assert len(raw) == len(y_all), f"length mismatch: raw={len(raw)}, labels={len(y_all)}"

    Xn = normalize(raw).to_numpy(dtype=np.float64)
    print(f"[knn v2] wallets={len(raw):,}, features={len(CLASSIFY_FEATURES_V2)}, "
          f"classes={np.bincount(y_all).tolist()}", flush=True)

    # Subsample for KNN (20k train + 5k val + 5k test, same as V1 run)
    rng = np.random.default_rng(123)
    N = 30_000
    idx = rng.choice(len(Xn), size=min(N, len(Xn)), replace=False)
    Xs, ys = Xn[idx], y_all[idx]

    tr, va, te = train_val_test_split(len(Xs), ratios=(0.7, 0.15, 0.15), random_state=42)
    Xtr, ytr = Xs[tr], ys[tr]
    Xva, yva = Xs[va], ys[va]
    Xte, yte = Xs[te], ys[te]
    print(f"[knn v2] split: train={len(tr):,}  val={len(va):,}  test={len(te):,}", flush=True)

    # Baseline: majority-class macro-F1 (and per-class)
    majority = int(np.bincount(ytr).argmax())
    yte_maj = np.full_like(yte, majority)
    baseline_f1 = macro_f1(yte, yte_maj)
    baseline_acc = accuracy(yte, yte_maj)
    print(f"[knn v2] baseline (majority={majority}): acc={baseline_acc:.4f}  macro-F1={baseline_f1:.4f}",
          flush=True)

    # Tune k
    tune_rows = []
    for k in [3, 5, 7, 9, 11, 13, 15, 17, 19]:
        t0 = time.time()
        knn = KNNClassifier(k=k).fit(Xtr, ytr)
        yv = knn.predict(Xva)
        rec = {
            "k": k,
            "val_accuracy": accuracy(yva, yv),
            "val_macro_f1": macro_f1(yva, yv),
            "runtime_sec": time.time() - t0,
        }
        tune_rows.append(rec)
        print(f"  k={k}: val_acc={rec['val_accuracy']:.4f}  val_f1={rec['val_macro_f1']:.4f}  "
              f"t={rec['runtime_sec']:.1f}s", flush=True)

    tune_df = pd.DataFrame(tune_rows)
    tune_df.to_csv(RESULTS / "knn_v2_tuning.csv", index=False)
    best_k = int(tune_df.loc[tune_df["val_macro_f1"].idxmax(), "k"])
    print(f"[knn v2] best k by val_macro_f1 = {best_k}", flush=True)

    # Final test
    knn_best = KNNClassifier(k=best_k).fit(Xtr, ytr)
    ypred = knn_best.predict(Xte)
    cm = confusion_matrix(yte, ypred)
    test_acc = accuracy(yte, ypred)
    test_f1 = macro_f1(yte, ypred)
    print(f"[knn v2] test: acc={test_acc:.4f}  macro-F1={test_f1:.4f}", flush=True)

    metrics = {
        "best_k": best_k,
        "test_accuracy": test_acc,
        "test_macro_f1": test_f1,
        "baseline_majority_accuracy": baseline_acc,
        "baseline_majority_macro_f1": baseline_f1,
        "n_train": int(len(tr)),
        "n_val": int(len(va)),
        "n_test": int(len(te)),
        "n_classes": int(y_all.max() + 1),
        "features": CLASSIFY_FEATURES_V2,
        "target_labels_source": "kmeans_v2_labels.npy",
        "confusion_matrix": cm.tolist(),
        "class_sizes": [int(x) for x in np.bincount(y_all)],
    }
    (RESULTS / "knn_v2_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"[knn v2] wrote {RESULTS / 'knn_v2_metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
