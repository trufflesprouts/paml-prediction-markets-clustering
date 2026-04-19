"""From-scratch NumPy implementations of clustering + classification + metrics.

No sklearn.  Only numpy + pandas.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

# Some Apple Silicon BLAS configs emit spurious "divide by zero / overflow / invalid
# in matmul" warnings on large dense products.  The math is fine; the numerical
# noise is not actionable.
warnings.filterwarnings("ignore", message=".*encountered in matmul.*", category=RuntimeWarning)


# --------------------------------------------------------------------------- #
# KMeans                                                                      #
# --------------------------------------------------------------------------- #

@dataclass
class KMeansResult:
    labels: np.ndarray
    centers: np.ndarray
    inertia: float
    n_iter: int
    seed: int


class KMeansClustering:
    """Lloyd's algorithm with k-means++ or random init, multiple restarts."""

    def __init__(
        self,
        n_clusters: int,
        n_init: int = 10,
        max_iter: int = 300,
        tol: float = 1e-4,
        init: str = "random",
        random_state: Optional[int] = None,
    ) -> None:
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.init = init
        self.random_state = random_state

        self.labels_: Optional[np.ndarray] = None
        self.cluster_centers_: Optional[np.ndarray] = None
        self.inertia_: Optional[float] = None
        self.n_iter_: Optional[int] = None
        self.history_: list[KMeansResult] = []

    def _init_centers(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        n = X.shape[0]
        if self.init == "random":
            idx = rng.choice(n, size=self.n_clusters, replace=False)
            return X[idx].copy()
        if self.init == "kmeans++":
            centers = np.empty((self.n_clusters, X.shape[1]), dtype=X.dtype)
            first = rng.integers(0, n)
            centers[0] = X[first]
            d2 = np.sum((X - centers[0]) ** 2, axis=1)
            for k in range(1, self.n_clusters):
                probs = d2 / d2.sum()
                choice = rng.choice(n, p=probs)
                centers[k] = X[choice]
                new_d2 = np.sum((X - centers[k]) ** 2, axis=1)
                d2 = np.minimum(d2, new_d2)
            return centers
        raise ValueError(f"unknown init: {self.init}")

    def _single_run(self, X: np.ndarray, seed: int) -> KMeansResult:
        rng = np.random.default_rng(seed)
        centers = self._init_centers(X, rng)
        prev_inertia = np.inf
        labels = np.zeros(X.shape[0], dtype=np.int32)
        n_iter = 0
        for it in range(self.max_iter):
            # assign
            # ||x - c||^2 = ||x||^2 - 2 x.c + ||c||^2
            x_norm = np.sum(X * X, axis=1)[:, None]
            c_norm = np.sum(centers * centers, axis=1)[None, :]
            d2 = x_norm - 2.0 * X @ centers.T + c_norm
            labels = np.argmin(d2, axis=1).astype(np.int32)
            # compute inertia (sum of chosen distances, clipped at 0)
            inertia = float(np.maximum(d2[np.arange(len(X)), labels], 0).sum())
            # update
            new_centers = np.zeros_like(centers)
            for k in range(self.n_clusters):
                mask = labels == k
                if mask.any():
                    new_centers[k] = X[mask].mean(axis=0)
                else:
                    # re-seed empty cluster to the farthest point
                    far_idx = int(np.argmax(np.min(d2, axis=1)))
                    new_centers[k] = X[far_idx]
            shift = float(np.sqrt(((new_centers - centers) ** 2).sum()))
            centers = new_centers
            n_iter = it + 1
            if abs(prev_inertia - inertia) < self.tol and shift < self.tol:
                break
            prev_inertia = inertia
        return KMeansResult(labels=labels, centers=centers, inertia=inertia, n_iter=n_iter, seed=seed)

    def fit(self, X: np.ndarray) -> "KMeansClustering":
        X = np.asarray(X, dtype=np.float64)
        if self.random_state is None:
            seeds = np.random.SeedSequence().spawn(self.n_init)
            seeds = [int(s.entropy) % (2**31 - 1) for s in seeds]
        else:
            seeds = list(range(self.random_state, self.random_state + self.n_init))

        best: Optional[KMeansResult] = None
        self.history_ = []
        for seed in seeds:
            run = self._single_run(X, seed)
            self.history_.append(run)
            if best is None or run.inertia < best.inertia:
                best = run
        assert best is not None
        self.labels_ = best.labels
        self.cluster_centers_ = best.centers
        self.inertia_ = best.inertia
        self.n_iter_ = best.n_iter
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        c = self.cluster_centers_
        x_norm = np.sum(X * X, axis=1)[:, None]
        c_norm = np.sum(c * c, axis=1)[None, :]
        d2 = x_norm - 2.0 * X @ c.T + c_norm
        return np.argmin(d2, axis=1).astype(np.int32)


# --------------------------------------------------------------------------- #
# Hierarchical (Ward) clustering                                              #
# --------------------------------------------------------------------------- #

class HierarchicalClustering:
    """Ward linkage agglomerative clustering.

    For scalability: if n > subsample_size, fit on a random subsample, then
    assign remaining points via nearest-centroid of subsample-cluster means.
    Implementation is O(n^2) memory for the subsample, which is fine for n=2000.
    """

    def __init__(
        self,
        n_clusters: int,
        subsample_size: int = 2000,
        random_state: int = 0,
    ) -> None:
        self.n_clusters = n_clusters
        self.subsample_size = subsample_size
        self.random_state = random_state

        self.labels_: Optional[np.ndarray] = None
        self.cluster_centers_: Optional[np.ndarray] = None
        self.subsample_idx_: Optional[np.ndarray] = None
        self.subsample_labels_: Optional[np.ndarray] = None

    def _ward_fit(self, X: np.ndarray) -> np.ndarray:
        """Ward's method: at each step merge the two clusters that minimize the
        increase in total within-cluster variance.

        Delta (increase in inertia) for merging clusters a,b with sizes na,nb
        and centroids ca,cb:
            delta = (na*nb / (na+nb)) * ||ca - cb||^2

        We iteratively merge pairs (na, nb, ca, cb) -> one cluster.  Maintain a
        distance matrix D[i,j] = delta between active clusters i,j.

        Lance-Williams update for Ward linkage when merging i,j -> t, and any
        other cluster k:
            d(t,k) = ((n_i + n_k) * d(i,k) + (n_j + n_k) * d(j,k) - n_k * d(i,j))
                     / (n_i + n_j + n_k)
        """
        n = X.shape[0]
        sizes = np.ones(n, dtype=np.float64)
        active = np.ones(n, dtype=bool)

        # initial D[i,j] = 0.5 * ||xi - xj||^2 (Ward's delta for singleton clusters)
        with np.errstate(over="ignore", invalid="ignore"):
            x_norm = np.sum(X * X, axis=1)
            D = x_norm[:, None] - 2.0 * X @ X.T + x_norm[None, :]
        D = 0.5 * np.maximum(D, 0)
        np.fill_diagonal(D, np.inf)

        merges: list[tuple[int, int]] = []
        merges_needed = n - self.n_clusters
        for _ in range(merges_needed):
            flat_idx = int(np.argmin(D))
            i, j = divmod(flat_idx, n)
            if i == j or not np.isfinite(D[i, j]):
                break
            if i > j:
                i, j = j, i
            ni, nj = sizes[i], sizes[j]
            d_ij = D[i, j]

            # vectorized Lance-Williams update over all other active k
            k_mask = active.copy()
            k_mask[i] = False
            k_mask[j] = False
            if k_mask.any():
                nk = sizes[k_mask]
                d_ik = D[i, k_mask]
                d_jk = D[j, k_mask]
                new_d = ((ni + nk) * d_ik + (nj + nk) * d_jk - nk * d_ij) / (ni + nj + nk)
                D[i, k_mask] = new_d
                D[k_mask, i] = new_d

            sizes[i] = ni + nj
            active[j] = False
            D[j, :] = np.inf
            D[:, j] = np.inf
            merges.append((i, j))

        # resolve cluster membership via union-find with path halving
        parent = np.arange(n, dtype=np.int64)
        for (i, j) in merges:
            parent[j] = i
        # path halving
        for _ in range(int(np.log2(n)) + 2):
            parent = parent[parent]

        unique = np.unique(parent)
        remap = {int(u): i for i, u in enumerate(unique)}
        labels = np.array([remap[int(p)] for p in parent], dtype=np.int32)
        return labels

    def fit(self, X: np.ndarray) -> "HierarchicalClustering":
        X = np.asarray(X, dtype=np.float64)
        n = X.shape[0]
        rng = np.random.default_rng(self.random_state)
        if n > self.subsample_size:
            sub_idx = rng.choice(n, size=self.subsample_size, replace=False)
        else:
            sub_idx = np.arange(n)
        X_sub = X[sub_idx]
        sub_labels = self._ward_fit(X_sub)

        # compute centroids from sub
        centers = np.zeros((self.n_clusters, X.shape[1]))
        for k in range(self.n_clusters):
            mask = sub_labels == k
            if mask.any():
                centers[k] = X_sub[mask].mean(axis=0)
            else:
                # should not happen, but fall back to random point
                centers[k] = X_sub[rng.integers(0, len(X_sub))]

        # assign all points to nearest centroid
        x_norm = np.sum(X * X, axis=1)[:, None]
        c_norm = np.sum(centers * centers, axis=1)[None, :]
        d2 = x_norm - 2.0 * X @ centers.T + c_norm
        labels = np.argmin(d2, axis=1).astype(np.int32)

        self.labels_ = labels
        self.cluster_centers_ = centers
        self.subsample_idx_ = sub_idx
        self.subsample_labels_ = sub_labels
        return self


# --------------------------------------------------------------------------- #
# KNN Classifier                                                              #
# --------------------------------------------------------------------------- #

class KNNClassifier:
    """Euclidean-distance KNN with majority voting (ties broken by lowest label)."""

    def __init__(self, k: int = 5) -> None:
        self.k = k
        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.classes_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNNClassifier":
        self.X_train = np.asarray(X, dtype=np.float64)
        self.y_train = np.asarray(y).astype(np.int32)
        self.classes_ = np.sort(np.unique(self.y_train))
        return self

    def _distances(self, X: np.ndarray) -> np.ndarray:
        xn = np.sum(X * X, axis=1)[:, None]
        tn = np.sum(self.X_train * self.X_train, axis=1)[None, :]
        d2 = xn - 2.0 * X @ self.X_train.T + tn
        return np.maximum(d2, 0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        d2 = self._distances(X)
        # take k smallest per row
        idx = np.argpartition(d2, kth=min(self.k, d2.shape[1] - 1), axis=1)[:, : self.k]
        nearest_labels = self.y_train[idx]
        n_classes = self.classes_.max() + 1
        preds = np.zeros(X.shape[0], dtype=np.int32)
        for i in range(X.shape[0]):
            counts = np.bincount(nearest_labels[i], minlength=n_classes)
            preds[i] = int(np.argmax(counts))
        return preds

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        d2 = self._distances(X)
        idx = np.argpartition(d2, kth=min(self.k, d2.shape[1] - 1), axis=1)[:, : self.k]
        nearest_labels = self.y_train[idx]
        n_classes = self.classes_.max() + 1
        probs = np.zeros((X.shape[0], n_classes), dtype=np.float64)
        for i in range(X.shape[0]):
            counts = np.bincount(nearest_labels[i], minlength=n_classes)
            probs[i] = counts / counts.sum()
        return probs


# --------------------------------------------------------------------------- #
# Metrics                                                                     #
# --------------------------------------------------------------------------- #

def silhouette_score(
    X: np.ndarray,
    labels: np.ndarray,
    sample_size: int = 3000,
    random_state: int = 0,
) -> float:
    """Silhouette coefficient.

    For each point i in sample:
        a(i) = mean dist to other points in same cluster
        b(i) = min over other clusters of mean dist to points in that cluster
        s(i) = (b - a) / max(a, b)
    Returns mean s.
    """
    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray(labels).astype(np.int32)
    n = len(X)
    unique = np.unique(labels)
    if len(unique) < 2:
        return 0.0

    rng = np.random.default_rng(random_state)
    if n > sample_size:
        sample = rng.choice(n, size=sample_size, replace=False)
    else:
        sample = np.arange(n)

    Xs = X[sample]
    ys = labels[sample]
    # distances from each sampled point to ALL points in X (for robust cluster stats)
    xs_norm = np.sum(Xs * Xs, axis=1)[:, None]
    all_norm = np.sum(X * X, axis=1)[None, :]
    d = xs_norm - 2.0 * Xs @ X.T + all_norm
    d = np.sqrt(np.maximum(d, 0))

    s = np.zeros(len(sample), dtype=np.float64)
    for i, point_idx in enumerate(sample):
        my_label = ys[i]
        a_mask = labels == my_label
        a_mask[point_idx] = False  # exclude self
        if a_mask.any():
            a = d[i, a_mask].mean()
        else:
            a = 0.0
        b = np.inf
        for other in unique:
            if other == my_label:
                continue
            other_mask = labels == other
            if other_mask.any():
                b_candidate = d[i, other_mask].mean()
                if b_candidate < b:
                    b = b_candidate
        if b == np.inf:
            s[i] = 0.0
        else:
            denom = max(a, b)
            s[i] = (b - a) / denom if denom > 0 else 0.0
    return float(s.mean())


def within_cluster_heterogeneity(X: np.ndarray, labels: np.ndarray) -> float:
    """Mean within-cluster standard deviation (mean across features and clusters).

    A lower value means clusters are tighter.
    """
    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray(labels)
    vals = []
    for k in np.unique(labels):
        mask = labels == k
        if mask.sum() < 2:
            continue
        vals.append(X[mask].std(axis=0, ddof=0).mean())
    if not vals:
        return 0.0
    return float(np.mean(vals))


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float((y_true == y_pred).mean())


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(np.int32)
    y_pred = np.asarray(y_pred).astype(np.int32)
    classes = np.unique(np.concatenate([y_true, y_pred]))
    f1s = []
    for c in classes:
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        if tp + fp == 0 or tp + fn == 0:
            f1s.append(0.0)
            continue
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        if prec + rec == 0:
            f1s.append(0.0)
        else:
            f1s.append(2 * prec * rec / (prec + rec))
    return float(np.mean(f1s))


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: Optional[int] = None) -> np.ndarray:
    y_true = np.asarray(y_true).astype(np.int32)
    y_pred = np.asarray(y_pred).astype(np.int32)
    if n_classes is None:
        n_classes = int(max(y_true.max(), y_pred.max())) + 1
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


# --------------------------------------------------------------------------- #
# Cluster profiling                                                           #
# --------------------------------------------------------------------------- #

def profile_clusters(
    df: pd.DataFrame,
    labels: np.ndarray,
    features: list[str],
) -> tuple[pd.DataFrame, dict[int, str]]:
    """Per-cluster mean of each feature + a one-sentence persona string.

    Persona heuristic: compare each cluster's mean to the global mean,
    pick the top 2 most-elevated and top 1 most-suppressed features.
    """
    df = df.reset_index(drop=True).copy()
    df["__cluster"] = labels

    profile = df.groupby("__cluster")[features].mean()
    global_mean = df[features].mean()
    global_std = df[features].std(ddof=0).replace(0, np.nan)
    z = (profile - global_mean) / global_std

    personas: dict[int, str] = {}
    for k in sorted(profile.index):
        z_row = z.loc[k].dropna().sort_values(ascending=False)
        if len(z_row) == 0:
            personas[int(k)] = f"Cluster {k}: n={int((labels == k).sum())} wallets."
            continue
        top_high = z_row.head(2)
        top_low = z_row.tail(1)
        def _fmt(s: pd.Series, direction: str) -> list[str]:
            out = []
            for feat, zval in s.items():
                out.append(f"{direction} {feat} (z={zval:+.2f})")
            return out
        desc = ", ".join(_fmt(top_high, "high") + _fmt(top_low, "low"))
        n_k = int((labels == k).sum())
        personas[int(k)] = f"Cluster {k} (n={n_k}): {desc}."

    profile = profile.reset_index().rename(columns={"__cluster": "cluster"})
    profile["n"] = [int((labels == k).sum()) for k in profile["cluster"]]
    profile["persona"] = profile["cluster"].map(personas)
    return profile, personas


# --------------------------------------------------------------------------- #
# Train/val/test split                                                        #
# --------------------------------------------------------------------------- #

def train_val_test_split(
    n: int,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    perm = rng.permutation(n)
    n_tr = int(ratios[0] * n)
    n_va = int(ratios[1] * n)
    return perm[:n_tr], perm[n_tr : n_tr + n_va], perm[n_tr + n_va :]


# --------------------------------------------------------------------------- #
# PCA (via SVD)                                                               #
# --------------------------------------------------------------------------- #

def pca_2d(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (2D projection, explained variance ratio [2]).  Uses np.linalg.svd."""
    X = np.asarray(X, dtype=np.float64)
    Xc = X - X.mean(axis=0, keepdims=True)
    # thin SVD: Xc = U S Vt
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    proj = Xc @ Vt[:2].T
    evr = (S ** 2) / (S ** 2).sum()
    return proj, evr[:2]
