"""Evaluation metrics (regression, classification, recruitment ranking) and paired cluster bootstrap.

All functions are pure NumPy/SciPy so they can be unit-tested against hand-computed values.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import stats
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


# ---------------- regression ----------------
def mae(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.mean(np.abs(y - p)))


def rmse(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.sqrt(np.mean((y - p) ** 2)))


def spearman(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    if len(y) < 3 or np.ptp(y) == 0 or np.ptp(p) == 0:
        return float("nan")
    return float(stats.spearmanr(y, p)[0])


def interval_coverage(y, lo, hi) -> float:
    y, lo, hi = (np.asarray(a, float) for a in (y, lo, hi))
    return float(np.mean((y >= lo) & (y <= hi)))


# ---------------- ranking / recruitment ----------------
def ndcg_at_k(relevance_true, score, k: int = 10) -> float:
    """NDCG@k with *gain = relevance*; relevance must be non-negative (shift/scale before calling).
    Ties in `score` are broken pessimistically-neutral by averaging over stable order (stable sort)."""
    rel = np.asarray(relevance_true, float)
    s = np.asarray(score, float)
    if rel.min() < 0:
        raise ValueError("relevance must be non-negative")
    k = min(k, len(rel))
    order = np.argsort(-s, kind="stable")[:k]
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = float(np.sum(rel[order] * disc))
    ideal = float(np.sum(np.sort(rel)[::-1][:k] * disc))
    return dcg / ideal if ideal > 0 else float("nan")


def relevance_from_values(values) -> np.ndarray:
    """Non-negative gains: values shifted so the cohort minimum is 0 (rank-preserving, scale-free)."""
    v = np.asarray(values, float)
    return v - v.min()


def topk_overlap(score_a, score_b, k: int) -> float:
    a = set(np.argsort(-np.asarray(score_a, float), kind="stable")[:k])
    b = set(np.argsort(-np.asarray(score_b, float), kind="stable")[:k])
    return len(a & b) / k


def precision_at_k(y_bin, score, k: int) -> float:
    y_bin, score = np.asarray(y_bin), np.asarray(score, float)
    k = min(k, len(y_bin))
    top = np.argsort(-score, kind="stable")[:k]
    return float(np.mean(y_bin[top]))


def recall_at_k(y_bin, score, k: int) -> float:
    y_bin, score = np.asarray(y_bin), np.asarray(score, float)
    pos = y_bin.sum()
    if pos == 0:
        return float("nan")
    top = np.argsort(-score, kind="stable")[: min(k, len(y_bin))]
    return float(y_bin[top].sum() / pos)


def shortlist_outcome(y_future, score, k: int) -> float:
    """Mean realised outcome of the k highest-scored candidates (outcome quality of a shortlist)."""
    y, s = np.asarray(y_future, float), np.asarray(score, float)
    return float(np.mean(y[np.argsort(-s, kind="stable")[: min(k, len(y))]]))


def rank_stability(score_a, score_b) -> float:
    """Spearman rank correlation between two score vectors on the same candidates (e.g. two seeds / folds)."""
    return spearman(score_a, score_b)


# ---------------- classification ----------------
def pr_auc(y_bin, p) -> float:
    return float(average_precision_score(y_bin, p)) if 0 < np.sum(y_bin) < len(y_bin) else float("nan")


def roc_auc(y_bin, p) -> float:
    return float(roc_auc_score(y_bin, p)) if 0 < np.sum(y_bin) < len(y_bin) else float("nan")


def brier(y_bin, p) -> float:
    return float(brier_score_loss(y_bin, np.clip(p, 0, 1)))


def brier_skill(y_bin, p, base_rate: float | None = None) -> float:
    base = np.mean(y_bin) if base_rate is None else base_rate
    ref = float(np.mean((np.asarray(y_bin) - base) ** 2))
    return 1.0 - brier(y_bin, p) / ref if ref > 0 else float("nan")


def calibration_table(y_bin, p, bins: int = 5) -> list[dict]:
    y_bin, p = np.asarray(y_bin, float), np.asarray(p, float)
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p > lo) & (p <= hi) if np.isfinite(lo) else (p <= hi)
        if m.sum():
            rows.append({"mean_pred": float(p[m].mean()), "obs_rate": float(y_bin[m].mean()), "n": int(m.sum())})
    return rows


# ---------------- uncertainty ----------------
def cluster_bootstrap(stat_fn, clusters, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05):
    """Percentile CI of stat_fn(index_array) resampling whole clusters (e.g. players) with replacement.

    stat_fn receives an integer index array into the original rows."""
    clusters = np.asarray(clusters)
    uniq, inv = np.unique(clusters, return_inverse=True)
    members = [np.flatnonzero(inv == i) for i in range(len(uniq))]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([members[i] for i in pick])
        v = stat_fn(idx)
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            vals.append(v)
    vals = np.asarray(vals)
    if len(vals) == 0:
        return float(stat_fn(np.arange(len(clusters)))), float('nan'), float('nan')
    point = stat_fn(np.arange(len(clusters)))
    return float(point), float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))
