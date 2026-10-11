"""Aggregation of out-of-sample predictions into metric tables with cluster-bootstrap uncertainty."""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.evaluation import metrics as M

KEY = ["player_id", "test_year"]


def pooled_metrics(pred: pd.DataFrame, n_boot: int = 1000, seed: int = 0) -> dict:
    y, p = pred["pi_next"].to_numpy(), pred["yhat"].to_numpy()
    r, lo, hi = M.cluster_bootstrap(lambda i: M.rmse(y[i], p[i]), pred["player_id"].to_numpy(), n_boot, seed)
    return {"n": len(pred), "n_players": pred["player_id"].nunique(), "rmse": r, "rmse_lo": lo, "rmse_hi": hi,
            "mae": M.mae(y, p), "spearman": M.spearman(y, p)}


def summary_table(pred: pd.DataFrame, by=("name",), n_boot: int = 1000) -> pd.DataFrame:
    rows = []
    for k, g in pred.groupby(list(by)):
        k = k if isinstance(k, tuple) else (k,)
        rows.append({**dict(zip(by, k)), **pooled_metrics(g, n_boot)})
    return pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)


def paired_rmse_diff(pred: pd.DataFrame, a: str, b: str, n_boot: int = 2000, seed: int = 0) -> dict:
    """RMSE(a) - RMSE(b) on identical rows; negative favours a. Cluster bootstrap over players."""
    pa = pred[pred.name == a].set_index(KEY).sort_index()
    pb = pred[pred.name == b].set_index(KEY).sort_index()
    idx = pa.index.intersection(pb.index)
    pa, pb = pa.loc[idx], pb.loc[idx]
    y = pa["pi_next"].to_numpy()
    ea, eb = pa["yhat"].to_numpy(), pb["yhat"].to_numpy()
    clusters = idx.get_level_values("player_id").to_numpy()
    d, lo, hi = M.cluster_bootstrap(lambda i: M.rmse(y[i], ea[i]) - M.rmse(y[i], eb[i]), clusters, n_boot, seed)
    return {"a": a, "b": b, "n": len(idx), "rmse_diff": d, "lo": lo, "hi": hi,
            "verdict": "a better" if hi < 0 else ("b better" if lo > 0 else "no demonstrated difference")}


def per_fold(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (name, ty), g in pred.groupby(["name", "test_year"]):
        rows.append({"name": name, "test_year": ty, "n": len(g), "rmse": M.rmse(g.pi_next, g.yhat),
                     "mae": M.mae(g.pi_next, g.yhat), "spearman": M.spearman(g.pi_next, g.yhat),
                     "small_fold": len(g) < 15})
    return pd.DataFrame(rows)


def ranking_table(pred: pd.DataFrame, ks=(3, 5)) -> pd.DataFrame:
    """Recruitment-style metrics per test fold: all eligible outfield candidates of the fold are ranked."""
    rows = []
    for (name, ty), g in pred.groupby(["name", "test_year"]):
        rel = M.relevance_from_values(g.pi_next)
        score = g["score"].to_numpy() if g["score"].notna().all() else g["yhat"].to_numpy()
        oracle = g["pi_next"].to_numpy()
        for k in ks:
            rows.append({"name": name, "test_year": ty, "k": k, "n": len(g),
                         "ndcg": M.ndcg_at_k(rel, score, k), "shortlist_mean_outcome": M.shortlist_outcome(oracle, score, k),
                         "cohort_mean_outcome": float(oracle.mean()), "overlap_with_true_topk": M.topk_overlap(score, oracle, k)})
    return pd.DataFrame(rows)


def ranking_summary(rt: pd.DataFrame) -> pd.DataFrame:
    g = rt.groupby(["name", "k"]).agg(folds=("test_year", "nunique"), ndcg=("ndcg", "mean"),
                                        shortlist=("shortlist_mean_outcome", "mean"),
                                        cohort=("cohort_mean_outcome", "mean"),
                                        overlap=("overlap_with_true_topk", "mean")).reset_index()
    g["shortlist_lift"] = g["shortlist"] - g["cohort"]
    return g
