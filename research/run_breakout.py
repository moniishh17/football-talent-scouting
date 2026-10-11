"""Exploratory breakout classification (label = top 20 % performance improvement within position group).

Pre-specified feasibility gate (reports/03_targets.md): the experiment is *reported as exploratory* when the estimated signal share of
the change score is below 0.3 (split-half noise estimate), because the label is then dominated by measurement noise.
Label threshold: 80th percentile of `delta = PI[t+1] - PI[t]` within position group, computed on TRAINING pairs only.
Run: python -m research.run_breakout --design a|b
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from research.config import PROCESSED, SEED
from research.evaluation import metrics as M
from research.evaluation import rolling as R
from research.run_design_b import config as b_config

try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover
    LGBMClassifier = None

Q = 0.80


def clf_zoo(seed=SEED):
    z = {"logit": (make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)), {"logisticregression__C": [0.03, 0.3, 3.0]}),
         "rf": (RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=seed, n_jobs=1), {"min_samples_leaf": [3, 8, 15]})}
    if LGBMClassifier is not None:
        z["lgbm"] = (LGBMClassifier(n_estimators=120, learning_rate=0.03, num_leaves=4, min_child_samples=10, subsample=0.8,
                                    subsample_freq=1, random_state=seed, verbose=-1, n_jobs=1),
                     {"n_estimators": [80, 200], "min_child_samples": [8, 25]})
    return z


def label(df: pd.DataFrame, thresholds: dict) -> np.ndarray:
    return (df["delta"].to_numpy() >= df["position_group"].map(thresholds).to_numpy()).astype(int)


def run(design: str):
    if design == "a":
        panel = pd.read_parquet(PROCESSED / "design_a_player_seasons.parquet")
        cfg, years = R.Config(), range(2009, 2020)
        fsets = ["pi+profile+agepos", "pi+profile+agepos+exposure+context+history"]
    else:
        panel = pd.read_parquet(PROCESSED / "design_b_blocks.parquet")
        cfg, years = b_config(), range(2, 4)
        fsets = ["pi+profile+agepos+exposure+context+league", "pi+profile+agepos+exposure+context+league+history"]
    rows = []
    for y in years:
        prep = R.prepare_fold(panel, y, cfg)
        if prep is None:
            continue
        train, test, _ = prep
        thr = train.groupby("position_group")["delta"].quantile(Q).to_dict()
        ytr, yte = label(train, thr), label(test, thr)
        base = pd.DataFrame({"player_id": test.player_id, "season_year": test.season_year, "y": yte, "test_year": y,
                             "position_group": test.position_group, "age": test.age})
        # heuristic baselines (scores; higher = more likely breakout)
        for name, sc in (("prevalence", np.full(len(test), ytr.mean())), ("low_current_pi", -test["pi"].to_numpy()),
                         ("young_age", -test["age"].to_numpy()), ("persistence_pi", test["pi"].to_numpy()),
                         ("mcda_weighted_sum_inverse", -test["mcda_weighted_sum"].fillna(0).to_numpy())):
            rows.append(base.assign(name=name, p=sc, is_prob=(name == "prevalence")))
        for mname, (est, grid) in clf_zoo().items():
            for fs in fsets:
                Xtr, Xte = R.design_matrix(train, fs), R.design_matrix(test, fs)
                best, bp = (np.inf, {})
                from sklearn.model_selection import GroupKFold
                folds = list(GroupKFold(n_splits=4).split(Xtr, ytr, train["player_id"]))
                for p in R._grid(grid):
                    err = []
                    for a, b in folds:
                        if ytr[a].min() == ytr[a].max():
                            continue
                        pr = clone(est).set_params(**p).fit(Xtr.iloc[a], ytr[a]).predict_proba(Xtr.iloc[b])[:, 1]
                        err.append(np.mean((pr - ytr[b]) ** 2))
                    if err and np.mean(err) < best:
                        best, bp = np.mean(err), p
                m = clone(est).set_params(**bp).fit(Xtr, ytr)
                rows.append(base.assign(name=f"{mname}|{fs}", p=m.predict_proba(Xte)[:, 1], is_prob=True))
    return pd.concat(rows, ignore_index=True)


def summarize(pred: pd.DataFrame, k_frac: float = 0.2) -> pd.DataFrame:
    out = []
    for name, g in pred.groupby("name"):
        y, p = g["y"].to_numpy(), g["p"].to_numpy()
        prev = y.mean()
        k = max(1, int(round(k_frac * len(g))))
        lo_hi = M.cluster_bootstrap(lambda i: M.pr_auc(y[i], p[i]), g["player_id"].to_numpy(), 500, 0)
        out.append({"name": name, "n": len(g), "prevalence": prev, "pr_auc": lo_hi[0], "pr_auc_lo": lo_hi[1], "pr_auc_hi": lo_hi[2],
                    "roc_auc": M.roc_auc(y, p), "precision_at_20pct": M.precision_at_k(y, p, k), "recall_at_20pct": M.recall_at_k(y, p, k),
                    "brier": M.brier(y, p) if g["is_prob"].iloc[0] else np.nan,
                    "brier_skill": M.brier_skill(y, p, prev) if g["is_prob"].iloc[0] else np.nan})
    return pd.DataFrame(out).sort_values("pr_auc", ascending=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--design", choices=["a", "b"], default="a")
    a = ap.parse_args()
    pred = run(a.design)
    pred.to_parquet(PROCESSED / f"predictions_breakout_{a.design}.parquet", index=False)
    pd.set_option("display.width", 200)
    print(summarize(pred).round(3).to_string())
