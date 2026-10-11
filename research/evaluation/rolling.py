"""Rolling-origin (expanding-window) evaluation for next-season performance, Design A.

Protocol for test transition  y -> y+1  (scouting date = end of season y):
  * reference statistics (z-score means/sds, shrinkage priors) are fitted on seasons <= y only;
  * training pairs are (t -> t+1) with t+1 <= y, i.e. outcomes already observed by the scouting date;
  * hyper-parameters are chosen by an inner chronological split of the *training* pairs
    (validate on the last two training transitions, fit on earlier ones);
  * the test pairs (t = y) are touched only for the final prediction and metrics.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from research.config import SEED
from research.models import baselines as bl
from research.targets import performance_index as P

try:
    from lightgbm import LGBMRegressor
except Exception:  # pragma: no cover
    LGBMRegressor = None

OUTFIELD = P.OUTFIELD


@dataclass
class Config:
    min_minutes: float = 450.0            # exposure threshold for scouting-season rows
    min_minutes_target: float = 450.0     # exposure threshold for the observed next season
    shrink: bool = False                  # empirical-Bayes shrinkage of the profile rates (features AND target?) -> features only
    shrink_target: bool = False
    group_col: str = "position_group"     # reference grouping: 'position_group' or 'primary_role'
    require_age: bool = True              # identical eligible cohort across feature sets
    first_test_year: int = 2009
    last_test_year: int = 2019
    extra: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------- pairs
def build_pairs(panel: pd.DataFrame, ref: P.Reference, cfg: Config, max_year: int,
                ref_target: P.Reference | None = None) -> pd.DataFrame:
    """One row per (player, t -> t+1) with features at t and targets at t+1, for t+1 <= max_year."""
    d = panel[panel.season_year <= max_year].copy()
    d = d[d.position_group.isin(OUTFIELD)]
    raw = d
    d = P.performance_index(raw, ref)                                # features (ref.shrunk decides)
    # target index is recomputed from the UNMODIFIED rates (performance_index overwrites rate columns when shrinking)
    # the target uses its OWN reference (fitted on the same training window) so that its scale matches its rate type
    tgt = d if ref_target is None or ref_target is ref else P.performance_index(raw, ref_target, use_shrunk=ref_target.shrunk)
    cur = d[d.minutes >= cfg.min_minutes].copy()
    nxt = tgt[tgt.minutes >= cfg.min_minutes_target][["player_id", "season_year", "pi", "minutes", "position_group", "age"]]
    nxt = nxt.assign(season_year=nxt.season_year - 1).rename(
        columns={"pi": "pi_next", "minutes": "minutes_next", "position_group": "pos_next", "age": "age_next"})
    pairs = cur.merge(nxt, on=["player_id", "season_year"], how="inner")
    pairs["delta"] = pairs["pi_next"] - pairs["pi"]
    # history (previous season) -- available at the scouting date
    prev = d[["player_id", "season_year", "pi", "minutes"]].copy()
    prev["season_year"] += 1
    prev = prev.rename(columns={"pi": "pi_prev", "minutes": "minutes_prev"})
    pairs = pairs.merge(prev, on=["player_id", "season_year"], how="left")
    pairs["has_prev"] = pairs["pi_prev"].notna().astype(float)
    pairs["pi_prev_f"] = pairs["pi_prev"].fillna(pairs["pi"])
    pairs["minutes_prev_f"] = pairs["minutes_prev"].fillna(pairs["minutes"])
    pairs["target_year"] = pairs["season_year"] + 1
    if cfg.require_age:
        pairs = pairs[pairs["age"].notna()]
    pairs = pairs.reset_index(drop=True)
    return pairs


def eligible_all(panel: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    return panel


# ----------------------------------------------------------------------------- features
Z_COLS = [f"z_{m}" for m in P.PROFILE]


def design_matrix(df: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    parts = feature_set.split("+")
    for part in parts:
        if part == "pi":
            X["pi"] = df["pi"]
        elif part == "profile":
            for c in Z_COLS:
                X[c] = df[c].fillna(0.0)
        elif part == "agepos":
            X["age"] = df["age"]
            X["age2"] = (df["age"] - 27.0) ** 2
            for g in OUTFIELD:
                X[f"pos_{g}"] = (df["position_group"] == g).astype(float)
        elif part == "exposure":
            X["log_minutes"] = np.log(df["minutes"])
            X["availability"] = df["availability"]
        elif part == "context":
            for c in ("team_poss", "team_gd_pm", "team_xgd_pm"):
                X[c] = df[c]
        elif part == "league":
            for c in (11, 2, 12, 7):
                X[f"league_{c}"] = (df["competition_id"] == c).astype(float)
        elif part == "history":
            X["pi_prev"] = df["pi_prev_f"]
            X["has_prev"] = df["has_prev"]
            X["log_minutes_prev"] = np.log(df["minutes_prev_f"])
        elif part == "valuation":
            X["log_value"] = np.log(df["value_t"].fillna(df["value_t"].median()))
            X["value_missing"] = df["value_t"].isna().astype(float)
        else:
            raise KeyError(part)
    return X


# ----------------------------------------------------------------------------- models
def model_zoo(seed: int = SEED) -> dict:
    """name -> (estimator, param grid). Grids are deliberately tiny (n is small)."""
    zoo = {
        "ridge": (make_pipeline(StandardScaler(), Ridge()), {"ridge__alpha": [1.0, 10.0, 100.0]}),
        "enet": (make_pipeline(StandardScaler(), ElasticNet(max_iter=20000, random_state=seed)),
                 {"elasticnet__alpha": [0.01, 0.05, 0.2], "elasticnet__l1_ratio": [0.2, 0.7]}),
        "rf": (RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=1, max_features=0.5),
               {"min_samples_leaf": [3, 8]}),
    }
    if LGBMRegressor is not None:
        zoo["lgbm"] = (LGBMRegressor(n_estimators=150, learning_rate=0.03, num_leaves=4, min_child_samples=10,
                                     subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=seed,
                                     verbose=-1, n_jobs=1),
                       {"n_estimators": [100, 250], "min_child_samples": [8, 20]})
    return zoo


def _grid(grid: dict):
    keys = list(grid)
    out = [{}]
    for k in keys:
        out = [{**o, k: v} for o in out for v in grid[k]]
    return out


def fit_with_tuning(est, grid, X: pd.DataFrame, y: np.ndarray, years: np.ndarray, base: np.ndarray,
                    groups: np.ndarray | None = None):
    """Choose hyper-parameters on an inner chronological split, then refit on all training pairs.

    `base` is added back to predictions (0 for level targets, pi for delta targets) before scoring RMSE on the
    *level* scale, which is the evaluation scale."""
    uniq = np.unique(years)
    params = {}
    if len(uniq) < 4 and len(grid) and groups is not None and len(y) >= 60:
        # too few training transitions for a chronological inner split (Design B): tune by player-grouped CV
        # inside the training window only (no test-period information is touched).
        from sklearn.model_selection import GroupKFold
        folds = list(GroupKFold(n_splits=4).split(X, y, groups))
        best = (np.inf, {})
        for p in _grid(grid):
            err = np.mean([np.mean((clone(est).set_params(**p).fit(X.iloc[a], y[a]).predict(X.iloc[b]) - y[b]) ** 2)
                           for a, b in folds])
            if err < best[0]:
                best = (err, p)
        params = best[1]
    elif len(uniq) >= 4 and len(grid):
        val_years = uniq[-2:]
        tr, va = np.isin(years, uniq[:-2]), np.isin(years, val_years)
        if tr.sum() >= 25 and va.sum() >= 10:
            best = (np.inf, {})
            for p in _grid(grid):
                m = clone(est).set_params(**p).fit(X[tr], y[tr])
                err = np.sqrt(np.mean((m.predict(X[va]) + base[va] - (y[va] + base[va])) ** 2))
                if err < best[0]:
                    best = (err, p)
            params = best[1]
    model = clone(est).set_params(**params).fit(X, y)
    return model, params


# ----------------------------------------------------------------------------- one fold
def prepare_fold(panel: pd.DataFrame, test_year: int, cfg: Config):
    """Leakage-controlled train/test pair tables for the scouting date at the end of `test_year`."""
    seen = panel[panel.season_year <= test_year]
    ref = P.fit_reference(seen[seen.position_group.isin(OUTFIELD)], cfg.min_minutes, cfg.group_col, shrink=cfg.shrink)
    ref_t = ref if cfg.shrink_target == cfg.shrink else P.fit_reference(
        seen[seen.position_group.isin(OUTFIELD)], cfg.min_minutes, cfg.group_col, shrink=cfg.shrink_target)
    pairs = build_pairs(panel, ref, cfg, max_year=test_year + 1, ref_target=ref_t)
    train = pairs[pairs.target_year <= test_year].reset_index(drop=True)
    test = pairs[pairs.target_year == test_year + 1].reset_index(drop=True)
    if len(test) == 0 or len(train) < 20:
        return None
    # MCDA scores for the legacy baselines (ranked within season x position over eligible rows)
    elig = panel[(panel.season_year <= test_year) & (panel.minutes >= cfg.min_minutes) & panel.position_group.isin(OUTFIELD)
                 & (panel.age.notna() if cfg.require_age else True)].copy()
    elig.index = np.arange(len(elig))
    mc = bl.mcda_scores(elig)
    mce = bl.mcda_scores(elig, bl.equal_weights())
    keyed = elig[["player_id", "season_year"]].reset_index(drop=True)
    for a in bl.LEGACY_ALGOS:
        keyed[f"mcda_{a}"] = mc[a].to_numpy()
    keyed["equal_weight"] = mce["weighted_sum"].to_numpy()
    train = train.merge(keyed, on=["player_id", "season_year"], how="left")
    test = test.merge(keyed, on=["player_id", "season_year"], how="left")
    return train, test, ref


def run_fold(panel: pd.DataFrame, test_year: int, cfg: Config, specs: list[dict], seed: int = SEED) -> pd.DataFrame:
    """specs: [{'name','kind':'ml'|'baseline','feature_set','model','formulation'}] -> predictions frame."""
    prep = prepare_fold(panel, test_year, cfg)
    if prep is None:
        return pd.DataFrame()
    train, test, _ = prep
    zoo = model_zoo(seed)
    out = []
    for s in specs:
        form = s.get("formulation", "level")
        y_tr = (train["pi_next"] if form == "level" else train["delta"]).to_numpy()
        base_tr = np.zeros(len(train)) if form == "level" else train["pi"].to_numpy()
        base_te = np.zeros(len(test)) if form == "level" else test["pi"].to_numpy()
        info = {}
        if s["kind"] == "baseline":
            b = s["model"]
            if b == "mean":
                yhat = np.full(len(test), train["pi_next"].mean())
            elif b == "mean_by_pos":
                m = train.groupby("position_group")["pi_next"].mean()
                yhat = test["position_group"].map(m).fillna(train["pi_next"].mean()).to_numpy()
            elif b == "persistence":
                yhat = test["pi"].to_numpy()
            elif b == "persistence_cal":
                lr = LinearRegression().fit(train[["pi"]], train["pi_next"]); yhat = lr.predict(test[["pi"]])
            elif b == "age_pos":
                Xtr, Xte = design_matrix(train, "agepos"), design_matrix(test, "agepos")
                m = Ridge(alpha=1.0).fit(Xtr, train["pi_next"]); yhat = m.predict(Xte)
            elif b.startswith("mcda_") or b == "equal_weight":
                ok = train[b].notna()
                Xtr = pd.concat([train.loc[ok, [b]], pd.get_dummies(train.loc[ok, "position_group"]).astype(float)], axis=1)
                Xte = pd.concat([test[[b]], pd.get_dummies(test["position_group"]).astype(float)], axis=1).reindex(columns=Xtr.columns, fill_value=0.0)
                m = LinearRegression().fit(Xtr, train.loc[ok, "pi_next"]); yhat = m.predict(Xte.fillna(0.0))
                info["score"] = test[b].to_numpy()
            else:
                raise KeyError(b)
        else:
            est, grid = zoo[s["model"]]
            Xtr, Xte = design_matrix(train, s["feature_set"]), design_matrix(test, s["feature_set"])
            model, params = fit_with_tuning(est, grid, Xtr, y_tr, train["target_year"].to_numpy(), base_tr,
                                            train["player_id"].to_numpy())
            yhat = model.predict(Xte) + base_te
            info["params"] = str(params)
        res = test[["player_id", "player_name", "season_year", "position_group", "primary_role", "age", "minutes",
                    "minutes_next", "pi", "pi_next", "delta", "team_poss", "pos_next"]].copy()
        res["yhat"] = yhat
        res["name"] = s["name"]
        res["formulation"] = form
        res["test_year"] = test_year
        res["n_train"] = len(train)
        res["params"] = info.get("params", "")
        res["score"] = info["score"] if "score" in info else np.nan
        out.append(res)
    return pd.concat(out, ignore_index=True)


def run_all(panel: pd.DataFrame, cfg: Config, specs: list[dict]) -> pd.DataFrame:
    frames = [run_fold(panel, y, cfg, specs) for y in range(cfg.first_test_year, cfg.last_test_year + 1)]
    frames = [f for f in frames if len(f)]
    return pd.concat(frames, ignore_index=True)
