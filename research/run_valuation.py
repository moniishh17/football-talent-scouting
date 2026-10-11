"""Secondary target: 12-month change in log Transfermarkt market value (Design A players).  Run: python -m research.run_valuation

Label  y = log(V[t + 12 m] / V[t]) with V[t] = last valuation at/before the scouting date (end of season t, <= 200 days
old) and V[t+12 m] = valuation closest to season_end + 365 d within +/- 120 d. Valuation is a *crowd-sourced proxy*,
not transfer profit or ability. Only tier-A crosswalk players are used.
Rolling origin: test scouting season y; training rows t <= y-1 (their 12-month labels are observed by the end of y).
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge

from research.config import PROCESSED, SEED
from research.evaluation import metrics as M
from research.evaluation import rolling as R
from research.features import panel as PN
from research.targets import performance_index as P

FIRST_TEST, LAST_TEST, DEV_MAX = 2012, 2020, 2016
ZERO_BASELINES = ["zero_change", "mean_change", "age_pos_value"]
FEATURE_SETS = {
    "perf": "pi+profile",
    "perf+agepos": "pi+profile+agepos",
    "perf+agepos+exposure+context": "pi+profile+agepos+exposure+context",
    "valuation_only": "valuation",
    "agepos+valuation": "agepos+valuation",
    "perf+agepos+valuation": "pi+profile+agepos+valuation",
    "perf+agepos+valuation+trend": "pi+profile+agepos+valuation+valtrend",
}


def prev_value(df: pd.DataFrame, val: pd.DataFrame, lag_days: int = 365, max_age_days: int = 200) -> pd.Series:
    by = {k: g for k, g in val.groupby("tm_id")}
    out = []
    for r in df.itertuples():
        g = by.get(r.tm_id)
        if g is None:
            out.append(np.nan); continue
        cut = r.season_end - pd.Timedelta(days=lag_days)
        past = g[g.date <= cut]
        out.append(float(past.market_value_in_eur.iloc[-1]) if len(past) and (cut - past.date.iloc[-1]).days <= max_age_days else np.nan)
    return pd.Series(out, index=df.index)


def build_panel() -> pd.DataFrame:
    a = pd.read_parquet(PROCESSED / "design_a_player_seasons.parquet")
    a = a[(a.position_group != "GK") & a.tm_id.notna()].copy()
    val = PN.load_valuations()
    a = PN.attach_valuation(a, val)
    a["value_prev"] = prev_value(a, val)
    return a.reset_index(drop=True)


def _extend_design(X: pd.DataFrame, df: pd.DataFrame, parts: list[str]) -> pd.DataFrame:
    if "valtrend" in parts:
        tr = np.log(df["value_t"] / df["value_prev"])
        X["val_trend"] = tr.fillna(0.0)
        X["val_trend_missing"] = tr.isna().astype(float)
    return X


def matrix(df: pd.DataFrame, fs: str) -> pd.DataFrame:
    parts = fs.split("+")
    base = "+".join(p for p in parts if p != "valtrend")
    X = R.design_matrix(df, base)
    return _extend_design(X, df, parts)


def run(first=FIRST_TEST, last=LAST_TEST, min_minutes=450.0, seed=SEED) -> pd.DataFrame:
    panel = build_panel()
    cfg = R.Config(min_minutes=min_minutes)
    out = []
    zoo = R.model_zoo(seed)
    for y in range(first, last + 1):
        seen = panel[panel.season_year <= y]
        ref = P.fit_reference(seen, min_minutes, "position_group", shrink=cfg.shrink)
        d = P.performance_index(panel[panel.season_year <= y], ref)
        d = d[(d.minutes >= min_minutes) & d.age.notna() & d.value_t.notna() & d.value_future.notna()].copy()
        d["y"] = np.log(d.value_future / d.value_t)
        d["pi_prev_f"], d["has_prev"], d["minutes_prev_f"] = d["pi"], 0.0, d["minutes"]
        d = d.reset_index(drop=True)
        train, test = d[d.season_year <= y - 1], d[d.season_year == y]
        if len(train) < 25 or len(test) == 0:
            continue
        rows = []

        def emit(name, yhat, params=""):
            r = test[["player_id", "player_name", "season_year", "position_group", "age", "minutes", "value_t", "pi"]].copy()
            r["y"], r["yhat"], r["name"], r["test_year"], r["n_train"], r["params"] = test["y"].to_numpy(), yhat, name, y, len(train), params
            rows.append(r)

        emit("zero_change", np.zeros(len(test)))
        emit("mean_change", np.full(len(test), train.y.mean()))
        Xa, Xb = matrix(train, "agepos+valuation"), matrix(test, "agepos+valuation")
        emit("age_pos_value", Ridge(alpha=1.0).fit(Xa, train.y).predict(Xb))
        for mname in ("ridge", "enet", "rf", "lgbm"):
            est, grid = zoo[mname]
            for label, fs in FEATURE_SETS.items():
                Xtr, Xte = matrix(train, fs), matrix(test, fs)
                model, params = R.fit_with_tuning(est, grid, Xtr, train.y.to_numpy(), train.season_year.to_numpy() + 1,
                                                  np.zeros(len(train)), train.player_id.to_numpy())
                emit(f"{mname}|{label}", model.predict(Xte), str(params))
        out.extend(rows)
    pred = pd.concat(out, ignore_index=True)
    pred["period"] = np.where(pred.test_year <= DEV_MAX, "development", "confirmation")
    return pred


def main():
    pred = run()
    pred.to_parquet(PROCESSED / "predictions_valuation.parquet", index=False)
    print(pred.shape, pred.groupby("period").test_year.nunique().to_dict())


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    main()
