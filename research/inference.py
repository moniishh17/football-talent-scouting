"""Freeze the development-selected Design-A model, save artifacts, and score new player-seasons.

fit_and_freeze():  fit reference statistics and the selected model on ALL transitions observed up to the latest
                    scouting season, derive conformal half-widths from the rolling out-of-sample residuals, and save
                    artifacts/design_a_frozen/{model.joblib, reference.joblib, metadata.json}.
score():            apply a frozen model to player-season rows (needs only data available at the scouting date).
No test-period outcome influences the choice of model: the configuration comes from reports/tables/selected_configs.json
(chosen on the development period only).
"""
from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from research.config import ARTIFACTS, PROCESSED, REPORTS, SEED
from research.evaluation import rolling as R
from research.targets import performance_index as P

FROZEN_DIR = ARTIFACTS / "design_a_frozen"
COMPONENT_LABELS = {"npxg_p90": "non-penalty xG", "xa_p90": "expected assists", "np_goals_p90": "non-penalty goals",
                    "key_passes_p90": "key passes", "dribbles_completed_p90": "completed dribbles",
                    "carries_progressive_p90": "progressive carries", "passes_progressive_p90": "progressive passes",
                    "passes_into_box_p90": "passes into box", "def_actions_p90": "defensive actions",
                    "def_actions_mid_p90": "ball-winning actions", "pass_completion": "pass completion",
                    "dribbled_past_p90": "rarely dribbled past", "aerials_lost_p90": "few aerial duels lost"}


def _model_spec() -> tuple[str, str, str]:
    sel = json.loads((REPORTS / "tables" / "selected_configs.json").read_text())["A"]["overall"]
    family, fs, form = sel.split("|")
    return family, fs, form


def conformal_halfwidth(pred_path: Path, name: str, alpha: float) -> tuple[float, int]:
    p = pd.read_parquet(pred_path)
    res = (p[p.name == name].pi_next - p[p.name == name].yhat).abs().to_numpy()
    n = len(res)
    return float(np.quantile(res, min(1.0, np.ceil((n + 1) * (1 - alpha)) / n))), n


def fit_and_freeze(panel: pd.DataFrame | None = None, out_dir: Path = FROZEN_DIR, latest_year: int | None = None) -> dict:
    panel = pd.read_parquet(PROCESSED / "design_a_player_seasons.parquet") if panel is None else panel
    family, fs, form = _model_spec()
    cfg = R.Config()
    latest_year = int(panel.season_year.max()) if latest_year is None else latest_year
    seen = panel[panel.season_year <= latest_year]
    ref = P.fit_reference(seen[seen.position_group.isin(P.OUTFIELD)], cfg.min_minutes, cfg.group_col, shrink=cfg.shrink)
    pairs = R.build_pairs(panel, ref, cfg, max_year=latest_year)
    train = pairs[pairs.target_year <= latest_year].reset_index(drop=True)
    est, grid = R.model_zoo(SEED)[family]
    y = (train["pi_next"] if form == "level" else train["delta"]).to_numpy()
    base = np.zeros(len(train)) if form == "level" else train["pi"].to_numpy()
    X = R.design_matrix(train, fs)
    model, params = R.fit_with_tuning(est, grid, X, y, train["target_year"].to_numpy(), base, train["player_id"].to_numpy())
    name = f"{family}|{fs}|{form}"
    pred_path = PROCESSED / "predictions_design_a.parquet"
    q80, n_cal = conformal_halfwidth(pred_path, name, 0.2)
    q90, _ = conformal_halfwidth(pred_path, name, 0.1)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / "model.joblib")
    joblib.dump(ref, out_dir / "reference.joblib")
    data_hash = hashlib.sha256(pd.util.hash_pandas_object(panel[["player_id", "season_year", "minutes"]], index=False).values.tobytes()).hexdigest()
    meta = {"model": name, "family": family, "feature_set": fs, "formulation": form, "tuned_params": params, "seed": SEED,
            "training_pairs": int(len(train)), "training_target_years": [int(train.target_year.min()), int(train.target_year.max())],
            "latest_scouting_season_start_year": latest_year, "min_minutes": cfg.min_minutes,
            "reference_group_col": cfg.group_col, "conformal_halfwidth_80": q80, "conformal_halfwidth_90": q90,
            "conformal_calibration_n": n_cal, "selection_rule": "lowest pooled RMSE in the development period (test years 2009-2014)",
            "interpretation": "development-selected model is a linear persistence model on the performance index; "
                              "richer feature sets did not demonstrably improve held-out RMSE (see reports/05_results.md).",
            "panel_hash_sha256": data_hash, "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                                                          "pandas": pd.__version__, "numpy": np.__version__}}
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=1))
    return meta


def load_frozen(out_dir: Path = FROZEN_DIR) -> dict:
    return {"model": joblib.load(out_dir / "model.joblib"), "ref": joblib.load(out_dir / "reference.joblib"),
            "meta": json.loads((out_dir / "metadata.json").read_text())}


def score(frozen: dict, df: pd.DataFrame, min_minutes: float | None = None) -> pd.DataFrame:
    """Predict next-season index for outfield player-seasons in `df` (one row per player-season, as built by
    research.features.panel). Uses only information in `df` plus the frozen reference/model."""
    meta, ref, model = frozen["meta"], frozen["ref"], frozen["model"]
    mm = meta["min_minutes"] if min_minutes is None else min_minutes
    d = df[df.position_group.isin(P.OUTFIELD) & (df.minutes >= mm)].copy()
    d = P.performance_index(d, ref)
    need_age = "age" in meta["feature_set"]
    if need_age:
        d = d[d.age.notna()]
    d["pi_prev_f"], d["has_prev"], d["minutes_prev_f"] = d["pi"], 0.0, d["minutes"]
    X = R.design_matrix(d, meta["feature_set"])
    yhat = model.predict(X) + (d["pi"].to_numpy() if meta["formulation"] == "delta" else 0.0)
    d["pred_next_pi"] = yhat
    d["pred_change"] = yhat - d["pi"]
    for lvl, key in ((80, "conformal_halfwidth_80"), (90, "conformal_halfwidth_90")):
        d[f"pred_lo_{lvl}"] = yhat - meta[key]
        d[f"pred_hi_{lvl}"] = yhat + meta[key]
    return d.reset_index(drop=True)


def explain(row: pd.Series, position_group: str, top: int = 2) -> str:
    comps = P.COMPONENTS[position_group]
    contrib = {m: sign * row[f"z_{m}"] for m, sign in comps.items()}
    best = sorted(contrib.items(), key=lambda kv: -kv[1])[:top]
    worst = min(contrib.items(), key=lambda kv: kv[1])
    s = ", ".join(f"{COMPONENT_LABELS[m]} ({v:+.1f} sd)" for m, v in best)
    return f"Index driven by {s}; weakest component {COMPONENT_LABELS[worst[0]]} ({worst[1]:+.1f} sd)."
