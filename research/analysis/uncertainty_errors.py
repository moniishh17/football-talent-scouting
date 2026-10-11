"""Prediction-interval coverage (split conformal, time-respecting) and subgroup error analysis.

Intervals: for test fold y the half-width is the ceil((n+1)(1-a))/n empirical quantile of |residual| of the SAME model on
strictly EARLIER folds (so calibration data precede the prediction date). Exchangeability across seasons is an
assumption, not a guarantee; empirical coverage is therefore reported, not assumed.
Run: python -m research.analysis.uncertainty_errors
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from research.analysis import results as RS
from research.config import REPORTS
from research.evaluation import metrics as M

TABLES = REPORTS / "tables"


def conformal_intervals(pred: pd.DataFrame, alpha: float = 0.2, min_calib: int = 30, by: str | None = None) -> pd.DataFrame:
    pred = pred.sort_values("test_year").copy()
    lo, hi, used = [], [], []
    years = sorted(pred.test_year.unique())
    for y in years:
        cur = pred[pred.test_year == y]
        past = pred[pred.test_year < y]
        res = (past.pi_next - past.yhat).abs()
        for idx, row in cur.iterrows():
            r = res
            if by is not None:
                r = res[past[by] == row[by]]
                if len(r) < min_calib:
                    r = res
            if len(r) < min_calib:
                lo.append(np.nan); hi.append(np.nan); used.append(len(r)); continue
            q = np.quantile(r, min(1.0, np.ceil((len(r) + 1) * (1 - alpha)) / len(r)))
            lo.append(row.yhat - q); hi.append(row.yhat + q); used.append(len(r))
    out = pred.copy()
    out["lo"], out["hi"], out["n_calib"] = lo, hi, used
    return out


def coverage_table(pred: pd.DataFrame, name: str, alphas=(0.2, 0.1)) -> pd.DataFrame:
    rows = []
    for a in alphas:
        for by in (None, "position_group"):
            c = conformal_intervals(pred[pred.name == name], a, by=by).dropna(subset=["lo"])
            rows.append({"name": name, "nominal": 1 - a, "grouping": by or "global", "n_evaluated": len(c),
                         "empirical_coverage": M.interval_coverage(c.pi_next, c.lo, c.hi),
                         "mean_width": float((c.hi - c.lo).mean())})
    return pd.DataFrame(rows)


def band(s: pd.Series, edges, labels):
    return pd.cut(s, edges, labels=labels, right=False)


def subgroup_errors(p: pd.DataFrame, name: str, extra_group: str | None = None, min_n: int = 20) -> pd.DataFrame:
    g = p[p.name == name].copy()
    g["resid"] = g.pi_next - g.yhat
    g["age_band"] = band(g.age, [0, 23, 26, 29, 99], ["<23", "23-25", "26-28", "29+"])
    g["minutes_band"] = band(g.minutes, [0, 900, 1800, 2700, 1e9], ["<900", "900-1800", "1800-2700", "2700+"])
    g["role"] = g["primary_role"]
    g["coverage_level"] = pd.cut(g["matches_share"], [0, 0.5, 0.8, 1.01], labels=["<50% of team matches", "50-80%", ">80%"], right=False)
    rows = []
    groups = ["position_group", "role", "age_band", "minutes_band", "coverage_level"] + ([extra_group] if extra_group else [])
    for col in groups:
        for k, d in g.groupby(col, observed=True):
            rows.append({"name": name, "dimension": col, "group": str(k), "n": len(d), "rmse": M.rmse(d.pi_next, d.yhat),
                         "bias_mean_resid": float(d.resid.mean()), "spearman": M.spearman(d.pi_next, d.yhat),
                         "small_sample": len(d) < min_n})
    return pd.DataFrame(rows)


def minutes_bias(p: pd.DataFrame, name: str, k: int | None = None) -> dict:
    """Does the model favour players with more minutes? (a) partial correlation of prediction with minutes given pi,
    (b) mean minutes of predicted top-k vs true top-k."""
    g = p[p.name == name]
    X = np.column_stack([np.ones(len(g)), g.pi.to_numpy(), np.log(g.minutes.to_numpy())])
    coef, *_ = np.linalg.lstsq(X, g.yhat.to_numpy(), rcond=None)
    out = {"name": name, "n": len(g), "yhat_on_log_minutes_given_pi": float(coef[2])}
    sel_pred, sel_true = [], []
    for y, f in g.groupby("test_year"):
        kk = k or max(3, int(round(0.2 * len(f))))
        sel_pred.append(f.sort_values("yhat", ascending=False).head(kk).minutes.mean())
        sel_true.append(f.sort_values("pi_next", ascending=False).head(kk).minutes.mean())
    out.update({"mean_minutes_pred_topk": float(np.mean(sel_pred)), "mean_minutes_true_topk": float(np.mean(sel_true)),
                "mean_minutes_cohort": float(g.minutes.mean())})
    return out


def attach_panel_cols(p: pd.DataFrame, design: str) -> pd.DataFrame:
    """Add competition and data-coverage columns from the scouting-season panel (keyed by player, season/block)."""
    from research.config import PROCESSED
    f = "design_a_player_seasons.parquet" if design == "A" else "design_b_blocks.parquet"
    pan = pd.read_parquet(PROCESSED / f)[["player_id", "season_year", "competition_id", "matches_share", "availability"]]
    pan = pan.drop_duplicates(["player_id", "season_year"])
    return p.merge(pan, on=["player_id", "season_year"], how="left")


def main():
    sel = json.loads((TABLES / "selected_configs.json").read_text())
    cov, err, mb = [], [], []
    for d in ("A", "B"):
        p = attach_panel_cols(RS.load(d), d)
        for name in dict.fromkeys(["persistence_cal", sel[d]["overall"]]):
            c = coverage_table(p, name)
            cov.append(c.assign(design=d))
            e = subgroup_errors(p, name, "competition_id" if d == "B" else None)
            err.append(e.assign(design=d))
            mb.append({**minutes_bias(p, name), "design": d})
    pd.concat(cov).to_csv(TABLES / "uncertainty_coverage.csv", index=False)
    pd.concat(err).to_csv(TABLES / "error_by_subgroup.csv", index=False)
    pd.DataFrame(mb).to_csv(TABLES / "minutes_bias.csv", index=False)
    pd.set_option("display.width", 220)
    print(pd.concat(cov).round(3).to_string())
    print(pd.DataFrame(mb).round(3).to_string())


if __name__ == "__main__":
    main()
