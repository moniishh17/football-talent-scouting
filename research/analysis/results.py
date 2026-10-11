"""Turn out-of-sample prediction files into the result tables used in the report (reports/tables/*.csv).

Selection rule (pre-specified, see reports/03_targets.md): within each model family and overall, the configuration with
the lowest pooled RMSE in the DEVELOPMENT period is 'selected'; the CONFIRMATION period is only ever reported.
Run: python -m research.analysis.results
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from research.config import PROCESSED, REPORTS
from research.evaluation import metrics as M
from research.evaluation import report as RP

TABLES = REPORTS / "tables"
FAMILIES = ("ridge", "enet", "rf", "lgbm")
BASELINES = ["mean", "mean_by_pos", "age_pos", "persistence", "persistence_cal", "equal_weight", "mcda_weighted_sum",
             "mcda_topsis", "mcda_vikor", "zero_change", "mean_change", "age_pos_value"]


def load(design: str) -> pd.DataFrame:
    if design == "A":
        p = pd.read_parquet(PROCESSED / "predictions_design_a.parquet")
    elif design == "B":
        p = pd.read_parquet(PROCESSED / "predictions_design_b.parquet")
        p["period"] = np.where(p.test_year <= 2, "development", "confirmation")
    else:
        p = pd.read_parquet(PROCESSED / "predictions_valuation.parquet").rename(columns={"y": "pi_next"})
    return p


def _rmse_by(p: pd.DataFrame) -> pd.Series:
    return p.groupby("name").apply(lambda g: M.rmse(g.pi_next, g.yhat), include_groups=False)


def select(p: pd.DataFrame) -> dict:
    dev = _rmse_by(p[p.period == "development"])
    ml = dev[[n for n in dev.index if "|" in n]]
    sel = {"overall": ml.idxmin()}
    for f in FAMILIES:
        sub = ml[[n for n in ml.index if n.startswith(f + "|")]]
        if len(sub):
            sel[f] = sub.idxmin()
    return sel


def main_table(p: pd.DataFrame, sel: dict, n_boot=1000) -> pd.DataFrame:
    names = [b for b in BASELINES if b in set(p.name)] + list(dict.fromkeys(sel.values()))
    rows = []
    for n in names:
        row = {"name": n, "selected_as": ",".join(k for k, v in sel.items() if v == n)}
        for tag, sub in (("dev", p[(p.name == n) & (p.period == "development")]),
                         ("confirm", p[(p.name == n) & (p.period == "confirmation")]),
                         ("all", p[p.name == n])):
            m = RP.pooled_metrics(sub, n_boot)
            row.update({f"{tag}_n": m["n"], f"{tag}_rmse": m["rmse"], f"{tag}_rmse_lo": m["rmse_lo"], f"{tag}_rmse_hi": m["rmse_hi"],
                        f"{tag}_mae": m["mae"], f"{tag}_spearman": m["spearman"]})
        rows.append(row)
    return pd.DataFrame(rows)


def paired_table(p: pd.DataFrame, sel: dict, refs: list[str], n_boot=2000) -> pd.DataFrame:
    rows = []
    for cand in dict.fromkeys(sel.values()):
        for ref in refs:
            if ref not in set(p.name):
                continue
            for tag, sub in (("confirm", p[p.period == "confirmation"]), ("all", p)):
                rows.append({**RP.paired_rmse_diff(sub, cand, ref, n_boot), "scope": tag})
    return pd.DataFrame(rows)


def ablation_table(p: pd.DataFrame, steps: list[tuple[str, str]], families=("ridge", "rf", "lgbm"), tail="|level", n_boot=1000):
    """steps: [(label, feature_set_or_name_fragment)] in cumulative order; names f'{family}|{fs}{tail}'."""
    rows = []
    for fam in families:
        prev = None
        for label, fs in steps:
            name = f"{fam}|{fs}{tail}"
            if name not in set(p.name):
                continue
            row = {"family": fam, "step": label, "name": name}
            for tag, sub in (("dev", p[(p.name == name) & (p.period == "development")]), ("all", p[p.name == name])):
                m = RP.pooled_metrics(sub, n_boot)
                row.update({f"{tag}_rmse": m["rmse"], f"{tag}_lo": m["rmse_lo"], f"{tag}_hi": m["rmse_hi"]})
            if prev:
                d = RP.paired_rmse_diff(p, name, prev, n_boot)
                row.update({"delta_vs_prev": d["rmse_diff"], "delta_lo": d["lo"], "delta_hi": d["hi"], "verdict_vs_prev": d["verdict"]})
            rows.append(row)
            prev = name
    return pd.DataFrame(rows)


def run_design(design: str, ks, steps, tail="|level", refs=None):
    p = load(design)
    sel = select(p)
    TABLES.mkdir(parents=True, exist_ok=True)
    main = main_table(p, sel)
    main.to_csv(TABLES / f"{design}_main.csv", index=False)
    refs = refs or ["persistence_cal", "mcda_weighted_sum", "mean", "age_pos"]
    pt = paired_table(p, sel, refs)
    pt.to_csv(TABLES / f"{design}_paired.csv", index=False)
    if steps:
        ablation_table(p, steps, tail=tail).to_csv(TABLES / f"{design}_ablation.csv", index=False)
    folds = RP.per_fold(p[p.name.isin(["persistence_cal", "mean", "mcda_weighted_sum"] + list(sel.values()))])
    folds.to_csv(TABLES / f"{design}_folds.csv", index=False)
    if design != "V":
        rt = RP.ranking_table(p[p.name.isin(["mean", "persistence_cal", "equal_weight", "mcda_weighted_sum", "mcda_topsis", "mcda_vikor"] + list(sel.values()))], ks=ks)
        RP.ranking_summary(rt).to_csv(TABLES / f"{design}_ranking.csv", index=False)
    # delta-skill: how much of the CHANGE (pi_next - pi) is explained, beyond regression to the mean
    if design != "V":
        rows = []
        for n in list(dict.fromkeys(["persistence_cal", "persistence", "age_pos"] + list(sel.values()))):
            g = p[p.name == n]
            dh = (g.yhat - g.pi).to_numpy()
            d = g.delta.to_numpy()
            r, lo, hi = M.cluster_bootstrap(lambda i: M.spearman(d[i], dh[i]), g.player_id.to_numpy(), 500, 0)
            rows.append({"name": n, "n": len(g), "spearman_delta": r, "lo": lo, "hi": hi,
                         "r2_delta": 1 - np.sum((d - dh) ** 2) / np.sum((d - d.mean()) ** 2)})
        pd.DataFrame(rows).to_csv(TABLES / f"{design}_delta_skill.csv", index=False)
    return sel


def main():
    from research.run_design_a import FEATURE_SETS as FA
    from research.run_design_b import FEATURE_SETS as FB
    names = {"pi": "PI only", "pi+profile": "+ profile z-scores", "pi+profile+agepos": "+ age, position",
             "pi+profile+agepos+exposure": "+ exposure", "pi+profile+agepos+exposure+context": "+ team context",
             "pi+profile+agepos+exposure+context+history": "+ history (t-1)",
             "pi+profile+agepos+exposure+context+league": "+ league", "pi+profile+agepos+exposure+context+league+history": "+ history (t-1)"}
    sel = {}
    sel["A"] = run_design("A", (3, 5), [(names[f], f) for f in FA])
    sel["B"] = run_design("B", (10, 25), [(names[f], f) for f in FB])
    vsteps = [("perf", "perf"), ("+ age, position", "perf+agepos"), ("+ exposure, context", "perf+agepos+exposure+context"),
              ("valuation only", "valuation_only"), ("age+pos+valuation", "agepos+valuation"), ("perf+age+pos+valuation", "perf+agepos+valuation"),
              ("+ valuation trend", "perf+agepos+valuation+trend")]
    sel["V"] = run_design("V", (), vsteps, tail="", refs=["age_pos_value", "mean_change", "zero_change"])
    (TABLES / "selected_configs.json").write_text(json.dumps(sel, indent=1))
    print(json.dumps(sel, indent=1))


if __name__ == "__main__":
    main()
