"""Does the 'MCDA < persistence' conclusion depend on the target definition?

Re-evaluates rankers against the LEGACY-style future index (the repository's original target: position weights without minutes,
min-max within position cohort) as well as the new performance index. Predictors: legacy MCDA scores at t, the legacy index at t
(persistence in the legacy metric), and the new index at t. Scope: Design A and B pairs (eligible, age known), same folds as the
main experiments but WITHOUT any model fitting (all predictors are fixed formulas, so no leakage is possible).
Run: python -m research.analysis.target_sensitivity
"""
import numpy as np
import pandas as pd

from processing.scouting_channel import POSITION_WEIGHTS, _normalize_group, PlayerProfile
from research.config import PROCESSED, REPORTS
from research.evaluation import metrics as M
from research.evaluation import rolling as R
from research.models import baselines as bl
from research.run_design_b import config as b_config


def legacy_index(df: pd.DataFrame) -> pd.Series:
    """Legacy future-performance index per row, min-max within (season_year, position_group) cohort."""
    m = bl.legacy_metrics(df)
    m["goals_assists_per90"] = m.goals_per90 + m.assists_per90
    out = pd.Series(np.nan, index=df.index)
    for (_, pos), idx in df.groupby(["season_year", "position_group"]).groups.items():
        if pos == "GK" or len(idx) < 2:
            continue
        w = {c: v for c, v in POSITION_WEIGHTS[pos].items() if c != "minutes"}
        norm = {c: v / sum(w.values()) for c, v in w.items()}
        profs = [PlayerProfile(str(i), str(i), pos, {k: float(m.at[i, k]) for k in m.columns}) for i in idx]
        mat = _normalize_group(profs, tuple(norm))
        for i, row in zip(idx, mat):
            out.at[i] = sum(row[c] * norm[c] for c in norm)
    return out


def run(design: str):
    if design == "A":
        panel, cfg, years = pd.read_parquet(PROCESSED / "design_a_player_seasons.parquet"), R.Config(), range(2009, 2020)
    else:
        panel, cfg, years = pd.read_parquet(PROCESSED / "design_b_blocks.parquet"), b_config(), range(2, 4)
    rows = []
    for y in years:
        prep = R.prepare_fold(panel, y, cfg)
        if prep is None:
            continue
        _, test, _ = prep
        d = panel[(panel.minutes >= cfg.min_minutes_target) & panel.position_group.isin(R.OUTFIELD)].copy().reset_index(drop=True)
        d["legacy_idx"] = legacy_index(d)
        cur = d[["player_id", "season_year", "legacy_idx"]].rename(columns={"legacy_idx": "legacy_idx_t"})
        nxt = d[["player_id", "season_year", "legacy_idx"]].assign(season_year=lambda x: x.season_year - 1).rename(columns={"legacy_idx": "legacy_idx_next"})
        t = test.merge(cur, on=["player_id", "season_year"], how="left").merge(nxt, on=["player_id", "season_year"], how="left")
        rows.append(t.assign(test_year=y))
    t = pd.concat(rows).dropna(subset=["legacy_idx_next", "legacy_idx_t"])
    res = []
    for target, tname in (("pi_next", "new performance index"), ("legacy_idx_next", "legacy index (original target)")):
        for pred, pname in (("mcda_weighted_sum", "legacy MCDA: WSM"), ("mcda_topsis", "legacy MCDA: TOPSIS"), ("mcda_vikor", "legacy MCDA: VIKOR"),
                            ("equal_weight", "equal-weight legacy"), ("legacy_idx_t", "persistence of legacy index"), ("pi", "persistence of new index")):
            g = t.dropna(subset=[pred])
            r, lo, hi = M.cluster_bootstrap(lambda i: M.spearman(g[target].to_numpy()[i], g[pred].to_numpy()[i]), g.player_id.to_numpy(), 500, 0)
            res.append({"design": design, "target": tname, "predictor": pname, "n": len(g), "spearman": r, "lo": lo, "hi": hi})
    return pd.DataFrame(res)


if __name__ == "__main__":
    out = pd.concat([run("A"), run("B")])
    out.to_csv(REPORTS / "tables" / "target_sensitivity.csv", index=False)
    pd.set_option("display.width", 200)
    print(out.round(3).to_string())
