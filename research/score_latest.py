"""Latest-season candidate assessment with the FROZEN model.  Run: python -m research.score_latest

1. In-domain: La Liga 2020/21 (Barcelona matches; the last season of the training panel) -> predicts 2021/22.
2. Out-of-domain (exploratory, no validation): Ligue 1 2022/23 (PSG matches) and Bundesliga 2023/24 (Leverkusen matches), the
   latest club seasons in StatsBomb Open Data (no 2025-era men's club data exists there).
Outputs: reports/candidates_in_domain_la_liga_2020_21.csv and reports/candidates_out_of_domain.csv
These are model outputs for research review, NOT evidence that any prediction is correct and NOT claims about future stardom.
"""
from __future__ import annotations

import sys
import numpy as np
import pandas as pd

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="backslashreplace")

from research import inference as I
from research.config import PROCESSED, REPORTS
from research.features import panel as PN
from research.features import player_season as ps

OUT_OF_DOMAIN = [((7, 235), "Ligue 1", "2022/2023"), ((9, 281), "Bundesliga", "2023/2024")]


def other_panel(comp: int, season: int) -> pd.DataFrame:
    pm, tm = ps.load_player_match([(comp, season)])
    team = tm.team_id.value_counts().idxmax()          # the club whose matches the file contains
    keep = set(tm.loc[tm.team_id == team, "match_id"])
    pm, tm = pm[pm.match_id.isin(keep)], tm[tm.match_id.isin(keep)]
    df = ps.add_rates(ps.aggregate(pm, tm))
    df = df[df.team_id == team].copy()
    df = df.merge(PN.season_meta(), on=["competition_id", "season_id"], how="left")
    return PN.attach_age(df, PN.load_crosswalk())


def prior_seasons(panel: pd.DataFrame, pid, year) -> int:
    return int(((panel.player_id == pid) & (panel.season_year < year) & (panel.minutes >= 450)).sum())


def shape(scored: pd.DataFrame, panel: pd.DataFrame, domain: str, val_pred: pd.DataFrame | None) -> pd.DataFrame:
    s = scored.copy()
    s["cohort_n"] = len(s)
    s["rank_current_index"] = s.groupby("position_group").pi.rank(ascending=False, method="min").astype(int)
    s["rank_pred_next_index"] = s.groupby("position_group").pred_next_pi.rank(ascending=False, method="min").astype(int)
    s["pred_change_vs_cohort_mean"] = s.pred_change - s.groupby("position_group").pred_change.transform("mean")
    s["n_prior_seasons_observed"] = [prior_seasons(panel, r.player_id, r.season_year) for r in s.itertuples()]
    s["minutes_band"] = pd.cut(s.minutes, [0, 900, 1800, 2700, 1e9], labels=["450-900", "900-1800", "1800-2700", "2700+"], right=False).astype(str)
    s["age_available"] = s.age.notna()
    s["explanation"] = [I.explain(r, r.position_group) for _, r in s.iterrows()]
    s["domain"] = domain
    s["interpretation_note"] = ("Predicted next-season index = calibrated persistence of the current index (regression to the mean included). "
                                "'High predicted development' means above-cohort predicted change, which is largely mean reversion; "
                                "no breakout probability is reported because the breakout label was not shown to be learnable beyond base rate.")
    if val_pred is not None:
        s = s.merge(val_pred, on="player_id", how="left")
    cols = ["domain", "competition_name", "season_name", "player_id", "player_name", "team", "position_group", "primary_role", "age",
            "age_available", "minutes", "matches", "matches_share", "minutes_band", "n_prior_seasons_observed", "pi", "npxg_p90", "xa_p90",
            "passes_progressive_p90", "carries_progressive_p90", "def_actions_p90", "pass_completion", "pred_next_pi", "pred_lo_80",
            "pred_hi_80", "pred_lo_90", "pred_hi_90", "pred_change", "pred_change_vs_cohort_mean", "rank_current_index",
            "rank_pred_next_index", "cohort_n", "explanation", "interpretation_note"]
    cols += [c for c in ("value_t_eur", "pred_log_value_change_12m_ESTIMATE", "pred_value_lo80_eur_ESTIMATE", "pred_value_hi80_eur_ESTIMATE") if c in s.columns]
    return s[[c for c in cols if c in s.columns]].sort_values(["position_group", "rank_pred_next_index"]).reset_index(drop=True)


def valuation_estimates(panel: pd.DataFrame, year: int) -> pd.DataFrame | None:
    """Valuation forecast for scouting season `year` = the selected valuation model's fold prediction for that year
    (trained only on scouting seasons <= year-1), with conformal width from earlier folds."""
    import json
    vp = pd.read_parquet(PROCESSED / "predictions_valuation.parquet")
    name = json.loads((REPORTS / "tables" / "selected_configs.json").read_text())["V"]["overall"]
    p = vp[vp.name == name]
    cur, past = p[p.test_year == year], p[p.test_year < year]
    if cur.empty or len(past) < 30:
        return None
    q = float(np.quantile((past.y - past.yhat).abs(), min(1.0, np.ceil((len(past) + 1) * 0.8) / len(past))))
    out = pd.DataFrame({"player_id": cur.player_id.to_numpy(), "value_t_eur": cur.value_t.to_numpy(),
                        "pred_log_value_change_12m_ESTIMATE": cur.yhat.to_numpy()})
    out["pred_value_lo80_eur_ESTIMATE"] = out.value_t_eur * np.exp(out.pred_log_value_change_12m_ESTIMATE - q)
    out["pred_value_hi80_eur_ESTIMATE"] = out.value_t_eur * np.exp(out.pred_log_value_change_12m_ESTIMATE + q)
    return out


def main():
    frozen = I.load_frozen()
    panel = pd.read_parquet(PROCESSED / "design_a_player_seasons.parquet")
    latest = frozen["meta"]["latest_scouting_season_start_year"]
    cand = panel[panel.season_year == latest]
    scored = I.score(frozen, cand)
    inn = shape(scored, panel, "in-domain (Barcelona matches, La Liga)", valuation_estimates(panel, latest))
    inn.to_csv(REPORTS / "candidates_in_domain_la_liga_2020_21.csv", index=False)
    outs = []
    for (comp, season), cname, sname in OUT_OF_DOMAIN:
        op = other_panel(comp, season)
        sc = I.score(frozen, op)
        outs.append(shape(sc, op, f"OUT-OF-DOMAIN ({cname} {sname}, single club; reference/model trained on Barcelona seasons; unvalidated)", None))
    pd.concat(outs).to_csv(REPORTS / "candidates_out_of_domain.csv", index=False)
    pd.set_option("display.width", 220, "display.max_columns", 30)
    print(inn[["player_name", "position_group", "age", "minutes", "pi", "pred_next_pi", "pred_lo_80", "pred_hi_80", "pred_change_vs_cohort_mean"]].round(2).head(20).to_string())
    print(len(inn), sum(len(o) for o in outs))


if __name__ == "__main__":
    main()
