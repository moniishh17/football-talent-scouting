"""Baselines: constant, persistence, age/position, and the repository's original MCDA rankers.

The MCDA rankers (WSM / TOPSIS / VIKOR) are *reused unchanged* from processing.scouting_channel; only the
adapter that builds their legacy metric set from the new player-season table lives here. Legacy quirks are kept
on purpose (e.g. an undefined shots_on_target_rate becomes 0, minutes is a criterion) so the baseline is the
method as originally implemented.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from processing.scouting_channel import PlayerProfile, POSITION_WEIGHTS, _rank_group

LEGACY_ALGOS = ("weighted_sum", "topsis", "vikor")
LEGACY_METRIC_KEYS = ("pass_completion", "goals_per90", "assists_per90", "xg_per90", "shots_on_target_rate",
                      "def_actions_per90", "minutes")


def legacy_metrics(df: pd.DataFrame) -> pd.DataFrame:
    n90 = df["minutes"] / 90.0
    out = pd.DataFrame(index=df.index)
    out["pass_completion"] = df["pass_completion"].fillna(0.0)
    out["goals_per90"] = df["goals"] / n90
    out["assists_per90"] = df["assists"] / n90
    out["xg_per90"] = df["xg"] / n90
    out["shots_on_target_rate"] = df["shot_accuracy"].fillna(0.0)          # legacy: undefined -> 0
    out["def_actions_per90"] = (df["pressures"] + df["interceptions"] + df["blocks"] + df["clearances"]
                                + df["ball_recoveries"] + df["tackles"]) / n90
    out["minutes"] = df["minutes"]
    out["saves_per90"] = df["gk_saves"] / n90
    out["clean_sheets_per90"] = df["clean_sheet"] / n90
    return out


def mcda_scores(df: pd.DataFrame, weights_override: dict | None = None, cohort_cols=("season_year",)) -> pd.DataFrame:
    """Score every row with the three legacy rankers, ranking within (cohort_cols x position_group).

    Returns a frame indexed like df with columns weighted_sum / topsis / vikor (NaN for GK)."""
    metrics = legacy_metrics(df)
    out = pd.DataFrame(np.nan, index=df.index, columns=LEGACY_ALGOS)
    keys = list(cohort_cols) + ["position_group"]
    for _, idx in df.groupby(keys).groups.items():
        pos = df.loc[idx[0], "position_group"]
        if pos not in POSITION_WEIGHTS or pos == "GK" or len(idx) < 2:
            continue
        profs = []
        for i in idx:
            m = {k: float(metrics.at[i, k]) for k in metrics.columns}
            m["goals_assists_per90"] = m["goals_per90"] + m["assists_per90"]
            profs.append(PlayerProfile(str(i), str(i), pos, m))
        w = weights_override[pos] if weights_override else None
        ranked = _rank_group(profs, w)
        for algo in LEGACY_ALGOS:
            for r in ranked[algo]:
                out.at[int(r["player_id"]), algo] = r["score"]
    return out


def equal_weights() -> dict:
    return {pos: {c: 1.0 / len(w) for c in w} for pos, w in POSITION_WEIGHTS.items()}
