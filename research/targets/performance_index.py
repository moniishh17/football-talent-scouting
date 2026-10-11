"""Pre-specified sporting-performance index (the prediction target) and the shared feature panel.

The index is fixed *before* any model comparison (documented in reports/03_targets.md):
  PI(player-season) = mean over the position group's components of  sign * (rate - mu_g) / sd_g,
with mu_g, sd_g estimated on a *reference* (training-window) sample of player-seasons with enough minutes.
Equal weights; minutes are never a component (exposure is evaluated separately). GK are excluded because
there are ~1-2 per team-season and the event schema gives no post-shot xG.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from research.features import shrinkage

# component -> sign (rates are per 90 unless noted)
COMPONENTS: dict[str, dict[str, int]] = {
    "FWD": {"npxg_p90": 1, "xa_p90": 1, "np_goals_p90": 1, "key_passes_p90": 1,
            "dribbles_completed_p90": 1, "carries_progressive_p90": 1},
    "MID": {"xa_p90": 1, "key_passes_p90": 1, "passes_progressive_p90": 1, "carries_progressive_p90": 1,
            "passes_into_box_p90": 1, "def_actions_mid_p90": 1},
    "DEF": {"def_actions_p90": 1, "passes_progressive_p90": 1, "carries_progressive_p90": 1,
            "pass_completion": 1, "dribbled_past_p90": -1, "aerials_lost_p90": -1},
}
# Common profile panel used as model features (z-scored within position group), for all outfield players.
PROFILE = ["npxg_p90", "xa_p90", "np_goals_p90", "key_passes_p90", "dribbles_completed_p90",
           "carries_progressive_p90", "passes_progressive_p90", "passes_into_box_p90", "def_actions_p90",
           "def_actions_mid_p90", "pass_completion", "dribbled_past_p90", "aerials_lost_p90", "pressures_p90"]
POISSON_COUNT = {  # rate column -> count column feeding the Poisson shrinker
    "npxg_p90": "npxg", "xa_p90": "xa", "np_goals_p90": "np_goals", "key_passes_p90": "key_passes",
    "dribbles_completed_p90": "dribbles_completed", "carries_progressive_p90": "carries_progressive",
    "passes_progressive_p90": "passes_progressive", "passes_into_box_p90": "passes_into_box",
    "def_actions_p90": "def_actions", "def_actions_mid_p90": "def_actions_mid",
    "dribbled_past_p90": "dribbled_past", "aerials_lost_p90": "aerials_lost", "pressures_p90": "pressures",
}
BINOM = {"pass_completion": ("passes_completed", "passes")}
OUTFIELD = ("DEF", "MID", "FWD")


@dataclass
class Reference:
    """Training-window statistics: z-score means/sds and shrinkage parameters (all fitted on train only)."""
    group_col: str = "position_group"
    stats: dict = field(default_factory=dict)         # (group, metric) -> (mu, sd)
    shrink: dict = field(default_factory=dict)
    shrunk: bool = False


def _rates(df: pd.DataFrame, ref: Reference | None, use_shrunk: bool) -> pd.DataFrame:
    if not use_shrunk:
        return df
    out = shrinkage.apply(df, ref.shrink, ref.group_col, POISSON_COUNT, BINOM, suffix="__s")
    for m in [*POISSON_COUNT, *BINOM]:
        out[m + "__raw"] = out[m]
        out[m] = out[m + "__s"]
    return out


def fit_reference(train: pd.DataFrame, min_minutes: float, group_col: str = "position_group",
                  shrink: bool = False) -> Reference:
    ref = Reference(group_col=group_col, shrunk=shrink)
    t = train[(train["minutes"] >= min_minutes) & (train["position_group"].isin(OUTFIELD))]
    ref.shrink = shrinkage.fit(train[train["position_group"].isin(OUTFIELD)], group_col, POISSON_COUNT, BINOM)
    t = _rates(t, ref, shrink)
    for grp, g in t.groupby(group_col):
        for m in PROFILE:
            x = g[m].dropna()
            sd = x.std(ddof=1)
            ref.stats[(grp, m)] = (float(x.mean()), float(sd) if sd and sd > 0 else 1.0)
    return ref


def _group_for_components(df: pd.DataFrame, group_col: str) -> pd.Series:
    """Components are defined per 3 outfield groups even if z-scoring uses finer roles."""
    return df["position_group"]


def zscores(df: pd.DataFrame, ref: Reference, use_shrunk: bool | None = None) -> pd.DataFrame:
    """Add z_<metric> for every PROFILE metric, standardised within the reference group (NaN -> 0)."""
    use_shrunk = ref.shrunk if use_shrunk is None else use_shrunk
    out = _rates(df, ref, use_shrunk).copy()
    for m in PROFILE:
        z = np.full(len(out), np.nan)
        for grp in out[ref.group_col].unique():
            if (grp, m) not in ref.stats:
                continue
            mu, sd = ref.stats[(grp, m)]
            mask = (out[ref.group_col] == grp).to_numpy()
            z[mask] = (out.loc[mask, m].to_numpy() - mu) / sd
        out[f"z_{m}"] = z
    return out


def performance_index(df: pd.DataFrame, ref: Reference, use_shrunk: bool | None = None) -> pd.DataFrame:
    """Return df with z_* columns and `pi` (NaN for GK / unknown groups)."""
    out = zscores(df, ref, use_shrunk)
    pi = np.full(len(out), np.nan)
    for grp, comp in COMPONENTS.items():
        mask = (out["position_group"] == grp).to_numpy()
        if not mask.any():
            continue
        cols = np.column_stack([sign * out.loc[mask, f"z_{m}"].fillna(0.0).to_numpy() for m, sign in comp.items()])
        pi[mask] = cols.mean(axis=1)
    out["pi"] = pi
    return out
