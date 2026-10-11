"""Cached per-match records -> player-match and player-season feature tables.

Every feature is a deterministic function of match events *inside the scouting window*; no future
information enters (windows are defined by competition-season and, for the within-season design, by
`match_week`). See reports/04_feature_schema.md for the definition of every column.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.config import CACHE, PROCESSED
from research.ingest.events import COUNT_FIELDS, ROLE_GROUP

POS_ORDER = ("GK", "DEF", "MID", "FWD")
# derived counts that are sums of base counts (kept as counts so shrinkage can treat them as Poisson)
DERIVED_COUNTS = {
    "def_actions": ("tackles", "interceptions", "blocks", "clearances"),            # defender-oriented
    "def_actions_mid": ("tackles", "interceptions", "ball_recoveries"),             # midfield ball-winning
}


def load_player_match(comp_seasons: list[tuple[int, int]] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (player_match, team_match) frames from the per-match cache."""
    prow, trow = [], []
    wanted = set(comp_seasons) if comp_seasons else None
    for f in sorted((CACHE / "player_match").glob("*.json.gz")):
        with gzip.open(f, "rb") as fh:
            d = json.loads(fh.read())
        if wanted and (d["competition_id"], d["season_id"]) not in wanted:
            continue
        meta = {"match_id": d["match_id"], "competition_id": d["competition_id"], "season_id": d["season_id"],
                "match_date": d["match_date"], "match_week": d.get("match_week")}
        for p in d["players"]:
            rm = p["role_minutes"]
            row = {**meta, **{k: v for k, v in p.items() if k != "role_minutes"}}
            for role in ROLE_GROUP:
                row[f"min_{role}"] = rm.get(role, 0.0)
            prow.append(row)
        for t in d["teams"]:
            trow.append({**meta, **t})
    pm, tm = pd.DataFrame(prow), pd.DataFrame(trow)
    pm["match_date"] = pd.to_datetime(pm["match_date"])
    tm["match_date"] = pd.to_datetime(tm["match_date"])
    return pm, tm


def _sum_cols() -> list[str]:
    return ["minutes", *COUNT_FIELDS, "clean_sheet", *[f"min_{r}" for r in ROLE_GROUP]]


def aggregate(pm: pd.DataFrame, tm: pd.DataFrame, window_col: str | None = None) -> pd.DataFrame:
    """Aggregate to one row per (player, competition, season[, window]).

    window_col: optional column of `pm` (e.g. 'half') to additionally group by.
    A player who changed team in-window is assigned to the team with most minutes.
    """
    pm = pm.copy()
    for new, parts in DERIVED_COUNTS.items():
        pm[new] = pm[list(parts)].sum(axis=1)
    keys = ["player_id", "competition_id", "season_id"] + ([window_col] if window_col else [])
    sum_cols = _sum_cols() + list(DERIVED_COUNTS)
    g = pm.groupby(keys)
    out = g[sum_cols].sum()
    out["matches"] = g["match_id"].nunique()
    out["first_match"] = g["match_date"].min()
    out["last_match"] = g["match_date"].max()
    out["n_teams"] = g["team_id"].nunique()
    names = g[["player_name", "player_nickname", "country"]].first()
    out = out.join(names)
    # team with most minutes
    tmin = pm.groupby(keys + ["team_id", "team"])["minutes"].sum().reset_index().sort_values("minutes", ascending=False)
    top = tmin.drop_duplicates(keys)[keys + ["team_id", "team"]].set_index(keys)
    out = out.join(top)
    out = out.reset_index()
    role_cols = [f"min_{r}" for r in ROLE_GROUP]
    out["primary_role"] = out[role_cols].idxmax(axis=1).str.replace("min_", "", regex=False)
    grp_min = pd.DataFrame({p: out[[f"min_{r}" for r, gname in ROLE_GROUP.items() if gname == p]].sum(axis=1) for p in POS_ORDER})
    out["position_group"] = grp_min.idxmax(axis=1)
    out["position_purity"] = grp_min.max(axis=1) / out["minutes"]
    # team context over matches inside the same window
    tm = tm.copy()
    if window_col:
        wmap = pm.drop_duplicates("match_id").set_index("match_id")[window_col]
        tm[window_col] = tm["match_id"].map(wmap)
    tk = ["team_id", "competition_id", "season_id"] + ([window_col] if window_col else [])
    tg = tm.groupby(tk)
    ctx = pd.DataFrame({
        "team_matches": tg["match_id"].nunique(),
        "team_poss": tg["passes"].sum() / (tg["passes"].sum() + tg["opp_passes"].sum()),
        "team_gd_pm": (tg["goals_for"].sum() - tg["goals_against"].sum()) / tg["match_id"].nunique(),
        "team_xgd_pm": (tg["xg_for"].sum() - tg["xg_against"].sum()) / tg["match_id"].nunique(),
        "team_minutes_avail": tg["match_minutes"].sum(),
    }).reset_index()
    out = out.merge(ctx, on=tk, how="left")
    out["availability"] = out["minutes"] / out["team_minutes_avail"]       # share of the team's observed minutes
    out["matches_share"] = out["matches"] / out["team_matches"]
    return out


def add_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Per-90 rates (counts / minutes * 90) and ratio features that keep NaN when the denominator is tiny."""
    df = df.copy()
    n90 = df["minutes"] / 90.0
    for c in [*COUNT_FIELDS, "def_actions", "def_actions_mid"]:
        df[f"{c}_p90"] = df[c] / n90
    df["pass_completion"] = np.where(df["passes"] >= 1, df["passes_completed"] / df["passes"].replace(0, np.nan), np.nan)
    df["shot_accuracy"] = np.where(df["shots"] >= 1, df["shots_on_target"] / df["shots"].replace(0, np.nan), np.nan)
    df["xg_per_shot"] = df["npxg"] / (df["shots"] - df["penalties_taken"]).replace(0, np.nan)
    df["dribble_success"] = df["dribbles_completed"] / df["dribbles"].replace(0, np.nan)
    df["avg_pass_length"] = df["pass_length_sum"] / df["passes"].replace(0, np.nan)
    df["gk_save_pct"] = df["gk_saves"] / (df["gk_saves"] + df["gk_goals_conceded"]).replace(0, np.nan)
    df["season_end"] = df.groupby(["competition_id", "season_id"])["last_match"].transform("max")
    return df


def build(comp_seasons: list[tuple[int, int]], by_half: bool = False, split_week: int | None = None) -> pd.DataFrame:
    pm, tm = load_player_match(comp_seasons)
    if by_half:
        # first half = match_week <= split_week (default: half of the max match_week of the competition-season)
        mw = pm.groupby(["competition_id", "season_id"])["match_week"].transform("max")
        cut = split_week if split_week else (mw // 2)
        pm["half"] = np.where(pm["match_week"] <= cut, "H1", "H2")
        return add_rates(aggregate(pm, tm, "half"))
    return add_rates(aggregate(pm, tm))


def season_labels() -> pd.DataFrame:
    comps = pd.DataFrame(json.loads(gzip.open(CACHE / "meta" / "competitions.gz").read()))
    return comps[["competition_id", "season_id", "competition_name", "season_name"]]


def save(df: pd.DataFrame, name: str) -> Path:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    path = PROCESSED / f"{name}.parquet"
    df.to_parquet(path, index=False)
    return path
