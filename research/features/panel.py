"""Player-season panels used by the two study designs.

Design A (next-season, "Barcelona-match panel"): La Liga 2004/05-2020/21, only matches involving FC Barcelona
  (StatsBomb's open La Liga coverage is Barcelona-centric; 2015/16 is restricted to Barcelona matches for a
  consistent observation process). Cohort = players whose most-minutes team is Barcelona.
Design B (within-season, multi-league): 2015/16 full seasons of La Liga, Premier League, Serie A, Ligue 1;
  features from matchweeks 1..H, outcome from H+1..end.
Age and market values come from the documented Transfermarkt crosswalk (tier A only).
"""
from __future__ import annotations

import gzip
import json

import numpy as np
import pandas as pd

from research.config import CACHE, PROCESSED, TRANSFERMARKT
from research.features import player_season as ps

BARCA_TEAM_ID = 217
LALIGA = 11
LALIGA_SEASONS = (37, 38, 39, 40, 41, 21, 22, 23, 24, 25, 26, 27, 2, 1, 4, 42, 90)
DESIGN_B = {11: 27, 2: 27, 12: 27, 7: 27}      # competition_id -> season_id (2015/16)


def season_meta() -> pd.DataFrame:
    comps = pd.DataFrame(json.loads(gzip.open(CACHE / "meta" / "competitions.gz").read()))
    comps["season_year"] = comps.season_name.str[:4].astype(int)
    return comps[["competition_id", "season_id", "competition_name", "season_name", "season_year"]]


def load_crosswalk(tiers=("A",)) -> pd.DataFrame:
    cw = pd.read_csv(PROCESSED / "crosswalk.csv")
    cw = cw[cw.tier.isin(tiers) & cw.tm_id.notna()].copy()
    cw["dob"] = pd.to_datetime(cw["date_of_birth"])
    return cw[["sb_player_id", "tm_id", "dob", "tm_position"]].rename(columns={"sb_player_id": "player_id"})


def load_valuations() -> pd.DataFrame:
    v = pd.read_csv(TRANSFERMARKT / "player_valuations.csv", usecols=["player_id", "date", "market_value_in_eur"], parse_dates=["date"])
    return v.rename(columns={"player_id": "tm_id"}).sort_values(["tm_id", "date"])


def attach_age(df: pd.DataFrame, cw: pd.DataFrame) -> pd.DataFrame:
    out = df.merge(cw, on="player_id", how="left")
    out["age"] = (out["season_end"] - out["dob"]).dt.days / 365.25
    return out


def attach_valuation(df: pd.DataFrame, val: pd.DataFrame, horizon_days: int = 365, max_age_days: int = 200,
                     tol_days: int = 120) -> pd.DataFrame:
    """value_at_t: last valuation at or before season_end (<= max_age_days old).
    value_future: valuation closest to season_end + horizon within +/- tol_days (strictly later than season_end)."""
    out = df.copy()
    vt, vf, vt_date = [], [], []
    by = {k: g for k, g in val.groupby("tm_id")}
    for r in out.itertuples():
        g = by.get(getattr(r, "tm_id"))
        if g is None or pd.isna(r.tm_id):
            vt.append(np.nan); vf.append(np.nan); vt_date.append(pd.NaT); continue
        past = g[g.date <= r.season_end]
        if len(past) and (r.season_end - past.date.iloc[-1]).days <= max_age_days:
            vt.append(float(past.market_value_in_eur.iloc[-1])); vt_date.append(past.date.iloc[-1])
        else:
            vt.append(np.nan); vt_date.append(pd.NaT)
        target = r.season_end + pd.Timedelta(days=horizon_days)
        fut = g[(g.date > r.season_end) & ((g.date - target).abs() <= pd.Timedelta(days=tol_days))]
        if len(fut):
            j = (fut.date - target).abs().idxmin()
            vf.append(float(fut.loc[j, "market_value_in_eur"]))
        else:
            vf.append(np.nan)
    out["value_t"], out["value_t_date"], out["value_future"] = vt, vt_date, vf
    return out


def design_a(min_minutes_keep: float = 90.0) -> pd.DataFrame:
    """Barcelona-match panel: one row per (player, season) for Barcelona-primary players."""
    pm, tm = ps.load_player_match([(LALIGA, s) for s in LALIGA_SEASONS])
    barca_matches = set(tm.loc[tm.team_id == BARCA_TEAM_ID, "match_id"])
    pm, tm = pm[pm.match_id.isin(barca_matches)], tm[tm.match_id.isin(barca_matches)]
    df = ps.add_rates(ps.aggregate(pm, tm))
    df = df[(df.team_id == BARCA_TEAM_ID) & (df.minutes >= min_minutes_keep)].copy()
    df = df.merge(season_meta(), on=["competition_id", "season_id"], how="left")
    df = attach_age(df, load_crosswalk())
    return df.sort_values(["season_year", "player_id"]).reset_index(drop=True)


def design_b_blocks(n_blocks: int = 4) -> pd.DataFrame:
    """Within-season panel of 4 consecutive blocks of matchweeks (~9-10 each) for the 2015/16 full leagues.

    `season_year` is re-used as the time index (= block number 1..4) so that the same rolling-origin code
    applies: scouting at the end of block b, outcome = block b+1."""
    pm, tm = ps.load_player_match([(c, s) for c, s in DESIGN_B.items()])
    mw = pm.groupby(["competition_id", "season_id"])["match_week"].transform("max")
    pm["block"] = np.minimum(np.ceil(pm["match_week"] / (mw / n_blocks)).astype(int), n_blocks)
    wmap = pm.drop_duplicates("match_id").set_index("match_id")["block"]
    df = ps.aggregate(pm, tm, "block")
    df = ps.add_rates(df)
    df = df.merge(season_meta().drop(columns="season_year"), on=["competition_id", "season_id"], how="left")
    df["season_end"] = df.groupby(["competition_id", "season_id"])["last_match"].transform("max")
    df = attach_age(df, load_crosswalk())
    df["season_year"] = df["block"]
    return df.reset_index(drop=True)


def design_b() -> pd.DataFrame:
    """Within-season panel: rows per (player, competition, window in {H1,H2}) for the 2015/16 full leagues."""
    pm, tm = ps.load_player_match([(c, s) for c, s in DESIGN_B.items()])
    df = ps.build([(c, s) for c, s in DESIGN_B.items()], by_half=True)
    df = df.merge(season_meta(), on=["competition_id", "season_id"], how="left")
    df = attach_age(df, load_crosswalk())
    return df.reset_index(drop=True)
