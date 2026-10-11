"""Phase-2 data feasibility audit (metadata + lineups only; no event files downloaded).

Usage: python -m research.ingest.audit
Outputs (data/research/audit/): competition_seasons.csv, player_seasons_lineup.csv,
linkage.csv, and a console summary. Lineup-based minutes are the exposure measure.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from research.config import AUDIT
from research.ingest.fetch import cached_json


def clock(value, default):
    if value is None:
        return default
    m, s = value.split(":", 1)
    return int(m) + int(s) / 60.0


def competitions() -> pd.DataFrame:
    df = pd.DataFrame(cached_json("meta", "competitions.json", "competitions"))
    return df


def match_table(comp_id: int, season_id: int) -> list[dict]:
    return cached_json("matches", f"matches/{comp_id}/{season_id}.json", f"matches_{comp_id}_{season_id}")


def lineup_rows(match: dict, comp_id: int, season_id: int) -> list[dict]:
    lineups = cached_json("lineups", f"lineups/{match['match_id']}.json", f"lineups_{match['match_id']}")
    rows = []
    for team in lineups:
        for p in team["lineup"]:
            segs = p.get("positions") or []
            minutes = 0.0
            for s in segs:
                minutes += max(0.0, clock(s.get("to"), 95.0) - clock(s.get("from"), 0.0))
            rows.append({"competition_id": comp_id, "season_id": season_id, "match_id": match["match_id"],
                         "team": team["team_name"], "player_id": p["player_id"],
                         "player_name": p["player_name"], "appeared": bool(segs), "minutes_approx": minutes})
    return rows


def run(male_clubs_only: bool = True, workers: int = 12) -> dict[str, pd.DataFrame]:
    AUDIT.mkdir(parents=True, exist_ok=True)
    comps = competitions()
    sel = comps[(comps.competition_gender == "male") & (~comps.competition_youth)]
    if male_clubs_only:
        sel = sel[~sel.competition_international]
    cs_rows, ps_frames = [], []
    for c in sel.itertuples():
        try:
            matches = match_table(c.competition_id, c.season_id)
        except Exception as exc:
            print("skip", c.competition_name, c.season_name, exc)
            continue
        if not matches:
            continue
        dates = pd.to_datetime([m["match_date"] for m in matches])
        teams = {m["home_team"]["home_team_id"] for m in matches} | {m["away_team"]["away_team_id"] for m in matches}
        with ThreadPoolExecutor(workers) as pool:
            rows = [r for chunk in pool.map(lambda m: lineup_rows(m, c.competition_id, c.season_id), matches) for r in chunk]
        lp = pd.DataFrame(rows)
        lp = lp[lp.appeared]
        agg = lp.groupby(["player_id", "player_name"], as_index=False).agg(
            minutes=("minutes_approx", "sum"), matches=("match_id", "nunique"), team=("team", "first"),
            n_teams=("team", "nunique"))
        agg["competition_id"], agg["season_id"] = c.competition_id, c.season_id
        agg["competition"], agg["season"] = c.competition_name, c.season_name
        ps_frames.append(agg)
        cs_rows.append({"competition_id": c.competition_id, "season_id": c.season_id,
                        "competition": c.competition_name, "season": c.season_name,
                        "international": bool(c.competition_international),
                        "matches": len(matches), "teams": len(teams),
                        "date_min": dates.min().date(), "date_max": dates.max().date(),
                        "players": agg.player_id.nunique(),
                        "players_ge450": int((agg.minutes >= 450).sum()),
                        "players_ge900": int((agg.minutes >= 900).sum()),
                        "has_lineups": True, "has_events": True,
                        "has_360": c.match_available_360 is not None and str(c.match_available_360) != "nan"})
        print(f"{c.competition_name} {c.season_name}: {len(matches)} matches, {agg.player_id.nunique()} players")
    cs = pd.DataFrame(cs_rows).sort_values(["competition", "season"])
    ps = pd.concat(ps_frames, ignore_index=True)
    cs.to_csv(AUDIT / "competition_seasons.csv", index=False)
    ps.to_csv(AUDIT / "player_seasons_lineup.csv", index=False)
    return {"competition_seasons": cs, "player_seasons": ps}


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    run()
