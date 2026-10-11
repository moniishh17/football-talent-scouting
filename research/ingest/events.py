"""Event-level feature extraction: one StatsBomb match -> per-player and per-team count rows.

Schema notes (verified against the live event JSON, see reports/feature_schema.md):
  * Coordinates are 120x80 with the acting team always attacking toward x=120; goal centre = (120, 40).
  * Pass.outcome is absent for completed passes; shot.key_pass_id links a shot to the key pass (-> xA).
  * Period 5 is the penalty shoot-out and is excluded from all counts (the legacy pipeline counted it).
  * Carries covering a distance implausible for the recorded duration (> MAX_CARRY_SPEED yd/s) are a
    known tracking artefact (observed: a 49-yard carry in 1.2 s) and are excluded from carry features.
Results are cached per match (data/research/cache/player_match/<match_id>.json.gz) and the sha256 of the
source event file is stored in the cache record, so re-runs are resumable and provenance is preserved.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from research.config import CACHE, OPEN_DATA_BASE
from research.ingest.fetch import cached_json, http_get, record_manifest

GOAL = (120.0, 40.0)
MAX_CARRY_SPEED = 12.0          # yards / second
PROGRESSIVE_YARDS = 10.0        # moves the ball >= 10 yds closer to the opposition goal (completed pass / carry)
COUNT_FIELDS = (
    "passes", "passes_completed", "passes_progressive", "passes_final_third", "passes_into_box", "passes_long",
    "crosses", "through_balls", "key_passes", "assists", "xa", "pass_length_sum",
    "shots", "shots_on_target", "goals", "np_goals", "xg", "npxg", "penalties_taken",
    "carries", "carries_progressive", "carry_dist_progressive", "dribbles", "dribbles_completed",
    "ball_receipts", "dispossessed", "miscontrols",
    "pressures", "tackles", "tackles_won", "interceptions", "blocks", "clearances", "ball_recoveries",
    "fouls_committed", "fouls_won", "aerials_lost", "dribbled_past",
    "gk_shots_faced", "gk_saves", "gk_goals_conceded", "gk_claims", "gk_punches", "gk_sweeper",
)
ROLE_MAP = [  # (substring, role) evaluated in order
    ("Goalkeeper", "GK"), ("Wing Back", "FB"), ("Back", "FB"), ("Center Back", "CB"),
    ("Defensive Midfield", "DM"), ("Attacking Midfield", "AM"), ("Center Midfield", "CM"),
    ("Midfield", "WM"), ("Wing", "W"), ("Forward", "CF"), ("Striker", "CF"),
]
ROLE_GROUP = {"GK": "GK", "CB": "DEF", "FB": "DEF", "DM": "MID", "CM": "MID", "AM": "MID", "WM": "MID",
              "W": "FWD", "CF": "FWD"}


def role_of(position_name: str) -> str:
    # "Center Back" contains "Back": handle it before the generic "Back" rule.
    if "Center Back" in (position_name or ""):
        return "CB"
    for token, role in ROLE_MAP:
        if token in (position_name or ""):
            return role
    return "CM"


def _clock(value, default):
    if value is None:
        return default
    m, s = value.split(":", 1)
    return int(m) + int(s) / 60.0


def _dist_goal(loc) -> float:
    return math.hypot(GOAL[0] - loc[0], GOAL[1] - loc[1])


def process_match(match: dict, comp_id: int, season_id: int, events: list[dict], lineups: list[dict]) -> dict:
    events = [e for e in events if int(e.get("period", 0)) <= 4]
    end_clock = max((e["minute"] + e.get("second", 0) / 60.0 for e in events), default=90.0)
    players: dict[int, dict] = {}
    for team in lineups:
        for p in team["lineup"]:
            segs = p.get("positions") or []
            if not segs:
                continue
            role_min = defaultdict(float)
            for s in segs:
                role_min[role_of(s["position"])] += max(0.0, _clock(s.get("to"), end_clock) - _clock(s.get("from"), 0.0))
            minutes = sum(role_min.values())
            if minutes <= 0:
                continue
            players[p["player_id"]] = {
                "player_id": p["player_id"], "player_name": p["player_name"],
                "player_nickname": p.get("player_nickname"), "country": (p.get("country") or {}).get("name"),
                "team_id": team["team_id"], "team": team["team_name"], "minutes": minutes,
                "role_minutes": dict(role_min), **{f: 0.0 for f in COUNT_FIELDS},
            }
    pass_owner, pass_xa = {}, defaultdict(float)
    team_stats = defaultdict(lambda: {"passes": 0, "shots": 0, "xg": 0.0})
    for e in events:
        t = e["type"]["name"]
        team_id = e["team"]["id"]
        if t == "Pass":
            team_stats[team_id]["passes"] += 1
            pass_owner[e["id"]] = e["player"]["id"] if e.get("player") else None
        elif t == "Shot":
            team_stats[team_id]["shots"] += 1
            team_stats[team_id]["xg"] += float(e["shot"].get("statsbomb_xg") or 0.0)
            kp = e["shot"].get("key_pass_id")
            if kp:
                pass_xa[kp] += float(e["shot"].get("statsbomb_xg") or 0.0)
    for pid_pass, xa in pass_xa.items():
        owner = pass_owner.get(pid_pass)
        if owner in players:
            players[owner]["xa"] += xa
    for e in events:
        pl = e.get("player")
        if not pl or pl["id"] not in players:
            continue
        r = players[pl["id"]]
        t = e["type"]["name"]
        loc = e.get("location")
        if t == "Pass":
            p = e["pass"]
            r["passes"] += 1
            done = "outcome" not in p
            r["passes_completed"] += done
            r["pass_length_sum"] += p.get("length", 0.0)
            if p.get("length", 0.0) >= 32.8:     # ~30 m
                r["passes_long"] += 1
            r["crosses"] += bool(p.get("cross"))
            r["through_balls"] += bool(p.get("through_ball"))
            r["key_passes"] += bool(p.get("shot_assist")) or bool(p.get("goal_assist"))
            r["assists"] += bool(p.get("goal_assist"))
            end = p.get("end_location")
            if done and loc and end:
                if _dist_goal(loc) - _dist_goal(end) >= PROGRESSIVE_YARDS:
                    r["passes_progressive"] += 1
                if end[0] >= 80 and loc[0] < 80:
                    r["passes_final_third"] += 1
                if end[0] >= 102 and 18 <= end[1] <= 62 and not (loc[0] >= 102 and 18 <= loc[1] <= 62):
                    r["passes_into_box"] += 1
        elif t == "Shot":
            s = e["shot"]
            xg = float(s.get("statsbomb_xg") or 0.0)
            outcome = (s.get("outcome") or {}).get("name", "")
            is_pen = (s.get("type") or {}).get("name") == "Penalty"
            r["shots"] += 1
            r["xg"] += xg
            r["goals"] += outcome == "Goal"
            r["shots_on_target"] += outcome in {"Goal", "Saved", "Saved To Post"}
            if is_pen:
                r["penalties_taken"] += 1
            else:
                r["npxg"] += xg
                r["np_goals"] += outcome == "Goal"
        elif t == "Carry":
            end = (e.get("carry") or {}).get("end_location")
            dur = e.get("duration") or 0.0
            if loc and end:
                d = math.hypot(end[0] - loc[0], end[1] - loc[1])
                if dur > 0 and d / dur > MAX_CARRY_SPEED:
                    continue
                r["carries"] += 1
                gain = _dist_goal(loc) - _dist_goal(end)
                if gain >= PROGRESSIVE_YARDS:
                    r["carries_progressive"] += 1
                    r["carry_dist_progressive"] += gain
        elif t == "Dribble":
            r["dribbles"] += 1
            r["dribbles_completed"] += (e["dribble"].get("outcome") or {}).get("name") == "Complete"
        elif t == "Ball Receipt*":
            r["ball_receipts"] += 1
        elif t == "Dispossessed":
            r["dispossessed"] += 1
        elif t == "Miscontrol":
            r["miscontrols"] += 1
        elif t == "Pressure":
            r["pressures"] += 1
        elif t == "Duel":
            d = e.get("duel") or {}
            dt = (d.get("type") or {}).get("name")
            if dt == "Tackle":
                r["tackles"] += 1
                r["tackles_won"] += (d.get("outcome") or {}).get("name") in {"Won", "Success In Play", "Success Out"}
            elif dt == "Aerial Lost":
                r["aerials_lost"] += 1
        elif t == "Interception":
            r["interceptions"] += 1
        elif t == "Block":
            r["blocks"] += 1
        elif t == "Clearance":
            r["clearances"] += 1
        elif t == "Ball Recovery":
            r["ball_recoveries"] += 1
        elif t == "Foul Committed":
            r["fouls_committed"] += 1
        elif t == "Foul Won":
            r["fouls_won"] += 1
        elif t == "Dribbled Past":
            r["dribbled_past"] += 1
        elif t == "Goal Keeper":
            k = (e.get("goalkeeper") or {}).get("type", {}).get("name", "")
            if k == "Shot Faced":
                r["gk_shots_faced"] += 1
            elif k in {"Shot Saved", "Penalty Saved", "Shot Saved To Post"}:
                r["gk_saves"] += 1
            elif k == "Goal Conceded":
                r["gk_goals_conceded"] += 1
            elif k == "Collected":
                r["gk_claims"] += 1
            elif k == "Punch":
                r["gk_punches"] += 1
            elif k == "Keeper Sweeper":
                r["gk_sweeper"] += 1
    home, away = match["home_team"], match["away_team"]
    scores = {home["home_team_id"]: (match["home_score"], match["away_score"]),
              away["away_team_id"]: (match["away_score"], match["home_score"])}
    teams = []
    for tid, name, opp in ((home["home_team_id"], home["home_team_name"], away["away_team_id"]),
                           (away["away_team_id"], away["away_team_name"], home["home_team_id"])):
        gf, ga = scores[tid]
        own, other = team_stats[tid], team_stats[opp]
        teams.append({"team_id": tid, "team": name, "opponent_id": opp, "is_home": tid == home["home_team_id"],
                      "goals_for": gf, "goals_against": ga, "xg_for": own["xg"], "xg_against": other["xg"],
                      "passes": own["passes"], "opp_passes": other["passes"], "match_minutes": end_clock})
    for r in players.values():
        r["clean_sheet"] = bool(r["role_minutes"].get("GK", 0) >= 60 and scores[r["team_id"]][1] == 0)
    return {"match_id": match["match_id"], "competition_id": comp_id, "season_id": season_id,
            "match_date": match["match_date"], "match_week": match.get("match_week"),
            "players": list(players.values()), "teams": teams}


def ensure_match(match: dict, comp_id: int, season_id: int) -> dict:
    path = CACHE / "player_match" / f"{match['match_id']}.json.gz"
    if path.exists():
        with gzip.open(path, "rb") as fh:
            return json.loads(fh.read())
    url = f"{OPEN_DATA_BASE}/events/{match['match_id']}.json"
    raw = http_get(url)
    events = json.loads(raw)
    lineups = cached_json("lineups", f"lineups/{match['match_id']}.json", f"lineups_{match['match_id']}")
    rec = process_match(match, comp_id, season_id, events, lineups)
    rec["source"] = {"url": url, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
                     "fetched_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with gzip.open(tmp, "wb") as fh:
        fh.write(json.dumps(rec).encode())
    tmp.replace(path)
    record_manifest({"url": url, "kind": "events(derived-cached)", **rec["source"]})
    return rec


def ingest(comp_id: int, season_ids: list[int], workers: int = 8) -> int:
    n = 0
    for sid in season_ids:
        matches = cached_json("matches", f"matches/{comp_id}/{sid}.json", f"matches_{comp_id}_{sid}")
        with ThreadPoolExecutor(workers) as pool:
            list(pool.map(lambda m: ensure_match(m, comp_id, sid), matches))
        n += len(matches)
        print(f"comp {comp_id} season {sid}: {len(matches)} matches cached")
    return n


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--competition", type=int, required=True)
    ap.add_argument("--seasons", type=int, nargs="+", required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    ingest(a.competition, a.seasons, a.workers)
