"""Download and aggregate StatsBomb Open Data for the 2022 FIFA World Cup.

Run from the project directory with:
    python -m processing.prepare_statsbomb_dataset

Only the derived player-season CSV is retained. Source event and lineup JSON is
read from the official open-data repository and processed in memory.
"""

import csv
import argparse
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from pathlib import Path


BASE_URL = "https://raw.githubusercontent.com/hudl/open-data/master/data"
COMPETITION_ID = 43
SEASON_ID = 106
MIN_MINUTES = 180
OUTPUT_PATH = (Path(__file__).resolve().parents[1] / "data" / "generated"
               / "statsbomb_world_cup_2022_players.csv")
BUNDLED_DATASET_PATH = Path(__file__).resolve().parents[1] / "data" / "statsbomb_world_cup_2022_players.csv"


def _fetch_json(url: str):
    request = urllib.request.Request(url, headers={"User-Agent": "sports-cloud-research/1.0"})
    with urllib.request.urlopen(request, timeout=40) as response:
        return json.load(response)


def _fetch_match_data(match: dict) -> tuple[list[dict], list[dict]]:
    match_id = match["match_id"]
    events = _fetch_json(f"{BASE_URL}/events/{match_id}.json")
    lineups = _fetch_json(f"{BASE_URL}/lineups/{match_id}.json")
    return events, lineups


def _position_group(name: str) -> str:
    value = (name or "").lower()
    if "goalkeeper" in value:
        return "GK"
    if "back" in value:
        return "DEF"
    if any(token in value for token in ("forward", "wing", "striker")):
        return "FWD"
    return "MID"


def _clock_minutes(value: str | None, default: float = 90.0) -> float:
    if value is None:
        return default
    minutes, seconds = value.split(":", 1)
    return int(minutes) + int(seconds) / 60.0


def _player_record(player_id, name, team, country=""):
    return {
        "player_id": str(player_id), "name": name, "team": team, "country": country,
        "minutes": 0.0, "positions": defaultdict(float), "passes": 0,
        "completed_passes": 0, "shots": 0, "shots_on_target": 0,
        "goals": 0, "xg": 0.0, "assists": 0, "defensive_actions": 0,
        "keeper_saves": 0, "clean_sheet_matches": 0, "matches_played": 0,
    }


def aggregate_competition_season(competition_id: int, season_id: int, competition_name: str,
                                season_name: str, min_minutes: int = MIN_MINUTES) -> list[dict]:
    """Aggregate one StatsBomb competition-season into player-season rows."""
    matches = _fetch_json(f"{BASE_URL}/matches/{competition_id}/{season_id}.json")
    with ThreadPoolExecutor(max_workers=8) as pool:
        match_data = list(pool.map(_fetch_match_data, matches))
    records = {}
    for match, (events, lineups) in zip(matches, match_data):
        match_id = match["match_id"]
        final_whistle = max(
            (float(event.get("minute", 0)) + float(event.get("second", 0)) / 60.0
             for event in events if int(event.get("period", 0)) <= 4),
            default=90.0,
        )
        match_player_ids = set()
        match_goalkeepers = set()
        team_players = defaultdict(list)
        for team in lineups:
            team_name = team["team_name"]
            for player in team["lineup"]:
                player_id = str(player["player_id"])
                segments = player.get("positions", [])
                if not segments:
                    continue
                rec = records.setdefault(player_id, _player_record(
                    player_id, player.get("player_nickname") or player["player_name"],
                    team_name, (player.get("country") or {}).get("name", ""),
                ))
                match_player_ids.add(player_id)
                team_players[team_name].append(player_id)
                match_minutes = 0.0
                for segment in segments:
                    start = _clock_minutes(segment.get("from"), 0.0)
                    end = _clock_minutes(segment.get("to"), final_whistle)
                    duration = max(0.0, end - start)
                    group = _position_group(segment.get("position", ""))
                    rec["positions"][group] += duration
                    if group == "GK" and duration > 0:
                        match_goalkeepers.add(player_id)
                    match_minutes += duration
                rec["minutes"] += match_minutes
                rec["matches_played"] += 1

        if len(match.get("home_team", {})) == 0:
            continue
        home_name = match["home_team"]["home_team_name"]
        away_name = match["away_team"]["away_team_name"]
        for team_name, opponent_score in ((home_name, match["away_score"]), (away_name, match["home_score"])):
            if opponent_score == 0:
                for player_id in team_players[team_name]:
                    rec = records[player_id]
                    if player_id in match_goalkeepers:
                        rec["clean_sheet_matches"] += 1

        for event in events:
            player = event.get("player")
            if not player or str(player.get("id")) not in match_player_ids:
                continue
            rec = records[str(player["id"])]
            event_type = (event.get("type") or {}).get("name", "")
            if event_type == "Pass":
                rec["passes"] += 1
                if not (event.get("pass") or {}).get("outcome"):
                    rec["completed_passes"] += 1
                if (event.get("pass") or {}).get("goal_assist"):
                    rec["assists"] += 1
            elif event_type == "Shot":
                shot = event.get("shot") or {}
                outcome = (shot.get("outcome") or {}).get("name", "")
                rec["shots"] += 1
                rec["goals"] += outcome == "Goal"
                rec["shots_on_target"] += outcome in {"Goal", "Saved", "Saved To Post"}
                rec["xg"] += float(shot.get("statsbomb_xg", 0.0) or 0.0)
            elif event_type in {"Pressure", "Interception", "Block", "Clearance", "Ball Recovery"}:
                rec["defensive_actions"] += 1
            elif event_type == "Duel" and (event.get("duel") or {}).get("type", {}).get("name") == "Tackle":
                rec["defensive_actions"] += 1
            elif event_type == "Goalkeeper":
                keeper_type = (event.get("goalkeeper") or {}).get("type", {}).get("name", "")
                if "Saved" in keeper_type:
                    rec["keeper_saves"] += 1

    rows = []
    for rec in records.values():
        if rec["minutes"] < min_minutes:
            continue
        position = max(rec["positions"], key=rec["positions"].get)
        minutes = rec["minutes"]
        per90 = 90 / minutes
        rows.append({
            "competition_id": competition_id, "competition": competition_name,
            "season_id": season_id, "season": season_name,
            "player_id": rec["player_id"], "name": rec["name"], "team": rec["team"],
            "country": rec["country"], "position": position,
            "minutes": round(minutes, 2), "matches_played": rec["matches_played"],
            "pass_completion": round(rec["completed_passes"] / rec["passes"], 6) if rec["passes"] else 0.0,
            "goals_per90": round(rec["goals"] * per90, 6),
            "assists_per90": round(rec["assists"] * per90, 6),
            "xg_per90": round(rec["xg"] * per90, 6),
            "shots_on_target_rate": round(rec["shots_on_target"] / rec["shots"], 6) if rec["shots"] else 0.0,
            "def_actions_per90": round(rec["defensive_actions"] * per90, 6),
            "saves_per90": round(rec["keeper_saves"] * per90, 6),
            "clean_sheets_per90": round(rec["clean_sheet_matches"] * per90, 6),
        })
    return sorted(rows, key=lambda row: row["player_id"])


def aggregate_world_cup() -> list[dict]:
    return aggregate_competition_season(COMPETITION_ID, SEASON_ID, "FIFA World Cup", "2022")


def prepare_dataset(output_path: str | Path = OUTPUT_PATH) -> int:
    """Fetch and prepare the World Cup player table at the requested output path."""
    path = Path(output_path)
    if path.resolve() == BUNDLED_DATASET_PATH.resolve():
        raise ValueError("Refusing to overwrite the bundled offline dataset; choose another --output path.")
    rows = aggregate_world_cup()
    if not rows:
        raise RuntimeError("No players were returned by the StatsBomb source.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} players to {path}")
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description="Prepare a player-level StatsBomb World Cup CSV.")
    parser.add_argument("--output", default=str(OUTPUT_PATH), help="prepared player CSV output path")
    args = parser.parse_args()
    prepare_dataset(args.output)


if __name__ == "__main__":
    main()
