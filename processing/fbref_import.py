"""Prepare user-exported FBref player tables for the scouting ranker.

Reads local CSV exports only; it never scrapes or downloads FBref. Separate
table exports from one league-season may be joined by player, club, and league.
"""

import csv
import hashlib
import re
from pathlib import Path


def _key(value: str) -> str:
    value = (value or "").strip().lower()
    value = re.sub(r"^(standard|shooting|passing|defense|defensive|goalkeeping|keeper)[ _-]+", "", value)
    return re.sub(r"[^a-z0-9]+", "", value)


ALIASES = {
    "name": {"player", "name"}, "position": {"pos", "position"},
    "team": {"squad", "team", "club"}, "league": {"comp", "competition", "league"},
    "country": {"nation", "country"}, "minutes": {"min", "minutes"}, "season": {"season"},
    "goals": {"gls", "goals"}, "assists": {"ast", "assists"}, "xg": {"xg"},
    "shots": {"sh", "shots"}, "shots_on_target": {"sot", "shotsontarget"},
    "passes_completed": {"cmp", "passescompleted"}, "passes_attempted": {"att", "passesattempted"},
    "pass_completion": {"passcompletion"},
    "tackles": {"tkl", "tackles"}, "interceptions": {"int", "interceptions"},
    "saves": {"saves", "save"}, "clean_sheets": {"cs", "cleansheets"},
    "save_pct": {"savepct", "savepercentage"}, "xag": {"xag", "expectedassists"},
    "progressive_passes": {"prgp", "progressivepasses"},
    "progressive_carries": {"prgc", "progressivecarries"},
}
NORMALIZED_ALIASES = {name: {_key(alias) for alias in aliases} for name, aliases in ALIASES.items()}


def _number(value: str, field: str, path: Path, row_number: int):
    text = (value or "").strip().replace(",", "")
    if not text or text == "-":
        return None
    if text.endswith("%"):
        text = text[:-1].strip()
    try:
        return float(text)
    except ValueError as exc:
        raise ValueError(f"{path.name}, row {row_number}: {field} is not numeric: {value!r}") from exc


def _read_table(path: str | Path) -> list[dict[str, str]]:
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"FBref CSV export not found: {file_path}")
    with file_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"FBref CSV has no header: {file_path}")
        rows = []
        for raw in reader:
            mapped = {}
            for header, value in raw.items():
                normalized = _key(header or "")
                if "%" in (header or "") and normalized in {"cmp", "save"}:
                    mapped.setdefault("pass_completion" if normalized == "cmp" else "save_pct", value)
                    continue
                for field, aliases in NORMALIZED_ALIASES.items():
                    if normalized in aliases and value not in (None, ""):
                        mapped.setdefault(field, value)
            if mapped.get("name") and mapped.get("team"):
                rows.append(mapped)
    if not rows:
        raise ValueError(f"No player rows with Player and Squad columns found in {file_path}.")
    return rows


def prepare_fbref_exports(paths: list[str | Path], output_path: str | Path,
                          season: str | None = None) -> int:
    """Join local FBref table exports and write the ranker's canonical CSV."""
    if not paths:
        raise ValueError("Provide at least one local FBref CSV export.")
    joined: dict[tuple[str, str, str], dict[str, str]] = {}
    for path in paths:
        for row in _read_table(path):
            key = (row["name"].strip().casefold(), row["team"].strip().casefold(),
                   row.get("league", "").strip().casefold())
            target = joined.setdefault(key, {})
            for field, value in row.items():
                target.setdefault(field, value)
            target["_name"], target["_team"] = row["name"].strip(), row["team"].strip()
            target["_league"] = row.get("league", "").strip()

    prepared = []
    for row in joined.values():
        name, team, league = row["_name"], row["_team"], row["_league"]
        source_path = Path(paths[0])
        minutes = _number(row.get("minutes", ""), "Min", source_path, 2)
        position = row.get("position", "").strip()
        if not position or minutes is None or minutes <= 0:
            continue
        source_season = row.get("season", "").strip()
        if season and source_season and source_season != season:
            raise ValueError(f"Export contains season {source_season!r}, but --season is {season!r}.")
        selected_season = season or source_season
        if not selected_season:
            raise ValueError("Season is absent from the export; provide --season YYYY-YY.")

        def val(field):
            return _number(row.get(field, ""), field, source_path, 2)

        def rate90(field):
            count = val(field)
            return None if count is None else count * 90 / minutes

        goals90, assists90 = rate90("goals"), rate90("assists")
        completed, attempted = val("passes_completed"), val("passes_attempted")
        pass_rate = val("pass_completion")
        if pass_rate is not None and pass_rate > 1:
            pass_rate /= 100
        elif pass_rate is None and completed is not None and attempted:
            pass_rate = completed / attempted
        tackles, interceptions = val("tackles"), val("interceptions")
        def_rate = ((tackles or 0) + (interceptions or 0)) * 90 / minutes \
            if tackles is not None or interceptions is not None else None
        shots, on_target = val("shots"), val("shots_on_target")
        sot_rate = on_target / shots if shots and shots > 0 and on_target is not None else None
        save_pct = val("save_pct")
        if save_pct is not None and save_pct > 1:
            save_pct /= 100
        player_key = "|".join((name.casefold(), team.casefold(), league.casefold(), selected_season.casefold()))
        player_id = "fbref_" + hashlib.sha1(player_key.encode("utf-8")).hexdigest()[:14]
        prepared.append({
            "season": selected_season, "league": league, "source": "FBref user-exported CSV",
            "player_id": player_id, "name": name, "position": position, "team": team,
            "country": row.get("country", "").strip(), "minutes": round(minutes, 3),
            "pass_completion": pass_rate, "goals_per90": goals90, "assists_per90": assists90,
            "goals_assists_per90": goals90 + assists90 if goals90 is not None and assists90 is not None else None,
            "xg_per90": rate90("xg"), "shots_on_target_rate": sot_rate,
            "def_actions_per90": def_rate, "saves_per90": rate90("saves"),
            "clean_sheets_per90": rate90("clean_sheets"), "save_pct": save_pct,
            "xag_per90": rate90("xag"),
            "progressive_passes_per90": rate90("progressive_passes"),
            "progressive_carries_per90": rate90("progressive_carries"),
        })

    if not prepared:
        raise ValueError("No usable rows. Include Player, Squad, Pos, Min, and statistics columns.")
    leagues = {item["league"] for item in prepared if item["league"]}
    if len(leagues) > 1:
        raise ValueError(
            "FBref exports contain multiple competitions: " + ", ".join(sorted(leagues)) +
            ". Rank one competition at a time because the current model does not adjust for league strength."
        )
    from processing.scouting_channel import _normalize_position
    required = {
        "GK": ("pass_completion", "saves_per90", "clean_sheets_per90"),
        "DEF": ("pass_completion", "def_actions_per90", "goals_assists_per90"),
        "MID": ("pass_completion", "def_actions_per90", "xg_per90", "goals_assists_per90"),
        "FWD": ("goals_per90", "xg_per90", "shots_on_target_rate", "goals_assists_per90"),
    }
    for item in prepared:
        position = _normalize_position(item["position"])
        missing = [field for field in required[position] if item[field] is None]
        if missing:
            raise ValueError(
                f"Missing {', '.join(missing)} for {item['name']} ({position}). "
                "Add the relevant FBref Standard, Shooting, Passing, Defense, or Goalkeeping CSV export."
            )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(prepared[0]))
        writer.writeheader()
        writer.writerows(prepared)
    return len(prepared)
