"""StatsBomb -> Transfermarkt player crosswalk (needed for age and, optionally, market value).

StatsBomb Open Data contains no birth dates. Transfermarkt (local snapshot in ../transfermkt, a public
CC0 re-publication of Transfermarkt data) has date_of_birth and dated valuations. There is no shared ID,
so players are linked by name AND club evidence, never by name alone:

  1. Candidate generation: all name tokens of the Transfermarkt name appear in the StatsBomb legal name or
     nickname (accent-stripped, lower-cased). Mononyms must equal the nickname.
  2. Club evidence: the candidate has a valuation/transfer record at a club whose name matches the
     StatsBomb team within +/- 1 year of the StatsBomb season.
  3. Acceptance: tier A = exactly one candidate with club evidence; tier B = no club evidence available for
     any candidate but exactly one candidate shares the player's nationality; everything else is rejected
     (ambiguous / unmatched). Only tier A is used for modelling by default.

Output: data/research/processed/crosswalk.csv (+ crosswalk_audit in the console).
"""
from __future__ import annotations

import gzip
import json
import re
import unicodedata
from collections import defaultdict
from difflib import SequenceMatcher

import pandas as pd

from research.config import CACHE, PROCESSED, TRANSFERMARKT

STOP = {"de", "da", "do", "dos", "das", "del", "la", "le", "van", "von", "der", "den", "di", "i", "y", "e", "el", "al", "jr", "junior"}
CLUB_NOISE = {"fc", "cf", "cd", "ud", "sd", "ca", "rc", "rcd", "sc", "ac", "as", "us", "afc", "ssc", "sv", "vfb", "vfl", "tsg", "1", "club", "de", "del", "la", "real", "sporting", "balompie", "futbol", "deportivo", "atletico", "athletic"}


def strip(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s or "")).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9 ]", " ", s)


def name_tokens(s: str) -> set[str]:
    return {t for t in strip(s).split() if t not in STOP and len(t) > 0}


def club_tokens(s: str) -> set[str]:
    return {t for t in strip(s).split() if t not in CLUB_NOISE}


def club_match(sb_team: str, tm_club: str) -> bool:
    a, b = club_tokens(sb_team), club_tokens(tm_club)
    if a and b and (a <= b or b <= a):
        return True
    return SequenceMatcher(None, strip(sb_team).strip(), strip(tm_club).strip()).ratio() >= 0.8


def statsbomb_players() -> pd.DataFrame:
    """One row per (player_id, team, season) with names/nickname/country, from cached player-match records."""
    rows = {}
    for f in (CACHE / "player_match").glob("*.json.gz"):
        with gzip.open(f, "rb") as fh:
            d = json.loads(fh.read())
        for p in d["players"]:
            k = (p["player_id"], p["team"], d["competition_id"], d["season_id"])
            if k not in rows:
                rows[k] = {"player_id": p["player_id"], "name": p["player_name"], "nickname": p.get("player_nickname"),
                           "country": p.get("country"), "team": p["team"], "competition_id": d["competition_id"],
                           "season_id": d["season_id"], "first_date": d["match_date"]}
            else:
                rows[k]["first_date"] = min(rows[k]["first_date"], d["match_date"])
    return pd.DataFrame(rows.values())


def build() -> pd.DataFrame:
    sb = statsbomb_players()
    tm = pd.read_csv(TRANSFERMARKT / "players.csv", usecols=["player_id", "name", "date_of_birth", "country_of_citizenship", "position", "sub_position"])
    tm = tm.dropna(subset=["name"]).rename(columns={"player_id": "tm_id", "name": "tm_name"})
    tm["tok"] = tm.tm_name.map(name_tokens)
    val = pd.read_csv(TRANSFERMARKT / "player_valuations.csv", usecols=["player_id", "date", "current_club_name"], parse_dates=["date"])
    val = val.rename(columns={"player_id": "tm_id"})
    clubs_by_player = defaultdict(list)
    for r in val.itertuples(index=False):
        clubs_by_player[r.tm_id].append((r.date, r.current_club_name))
    tr = pd.read_csv(TRANSFERMARKT / "transfers.csv", usecols=["player_id", "transfer_date", "from_club_name", "to_club_name"], parse_dates=["transfer_date"])
    for r in tr.itertuples(index=False):
        for club in (r.from_club_name, r.to_club_name):
            clubs_by_player[r.player_id].append((r.transfer_date, club))
    # inverted index: token -> tm rows
    index = defaultdict(set)
    for i, toks in enumerate(tm.tok):
        for t in toks:
            index[t].add(i)
    out = []
    for pid, grp in sb.groupby("player_id"):
        g0 = grp.iloc[0]
        sb_tok = name_tokens(g0["name"]) | name_tokens(g0["nickname"] or "")
        nick_tok = name_tokens(g0["nickname"] or "")
        cand_idx = set()
        for t in sb_tok:
            cand_idx |= index.get(t, set())
        cands = []
        for i in cand_idx:
            tk = tm.tok.iat[i]
            if not tk or not tk <= sb_tok:
                continue
            if len(tk) == 1 and tk != nick_tok and not (tk <= nick_tok and len(nick_tok) == 1):
                continue
            cands.append(i)
        ev = {}
        for i in cands:
            tid = tm.tm_id.iat[i]
            ok = False
            for r in grp.itertuples():
                d0 = pd.Timestamp(r.first_date)
                for date, club in clubs_by_player.get(tid, []):
                    if abs((date - d0).days) <= 400 and isinstance(club, str) and club_match(r.team, club):
                        ok = True
                        break
                if ok:
                    break
            ev[i] = ok
        with_ev = [i for i in cands if ev[i]]
        tier, chosen = "none", None
        if len(with_ev) == 1:
            tier, chosen = "A", with_ev[0]
        elif len(with_ev) > 1:
            tier = "ambiguous"
        elif len(cands) >= 1:
            same_country = [i for i in cands if strip(tm.country_of_citizenship.iat[i]) == strip(g0["country"] or "")]
            if len(same_country) == 1 and len(cands) == 1:
                tier, chosen = "B", same_country[0]
            else:
                tier = "ambiguous" if len(cands) > 1 else "unverified"
        rec = {"sb_player_id": pid, "sb_name": g0["name"], "sb_nickname": g0["nickname"], "sb_country": g0["country"],
               "teams": "; ".join(sorted(set(grp.team))), "n_candidates": len(cands), "tier": tier}
        if chosen is not None:
            rec.update({"tm_id": int(tm.tm_id.iat[chosen]), "tm_name": tm.tm_name.iat[chosen],
                        "date_of_birth": tm.date_of_birth.iat[chosen], "tm_country": tm.country_of_citizenship.iat[chosen],
                        "tm_position": tm.position.iat[chosen]})
        out.append(rec)
    df = pd.DataFrame(out)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED / "crosswalk.csv", index=False)
    return df


if __name__ == "__main__":
    df = build()
    print(df.tier.value_counts())
