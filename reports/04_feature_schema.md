# 04 — Player-season dataset schema and feature definitions

Built by `research/ingest/events.py` (per match) → `research/features/player_season.py` (aggregation) → `research/features/panel.py` (designs, age, valuations).
Files: `data/research/processed/design_a_player_seasons.parquet` (370 rows; one per player × La Liga season, Barcelona-primary players), `design_b_blocks.parquet` (7,119 rows; one per player × block × league), `design_b_player_halves.parquet` (unused half-season variant), `crosswalk.csv`.

## Verification against the event schema
Checked on live StatsBomb event JSON (match 15978): event types present (`Pass`, `Carry`, `Shot`, `Dribble`, `Duel`, `Pressure`, `Ball Recovery`, `Block`, `Interception`, `Clearance`, `Foul Committed/Won`, `Dispossessed`, `Miscontrol`, `Dribbled Past`, `Goal Keeper`, `Ball Receipt*`); `pass.outcome` absent ⇒ completed; `pass.shot_assist`/`goal_assist`; `shot.statsbomb_xg`, `shot.type`, `shot.key_pass_id`; `duel.type` ∈ {Tackle, Aerial Lost}; `goalkeeper.type`; `period` 5 = shoot-out; lineups give `positions[from,to]` (minutes), `player_nickname`, `country`; **no birth date, no club history**.

## Per-match counts (`COUNT_FIELDS`, all restricted to periods 1–4)
| Column | Definition |
|---|---|
| minutes, role_minutes | Σ over lineup position segments of (to − from); `to = null` ⇒ last event clock of the match |
| passes, passes_completed | Pass events; completed ⇔ no `pass.outcome` |
| passes_progressive | completed pass moving the ball ≥ 10 yd closer to goal centre (120,40) |
| passes_final_third / passes_into_box | completed, start outside → end x ≥ 80 / end in box (x ≥ 102, 18 ≤ y ≤ 62) not starting in box |
| passes_long, crosses, through_balls | length ≥ 32.8 yd; `pass.cross`; `pass.through_ball` |
| key_passes, assists | `shot_assist` or `goal_assist`; `goal_assist` |
| xa | Σ xG of shots whose `key_pass_id` is the player's pass |
| shots, shots_on_target, goals, xg | Shot events; on target = Goal/Saved/Saved To Post; xG = `statsbomb_xg` (incl. penalties) |
| np_goals, npxg, penalties_taken | excluding `shot.type = Penalty` |
| carries, carries_progressive, carry_dist_progressive | Carry events; progressive = ≥ 10 yd closer to goal. **Carries faster than 12 yd/s are discarded** (tracking artefact: 49 yd in 1.2 s observed) |
| dribbles, dribbles_completed | Dribble events; outcome Complete |
| ball_receipts, dispossessed, miscontrols | event counts |
| pressures, tackles, tackles_won, interceptions, blocks, clearances, ball_recoveries | event counts (tackle = Duel type Tackle; won ⇔ Won/Success In Play/Success Out) |
| fouls_committed, fouls_won, aerials_lost, dribbled_past | event counts |
| gk_shots_faced, gk_saves, gk_goals_conceded, gk_claims, gk_punches, gk_sweeper | Goal Keeper event types (Shot Faced; Shot Saved/Penalty Saved/Shot Saved To Post; Goal Conceded; Collected; Punch; Keeper Sweeper) |
| clean_sheet | keeper with ≥ 60 GK minutes and 0 goals against |
Team rows per match: goals for/against, xG for/against (period ≤ 4), passes for/against, match length.

## Player-season columns (`aggregate` + `add_rates`)
`player_id` (StatsBomb, stable key), names, `team/team_id` (most minutes), `n_teams`, `minutes`, `matches`, `first_match`, `last_match`, all counts above summed, `position_group` (GK/DEF/MID/FWD by minutes; role→group map in `events.ROLE_GROUP`), `primary_role` (GK, CB, FB, DM, CM, AM, WM, W, CF), `position_purity` (share of minutes in the group), per-90 rates `<count>_p90 = count / minutes × 90`, ratios (`pass_completion`, `shot_accuracy`, `xg_per_shot`, `dribble_success`, `avg_pass_length`, `gk_save_pct`) that stay **NaN** when the denominator is 0 (never coded 0), `def_actions` (tackles+interceptions+blocks+clearances), `def_actions_mid` (tackles+interceptions+recoveries).
Context/quality: `team_matches`, `team_poss` (team pass share), `team_gd_pm`, `team_xgd_pm` (per match, in-window), `availability` (player minutes / team minutes observed), `matches_share`, `season_end`.
Identity/age/value: `tm_id`, `dob`, `age` (at `season_end`; NaN without tier-A link), `value_t`, `value_future`, `value_prev` (valuation panel only).

## Model feature sets (cumulative; `rolling.design_matrix`)
`pi` = performance index; `profile` = 14 z-scored rates (`PROFILE`) standardised within position group using the training reference (NaN→0); `agepos` = age, (age−27)², position dummies; `exposure` = log minutes, availability; `context` = team_poss, team_gd_pm, team_xgd_pm; `league` = competition dummies (Design B); `history` = PI[t−1] (filled with PI[t] if absent) + indicator + log minutes[t−1]; `valuation` = log V[t] (+ missing flag); `valtrend` = log(V[t]/V[t−12 m]).
Future targets are never present in any feature column (tested in `tests/test_research.py::TemporalLeakageTest`).

## Exclusions and known gaps
* No GK modelling; no age for most pre-2012 players (not in the Transfermarkt snapshot) ⇒ dropped by the age requirement.
* No tracking/360 data, no off-ball running, no aerial *wins* (events record only losses), no post-shot xG for goalkeepers.
* Team strength is proxied by pass share and goal/xG difference over the observed matches only; for Design A (always Barcelona) it varies only across seasons.
* Position is the dominant group by minutes; mid-season role changes are summarised, not modelled.
