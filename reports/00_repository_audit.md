# 00 — Repository audit and initial test report

Audit date: 2026-10-09. Scope: `talent-scouting/` (code, data, tests) and the sibling `transfermkt/` folder.

## 1. Inventory
| Area | Files | Notes |
|---|---|---|
| Ranking (MCDA) | `processing/scouting_channel.py` | WSM, TOPSIS, VIKOR on min-max-normalised, position-weighted criteria; Excel report writer; CLI. |
| Data prep | `processing/prepare_statsbomb_dataset.py`, `processing/fbref_import.py` | StatsBomb events→player-season aggregates (World Cup 2022; La Liga via temporal module). |
| Temporal pilot | `processing/temporal_validation.py`, `processing/algorithm_selection_analysis.py` | 5 La Liga seasons, adjacent-season rank validation, bootstrap over folds. |
| Out of scope | `aws/`, `shared/cloud_infra.py`, `template.yaml`, `processing/injury_channel.py`, `main.py` | Cloud simulation / deployment: left untouched. |
| Data | `data/statsbomb_world_cup_2022_players.csv` (346 players), `data/la_liga_player_seasons.csv` (440 rows), `data/results/*.xlsx` | WC CSV is kept as the offline demo/regression fixture. |
| Tests | 4 files, 16 tests | unittest-style, run with pytest. |
| External | `../transfermkt/` (players, player_valuations 656k rows 2000–2026, transfers, …) | Public Transfermarkt snapshot; no StatsBomb IDs. |

## 2. Initial test run (before any change)
`python -m pytest -q` → **16 passed in 35 s**. Python 3.12.1, pandas 2.3.2, scikit-learn 1.6.1, xgboost 3.3.0, lightgbm 4.7.0.

## 3. How the existing methods work (from the code, not the README)
- **Normalisation**: min-max within a position cohort; constant criteria → 0 (`_normalize_group`).
- **WSM** = Σ wᵢ·xᵢ. **TOPSIS**: ideal = weight vector (since x∈[0,1]), anti-ideal = 0, closeness = d⁻/(d⁺+d⁻). **VIKOR** (v=0.5): `1 − (v·Q_utility + (1−v)·Q_regret)`. All three implementations are mathematically coherent and are **preserved and reused** (they become baselines).
- Weights are researcher-chosen constants (`POSITION_WEIGHTS`) and include **minutes** (20–30 % of the score) — exposure is treated as ability.
- **Temporal validation**: link players by StatsBomb `player_id` across adjacent seasons, keep same-position players, rank on season *t* only, correlate with a season *t+1* index, NDCG@10 per (season-transition × position) fold; 13 folds, 5–17 players each.
- **Algorithm selection**: mean NDCG@10 over folds; paired bootstrap over folds.

## 4. Findings (methodological weaknesses / bugs)
| # | Finding | Evidence | Severity | Action |
|---|---|---|---|---|
| F1 | **Target is circular.** `_future_index` re-uses the ranking criteria and weights (minus minutes) to build the "future performance" label. The test asks "does a weighted sum of X at *t* predict the same weighted sum at *t+1*", i.e. it measures metric persistence, favouring the ranker that mirrors its own target. | `temporal_validation._future_index` | High | New, separately-defined target (`03_targets.md`); legacy rankers become baselines. |
| F2 | **NDCG@10 floor is high.** With cohorts of 5–17 players, NDCG@10 covers nearly the whole list. Random scores give mean NDCG@10 = **0.759** on the same 13 folds vs 0.876–0.888 for the three methods. The README's 0.88 is therefore not evidence of strong skill. | measured by shuffling scores through `evaluate()` (30 repeats) | High | Report NDCG against a random/persistence baseline; use k relative to cohort size; pooled cohorts. |
| F3 | **Selection on the future.** Future index is min-max normalised over only the players linked to the next season, and players are required to keep the same position and ≥180 min in both seasons. Conditioning on future survival/role biases folds and the normalisation uses future cohort information. | `evaluate()` | Medium | Document eligibility; new pipeline reports the eligible population and handles attrition explicitly. |
| F4 | **Winner chosen on the evaluation folds** by mean NDCG@10 with 13 folds; paired CI includes 0 (README admits). The Excel "provisional best method" banner is still driven by this. | `algorithm_selection_analysis.compare` | Medium | Keep for legacy report; research conclusions use new protocol. |
| F5 | **Missing coded as zero.** `shots_on_target_rate` = 0.0 when a player took no shots (167/440 La Liga rows have both rate 0 and xG 0); `pass_completion` = 0 when no passes. | `prepare_statsbomb_dataset.aggregate_competition_season` | Medium | New features keep NaN and carry shot/pass counts. |
| F6 | **Penalty shoot-out events counted.** Events are not filtered by period; period 5 shots count as goals/xG. Verified on the 2022 final (match 3869685): 8 of 38 shots are shoot-out shots (6 goals, 6.27 xG). Affects World Cup fixture rows of players who took kicks. | event JSON | Medium | Fixed in `research/ingest/events.py` (periods ≤ 4). The legacy CSV is left unchanged as a fixture. |
| F7 | **Low-exposure noise.** 77 % of the La Liga pilot rows have < 450 min; per-90 rates from 180 min are very noisy and unshrunk. 75 % of rows are non-Barcelona players seen in ≤ 2 matches (max 191 min; all matches are Barça games). | `la_liga_player_seasons.csv` | High | Exposure thresholds + empirical-Bayes shrinkage examined as ablation. |
| F8 | **Bug**: `python -m processing.scouting_channel` crashed with `NameError: argparse`. | direct run | Low | **Fixed** (`import argparse` added). |
| F9 | Existing tests write an `archive/` directory at the repo root (side effect of `_archive_previous_reports` from a temp dir inside the project). | `git status` after test run | Low | Noted; removed after runs. Not changed. |
| F10 | `defensive_actions` = pressures + interceptions + blocks + clearances + recoveries + tackles lumps very different actions; pressures dominate. | prepare script | Low | New features keep them separate. |
| F11 | Position = most-minutes group from lineup positions, assigned at season level; "Wing Back" → DEF, "Attacking Midfield" → MID. Role switches during a season are hidden. | prepare script | Low | New pipeline stores role-minute shares. |
| F12 | StatsBomb Open Data has **no birth dates**, so age was never available to the pilot. | lineup schema | — | Age obtained via a documented Transfermarkt crosswalk. |

## 5. Reusable pieces
- `_rank_group`, `_weighted_sum`, `_topsis`, `_vikor`, `_normalize_group`, `PlayerProfile` — reused unchanged as baselines (`research/models/baselines.py`).
- Excel writer (`_save_rankings_workbook`) — kept for presenting results.
- `_spearman`, `_ndcg` in `temporal_validation.py` — mathematically correct; re-implemented with tie handling and tests in `research/evaluation/metrics.py`; legacy tests untouched.
- `data/statsbomb_world_cup_2022_players.csv` — offline demo/regression fixture (not used for training).

## 6. Implementation plan (grounded in the above)
1. Cached ingestion of StatsBomb club seasons → per player-match counts with provenance (`research/ingest`).
2. Data feasibility audit (`02_data_feasibility.md`) decides cohorts.
3. Player-season table, targets, models, rolling-origin evaluation, ablations, uncertainty, shortlist (`research/…`).
4. Tests for features, leakage, metrics, inference; README with reproduction commands.
