# 05 — ScoutIQ Research: methods, results, limitations

*All numbers are produced by the commands in the README from cached StatsBomb Open Data (credit: StatsBomb, https://github.com/hudl/open-data) and a local Transfermarkt snapshot. Tables: `reports/tables/*.csv` (and `ScoutIQ_research_results.xlsx`); figures: `figures/{audit,eda,models}/`. Intervals are 95 % percentile bootstrap over **players** (clusters) unless stated. "No demonstrated difference" means the paired interval contains 0 — it is not evidence of equality.*

## 1. Summary (what can and cannot be said)

1. **Data are the binding constraint.** StatsBomb's open *club* data are single-club subsets (FC Barcelona matches in La Liga 2004/05–2020/21; PSG, Leverkusen, Inter Miami elsewhere) plus four complete leagues in 2015/16. No two consecutive seasons of any complete league exist, and no 2025-era men's club data exist. The only valid next-season panel is **Barcelona players**: 143 held-out player-season pairs from 11 chronological folds (62 in the confirmation period).
2. **Next-season performance (Design A):** performance is highly persistent (Spearman 0.82 between the index at *t* and *t+1*). **No supervised model demonstrably beats calibrated persistence** (confirmation RMSE 0.313–0.325 vs 0.321; paired differences −0.008 [−0.043, +0.027] at best). The age/position, exposure, team-context and history feature sets add nothing demonstrable. All baselines that ignore the player's own performance, and the three legacy MCDA rankers, are far worse (RMSE 0.54–0.60 vs 0.32).
3. **Short-horizon performance with n ≈ 1,400 (Design B, 4 leagues, next ~9-matchweek block):** richer models *do* beat calibrated persistence: random forest with profile + history features RMSE 0.412 [0.387, 0.438] vs 0.448 [0.420, 0.476] in the confirmation fold, paired difference −0.035 [−0.049, −0.022] (≈ 8 % lower error). The gain comes from using the **full profile of rates** and **earlier-window history** (i.e. averaging more noisy observations), not from age, position, exposure, team context or league.
4. **Change is mostly noise.** The estimated noise SD of a season-to-season change in the index (0.36) is as large as its observed SD (0.35). Predicted changes correlate only weakly with realised ones (Spearman 0.18 [0.04, 0.35] for persistence; 0.26 [0.11, 0.43] best model in Design A; 0.37 → 0.49 in Design B) and mostly reflect regression to the mean.
5. **Breakout labels (top-20 % improvement) are not a reliable construct here.** Classifiers beat the base rate in Design B (PR-AUC 0.44 vs prevalence 0.24, Brier skill ≈ 0.07) but **not** the zero-parameter heuristic "currently low index" (0.41, CI overlapping); in Design A they are indistinguishable from base rate.
6. **Market value (secondary, feasible):** valuation change is explained mostly by age and starting value; adding performance features gives a point-estimate improvement (confirmation RMSE 0.480 vs 0.510 for the age/position/value baseline) whose interval includes 0 (−0.030 [−0.075, +0.020], n = 75).
7. **The legacy pilot's conclusions do not survive scrutiny:** its target was circular (see `00_repository_audit.md`) and its NDCG@10 floor is 0.76 for random scores. Which ranker looks best depends on which index defines "future performance": each index predicts *its own* future well (ρ ≈ 0.49 legacy-on-legacy, 0.82 new-on-new) and the other poorly (ρ ≈ 0.13–0.28). No independent outcome (e.g. later transfer fee, contract, expert rating) exists in the data to adjudicate.
8. Latest-season research shortlist generated (La Liga 2020/21 Barcelona; exploratory out-of-domain Ligue 1 2022/23 PSG, Bundesliga 2023/24 Leverkusen). These are model outputs, not evidence of future success.

## 2. Methods in brief
* **Ingestion:** `research/ingest` — resumable cache (gzip per match; manifest with URL, bytes, sha256, fetch time). 2,135 matches processed. Event features per player-match (`04_feature_schema.md`), shoot-outs excluded, anomalous carries filtered.
* **Audit:** `02_data_feasibility.md` (all male, non-youth club competition-seasons in the open data; cohort sizes; linkage; missingness; what each season can support).
* **Identity/age/value:** StatsBomb `player_id` for linking seasons; Transfermarkt crosswalk by name tokens **and** club evidence (tier A: 3,939 of 4,705 players, 84 %; 85 % of Design A player-seasons). Ambiguous/unverified players are excluded.
* **Targets:** `03_targets.md`. **Models:** ridge, elastic net, random forest, LightGBM (one boosted-tree implementation, chosen for small tabular data and speed), each with level and delta formulations and 6–7 cumulative feature sets; baselines: train mean, mean by position, age/position ridge, raw and calibrated persistence, equal-weight legacy ranking, legacy WSM/TOPSIS/VIKOR (calibrated linearly to the target for RMSE; raw scores for ranking metrics).
* **Validation:** rolling-origin with a development period (selection) and confirmation period (reporting); see `03_targets.md` §4. Hyper-parameters tuned inside the training window only. Leakage controls are unit-tested (corrupting all post-scouting data leaves predictions unchanged).

## 3. Results

### 3.1 Design A — next-season index (Barcelona panel, n = 143; confirmation n = 62)
| Model | Dev RMSE | Confirm RMSE [95 % CI] | All-folds RMSE | Spearman (all) |
|---|---|---|---|---|
| Train mean | 0.609 | 0.566 [0.456, 0.686] | 0.591 | 0.01 |
| Age/position ridge | 0.627 | 0.567 [0.472, 0.667] | 0.602 | −0.05 |
| Legacy WSM / TOPSIS / VIKOR (calibrated) | 0.583 / 0.578 / 0.599 | 0.540 / 0.543 / 0.543 | 0.564 / 0.563 / 0.575 | 0.15 / 0.16 / 0.12 |
| Equal-weight legacy | 0.579 | 0.538 | 0.562 | 0.17 |
| Persistence (raw) | 0.339 | 0.332 [0.249, 0.435] | 0.336 | 0.82 |
| **Persistence (calibrated)** | 0.330 | 0.321 [0.248, 0.408] | 0.326 | 0.82 |
| Ridge on PI (dev-selected overall) | 0.328 | 0.324 [0.252, 0.410] | 0.326 | 0.82 |
| Elastic net, PI, delta | 0.331 | 0.325 | 0.329 | 0.82 |
| RF full + history (dev-selected RF) | 0.342 | 0.313 [0.253, 0.384] | 0.330 | 0.81 |
| LightGBM, PI, delta | 0.345 | 0.321 | 0.335 | 0.81 |

Paired (confirmation, vs calibrated persistence): ridge +0.003 [−0.001, 0.006]; enet +0.004 [−0.005, 0.014]; RF −0.008 [−0.043, 0.027]; LightGBM 0.000 [−0.017, 0.014] — **no demonstrated difference**. Versus legacy WSM all selected models are better by ≈ 0.23 RMSE (interval excludes 0). Fig. M1, M3, M4. The development-selected model is therefore effectively a linear persistence model (`ridge|pi|level`), which is what was frozen.

*Ablation* (cumulative; Fig. M2, `A_ablation.csv`): adding profile z-scores, age/position, exposure or team context never reduced held-out RMSE with an interval excluding 0 (ridge got significantly *worse* with age/position, +0.019 [0.003, 0.038], and team context). Only `history` helped the random forest (−0.016 [−0.027, −0.004]) and that forest still does not beat the one-feature ridge (0.330 vs 0.326). **Negative result** — plausible reasons: only ~100–170 training rows for 14–25 correlated features; the index is already a summary of the profile; age range at a single elite club is narrow (mean 27.7); next-season change is mostly noise (§3.4).

*Recruitment-style ranking* (cohort = all eligible outfield players in a test fold, ~10–17): top-3 shortlist outcome lift (mean realised index minus cohort mean) 0.72 for persistence/ridge/RF, 0.38 WSM, 0.39 TOPSIS, 0.30 VIKOR, 0.42 equal-weight, −0.21 train-mean (arbitrary order). NDCG@3: 0.96 vs 0.75–0.81 (`A_ranking.csv`). These folds have 10–17 players; treat as descriptive.

### 3.2 Design B — next ~9-matchweek block, four 2015/16 leagues (n = 1,382; confirmation n = 691)
| Model | Dev RMSE | Confirm RMSE [95 % CI] | All | Spearman |
|---|---|---|---|---|
| Train mean / age-position | 0.505 | 0.574 / 0.573 | 0.540 | 0.00 |
| Legacy WSM / TOPSIS / VIKOR | 0.498 / 0.501 / 0.497 | 0.567 / 0.569 / 0.564 | 0.533 / 0.536 / 0.532 | 0.14 / 0.11 / 0.16 |
| Persistence (raw) | 0.453 | 0.468 [0.438, 0.500] | 0.460 | 0.60 |
| Persistence (calibrated) | 0.407 | 0.448 [0.420, 0.476] | 0.428 | 0.59 |
| Ridge, full + history | 0.388 | 0.416 [0.391, 0.441] | 0.402 | 0.65 |
| **Random forest, full + history (dev-selected)** | **0.378** | **0.412 [0.387, 0.438]** | **0.395** | **0.66** |
| LightGBM, PI + profile | 0.402 | 0.431 [0.405, 0.457] | 0.417 | 0.62 |

Paired vs calibrated persistence, confirmation: RF −0.035 [−0.049, −0.022]; ridge −0.032 [−0.045, −0.019]; elastic net −0.033 [−0.046, −0.021]; LightGBM −0.017 [−0.027, −0.008]. Versus legacy WSM: −0.14 [−0.16, −0.12] (all folds).
*Ablation* (RF, `B_ablation.csv`): PI only 0.443 → + profile 0.412 (−0.031 [−0.040, −0.021]) → age/position −0.001 [−0.004, 0.001] → exposure −0.001 → team context −0.002 [−0.006, 0.001] → league +0.003 [0.001, 0.004] (slightly worse) → history −0.015 [−0.022, −0.010]. Ridge/LightGBM agree in sign: profile and history help; age/position, exposure and league do not (ridge exposure +0.005 worse). Delta skill: Spearman of predicted vs realised change 0.49 [0.46, 0.53] (RF) vs 0.37 [0.33, 0.41] (persistence); R² of change 0.26 vs 0.13. Top-10 shortlist lift 1.32 (RF) vs 1.14 (persistence) vs 0.37–0.66 (legacy rankers); NDCG@10 0.88 vs 0.80 vs 0.55–0.65. Caveat: only two test folds, a ~2-month horizon, and a small positive drift of outcomes relative to predictions (mean residual +0.05).

### 3.3 Sensitivity (`sensitivity.csv`, Fig. M10) — ratio of model RMSE to the variant's own persistence baseline
* **Exposure thresholds:** Design B RF ratio 0.925 [0.907, 0.946] at 360 min, 0.947 at 180 min, 0.965 [0.933, 1.001] at 600 min — gains shrink as the threshold rises (less noise left to remove). Design A: no model beats persistence at 270, 450 or 900 min (ratios 1.05–1.21).
* **Reliability adjustment (empirical-Bayes shrinkage of rates):** no material change (B RF 0.925 → 0.922; A ridge+profile 1.05 → 1.04). Not supported as a meaningful improvement here; the profile+history models already average noise.
* **Position definition (4 groups vs 8 roles as z-score reference):** same qualitative picture (B RF 0.934; A no gain).
* **Target definition** (`target_sensitivity.csv`): see Summary 7. Legacy MCDA scores correlate 0.49 (A) / 0.36 (B) with the legacy future index — about the same as persistence of the legacy index (0.49 / 0.45) — and 0.26–0.28 (A) / 0.13–0.19 (B) with the new index; the new index's own persistence is 0.82 / 0.60.

### 3.4 Noise, attrition, breakout
* Split-half reliability of the season index 0.81; implied noise SD of change 0.36 vs observed 0.35 → signal share of change variance ≈ 0 (`reliability_design_a.csv`, Fig. E9).
* Attrition (Design A): 202 of 275 eligible outfield player-seasons (73 %) have a next-season observation; the missing ones had lower current index (−0.14 vs +0.06, p = 0.008). Reported accuracy is conditional on retention at Barcelona; players who left (often after poorer seasons) are invisible.
* **Breakout (exploratory; gate failed for A)** (`run_breakout`, Fig. M7–M8). Design A (n = 143, prevalence 21.7 %): "low current index" PR-AUC 0.43 [0.31, 0.63]; best learned model 0.32 [0.19, 0.49]; Brier skill −0.03…+0.01. Design B (n = 1,382, prevalence 24 %): RF 0.44 [0.39, 0.49], LightGBM+history 0.43, logistic 0.41–0.42, "low current index" 0.41 [0.36, 0.46], young-age heuristic 0.26, inverted legacy WSM 0.32; Brier skill 0.06–0.08. Interpretation: a modest, real tendency for low-index players to rebound (regression to the mean) — not a learned "talent" signal.

### 3.5 Valuation target (n = 162 test rows; confirmation n = 75; `V_*.csv`, Fig. M1/M2)
Age/position/value baseline RMSE 0.494 (all) / 0.510 (confirm); mean change 0.553 / 0.583; zero change 0.602 / 0.665. Dev-selected RF (performance + age/position + valuation + trend) 0.453 / 0.480 [0.374, 0.583]. Paired vs the age/position/value baseline: confirmation −0.030 [−0.075, 0.020]; all folds −0.042 [−0.078, −0.008] (the all-fold figure includes the development period used for selection and is optimistic). Ablation: age/position is the dominant contributor (−0.05…−0.07 RMSE over performance only); **performance-only vs performance + historical valuation features** (`perf+agepos` vs `perf+agepos+valuation`): 0.469/0.458/0.464 vs 0.468/0.457/0.463 (ridge/RF/LightGBM) — no difference; valuation alone is worse than age/position alone. Valuations are a crowd proxy; label noise and the Barcelona-only sample (very high-value players) limit external validity.

### 3.6 Uncertainty and error analysis (`uncertainty_coverage.csv`, `error_by_subgroup.csv`, `minutes_bias.csv`)
* Split-conformal intervals (calibrated on earlier folds only): Design A nominal 80 % → 81.4 % (n = 102, mean width 0.83), 90 % → 91.2 %; Design B nominal 80 % → 73–76 %, 90 % → 85–88 % (calibrated on a single earlier block; under-covers). Exchangeability across seasons is assumed, not guaranteed.
* Subgroups (Design A, n = 143): RMSE by position DEF 0.34 / MID 0.33 / FWD 0.28; by minutes 900–1,800 0.41 vs 1,800–2,700 0.28; by data coverage (< 50 % of team matches) 0.43 vs (> 80 %) 0.26 — error is highest where exposure is lowest. Several cells (age < 23, minutes < 900, roles DM/CF/WM) have n < 20 and are not interpretable. Design B: forwards 0.44, midfielders 0.43, defenders 0.33; wingers/wide midfielders/attacking midfielders have the largest errors (0.51–0.55) and positive bias (+0.13–0.23: under-prediction); by league 0.37–0.41 (Premier League bias +0.11, Ligue 1 +0.02). Design B minutes bands are uninformative (blocks cap at ~950 min).
* Minutes bias: given the current index, predictions do not depend on minutes (coefficient on log minutes −0.003 A, −0.02 B). Predicted top-k lists contain high-minute players (A: 2,369 vs cohort 1,990) but exactly as many as the *realised* top-k (2,386): starters really do have higher indices.
* Feature importance is **not** reported as evidence of mechanism; none of the models is causal.

## 4. Latest-season candidate assessment (frozen model)
* **Model:** `ridge|pi|level` (development-selected; fit on all 173 age-known transitions through 2020/21; metadata and reference in `artifacts/design_a_frozen/`).
* **In-domain:** La Liga 2020/21, Barcelona outfield players ≥ 450 min (19 players) → `reports/candidates_in_domain_la_liga_2020_21.csv` with observed per-90 features, predicted 2021/22 index with 80 %/90 % conformal intervals (half-width 0.41/0.54 index units), rank, minutes band, prior seasons observed, data-quality flags, a model-grounded explanation and — where available — an *estimated* 12-month market-value change (from the validated-protocol valuation fold; confirmation interval vs the age baseline includes 0).
* **Exploratory out-of-domain** (no validation; reference/model trained on Barcelona seasons): Ligue 1 2022/23 PSG (19 players) and Bundesliga 2023/24 Leverkusen (17) → `candidates_out_of_domain.csv`.
* Wording: "highest predicted index / above-cohort predicted change **relative to this cohort**", with intervals that are wide (±0.4 on an index whose between-player SD is 0.6). Predicted *change* is almost entirely mean reversion (low-index players are predicted to rise). **No breakout probabilities or "future star" labels are produced**, because the breakout label failed its feasibility gate.
* Intervals mean "expected spread of next-season index among similar Barcelona players", not a guarantee for any individual.

## 5. Deviations from the original plan, bugs found, and honesty notes
1. Design B was first conceived as leave-one-league-out on a half-season split; I replaced it **before any results** with a four-block rolling origin because training on other leagues' same-period outcomes is not time-respecting.
2. Design A is restricted to matches involving Barcelona so that 2015/16 (complete season) has the same observation process as the other seasons.
3. A first exploratory run (13 specifications, folds 2009–2019) was inspected **before** the development/confirmation split was formalised; it already showed persistence beating richer models. The split (≤ 2014 / ≥ 2015), the index definition and selection rule were fixed before the full grid was run, but this exposure is disclosed.
4. Bugs found and fixed during the work (all before the final result tables): (a) role-level reference grouping filtered on the wrong column (all roles dropped); (b) the first shrinkage-variant implementation recomputed the "raw" target from already-shrunk columns and later compared a raw target to a shrunk reference (scale mismatch) — fixed by giving the target its own training-window reference and re-running the affected variants; (c) cluster-bootstrap returned empty arrays for constant predictors (now NaN). The main grids (no shrinkage) were not affected.
5. The valuation target was initially expected to be infeasible; the local Transfermarkt snapshot turned out to be time-aligned and linkable (≈ 95 % of tier-A player-seasons have both labels), so it was run. Terms-of-use status of that snapshot is for the user to confirm (`02_data_feasibility.md` §6).
6. Hyper-parameter grids are tiny by design; richer tuning on ~100–170 rows would mostly fit noise. Absence of an ML gain in Design A may partly reflect low power: the confirmation CI half-width (≈ ±0.035) is larger than any plausible gain.

## 6. Limitations
* Single-club panel (Barcelona; elite, possession-dominant, 25 years of squads) — results do not generalise to other clubs, leagues, or to academy "potential"; between-player SD of the index is large, which flatters persistence.
* n is small: 143 held-out pairs from ~60 unique players; folds of 10–17; one season-pair per player is far from independent (cluster bootstrap addresses within-player dependence, not between-season dependence).
* Outcomes missing not-at-random (departures); results conditional on retention.
* Index definition is a researcher choice; conclusions about rankers depend on it (§3.3). No independent ground truth for "talent" or "development".
* Age comes from name/club matching to Transfermarkt (84 % linked, 15 % of Design A rows lack age); pre-2012 players mostly unlinked.
* StatsBomb event definitions and xG model version differ across seasons; no 360 or tracking data; goalkeepers excluded.
* Design B horizon is ~2 months, mixes four leagues, and has two test folds.
* Valuation labels are crowd opinions; Transfermarkt terms not verified.

## 7. What would strengthen the study (future experiments)
* Complete multi-season league data (e.g. another provider with licence for research) so that next-season targets cover non-Barcelona players and transfers; a labelled outcome such as subsequent minutes at a higher-level club, transfer fee, or expert rating.
* A time-stamped valuation dataset with documented licence and explicit club/league fields, to model valuation growth with club-level controls.
* Larger, multi-club samples to power ablations for age/position/context (current CIs cannot exclude gains of ~0.03 RMSE).
* Pre-registered breakout definition after estimating its reliability on a larger sample.

## 8. Figure index
Audit: A1 coverage, A2 continuity. EDA: E1 minutes, E2 missingness, E3 distributions, E4 correlations, E5 age/development, E6 role profiles, E7 exposure noise, E8 targets, E9 reliability, E10 attrition, E11 confounders, E12 market values (`notebooks/01_eda.ipynb`). Models: M1 confirmation RMSE forest, M2 ablation, M3 per-fold, M4 predicted vs observed, M5 subgroup error, M6 shortlist lift, M7 breakout PR-AUC, M8 calibration, M9 interval coverage, M10 sensitivity.
