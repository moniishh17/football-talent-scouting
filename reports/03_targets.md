# 03 — Target definitions (fixed before model comparison)

All definitions below are implemented in `research/targets/performance_index.py`, `research/run_breakout.py`, `research/run_valuation.py`, and were not altered after seeing held-out results (one *bug fix* to the shrinkage-variant plumbing is documented in `05_results.md` §Deviations).

## 1. Performance index (PI) — primary construct

`PI = mean over the position group's components of  sign · (rate − μ_g) / σ_g`, equal weights; μ_g, σ_g from a **reference sample** (player-seasons ≥ min-minutes, same position group) **fitted on the training window only** (all seasons up to the scouting season). Minutes are never a component. Goalkeepers are excluded (1–2 per team; no post-shot xG in the event schema; n = 22 player-seasons in Design A).

| Group | Components (sign) |
|---|---|
| FWD | non-penalty xG/90 (+), xA/90 (+), non-penalty goals/90 (+), key passes/90 (+), completed dribbles/90 (+), progressive carries/90 (+) |
| MID | xA/90 (+), key passes/90 (+), progressive passes/90 (+), progressive carries/90 (+), passes into box/90 (+), ball-winning actions/90 [tackles+interceptions+recoveries] (+) |
| DEF | defensive actions/90 [tackles+interceptions+blocks+clearances] (+), progressive passes/90 (+), progressive carries/90 (+), pass completion (+), times dribbled past/90 (−), aerial duels lost/90 (−) |

Why this is *not* the legacy ranking score: different metric set (process/progression metrics instead of goals, raw pass completion and minutes), equal weights instead of researcher weights, no minutes term, z-scores against a fixed reference instead of cohort min-max. Test `test_index_is_not_the_legacy_ranking_score` asserts |Spearman| < 0.9 with the legacy WSM score; measured cross-metric agreement is only ρ ≈ 0.13–0.28 (`tables/target_sensitivity.csv`).

**Formulations.** A: `y = PI[t+1]` (level). B: `Δ = PI[t+1] − PI[t]`, predictions mapped back to the level scale for evaluation (RMSE/MAE are always on PI[t+1]). Both are run for every model.

**Eligibility and missing outcomes.** Scouting row: outfield player-season with ≥ 450 min (Design A) / ≥ 360 min (Design B), age known (tier-A crosswalk) — *identical cohorts for every model*. Outcome row: same player observed in the next season/block with ≥ the same threshold. A player with no such observation (left Barcelona, injured, benched) has **no label** and is not evaluated; for Design A this is 27 % of eligible player-seasons and those players had a *lower* current PI (−0.14 vs +0.06, Mann–Whitney p = 0.008): results are conditional on retention. Position changes: the outcome PI uses the player's position group *in the outcome season* (95 % keep the same group); no special handling beyond that.

**Measurement noise.** Odd/even matchweek split-half (n = 265 player-seasons): r = 0.68, Spearman–Brown reliability 0.81. The implied noise SD of a season-to-season *change* is 0.36 against an *observed* SD of change of 0.35, i.e. the estimated signal share of Δ's variance is ≈ 0 (`tables/reliability_design_a.csv`). The estimate is rough (odd/even halves share seasonal conditions), but it means next-season *change* is barely distinguishable from noise at this exposure.

**Sensitivity examined** (`tables/sensitivity.csv`): exposure thresholds (270/450/900 min for A; 180/360/600 for B), position reference by 4 groups vs 8 roles, empirical-Bayes shrinkage of features (and of target), target definition (new PI vs legacy index, `target_sensitivity.csv`).

## 2. Breakout label — secondary, exploratory
`breakout = 1` iff `Δ ≥ q80` of training-pair Δ **within position group** (threshold fixed on training pairs only; prevalence in the test folds is therefore not forced to 20 %: 21.7 % in Design A, 24 % in Design B).
**Pre-specified feasibility gate:** classification is reported as *exploratory* if the estimated signal share of Δ is < 0.3. The gate **failed** (≈ 0) for Design A, so the experiment is a documented falsification check, not a validated target. Minimum sizes used: ≥ 20 training pairs and ≥ 1 positive per training fold. Models: logistic regression, random forest, LightGBM, vs prevalence, "low current PI" (pure regression to the mean), "young age", inverted MCDA. Metrics: PR-AUC, ROC-AUC, Brier (+skill), precision/recall@top-20 %, calibration.

## 3. Market-valuation target — secondary
`y = log(V[t+12 m] / V[t])` from the local Transfermarkt snapshot (dated EUR valuations). `V[t]` = last valuation at/before the scouting date (end of season t) and ≤ 200 days old; `V[t+12 m]` = valuation closest to scouting date + 365 d within ± 120 d (strictly after the scouting date). Current/`players.csv` market value is **never** used. Rolling origin: test scouting season y, training rows t ≤ y−1 (their labels are observed by the end of y). Players need a tier-A crosswalk link, ≥ 450 min, age. Valuation is a crowd-sourced proxy, not transfer profit or ability.
Terms-of-use caveat: see `02_data_feasibility.md` §6 (user should confirm permission; data are not redistributed).

## 4. Validation protocol (all targets)
* Design A: rolling origin, test transitions y→y+1 for y = 2009…2019 (11 folds, 143 test pairs). **Development** = y ≤ 2014 (selection of model/feature set by pooled RMSE), **confirmation** = y ≥ 2015 (reported only).
* Design B: within-season blocks 1–4 (~9–10 matchweeks, 4 leagues, 2015/16); test block2→3 (train block1→2; *development*) and block3→4 (train 1→2, 2→3; *confirmation*).
* Valuation: test scouting seasons 2012…2020; development ≤ 2016, confirmation ≥ 2017.
* Reference statistics, shrinkage priors, hyper-parameters (inner chronological split; Design B: player-grouped CV inside training window) all use training-window data only.
* Uncertainty: cluster (player) bootstrap, 1,000–2,000 resamples; split-conformal intervals calibrated on earlier folds.
