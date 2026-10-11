# 01 — Literature review, gap, and pre-specified hypotheses

*Status: written before any model was fitted. Items marked **[verified]** were confirmed by a web lookup of title, venue and authors during this session; items marked **[from memory]** were not re-checked and must be verified before they are cited in a publication.*

## 1. What the literature covers

**Player evaluation from event data.**
- Pappalardo et al., *PlayeRank* (ACM TIST 10(5), 2019) **[verified]**: learns position-specific weights over event-derived features to rate and rank players per match, validated against expert-style rankings — i.e. an *evaluation* of current performance, not a forecast of future development.
- Decroos, Bransen, Van Haaren, Davis, *Actions Speak Louder than Goals* (VAEP, KDD 2019) **[verified]**: values every on-ball action by its effect on scoring/conceding probability. Again a measure of past contribution.
- Multi-criteria schemes (weighted sum, TOPSIS, VIKOR) are common in applied scouting write-ups; weights are usually analyst-chosen. This repository's original rankers are of this kind.

**Market-value modelling.**
- Müller, Simons, Weinmann, *Beyond crowd judgments* (EJOR 263(2), 2017) **[verified]**: multilevel regression on performance, age, position, club and popularity covariates reproduces Transfermarkt crowd values for ~4,200 top-five-league players over six seasons. This explains the *current* crowd value; it is not a time-ordered prediction of value change.
- Related work on determinants of Transfermarkt values (e.g. Herm et al. 2014 on crowd accuracy; Majewski 2016; Franceschi et al. 2024 systematic review) **[from memory]**. Common finding: age, position, league, club and recent minutes dominate; event statistics add modest increments.

**Talent identification / development with ML.**
- Recent reviews and application papers use random forests, SVMs and gradient boosting, mostly on *youth/academy* data (anthropometric, physical, technical tests) or video-game ratings **[search results; not individually verified]**. Reported accuracies are often on random splits of small samples, which the broader literature itself lists as a limitation (data quality, external validity, methodological rigor).
- Aging curves: players typically peak in the mid/late 20s (e.g. Dendir 2016) **[from memory]**.

## 2. Common evaluation weaknesses relevant here
1. Random row-level splits when the same player appears in several seasons (leakage by identity).
2. Predicting *current* value/rating rather than a later outcome.
3. Evaluating on survivors only (players observed in the next season) and ignoring the selection this induces.
4. Per-90 rates from small minutes treated as precise; no shrinkage or reliability analysis.
5. Reporting a single aggregate score with no uncertainty and no simple baselines (persistence, age curve).
6. Weights/targets that are functions of the same features being used to "validate" them (circularity) — this applies to the repository's pilot validation too (see `00_repository_audit.md`).

## 3. Gap this project can examine (and what it cannot)
With **only StatsBomb Open Data** (free, event-level, player IDs) the project can examine, under strict time-ordered evaluation, **whether event-derived features, age, position, exposure and team context add predictive value for a player's *next-season* performance beyond persistence and an age/position curve, and beyond MCDA rankings**. It cannot:
- claim novelty for the algorithms themselves (none is new),
- generalise to leagues: StatsBomb's open club data are dominated by FC Barcelona matches (see `02_data_feasibility.md`),
- make claims about *potential* of academy players or about transfers/profit.
Whether the combination "next-season, time-ordered, shrinkage-aware, with MCDA and persistence baselines, on public data" is novel was **not established**; no novelty claim is made.

## 4. Hypotheses and what would count as evidence

| ID | Hypothesis | Evidence **for** | Evidence **against** |
|---|---|---|---|
| H1 | Current performance index is a strong predictor of next-season index (persistence). | Spearman ≥ 0.5 pooled; persistence beats mean baseline in all/most test folds. | Persistence ≈ mean baseline. |
| H2 | Supervised ML beats persistence **and** age/position baseline for next-season performance. | Lower RMSE with paired-bootstrap CI (over test players, clustered by player) excluding 0, in ≥ most folds. | CI includes 0 or ML worse. Small differences (< CI width) are *not* evidence. |
| H3 | Supervised ML ranks next-season performers better than WSM/TOPSIS/VIKOR (NDCG@k, Spearman). | Paired difference CI excludes 0. | CI includes 0. |
| H4 | Age/position/context features add value over performance-only features (ablation). | Held-out improvement that survives CI. | No improvement (plausible: age range is narrow in a single elite club). |
| H5 | Reliability adjustment (shrinkage/exposure weighting) improves prediction. | Held-out improvement. | None. |
| H6 | Breakout (top-20 % improvement) is predictable above base rate. | PR-AUC ≫ prevalence with CI; Brier skill > 0. | PR-AUC ≈ prevalence (expected if improvement is mostly noise / regression to the mean). |
| H7 | Performance features predict future log market-value change. | Only testable if time-aligned Transfermarkt labels link to enough players. | Insufficient linkage or no signal. |

**Decision rules fixed before modelling:** primary comparison metric = RMSE on the next-season performance index (pooled over test transitions); ties/overlapping CIs are reported as "no demonstrated difference". Model configurations are chosen only on training-window data (inner chronological validation). Test-fold results never feed back into selection.
