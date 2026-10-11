# ScoutIQ: Football Talent Forecasting & Persistence Benchmarking Platform

[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-40%20passed-brightgreen.svg)](tests/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Repository](https://img.shields.io/badge/GitHub-moniishh17%2Ffootball--talent--scouting-blue.svg)](https://github.com/moniishh17/football-talent-scouting)

This repository contains the complete **ScoutIQ platform** for prospective football talent forecasting and autoregressive persistence benchmarking:
1. **Research & Modeling Pipeline** (`research/`): Rigorous rolling-origin temporal validation benchmarking regularized linear estimators (Ridge, Elastic Net), tree ensembles (Random Forest), and gradient boosting (LightGBM) against calibrated persistence across 17 seasons of StatsBomb telemetry and Transfermarkt economic valuations.
2. **Interactive Web Dashboard** (`app/streamlit_app.py`): Full-featured Streamlit analytics application with transparent ranking criteria, conformal uncertainty corridors, and interactive Plotly tactical radars.
3. **Legacy Multi-Criteria Ranker** (`processing/`): Production implementation of Weighted Sum (WSM), TOPSIS, and VIKOR algorithms with position-specific weighting.
4. **Cloud & Serverless Ingestion** (`aws/`, `template.yaml`): Event-driven AWS Lambda processor and AWS SAM deployment template.

## Where Things Are
| Path | Content |
|---|---|
| `The_Persistence_Trap_IEEE_Journal_Paper.docx` | Full 18-page IEEE journal paper with complete empirical benchmarks and tables |
| `reports/00_repository_audit.md` | Audit of legacy ranking methods and initial defect analysis |
| `reports/01_literature_and_hypotheses.md` | Literature review, theoretical gap, and pre-specified hypotheses (H1–H7) |
| `reports/02_data_feasibility.md` | Longitudinal data availability audit, entity resolution, and temporal splits |
| `reports/03_targets.md`, `04_feature_schema.md` | Performance Index mathematical construction and per-90 rate feature schemas |
| `reports/05_results.md` | **Comprehensive Research Report**: Empirical findings, ablations, conformal calibration, and limitations |
| `reports/tables/`, `reports/ScoutIQ_research_results.xlsx` | All empirical benchmark tables (CSV and consolidated Excel workbook) |
| `reports/candidates_*.csv` | Prospective scouting candidate shortlists scored with frozen model |
| `figures/audit`, `figures/eda`, `figures/models` | High-resolution publication figures and diagnostic plots |
| `artifacts/design_a_frozen/` | Serialized model (`model.joblib`), reference stats (`reference.joblib`), and metadata |
| `data/research/` | Processed panels (`design_a_player_seasons.parquet`, `design_b_blocks.parquet`) and backtest predictions |
| `research/` | Ingestion, feature engineering, targets, model zoo, conformal evaluation, and inference engines |
| `app/` | Multi-page Streamlit web application (`app/streamlit_app.py`, `app/data.py`) |
| `aws/`, `template.yaml` | AWS Lambda serverless handler and CloudFormation/SAM deployment manifest |
| `tests/` | 40 automated unit and integration tests (zero look-ahead leakage, model determinism, ranking checks) |

## Dashboard
```bash
pip install streamlit plotly
streamlit run app/streamlit_app.py
```
Pages: Overview, Players to watch (filters + four transparent ranking criteria + CSV download), Player detail (history, prediction interval, index drivers, valuation estimate), Model predictions (out-of-sample backtest for every model), Data & method. It reads the precomputed files in `reports/` and `data/research/processed/` and the frozen model in `artifacts/`.

## Reproduce (from `talent-scouting/`, Python 3.12, internet needed only for step 1)
```bash
pip install -r requirements-research.txt
# 1. Data: audit metadata+lineups, then cached event ingestion (resumable; ~2,100 matches, tens of minutes)
python -m research.ingest.audit
bash research/ingest/run_all.sh
python -m research.ingest.crosswalk                 # StatsBomb -> Transfermarkt ids (needs ../transfermkt/*.csv)
# 2. Panels
python -c "from research.features import panel as P; P.design_a().to_parquet('data/research/processed/design_a_player_seasons.parquet'); P.design_b_blocks().to_parquet('data/research/processed/design_b_blocks.parquet')"
python -m research.ingest.feasibility               # writes reports/02_data_feasibility.md
# 3. Experiments (each takes minutes; results are deterministic given SEED in research/config.py)
python -m research.run_design_a
python -m research.run_design_b
python -m research.run_valuation
python -m research.run_breakout --design a && python -m research.run_breakout --design b
python -m research.run_sensitivity
# 4. Tables, uncertainty, figures, notebook, workbook
python -m research.analysis.results
python -m research.analysis.uncertainty_errors
python -m research.analysis.sensitivity_table
python -m research.analysis.target_sensitivity
python -m research.analysis.eda
python -m research.analysis.model_figures
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
# 5. Freeze the development-selected model and score the latest seasons
python -c "from research import inference as I; I.fit_and_freeze()"
python -m research.score_latest
python -m research.analysis.export_excel
# 6. Tests (40: 16 legacy + 24 research incl. leakage, metrics, feature engineering, inference)
python -m pytest -q
```
Notes: `data/research/cache` is git-ignorable raw-derived data; the Transfermarkt CSVs live outside the repo and are **not redistributed** (confirm their terms of use before publishing). Credit StatsBomb for any published use of the open data (https://statsbomb.com/media-pack/).

---

# Legacy documentation (original ranking system)

> **Caution:** the "Time-based validation pilot" and algorithm-selection numbers below come from the original pilot, whose future-performance target re-uses the ranking criteria (circular) and whose NDCG@10 has a high random floor (0.76). See `reports/00_repository_audit.md`; they are retained for provenance only.

# Advanced Talent Scouting Ranking System

This repository contains a historical football talent-scouting ranker, a local cloud-architecture simulation, and an optional AWS serverless pipeline. It compares Weighted Sum (WSM), TOPSIS, and VIKOR, selects a provisional method using temporal validation, and generates one consolidated Excel report.

## Dataset and attribution

The input is derived from [StatsBomb Open Data](https://github.com/hudl/open-data), competition 43, season 106 (2022 FIFA World Cup). The preparation script downloads that tournament's match, lineup, and event JSON, then writes a player-level CSV to `data/statsbomb_world_cup_2022_players.csv`. It keeps players with at least 180 tournament minutes and aggregates event counts into per-90 rates. No raw event files are retained.

StatsBomb permits research use of selected open data and requests that published research, analysis, or insights credit StatsBomb and use its logo from the [StatsBomb Media Pack](https://statsbomb.com/media-pack/). Check the repository terms before redistribution or publication. The 2022 World Cup is a single international tournament, not a broad player labor-market sample; findings do not automatically generalize to club scouting or future performance.

The prepared World Cup CSV is stored separately from the Python code under `data/` so the default demo works offline. To fetch raw StatsBomb data and generate a fresh prepared CSV on request (internet access required), run:

```bash
python -m processing.prepare_statsbomb_dataset
```

This writes a new file to `data/generated/statsbomb_world_cup_2022_players.csv`; it leaves the bundled offline input at `data/statsbomb_world_cup_2022_players.csv` untouched. The integrated command below generates the CSV and immediately ranks it.

## Talent-scoring objective

The scouting objective is to prioritize players for further review using observed tournament performance: pass completion, goals and assists, expected goals, shots on target, defensive actions, goalkeeper saves and clean sheets, and minutes. The preparation script groups StatsBomb positions into goalkeeper, defender, midfielder, and forward buckets and uses a player's most-played bucket. Ranking is performed within each position; scores are relative to that tournament cohort.

Three algorithms are reported side by side:

| Method | How it ranks players | Main limitation |
|---|---|---|
| Weighted sum (WSM) | Sums within-position min-max normalized criteria multiplied by disclosed weights. | Assumes criteria can compensate linearly for one another. |
| TOPSIS | Ranks by relative closeness to the weighted ideal and distance from the anti-ideal. | Results depend on criteria, weights, and the cohort normalization. |
| VIKOR | Ranks a compromise between total weighted shortfall and the worst individual shortfall. | The compromise parameter and weights need sensitivity analysis. |

Core weights are explicitly listed in `processing/scouting_channel.py`. They are researcher-defined starting values, not expert-elicited or statistically estimated. When a cohort has complete FBref coverage, optional position-specific criteria add expected assists per 90, progressive passes per 90, progressive carries per 90, and goalkeeper save percentage. Optional weights are disclosed in the ranking CSV and run summary; incomplete optional fields are omitted for the whole position cohort, never imputed as zero. Thus all three methods in a run use the same declared criteria, while different data sources may have different available feature sets. Scores are ranking aids, not probabilities of future success. Rank agreement describes consensus, not predictive accuracy.

## Research recommendations

1. Define a target outcome before calling a method “best,” such as future minutes, contract progression, or performance over a specified horizon. “Best” means best against that outcome and decision cost, not highest score or agreement on one tournament.
2. Extend beyond this tournament with documented player-season observations and a time-aligned future outcome. Split by season or cohort, not randomly by row, to reduce leakage. Keep scouting-time features separate from future outcome labels.
3. Elicit position-specific weights from multiple scouts (for example, through AHP), document disagreement, and run weight sensitivity analysis. Consider criteria such as defensive actions, chance creation, weak-foot use, workload, league strength, and data confidence when available.
4. Evaluate ranking quality on held-out players with metrics aligned to the use case: NDCG@k or precision/recall@k for shortlist quality, Spearman/Kendall rank correlation, and subgroup calibration/fairness checks. Compare against simple baselines and report confidence intervals across seasons.
5. Report top-k overlap and rank stability across WSM, TOPSIS, and VIKOR. A consensus or learned ranking should only be proposed after validated outcome data and a pre-registered evaluation protocol.

These three methods are defensible transparent baselines, but the tournament data has no independent future-scouting outcome label, so this project cannot establish which method is empirically best. With outcome labels, a supervised ranking model (such as gradient-boosted trees with learning-to-rank objectives) could be evaluated as a later extension, with leakage controls and explainability checks.

## Time-based validation pilot

`processing/temporal_validation.py` evaluates adjacent La Liga seasons from 2016/17 through 2020/21. The source exposes only 33–36 matches per season in this period, not complete league seasons. The derived file currently contains 440 player-season rows after the 180-minute threshold. Players are joined by StatsBomb player ID and retained in a fold only if they appear in both adjacent seasons and remain in the same broad position group.

For each of four season transitions and each position group with at least five linked players, the ranking uses only prior-season features. The evaluation is conditional on a player being observed for at least 180 minutes in both seasons and remaining in the same broad position group; it does not score players who leave the sampled league data, miss the future threshold, or change role. Two next-season targets are evaluated: a position-weighted performance index built from future per-90 rates (excluding minutes), and next-season minutes played. The future index is a repeat-performance proxy, not an independent scouting label such as a contract, transfer, or external player grade. The normal scouting run refreshes these validation outputs under `data/results/` before choosing its primary method.

The current pilot averages 13 season/position folds. Mean results across those folds:

| Method | Future performance: Spearman | Future performance: NDCG@10 | Future minutes: Spearman | Future minutes: NDCG@10 |
|---|---:|---:|---:|---:|
| Weighted sum | 0.461 | 0.888 | **0.591** | **0.929** |
| TOPSIS | 0.446 | 0.876 | 0.545 | 0.896 |
| VIKOR | **0.473** | 0.886 | 0.554 | 0.897 |

These are exploratory averages, not proof that one algorithm is best. Fold sizes are small (5–17 linked players), coverage is incomplete and selected, and NDCG@10 covers the full candidate list when a fold has fewer than ten players. Larger multi-season data and independent future outcomes are needed before drawing a research conclusion. The current figures weakly favor VIKOR on future performance rank correlation and weighted sum on future minutes; neither is a robust winner.

For a scouting shortlist, `processing/algorithm_selection_analysis.py` pre-specifies future-performance NDCG@10 as the primary selection metric. Weighted Sum ranks first in this pilot (0.888 versus 0.886 for VIKOR and 0.876 for TOPSIS). Its paired-fold bootstrap difference from VIKOR is only +0.001 (95% interval −0.025 to +0.028), which includes zero. Weighted Sum is therefore the provisional best method under the stated rule, not a statistically established winner. The interval resamples only 13 season/position folds; the future performance index uses similar performance criteria to the ranking, and the dataset has selected partial-season coverage. This analysis cannot support a general claim of predictive superiority.

Run the selection comparison with:

```bash
python -m processing.algorithm_selection_analysis
```

It writes the per-method summary and paired-fold uncertainty estimates to `data/results/algorithm_selection_analysis.csv`. The normal `python main.py` run recomputes this analysis and publishes the selected method's full ranking after the three comparison rankings.

The script also changes each position weight one at a time by ±20%, renormalizes the weights, and compares shortlist membership with the baseline. The shortlist is capped at 10 candidates and uses the top 20% for smaller position groups (so a nine-player forward group uses top two, not all nine). In the current run, mean retention was 0.986 for weighted sum, 0.967 for TOPSIS, and 0.950 for VIKOR. Detailed results are in `data/weight_sensitivity_results.csv`; these figures describe shortlist membership stability under this perturbation, not model accuracy or unchanged rank order.

The default `python main.py` run prints all three complete rankings and refreshes the temporal-validation, shortlist-sensitivity, and algorithm-selection analyses. The standalone temporal command is available for research evaluation (building downloads source files; evaluation alone works offline):

```bash
python -m processing.temporal_validation --prepare
python -m processing.temporal_validation
```

## Architecture simulation

```text
main.py
├── shared/cloud_infra.py       # simulated compute, storage, gateway, orchestration
├── aws/lambda_function.py      # real S3-triggered AWS ranking handler
├── template.yaml               # SAM: private S3 bucket + Lambda + scoped IAM
└── processing/
    ├── scouting_channel.py     # real player metrics and three MCDA rankings
    ├── prepare_statsbomb_dataset.py # fetch and aggregate official open event data
    └── temporal_validation.py  # rolling next-season evaluation and weight sensitivity
```

The classes in `shared/cloud_infra.py` remain local simulations for the architecture demo. AWS is optional. The SAM template at `template.yaml` deploys a separate AWS workflow using Amazon S3, AWS Lambda, IAM, and CloudWatch Logs. Uploading a CSV under the S3 `incoming/` prefix invokes the Lambda ranker, which publishes the full comparison, selected-method ranking, selection analysis, and formatted workbook under `results/`. The trigger is restricted to the input prefix so result uploads cannot retrigger processing. The bucket blocks public access and Lambda can read only incoming objects and write only results objects.

### Deploy the live AWS ranker

Install and configure the AWS CLI and AWS SAM CLI for an AWS account and region. Then, from the project root:

```powershell
sam build
sam deploy --guided
```

Review and confirm the CloudFormation changes during the guided deploy. It creates a private S3 bucket, an S3-triggered Lambda function, and the Lambda execution role. AWS services can incur charges in your account. When deployment finishes, copy the `ScoutingBucketName` output. Upload a canonical player-season CSV to `incoming/` using the AWS Console or CLI:

```powershell
aws s3 cp "data/statsbomb_world_cup_2022_players.csv" "s3://YOUR_BUCKET/incoming/world_cup_2022.csv"
aws s3 ls "s3://YOUR_BUCKET/results/"
```

The upload triggers all three rankings and the validation-based selection; retrieve the comparison CSV, selected-method CSV, selection-analysis CSV, and Excel report from `results/`. For FBref, use the FBref importer to create a canonical CSV with the desired season and competition first, then upload that prepared CSV. AWS deployment has not been run from this development environment because it has no AWS CLI, SAM CLI, or configured AWS credentials. The Lambda handler is covered locally with an S3 test double.

## Run

Requires Python 3.10 or later and the `openpyxl` package for formatted Excel exports. Install the project dependency once:

```bash
python -m pip install -r requirements.txt
```

The prepared CSVs are included, so the demo and tests do not need internet access.

### Run in VS Code

1. Open **File → Open Folder…** and select the cloned `talent-scouting` repository folder (the folder containing `main.py`).
2. Install/enable Microsoft's Python extension if it is not already available, then select a Python 3.10+ interpreter with **Python: Select Interpreter**.
3. Open **Terminal → New Terminal**. VS Code should open the terminal in the project folder. Run:

```bash
python main.py
```

The console shows each full ranking in its own block, then reports the provisional best method and supporting evidence. Each default run recalculates rankings, temporal validation, sensitivity, and selection. The single generated deliverable is `data/results/FINAL_TALENT_SCOUTING_RESULTS.xlsx`; reports are generated locally and are not committed as pre-generated results. It contains the winner and caveat, applied weights, each algorithm's full ranking (the selected method is highlighted), algorithm comparison, temporal-validation folds, and sensitivity cases. The default rule selects the method with the highest mean future-performance NDCG@10 on adjacent-season La Liga folds.

### Rank a different historical dataset

Use the scouting-only mode to rank a prepared player-statistics CSV without running the other simulations. It regenerates the same final workbook on every run. To keep multiple experiments, pass a distinct `--rankings-output` workbook path:

```bash
python main.py --scouting-only --input "data/my_historical_players.csv"
```

To fetch and prepare the official 2022 World Cup data, then immediately rank it with all three methods:

```bash
python main.py --prepare-statsbomb
```

This downloads StatsBomb match, lineup, and event data, generates `data/generated/statsbomb_world_cup_2022_players.csv`, prints the full rankings, and saves them under `data/results/`. It requires internet access. The normal `python main.py` command continues to use the bundled CSV for an offline run.

The input must have one row per player for one season. The workbook places each method's complete ranking on its own worksheet. If the input contains more than one season, select one explicitly:

```bash
python main.py --scouting-only --input "data/player_seasons.csv" --season "2019/2020" --rankings-output "data/results/FINAL_TALENT_SCOUTING_RESULTS_2019_2020.xlsx"
```

Required common columns are `player_id`, `name`, `position`, and numeric `minutes`. Position values can be `GK`, `DEF`, `MID`, `FWD` or clear role names such as `Goalkeeper`, `Defender`, `Midfielder`, and `Forward`. The scoring metrics required depend on position:

| Position | Required metrics |
|---|---|
| GK | `pass_completion`, `saves_per90`, `clean_sheets_per90` |
| DEF | `pass_completion`, `def_actions_per90`, and `goals_assists_per90` (or both `goals_per90` and `assists_per90`) |
| MID | `pass_completion`, `def_actions_per90`, `xg_per90`, and `goals_assists_per90` (or both component rates) |
| FWD | `goals_per90`, `xg_per90`, `shots_on_target_rate`, and `goals_assists_per90` (or both component rates) |

`pass_completion` and `shots_on_target_rate` must be proportions from 0 to 1. The default minimum is 180 minutes; change it with `--min-minutes 90`. The program checks headers, numeric values, positions, and duplicate player IDs before ranking. It prints every player under each of the three algorithms and writes one consolidated Excel workbook. Extra columns are ignored. This expects already aggregated player statistics; raw match-event files need a source-specific preparation step first. Use `python main.py --help` to see all options.

### Rank a local FBref export

The project can prepare and rank FBref player-table CSV exports supplied by the user. It does not scrape or download FBref. Export the relevant player tables for the **same league and season** (Standard, Shooting, Passing, Defense, and, for goalkeepers, Goalkeeping), save them locally, and pass each path with a repeated `--fbref-csv` option. The importer joins rows by player, squad, and competition, derives per-90 rates from totals and minutes, and writes a canonical prepared input under `data/generated/`. Rankings include source, competition, and season metadata.

Example (PowerShell; replace paths and season with your exports):

```powershell
python main.py --fbref-csv "data/fbref_standard.csv" --fbref-csv "data/fbref_shooting.csv" --fbref-csv "data/fbref_passing.csv" --fbref-csv "data/fbref_defense.csv" --season "2022-2023" --min-minutes 180
```

Add `--fbref-csv "data/fbref_goalkeeping.csv"` for goalkeepers. The importer reports required fields that are missing from the supplied tables. `Min` is required; total goals, xG, xAG, tackles, interceptions, saves, clean sheets, progressive passes, and progressive carries are converted to per-90 using minutes. `Cmp%` or `Cmp` + `Att` supplies pass completion; `SoT` / `Sh` supplies shots-on-target rate; `Save%` supplies goalkeeper save percentage. These additional FBref features are included by the ranking model when the full position cohort has values. Core StatsBomb rankings continue to use their available criteria.

FBref states its site content is subject to its [Terms of Use](https://static.fbref.com/termsofuse.html), including limits on using its data to create substitute databases or services. Use exports only where your intended research and redistribution are permitted; record the source and export date in the paper. Opta data is not fetched by this project; its feeds require client portal access ([Opta Core](https://client.core.optasports.com/login)).

To run temporal validation or tests, use:

```bash
python -m processing.temporal_validation
python -m unittest discover -s tests -v
```

On Windows, use `py` in place of `python` if the Python launcher is configured that way. The `.vscode/launch.json` file also adds **Sports analytics demo**, **Time-based validation**, and **Run tests** entries under **Run and Debug**. Rebuilding the CSV datasets with `--prepare` requires internet access.
