"""
main.py
========
Entry point for the football talent-scouting ranking pipeline.

The default run produces complete WSM, TOPSIS, and VIKOR rankings,
then reports the provisional method selected by temporal validation.

Run with:  python main.py
"""

import argparse
import json
import sys
import time
import unicodedata
from pathlib import Path

from shared.cloud_infra import ElasticComputePool, DistributedStorage, APIGateway, Orchestrator
from processing.scouting_channel import (
    DEFAULT_DATASET, DEFAULT_FINAL_REPORT, run_scouting_pipeline, format_rankings_for_console,
)
from processing.prepare_statsbomb_dataset import OUTPUT_PATH as GENERATED_DATASET, prepare_dataset
from processing.injury_channel import run_injury_pipeline
from processing.temporal_validation import ALGORITHMS, TARGETS


def main():
    parser = argparse.ArgumentParser(description="Run the analytics demo or the scouting ranker on historical player data.")
    parser.add_argument("--scouting-only", action="store_true",
                        help="rank player statistics without running unrelated simulations")
    parser.add_argument("--full-demo", action="store_true",
                        help="also run temporal-validation and illustrative injury simulations")
    parser.add_argument("--input", help="prepared player-statistics CSV")
    parser.add_argument("--fbref-csv", action="append", dest="fbref_csv", metavar="PATH",
                        help="local FBref CSV export; repeat for Standard/Shooting/Passing/Defense/GK tables")
    parser.add_argument("--prepare-statsbomb", action="store_true",
                        help="download and prepare the StatsBomb 2022 World Cup CSV, then rank it")
    parser.add_argument("--dataset-output", default=str(GENERATED_DATASET),
                        help="where --prepare-statsbomb writes its generated input CSV")
    parser.add_argument("--rankings-output", dest="rankings_output",
                        help=f"final consolidated Excel workbook path (default: {DEFAULT_FINAL_REPORT})")
    parser.add_argument("--season", help="select one season if the input contains multiple seasons")
    parser.add_argument("--min-minutes", type=float, default=180,
                        help="minimum minutes needed for ranking (default: 180)")
    args = parser.parse_args()

    # Use escaped Unicode for names unavailable in a legacy Windows console code page.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="backslashreplace")

    if sum(bool(value) for value in (args.prepare_statsbomb, args.input, args.fbref_csv)) > 1:
        parser.error("Use only one of --prepare-statsbomb, --input, or --fbref-csv.")
    if args.full_demo and (args.scouting_only or args.prepare_statsbomb or args.input or args.fbref_csv
                           or args.rankings_output or args.season or args.min_minutes != 180):
        parser.error("--full-demo cannot be combined with scouting input or output options.")
    if not args.prepare_statsbomb and args.dataset_output != str(GENERATED_DATASET):
        parser.error("--dataset-output requires --prepare-statsbomb.")

    scouting_mode = not args.full_demo
    if scouting_mode:
        fbref_prepared = None
        if args.prepare_statsbomb:
            prepare_dataset(args.dataset_output)
            input_path = args.dataset_output
        elif args.fbref_csv:
            from processing.fbref_import import prepare_fbref_exports
            if not args.season:
                parser.error("--fbref-csv requires --season so the generated dataset is traceable.")
            season_slug = "".join(character.lower() if character.isalnum() else "_" for character in args.season).strip("_")
            fbref_prepared = Path(__file__).resolve().parent / "data" / "generated" / f"fbref_{season_slug}_players.csv"
            count = prepare_fbref_exports(args.fbref_csv, fbref_prepared, season=args.season)
            print(f"Prepared {count} players from local FBref exports: {fbref_prepared}")
            input_path = str(fbref_prepared)
        else:
            input_path = args.input or str(DEFAULT_DATASET)
        rankings_output = args.rankings_output
        storage = DistributedStorage()
        compute = ElasticComputePool(min_workers=2, max_workers=8)
        try:
            summary = run_scouting_pipeline(
                compute, storage, input_path, rankings_output, args.season, args.min_minutes,
            )
            _print_scouting_summary(summary)
            _print_full_rankings(summary["rankings"])
            _print_selected_result(summary)
        finally:
            compute.shutdown()
        return

    if (args.rankings_output is not None
            or args.season is not None or args.min_minutes != 180):
        parser.error("Custom scouting options require --scouting-only or --prepare-statsbomb.")

    print("=" * 70)
    print("HIGH PERFORMANCE CLOUD COMPUTING FOR SPORTS ANALYTICS")
    print("System Architecture Simulation")
    print("=" * 70)

    # ---- Unified Cloud Architecture (shared backbone) ----
    storage = DistributedStorage()
    compute = ElasticComputePool(min_workers=2, max_workers=8)
    gateway = APIGateway()
    orchestrator = Orchestrator(compute)

    # ---- Talent Scouting Channel (batch / HPC path) ----
    print("\n[TALENT SCOUTING CHANNEL] Starting ingestion + HPC processing...")
    t0 = time.time()
    scouting_job = orchestrator.schedule_batch_job(
        "scouting_pipeline", run_scouting_pipeline, compute, storage
    )
    scouting_summary = scouting_job.result()
    print(f"[TALENT SCOUTING CHANNEL] Done in {time.time() - t0:.2f}s "
          f"(compute pool scaled to {compute.current_workers} workers)")

    # ---- Injury Prevention Channel (streaming path) ----
    print("\n[INJURY PREVENTION CHANNEL] Starting IoT wearable simulation + real-time sync...")
    t0 = time.time()
    player_ids = [f"P{2000 + i}" for i in range(10)]
    injury_summary = run_injury_pipeline(player_ids, storage, n_readings=20)
    print(f"[INJURY PREVENTION CHANNEL] Done in {time.time() - t0:.2f}s")

    # ---- Secure API Gateway: expose both summaries through one entry point ----
    gateway.register_route("/scouting/summary", lambda: storage.get_latest("scouting:summary"))
    gateway.register_route("/injury/summary", lambda: storage.get_latest("injury:summary"))

    print("\n" + "=" * 70)
    print("RESULTS (retrieved via unified API Gateway)")
    print("=" * 70)

    print("\n--- /scouting/summary (shortlist view) ---")
    scouting_summary = gateway.request("/scouting/summary")
    _print_scouting_summary(scouting_summary)
    _print_full_rankings(scouting_summary["rankings"])
    _print_selected_result(scouting_summary)

    print("\n--- /scouting/temporal-validation ---")
    validation_rows = scouting_summary["validation_rows"]
    sensitivity_rows = scouting_summary["sensitivity_rows"]
    _print_validation_summary(validation_rows, sensitivity_rows)
    _print_validation_details(validation_rows, sensitivity_rows)

    print("\n--- /injury/summary ---")
    print(json.dumps(gateway.request("/injury/summary"), indent=2, default=str))

    print("\n--- Orchestrator job status ---")
    print(json.dumps(orchestrator.status(), indent=2, default=str))

    print(f"\nTotal API Gateway requests logged: {len(gateway.access_log())}")
    print(f"Distinct data-lake keys in Distributed Storage: {len(storage.keys())}")

    compute.shutdown()


def _print_scouting_summary(summary: dict) -> None:
    """Print a readable shortlist while leaving the full API payload intact."""
    print(f"Source: {summary['data_source']}")
    print(f"Players: {summary['total_players_scouted']} | scope: {summary['ranking_scope']}")
    print("\nTop 3 by position and method:")
    for position, methods in summary["top_candidates_by_algorithm_and_position"].items():
        for algorithm, candidates in methods.items():
            names = ", ".join(
                f"{_ascii_name(candidate['name'])} ({candidate['score']:.3f})"
                for candidate in candidates[:3]
            )
            print(f"  {position:>3} | {algorithm:<12} {names}")
    print("\nMethod rank agreement (Spearman):")
    print(json.dumps(summary["rank_agreement_spearman"], indent=2))
    print(f"\nResearch note: {summary['research_note']}")
    print(f"Final consolidated report: {summary['rankings_workbook_file']}")


def _print_selected_result(summary: dict) -> None:
    """Publish the selected method as the final core result after all rankings."""
    selection = summary["algorithm_selection"]
    print("\n" + "=" * 88)
    print("CORE RESULT | PROVISIONAL BEST ALGORITHM")
    print("=" * 88)
    print(f"Selected: {selection['method']}")
    print(f"Primary evidence: mean future-performance NDCG@10={selection['mean_score']:.4f} "
          f"across {selection['validation_folds']} historical season/position folds")
    print(f"Selection cohort: {selection['selection_dataset']}")
    print(f"Interpretation: {selection['interpretation']}")
    print(f"Selected method's full ranking is highlighted in the final workbook.")


def _print_validation_summary(rows: list[dict], sensitivity_rows: list[dict]) -> None:
    print("Mean metrics across season/position folds:")
    for target in TARGETS:
        print(f"  {target}:")
        for algorithm in ALGORITHMS:
            selected = [row for row in rows if row["target"] == target and row["algorithm"] == algorithm]
            if selected:
                spearman = sum(float(row["spearman"]) for row in selected) / len(selected)
                ndcg = sum(float(row["ndcg_at_10"]) for row in selected) / len(selected)
                print(f"    {algorithm:<12} Spearman={spearman:.3f}  NDCG@10={ndcg:.3f}  folds={len(selected)}")
    print("  plus/minus 20% one-weight-at-a-time shortlist retention:")
    for algorithm in ALGORITHMS:
        selected = [float(row["top_k_retention"]) for row in sensitivity_rows if row["algorithm"] == algorithm]
        if selected:
            print(f"    {algorithm:<12} {sum(selected) / len(selected):.3f}")
    print("Detailed folds and sensitivity cases are included in the final Excel workbook.")


def _print_full_rankings(rankings) -> None:
    """Print all full ranking blocks from the current in-memory run."""
    print("\n" + format_rankings_for_console(rankings))


def _print_validation_details(rows: list[dict], sensitivity_rows: list[dict]) -> None:
    print("\nAll temporal validation folds:")
    for row in rows:
        print(
            f"  {row['past_season']} -> {row['future_season']} | {row['position']:<3} | "
            f"{row['target']:<25} | {row['algorithm']:<12} | n={int(row['n_linked_players']):>2} | "
            f"Spearman={float(row['spearman']): .3f} | NDCG@10={float(row['ndcg_at_10']):.3f}"
        )
    print("\nAll one-at-a-time weight-sensitivity cases:")
    for row in sensitivity_rows:
        print(
            f"  {row['season']} | {row['position']:<3} | {row['algorithm']:<12} | "
            f"{row['perturbed_criterion']} {row['direction']} {abs(int(row['weight_change_pct']))}% | "
            f"top-{row['top_k']} retention={float(row['top_k_retention']):.3f} | "
            f"Jaccard={float(row['top_k_jaccard']):.3f}"
        )


def _ascii_name(value: str) -> str:
    """Keep shortlist names readable in Windows terminals with legacy encodings."""
    return unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")


if __name__ == "__main__":
    main()
