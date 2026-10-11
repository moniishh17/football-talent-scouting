"""Compare the three scouting methods under a pre-defined shortlist objective.

Primary criterion: mean fold NDCG@10 against the next-season performance proxy.
Secondary criteria: Spearman correlation, future minutes, and weight sensitivity.
The bootstrap resamples paired season/position folds; with few folds, intervals
are exploratory uncertainty summaries rather than population guarantees.
"""

import argparse
import csv
import random
import statistics
from collections import defaultdict
from pathlib import Path

from processing.temporal_validation import DATA_PATH, evaluate, weight_sensitivity


ALGORITHMS = ("weighted_sum", "topsis", "vikor")
LABELS = {"weighted_sum": "Weighted Sum", "topsis": "TOPSIS", "vikor": "VIKOR"}
PRIMARY_TARGET = "future_performance_index"
PRIMARY_METRIC = "ndcg_at_10"
OUTPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "results" / "algorithm_selection_analysis.csv"


def _bootstrap_interval(values, seed=17, iterations=20000):
    rng = random.Random(seed)
    means = sorted(statistics.mean(rng.choice(values) for _ in values) for _ in range(iterations))
    return means[int(0.025 * iterations)], means[int(0.975 * iterations)]


def compare(input_path=DATA_PATH, output_path=OUTPUT_PATH, *, validation_rows=None,
            sensitivity_rows=None, print_results=True):
    """Evaluate each method and return fold-level aggregates and paired uncertainty."""
    rows = evaluate(input_path) if validation_rows is None else validation_rows
    sensitivity = weight_sensitivity(input_path) if sensitivity_rows is None else sensitivity_rows
    if not rows:
        raise ValueError(f"No temporal-validation folds available in {input_path}.")
    validation = defaultdict(dict)
    for row in rows:
        key = (row["past_season"], row["future_season"], row["position"], row["target"])
        validation[key][row["algorithm"]] = row

    performance_folds = [key for key in validation if key[3] == PRIMARY_TARGET]
    minutes_folds = [key for key in validation if key[3] == "future_minutes"]
    if not performance_folds:
        raise ValueError(f"No {PRIMARY_TARGET} folds available for algorithm selection in {input_path}.")
    sensitivity_by_algorithm = defaultdict(list)
    for row in sensitivity:
        sensitivity_by_algorithm[row["algorithm"]].append(float(row["top_k_retention"]))

    result = []
    primary_means = {}
    for algorithm in ALGORITHMS:
        perf = [validation[key][algorithm] for key in performance_folds]
        mins = [validation[key][algorithm] for key in minutes_folds]
        primary_means[algorithm] = statistics.mean(float(row[PRIMARY_METRIC]) for row in perf)
        result.append({
            "record_type": "algorithm_summary",
            "algorithm": LABELS[algorithm],
            "primary_metric": "future performance proxy NDCG@10",
            "primary_mean": round(primary_means[algorithm], 6),
            "primary_rank": None,
            "future_performance_spearman_mean": round(statistics.mean(float(row["spearman"]) for row in perf), 6),
            "future_minutes_spearman_mean": round(statistics.mean(float(row["spearman"]) for row in mins), 6),
            "future_performance_ndcg10_mean": round(primary_means[algorithm], 6),
            "future_minutes_ndcg10_mean": round(statistics.mean(float(row["ndcg_at_10"]) for row in mins), 6),
            "mean_top10_retention_under_pm20pct_weights": round(statistics.mean(sensitivity_by_algorithm[algorithm]), 6),
            "performance_folds": len(perf),
            "linked_player_fold_observations": sum(int(row["n_linked_players"]) for row in perf),
            "comparison_against": "",
            "paired_primary_mean_difference": "",
            "paired_bootstrap_95ci_low": "",
            "paired_bootstrap_95ci_high": "",
        })

    ordered = sorted(ALGORITHMS, key=lambda alg: primary_means[alg], reverse=True)
    for rank, algorithm in enumerate(ordered, 1):
        next(row for row in result if row["algorithm"] == LABELS[algorithm])["primary_rank"] = rank

    for algorithm in ALGORITHMS:
        for comparator in ALGORITHMS:
            if algorithm == comparator:
                continue
            diffs = [
                float(validation[key][algorithm][PRIMARY_METRIC])
                - float(validation[key][comparator][PRIMARY_METRIC])
                for key in performance_folds
            ]
            low, high = _bootstrap_interval(diffs, seed=17 + ALGORITHMS.index(algorithm) * 5 + ALGORITHMS.index(comparator))
            result.append({
                "record_type": "paired_difference",
                "algorithm": LABELS[algorithm],
                "primary_metric": "future performance proxy NDCG@10",
                "primary_mean": "",
                "primary_rank": "",
                "future_performance_spearman_mean": "",
                "future_minutes_spearman_mean": "",
                "future_performance_ndcg10_mean": "",
                "future_minutes_ndcg10_mean": "",
                "mean_top10_retention_under_pm20pct_weights": "",
                "performance_folds": len(diffs),
                "linked_player_fold_observations": "",
                "comparison_against": LABELS[comparator],
                "paired_primary_mean_difference": round(statistics.mean(diffs), 6),
                "paired_bootstrap_95ci_low": round(low, 6),
                "paired_bootstrap_95ci_high": round(high, 6),
            })

    path = Path(output_path) if output_path is not None else None
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fields = list(result[0])
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(result)

    if print_results:
        print(f"Primary criterion: {PRIMARY_METRIC} for {PRIMARY_TARGET}; higher is better")
        print(f"Paired season/position folds: {len(performance_folds)}")
        for algorithm in ordered:
            row = next(item for item in result if item["algorithm"] == LABELS[algorithm]
                       and item["record_type"] == "algorithm_summary")
            print(f"{row['primary_rank']}. {row['algorithm']}: NDCG@10={row['primary_mean']:.3f}; "
                  f"Spearman={row['future_performance_spearman_mean']:.3f}; "
                  f"future-minutes Spearman={row['future_minutes_spearman_mean']:.3f}; "
                  f"top-10 retention={row['mean_top10_retention_under_pm20pct_weights']:.3f}")
        for comparator in ALGORITHMS:
            if comparator == ordered[0]:
                continue
            row = next(item for item in result if item["record_type"] == "paired_difference"
                       and item["algorithm"] == LABELS[ordered[0]]
                       and item["comparison_against"] == LABELS[comparator])
            print(f"{row['algorithm']} - {row['comparison_against']} NDCG@10: "
                  f"{row['paired_primary_mean_difference']:+.3f} "
                  f"(paired fold bootstrap 95% CI {row['paired_bootstrap_95ci_low']:+.3f} to "
                  f"{row['paired_bootstrap_95ci_high']:+.3f})")
        if path is not None:
            print(f"Detailed results: {path}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DATA_PATH), help="player-season CSV path")
    parser.add_argument("--output", default=str(OUTPUT_PATH), help="comparison CSV output path")
    args = parser.parse_args()
    compare(args.input, args.output)


if __name__ == "__main__":
    main()
