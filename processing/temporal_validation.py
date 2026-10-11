"""Rolling next-season evaluation for the three MCDA scouting methods.

Build the real player-season table:
    python -m processing.temporal_validation --prepare
Run the evaluation:
    python -m processing.temporal_validation
"""

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

from processing.prepare_statsbomb_dataset import aggregate_competition_season
from processing.scouting_channel import (
    ALGORITHMS, POSITION_WEIGHTS, PlayerProfile, _normalize_group, _rank_group,
)


COMPETITION_ID = 11
COMPETITION_NAME = "La Liga"
SEASONS = ((2, "2016/2017"), (1, "2017/2018"), (4, "2018/2019"),
           (42, "2019/2020"), (90, "2020/2021"))
MIN_MINUTES = 180
DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "la_liga_player_seasons.csv"
RESULT_PATH = Path(__file__).resolve().parents[1] / "data" / "temporal_validation_results.csv"
SENSITIVITY_PATH = Path(__file__).resolve().parents[1] / "data" / "weight_sensitivity_results.csv"
TARGETS = ("future_performance_index", "future_minutes")


def prepare_dataset(output_path: str | Path = DATA_PATH) -> int:
    rows = []
    for season_id, season_name in SEASONS:
        print(f"Fetching {COMPETITION_NAME} {season_name}…")
        rows.extend(aggregate_competition_season(
            COMPETITION_ID, season_id, COMPETITION_NAME, season_name, MIN_MINUTES,
        ))
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError("No player-seasons were returned by the StatsBomb source.")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} player-seasons to {path}")
    return len(rows)


def _read_seasons(path: str | Path) -> dict[str, list[PlayerProfile]]:
    grouped: dict[str, list[PlayerProfile]] = defaultdict(list)
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["season"] not in {name for _, name in SEASONS}:
                continue
            metrics = {key: float(value) for key, value in row.items()
                       if key not in {"competition_id", "competition", "season_id", "season",
                                      "player_id", "name", "team", "country", "position", "matches_played"}}
            metrics["goals_assists_per90"] = metrics.get("goals_per90", 0) + metrics.get("assists_per90", 0)
            profile = PlayerProfile(row["player_id"], row["name"], row["position"], metrics,
                                    row.get("team", ""), row.get("country", ""))
            grouped[row["season"]].append(profile)
    return grouped


def _average_ranks(values: list[float]) -> list[float]:
    ordered = sorted(range(len(values)), key=lambda index: values[index], reverse=True)
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(ordered):
        end = cursor + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[cursor]]:
            end += 1
        average_rank = (cursor + 1 + end) / 2
        for index in ordered[cursor:end]:
            ranks[index] = average_rank
        cursor = end
    return ranks


def _spearman(predicted: list[float], observed: list[float]) -> float:
    x, y = _average_ranks(predicted), _average_ranks(observed)
    mean_x, mean_y = sum(x) / len(x), sum(y) / len(y)
    cov = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y))
    den_x = sum((a - mean_x) ** 2 for a in x) ** 0.5
    den_y = sum((b - mean_y) ** 2 for b in y) ** 0.5
    return cov / (den_x * den_y) if den_x and den_y else 0.0


def _ndcg(order: list[int], relevance: list[float], k: int = 10) -> float:
    ideal = sorted(relevance, reverse=True)[:k]
    if not ideal or max(ideal) == 0:
        return 0.0
    dcg = sum(relevance[index] / math.log2(rank + 2) for rank, index in enumerate(order[:k]))
    idcg = sum(value / math.log2(rank + 2) for rank, value in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def _future_index(players: list[PlayerProfile]) -> dict[str, float]:
    """Future outcome: position-weighted rates, excluding future playing time."""
    weights = POSITION_WEIGHTS[players[0].position]
    criteria = tuple(criterion for criterion in weights if criterion != "minutes")
    norm_weights = {criterion: weights[criterion] / sum(weights[c] for c in criteria)
                    for criterion in criteria}
    matrix = _normalize_group(players, criteria)
    return {player.player_id: sum(row[c] * norm_weights[c] for c in criteria)
            for player, row in zip(players, matrix)}


def _shortlist_size(candidate_count: int) -> int:
    """Use up to ten candidates, or the top 20% for smaller position cohorts."""
    return min(10, max(1, math.ceil(candidate_count * 0.20)))


def evaluate(path: str | Path = DATA_PATH) -> list[dict]:
    seasons = _read_seasons(path)
    results = []
    for (past_id, past_name), (_, future_name) in zip(SEASONS, SEASONS[1:]):
        current = seasons.get(past_name, [])
        future = seasons.get(future_name, [])
        future_by_id = {player.player_id: player for player in future}
        for position, weights in POSITION_WEIGHTS.items():
            prior_group = [player for player in current if player.position == position]
            next_group = [future_by_id[player.player_id] for player in prior_group
                          if player.player_id in future_by_id
                          and future_by_id[player.player_id].position == position]
            if len(next_group) < 5:
                continue
            next_ids = {player.player_id for player in next_group}
            future_index = _future_index(next_group)
            future_minutes = {player.player_id: player.metrics["minutes"] for player in next_group}
            actual_by_target = {
                "future_performance_index": future_index,
                "future_minutes": future_minutes,
            }
            methods = _rank_group(prior_group)
            for algorithm in ALGORITHMS:
                full_ranking = methods[algorithm]
                eligible = [entry for entry in full_ranking if entry["player_id"] in next_ids]
                predicted_scores = [entry["score"] for entry in eligible]
                for target in TARGETS:
                    actual = actual_by_target[target]
                    raw_relevance = [actual[entry["player_id"]] for entry in eligible]
                    low, high = min(raw_relevance), max(raw_relevance)
                    relevance = [(value - low) / (high - low) if high > low else 0.0
                                 for value in raw_relevance]
                    order = list(range(len(eligible)))
                    spearman = _spearman(predicted_scores, raw_relevance)
                    results.append({
                        "past_season": past_name, "future_season": future_name,
                        "position": position, "algorithm": algorithm, "target": target,
                        "n_linked_players": len(eligible), "spearman": round(spearman, 6),
                        "ndcg_at_10": round(_ndcg(order, relevance, 10), 6),
                    })
    return results


def weight_sensitivity(path: str | Path = DATA_PATH, perturbation: float = 0.20) -> list[dict]:
    """Perturb each weight ±20%; compare a shortlist capped at 10 or 20% of cohort."""
    seasons = _read_seasons(path)
    results = []
    for _, season_name in SEASONS[:-1]:
        for position, base_weights in POSITION_WEIGHTS.items():
            group = [player for player in seasons.get(season_name, []) if player.position == position]
            if len(group) < 5:
                continue
            baseline = _rank_group(group)
            for criterion, base_value in base_weights.items():
                for direction, multiplier in (("down", 1.0 - perturbation), ("up", 1.0 + perturbation)):
                    changed = dict(base_weights)
                    changed[criterion] = base_value * multiplier
                    total = sum(changed.values())
                    changed = {key: value / total for key, value in changed.items()}
                    perturbed = _rank_group(group, changed)
                    for algorithm in ALGORITHMS:
                        k = _shortlist_size(len(group))
                        base_top = {row["player_id"] for row in baseline[algorithm][:k]}
                        changed_top = {row["player_id"] for row in perturbed[algorithm][:k]}
                        results.append({
                            "season": season_name, "position": position, "algorithm": algorithm,
                            "perturbed_criterion": criterion, "direction": direction,
                            "weight_change_pct": int(perturbation * 100 * (1 if direction == "up" else -1)),
                            "top_k": k,
                            "top_k_retention": round(len(base_top & changed_top) / k, 6),
                            "top_k_jaccard": round(len(base_top & changed_top) / len(base_top | changed_top), 6),
                        })
    return results


def _write_csv(rows: list[dict], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_and_save_validation(path: str | Path = DATA_PATH,
                           results_path: str | Path = RESULT_PATH,
                           sensitivity_path: str | Path = SENSITIVITY_PATH) -> tuple[list[dict], list[dict]]:
    """Recompute and persist temporal metrics and weight-sensitivity rows."""
    rows = evaluate(path)
    if not rows:
        raise RuntimeError("No season-to-season player overlaps were available for evaluation.")
    sensitivity_rows = weight_sensitivity(path)
    _write_csv(rows, results_path)
    _write_csv(sensitivity_rows, sensitivity_path)
    return rows, sensitivity_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true", help="download and aggregate the five La Liga seasons")
    parser.add_argument("--input", default=str(DATA_PATH), help="player-season CSV path")
    parser.add_argument("--output", default=str(RESULT_PATH), help="evaluation results CSV path")
    args = parser.parse_args()
    if args.prepare:
        prepare_dataset(args.input)
    sensitivity_output = Path(args.output).with_name(SENSITIVITY_PATH.name)
    rows, sensitivity_rows = run_and_save_validation(args.input, args.output, sensitivity_output)
    output = Path(args.output)
    print(f"Wrote {len(rows)} evaluation rows to {output}")
    _print_summary(rows)
    _print_sensitivity_summary(sensitivity_rows)


def _print_summary(rows: list[dict]):
    for target in TARGETS:
        print(f"\n{target} (mean across season/position folds)")
        for algorithm in ALGORITHMS:
            selected = [row for row in rows if row["target"] == target and row["algorithm"] == algorithm]
            if not selected:
                continue
            mean_spearman = sum(row["spearman"] for row in selected) / len(selected)
            mean_ndcg = sum(row["ndcg_at_10"] for row in selected) / len(selected)
            print(f"  {algorithm}: Spearman={mean_spearman:.3f}, NDCG@10={mean_ndcg:.3f}, folds={len(selected)}")


def _print_sensitivity_summary(rows: list[dict]):
    print("\nWeight sensitivity (mean shortlist retention after one-at-a-time +/-20% weight changes)")
    for algorithm in ALGORITHMS:
        selected = [row["top_k_retention"] for row in rows if row["algorithm"] == algorithm]
        if selected:
            print(f"  {algorithm}: {sum(selected) / len(selected):.3f}")
    print(f"Detailed sensitivity rows: {len(rows)}")


if __name__ == "__main__":
    main()
