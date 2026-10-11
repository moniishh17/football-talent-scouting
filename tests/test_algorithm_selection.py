import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from processing.algorithm_selection_analysis import compare


class AlgorithmSelectionTest(unittest.TestCase):
    def test_selects_highest_mean_primary_metric_and_writes_paired_analysis(self):
        validation_rows = []
        perf = {
            "weighted_sum": (0.95, 0.95),
            "topsis": (0.80, 0.70),
            "vikor": (0.89, 0.88),
        }
        for fold_index in range(2):
            for algorithm, ndcg_scores in perf.items():
                for target, score in (("future_performance_index", ndcg_scores[fold_index]),
                                      ("future_minutes", 0.7)):
                    validation_rows.append({
                        "past_season": f"s{fold_index}",
                        "future_season": f"s{fold_index + 1}",
                        "position": "FWD", "algorithm": algorithm, "target": target,
                        "n_linked_players": 12, "spearman": score, "ndcg_at_10": score,
                    })
        sensitivity_rows = [
            {"algorithm": algorithm, "top_k_retention": 0.95}
            for algorithm in perf
        ]
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            output_path = Path(directory) / "algorithm_selection.csv"
            with patch("processing.algorithm_selection_analysis._bootstrap_interval",
                       side_effect=lambda values, **kwargs: (min(values), max(values))):
                rows = compare(
                    "synthetic.csv", output_path, validation_rows=validation_rows,
                    sensitivity_rows=sensitivity_rows, print_results=False,
                )
            summaries = [row for row in rows if row["record_type"] == "algorithm_summary"]
            self.assertEqual(min(summaries, key=lambda row: row["primary_rank"])["algorithm"], "Weighted Sum")
            self.assertAlmostEqual(min(summaries, key=lambda row: row["primary_rank"])["primary_mean"], 0.95)
            self.assertTrue(output_path.exists())
            self.assertEqual(sum(row["record_type"] == "paired_difference" for row in rows), 6)


if __name__ == "__main__":
    unittest.main()
