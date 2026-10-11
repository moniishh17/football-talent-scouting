import unittest

from processing.temporal_validation import (
    DATA_PATH,
    _ndcg,
    _shortlist_size,
    _spearman,
    evaluate,
    weight_sensitivity,
)


class TemporalValidationMetricsTest(unittest.TestCase):
    def test_spearman_perfect_and_reversed_orders(self):
        self.assertAlmostEqual(_spearman([3, 2, 1], [30, 20, 10]), 1.0)
        self.assertAlmostEqual(_spearman([3, 2, 1], [10, 20, 30]), -1.0)

    def test_ndcg_rewards_relevant_candidates_at_the_top(self):
        self.assertAlmostEqual(_ndcg([0, 1, 2], [1.0, 0.5, 0.0], 3), 1.0)
        self.assertLess(_ndcg([2, 1, 0], [1.0, 0.5, 0.0], 3), 1.0)

    def test_temporal_validation_uses_only_adjacent_seasons_and_future_targets(self):
        rows = evaluate(DATA_PATH)
        self.assertTrue(rows)
        self.assertEqual({row["target"] for row in rows}, {"future_performance_index", "future_minutes"})
        self.assertEqual({row["algorithm"] for row in rows}, {"weighted_sum", "topsis", "vikor"})
        self.assertTrue(all(int(row["n_linked_players"]) >= 5 for row in rows))
        self.assertTrue(all(row["past_season"] != row["future_season"] for row in rows))

    def test_weight_sensitivity_reports_bounded_top_k_retention(self):
        rows = weight_sensitivity(DATA_PATH)
        self.assertTrue(rows)
        self.assertEqual({row["direction"] for row in rows}, {"up", "down"})
        self.assertTrue(all(0 <= float(row["top_k_retention"]) <= 1 for row in rows))
        self.assertEqual(_shortlist_size(9), 2)
        self.assertEqual(_shortlist_size(346), 10)
        forwards = [row for row in rows if row["position"] == "FWD"]
        self.assertTrue(forwards)
        self.assertTrue(all(int(row["top_k"]) < 9 for row in forwards))


if __name__ == "__main__":
    unittest.main()
