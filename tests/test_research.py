"""Tests for the research pipeline: metrics, feature engineering, shrinkage, temporal leakage, inference, baselines."""
import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from research.config import PROCESSED, ARTIFACTS
from research.evaluation import metrics as M
from research.features import shrinkage
from research.ingest.crosswalk import club_match, name_tokens
from research.ingest.events import process_match, role_of

PANEL = PROCESSED / "design_a_player_seasons.parquet"
FROZEN = ARTIFACTS / "design_a_frozen" / "metadata.json"


# ----------------------------------------------------------------------------- metrics
class MetricsTest(unittest.TestCase):
    def test_regression_metrics_hand_computed(self):
        y, p = [1.0, 2.0, 3.0], [1.0, 3.0, 5.0]
        self.assertAlmostEqual(M.mae(y, p), 1.0)
        self.assertAlmostEqual(M.rmse(y, p), np.sqrt(5 / 3))
        self.assertAlmostEqual(M.spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)
        self.assertAlmostEqual(M.spearman([1, 2, 3, 4], [4, 3, 2, 1]), -1.0)
        self.assertTrue(np.isnan(M.spearman([1, 1, 1], [1, 2, 3])))

    def test_ndcg_at_k(self):
        rel = np.array([3.0, 2.0, 0.0, 1.0])
        self.assertAlmostEqual(M.ndcg_at_k(rel, [4, 3, 1, 2], 4), 1.0)            # perfect order
        worst = M.ndcg_at_k(rel, [1, 2, 4, 3], 4)
        self.assertLess(worst, 1.0)
        # hand computation for k=2, order [index 1, index 0]: (2/1 + 3/log2(3)) / (3/1 + 2/log2(3))
        got = M.ndcg_at_k(rel, [3, 4, 0, 1], 2)
        self.assertAlmostEqual(got, (2 + 3 / np.log2(3)) / (3 + 2 / np.log2(3)))
        with self.assertRaises(ValueError):
            M.ndcg_at_k([-1.0, 1.0], [1, 2], 2)

    def test_topk_precision_recall_overlap(self):
        score = [0.9, 0.8, 0.1, 0.2]
        y = [1, 0, 0, 1]
        self.assertAlmostEqual(M.precision_at_k(y, score, 2), 0.5)
        self.assertAlmostEqual(M.recall_at_k(y, score, 2), 0.5)
        self.assertAlmostEqual(M.topk_overlap(score, [0.9, 0.1, 0.2, 0.8], 2), 0.5)
        self.assertAlmostEqual(M.shortlist_outcome([1, 2, 3, 4], [0, 0, 5, 9], 2), 3.5)

    def test_classification_metrics(self):
        y = np.array([0, 0, 1, 1])
        self.assertAlmostEqual(M.brier(y, [0.0, 0.0, 1.0, 1.0]), 0.0)
        self.assertAlmostEqual(M.brier(y, [0.5] * 4), 0.25)
        self.assertAlmostEqual(M.brier_skill(y, [0.5] * 4), 0.0)       # base rate is 0.5
        self.assertAlmostEqual(M.roc_auc(y, [0.1, 0.2, 0.8, 0.9]), 1.0)
        self.assertAlmostEqual(M.pr_auc(y, [0.1, 0.2, 0.8, 0.9]), 1.0)
        self.assertTrue(np.isnan(M.pr_auc([0, 0, 0], [0.1, 0.2, 0.3])))

    def test_interval_coverage_and_bootstrap(self):
        self.assertAlmostEqual(M.interval_coverage([1, 5], [0, 0], [2, 2]), 0.5)
        clusters = np.repeat(np.arange(20), 3)
        vals = np.arange(60, dtype=float)
        point, lo, hi = M.cluster_bootstrap(lambda i: float(vals[i].mean()), clusters, 300, 1)
        self.assertAlmostEqual(point, vals.mean())
        self.assertLess(lo, point)
        self.assertGreater(hi, point)


# ----------------------------------------------------------------------------- feature engineering
def _event(i, type_, team, player, minute=10, period=1, **extra):
    base = {"id": f"e{i}", "index": i, "period": period, "minute": minute, "second": 0, "type": {"name": type_},
            "team": {"id": team, "name": f"T{team}"}, "player": {"id": player, "name": f"P{player}"}, "location": [60.0, 40.0]}
    base.update(extra)
    return base


def _match():
    match = {"match_id": 1, "match_date": "2020-01-01", "match_week": 1,
             "home_team": {"home_team_id": 1, "home_team_name": "T1"}, "away_team": {"away_team_id": 2, "away_team_name": "T2"},
             "home_score": 1, "away_score": 0}
    lineups = [{"team_id": 1, "team_name": "T1", "lineup": [
                   {"player_id": 10, "player_name": "Ten", "player_nickname": None, "country": {"name": "Spain"},
                    "positions": [{"position": "Center Midfield", "from": "00:00", "to": None}]},
                   {"player_id": 11, "player_name": "Eleven", "player_nickname": None, "country": {"name": "Spain"},
                    "positions": [{"position": "Goalkeeper", "from": "00:00", "to": "45:00"}]}]},
               {"team_id": 2, "team_name": "T2", "lineup": [
                   {"player_id": 20, "player_name": "Twenty", "player_nickname": None, "country": {"name": "France"},
                    "positions": [{"position": "Right Back", "from": "00:00", "to": "30:00"}]}]}]
    events = [
        # completed progressive pass (20 yards closer to goal) that becomes a key pass
        _event(1, "Pass", 1, 10, location=[50.0, 40.0], **{"pass": {"length": 20.0, "end_location": [70.0, 40.0], "shot_assist": True}}),
        # shot from that pass: xG 0.3, open play, goal; key_pass_id references event e1
        _event(2, "Shot", 1, 12, location=[100.0, 40.0], shot={"statsbomb_xg": 0.3, "outcome": {"name": "Goal"}, "type": {"name": "Open Play"}, "key_pass_id": "e1"}),
        # incomplete pass (has outcome)
        _event(3, "Pass", 1, 10, location=[50.0, 40.0], **{"pass": {"length": 5.0, "end_location": [55.0, 40.0], "outcome": {"name": "Incomplete"}}}),
        # penalty shoot-out kick must be ignored (period 5)
        _event(4, "Shot", 1, 10, period=5, shot={"statsbomb_xg": 0.78, "outcome": {"name": "Goal"}, "type": {"name": "Penalty"}}),
        # plausible progressive carry vs impossible carry (49 yards in 1.2 s)
        _event(5, "Carry", 1, 10, location=[40.0, 40.0], duration=3.0, carry={"end_location": [60.0, 40.0]}),
        _event(6, "Carry", 1, 10, location=[76.3, 54.2], duration=1.2, carry={"end_location": [27.3, 23.9]}),
        _event(7, "Duel", 2, 20, duel={"type": {"name": "Tackle"}, "outcome": {"name": "Won"}}),
        _event(8, "Goal Keeper", 1, 11, **{"goalkeeper": {"type": {"name": "Shot Saved"}}}),
        _event(9, "Pressure", 1, 10, minute=89),
    ]
    return match, lineups, events


class EventFeatureTest(unittest.TestCase):
    def setUp(self):
        match, lineups, events = _match()
        self.rec = process_match(match, 11, 1, events, lineups)
        self.p = {p["player_id"]: p for p in self.rec["players"]}

    def test_passes_xa_and_progressive(self):
        p = self.p[10]
        self.assertEqual(p["passes"], 2)
        self.assertEqual(p["passes_completed"], 1)
        self.assertEqual(p["passes_progressive"], 1)
        self.assertEqual(p["key_passes"], 1)
        self.assertAlmostEqual(p["xa"], 0.3)

    def test_penalty_shootout_excluded_and_shot_counts(self):
        self.assertEqual(self.p[10]["shots"], 0)               # the shoot-out kick is period 5
        self.assertEqual(self.p[10]["penalties_taken"], 0)
        self.assertNotIn(12, self.p)                          # shooter 12 is not in the lineup, so his events are ignored

    def test_anomalous_carry_filtered(self):
        p = self.p[10]
        self.assertEqual(p["carries"], 1)                      # the 41 yd/s carry is dropped
        self.assertEqual(p["carries_progressive"], 1)

    def test_minutes_roles_and_gk(self):
        self.assertAlmostEqual(self.p[11]["minutes"], 45.0)
        self.assertEqual(self.p[20]["role_minutes"], {"FB": 30.0})
        self.assertEqual(self.p[11]["gk_saves"], 1)
        self.assertGreater(self.p[10]["minutes"], 85)          # played to the last event minute (89)
        self.assertFalse(self.p[11]["clean_sheet"])            # left at half time (< 60 GK minutes)

    def test_defensive_counts_and_team_rows(self):
        self.assertEqual(self.p[20]["tackles"], 1)
        self.assertEqual(self.p[20]["tackles_won"], 1)
        teams = {t["team_id"]: t for t in self.rec["teams"]}
        self.assertEqual(teams[1]["goals_for"], 1)
        self.assertAlmostEqual(teams[1]["xg_for"], 0.3)         # shoot-out xG excluded from team totals
        self.assertEqual(teams[1]["passes"], 2)

    def test_role_mapping(self):
        self.assertEqual(role_of("Left Center Back"), "CB")
        self.assertEqual(role_of("Right Wing Back"), "FB")
        self.assertEqual(role_of("Left Defensive Midfield"), "DM")
        self.assertEqual(role_of("Center Attacking Midfield"), "AM")
        self.assertEqual(role_of("Right Wing"), "W")
        self.assertEqual(role_of("Center Forward"), "CF")
        self.assertEqual(role_of("Goalkeeper"), "GK")


class CrosswalkHelpersTest(unittest.TestCase):
    def test_name_tokens_and_club_match(self):
        self.assertTrue(name_tokens("Lionel Messi") <= name_tokens("Lionel Andrés Messi Cuccittini"))
        self.assertFalse(name_tokens("Luis Suárez") <= name_tokens("Luis Alberto Díaz"))
        self.assertTrue(club_match("Barcelona", "FC Barcelona"))
        self.assertTrue(club_match("Atlético Madrid", "Club Atlético de Madrid"))
        self.assertFalse(club_match("Real Madrid", "FC Barcelona"))


# ----------------------------------------------------------------------------- shrinkage
class ShrinkageTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        n = 300
        minutes = rng.uniform(90, 3000, n)
        true = rng.gamma(4.0, 0.1, n)
        self.df = pd.DataFrame({"minutes": minutes, "grp": "A", "shots": rng.poisson(true * minutes / 90),
                                "passes": rng.integers(20, 2000, n)})
        self.df["succ"] = rng.binomial(self.df.passes, 0.8)

    def test_shrunk_rate_between_raw_and_prior_and_converges_with_exposure(self):
        poisson, binom = {"r": "shots"}, {"pc": ("succ", "passes")}
        params = shrinkage.fit(self.df, "grp", poisson, binom)
        out = shrinkage.apply(self.df, params, "grp", poisson, binom)
        raw = self.df.shots / (self.df.minutes / 90)
        prior = self.df.shots.sum() / (self.df.minutes.sum() / 90)
        short = self.df.minutes < 200
        self.assertTrue(((out.r_shr - prior).abs() <= (raw - prior).abs() + 1e-9).all())
        self.assertGreater(((raw - prior).abs() - (out.r_shr - prior).abs())[short].mean(),
                           ((raw - prior).abs() - (out.r_shr - prior).abs())[~short].mean())
        long_ = self.df.minutes > 2800
        self.assertLess((out.r_shr - raw)[long_].abs().mean(), (out.r_shr - raw)[short].abs().mean())

    def test_apply_uses_only_fitted_parameters(self):
        poisson, binom = {"r": "shots"}, {}
        params = shrinkage.fit(self.df.iloc[:150], "grp", poisson, binom)
        a = shrinkage.apply(self.df.iloc[150:], params, "grp", poisson, binom)
        later = self.df.copy()
        later.loc[later.index[:150], "shots"] = 999        # change data that is NOT in the fitted window
        b = shrinkage.apply(later.iloc[150:], params, "grp", poisson, binom)
        np.testing.assert_allclose(a.r_shr, b.r_shr)


# ----------------------------------------------------------------------------- leakage & pipeline
@unittest.skipUnless(PANEL.exists(), "processed Design A panel not built")
class TemporalLeakageTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from research.evaluation import rolling as R
        cls.R = R
        cls.panel = pd.read_parquet(PANEL)
        cls.cfg = R.Config()
        cls.specs = [{"name": "persistence_cal", "kind": "baseline", "model": "persistence_cal"},
                     {"name": "mcda_weighted_sum", "kind": "baseline", "model": "mcda_weighted_sum"},
                     {"name": "ridge|pi+profile+agepos|level", "kind": "ml", "model": "ridge",
                      "feature_set": "pi+profile+agepos", "formulation": "level"},
                     {"name": "rf|pi+profile|level", "kind": "ml", "model": "rf", "feature_set": "pi+profile", "formulation": "level"}]

    def test_pair_windows(self):
        train, test, ref = self.R.prepare_fold(self.panel, 2014, self.cfg)
        self.assertTrue((train.target_year <= 2014).all())
        self.assertTrue((test.target_year == 2015).all())
        self.assertTrue((test.season_year == 2014).all())
        self.assertGreater(len(train), 20)
        # the same player never appears in both with identical (player, season) rows
        key = lambda d: set(zip(d.player_id, d.season_year))
        self.assertFalse(key(train) & key(test))

    def test_test_outcomes_cannot_change_predictions(self):
        """Corrupt every observation after the scouting date -> predictions for that date must be identical."""
        year = 2014
        base = self.R.run_fold(self.panel, year, self.cfg, self.specs)
        corrupted = self.panel.copy()
        later = corrupted.season_year > year
        from research.targets import performance_index as P
        for col in P.PROFILE:
            corrupted.loc[later, col] = corrupted.loc[later, col] * 7 + 13
        again = self.R.run_fold(corrupted, year, self.cfg, self.specs)
        key = ["name", "player_id"]
        a = base.sort_values(key).reset_index(drop=True)
        b = again.sort_values(key).reset_index(drop=True)
        np.testing.assert_allclose(a.yhat.to_numpy(), b.yhat.to_numpy(), rtol=1e-9, atol=1e-9)
        # ... while the observed outcome (which lives in the corrupted period) did change
        self.assertFalse(np.allclose(a.pi_next.to_numpy(), b.pi_next.to_numpy()))

    def test_reference_depends_only_on_visible_seasons(self):
        from research.targets import performance_index as P
        seen = self.panel[self.panel.season_year <= 2012]
        r1 = P.fit_reference(seen, 450)
        more = pd.concat([seen, self.panel[self.panel.season_year > 2012]])
        r2 = P.fit_reference(more[more.season_year <= 2012], 450)
        self.assertEqual(r1.stats, r2.stats)
        r3 = P.fit_reference(self.panel, 450)
        self.assertNotEqual(r1.stats, r3.stats)

    def test_index_excludes_minutes_and_gk(self):
        from research.targets import performance_index as P
        for comps in P.COMPONENTS.values():
            self.assertNotIn("minutes", " ".join(comps))
        self.assertNotIn("GK", P.COMPONENTS)
        ref = P.fit_reference(self.panel, 450)
        d = P.performance_index(self.panel, ref)
        self.assertTrue(d[d.position_group == "GK"].pi.isna().all())
        self.assertTrue(d[d.position_group != "GK"].pi.notna().all())

    def test_index_is_not_the_legacy_ranking_score(self):
        from research.models import baselines as bl
        from research.targets import performance_index as P
        d = self.panel[(self.panel.minutes >= 450) & (self.panel.position_group != "GK")].copy()
        d = P.performance_index(d, P.fit_reference(self.panel, 450))
        d.index = np.arange(len(d))
        sc = bl.mcda_scores(d)["weighted_sum"]
        ok = sc.notna()
        rho = M.spearman(d.pi[ok], sc[ok])
        self.assertLess(abs(rho), 0.9)              # not a re-labelling of the ranking score

    def test_mcda_adapter_matches_legacy_ranker(self):
        from processing.scouting_channel import PlayerProfile, _rank_group
        from research.models import baselines as bl
        d = self.panel[(self.panel.season_year == 2017) & (self.panel.minutes >= 450) & (self.panel.position_group == "FWD")].copy()
        d.index = np.arange(len(d))
        sc = bl.mcda_scores(d)
        m = bl.legacy_metrics(d)
        profs = []
        for i in d.index:
            mm = {k: float(m.at[i, k]) for k in m.columns}
            mm["goals_assists_per90"] = mm["goals_per90"] + mm["assists_per90"]
            profs.append(PlayerProfile(str(i), str(i), "FWD", mm))
        direct = {int(r["player_id"]): r["score"] for r in _rank_group(profs)["topsis"]}
        np.testing.assert_allclose([direct[i] for i in d.index], sc["topsis"].to_numpy())
        self.assertTrue(((sc.dropna() >= 0) & (sc.dropna() <= 1)).all().all())


@unittest.skipUnless(FROZEN.exists() and PANEL.exists(), "frozen artifacts not built")
class InferenceTest(unittest.TestCase):
    def test_score_is_deterministic_and_outcome_free(self):
        from research import inference as I
        frozen = I.load_frozen()
        panel = pd.read_parquet(PANEL)
        latest = panel[panel.season_year == frozen["meta"]["latest_scouting_season_start_year"]]
        a = I.score(frozen, latest)
        b = I.score(frozen, latest.drop(columns=[c for c in ("value_t",) if c in latest]))
        np.testing.assert_allclose(a.pred_next_pi, b.pred_next_pi)
        self.assertTrue(np.isfinite(a.pred_next_pi).all())
        self.assertTrue((a.pred_hi_80 > a.pred_next_pi).all() and (a.pred_lo_80 < a.pred_next_pi).all())
        self.assertTrue(((a.pred_hi_90 - a.pred_lo_90) > (a.pred_hi_80 - a.pred_lo_80)).all())
        self.assertFalse(a.position_group.eq("GK").any())
        self.assertTrue((a.minutes >= frozen["meta"]["min_minutes"]).all())

    def test_metadata_records_reproducibility_fields(self):
        meta = json.loads(FROZEN.read_text())
        for key in ("seed", "model", "training_pairs", "panel_hash_sha256", "selection_rule", "conformal_halfwidth_80", "versions"):
            self.assertIn(key, meta)

    def test_selected_model_chosen_on_development_period_only(self):
        from research.analysis import results as RS
        p = RS.load("A")
        dev = p[p.period == "development"]
        sel = RS.select(p)["overall"]
        by = RS._rmse_by(dev)
        ml = by[[n for n in by.index if "|" in n]]
        self.assertEqual(sel, ml.idxmin())
        self.assertTrue(dev.test_year.max() <= 2014)


class LegacyRegressionTest(unittest.TestCase):
    def test_scouting_channel_cli_imports_argparse(self):
        import processing.scouting_channel as sc
        self.assertTrue(hasattr(sc, "argparse"))


if __name__ == "__main__":
    unittest.main()
