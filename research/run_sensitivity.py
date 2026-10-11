"""Sensitivity of the conclusions to exposure thresholds, position definition, shrinkage and target definition.

Run: python -m research.run_sensitivity
Each variant changes ONE choice relative to the base configuration and is compared with its own persistence_cal baseline
(RMSE of different targets/cohorts is not comparable across variants; the ratio to the variant's baseline is).
"""
from __future__ import annotations

import pandas as pd

from research.config import PROCESSED
from research.evaluation import rolling as R
from research.run_design_b import config as b_config

SPECS_A = [
    {"name": "mean", "kind": "baseline", "model": "mean"},
    {"name": "persistence_cal", "kind": "baseline", "model": "persistence_cal"},
    {"name": "mcda_weighted_sum", "kind": "baseline", "model": "mcda_weighted_sum"},
    {"name": "ridge|pi+profile|level", "kind": "ml", "model": "ridge", "feature_set": "pi+profile", "formulation": "level"},
    {"name": "ridge|pi+profile+agepos+exposure+context+history|level", "kind": "ml", "model": "ridge",
     "feature_set": "pi+profile+agepos+exposure+context+history", "formulation": "level"},
    {"name": "lgbm|pi+profile+agepos|level", "kind": "ml", "model": "lgbm", "feature_set": "pi+profile+agepos", "formulation": "level"},
]
SPECS_B = [
    {"name": "mean", "kind": "baseline", "model": "mean"},
    {"name": "persistence_cal", "kind": "baseline", "model": "persistence_cal"},
    {"name": "ridge|pi+profile|level", "kind": "ml", "model": "ridge", "feature_set": "pi+profile", "formulation": "level"},
    {"name": "ridge|pi+profile+agepos+exposure+context+league+history|level", "kind": "ml", "model": "ridge",
     "feature_set": "pi+profile+agepos+exposure+context+league+history", "formulation": "level"},
    {"name": "rf|pi+profile+agepos+exposure+context+league+history|level", "kind": "ml", "model": "rf",
     "feature_set": "pi+profile+agepos+exposure+context+league+history", "formulation": "level"},
]


def main(only: list[str] | None = None):
    a = pd.read_parquet(PROCESSED / "design_a_player_seasons.parquet")
    b = pd.read_parquet(PROCESSED / "design_b_blocks.parquet")
    out = []
    variants_a = {
        "base": R.Config(), "shrink_features": R.Config(shrink=True), "shrink_features_and_target": R.Config(shrink=True, shrink_target=True),
        "min_minutes_270": R.Config(min_minutes=270, min_minutes_target=270),
        "min_minutes_900": R.Config(min_minutes=900, min_minutes_target=900),
        "role_reference_groups": R.Config(group_col="primary_role"),
    }
    for name, cfg in variants_a.items():
        if only and name not in only:
            continue
        p = R.run_all(a, cfg, SPECS_A)
        out.append(p.assign(design="A", variant=name))
        print("A", name, p.shape)
    variants_b = {"base": b_config(), "shrink_features": b_config(shrink=True), "shrink_features_and_target": b_config(shrink=True, shrink_target=True),
                  "min_minutes_180": R.Config(min_minutes=180, min_minutes_target=180, first_test_year=2, last_test_year=3),
                  "min_minutes_600": R.Config(min_minutes=600, min_minutes_target=600, first_test_year=2, last_test_year=3),
                  "role_reference_groups": b_config(group_col="primary_role")}
    for name, cfg in variants_b.items():
        if only and name not in only:
            continue
        p = R.run_all(b, cfg, SPECS_B)
        out.append(p.assign(design="B", variant=name))
        print("B", name, p.shape)
    new = pd.concat(out, ignore_index=True)
    path = PROCESSED / "predictions_sensitivity.parquet"
    if only and path.exists():          # re-run of selected variants: replace only those rows
        old = pd.read_parquet(path)
        new = pd.concat([old[~old.variant.isin(only)], new], ignore_index=True)
    new.to_parquet(path, index=False)


if __name__ == "__main__":
    import sys
    main(sys.argv[1:] or None)
