"""Design B experiments (within-season rolling origin, 4 leagues, 2015/16).  Run: python -m research.run_design_b

Blocks of ~9-10 matchweeks (1..4). Test transitions: block2->block3 (train: block1->2) and block3->block4
(train: block1->2, block2->3). Time index `season_year` = block number. Horizon is ~2 months, so this design measures
short-horizon persistence/dynamics and the value of league context and exposure adjustments with larger n.
"""
from __future__ import annotations

import argparse

import pandas as pd

from research.config import PROCESSED
from research.evaluation import rolling as R

FEATURE_SETS = ["pi", "pi+profile", "pi+profile+agepos", "pi+profile+agepos+exposure",
                "pi+profile+agepos+exposure+context", "pi+profile+agepos+exposure+context+league",
                "pi+profile+agepos+exposure+context+league+history"]
ML_MODELS = ["ridge", "enet", "rf", "lgbm"]
BASELINES = ["mean", "mean_by_pos", "persistence", "persistence_cal", "age_pos",
             "mcda_weighted_sum", "mcda_topsis", "mcda_vikor", "equal_weight"]


def specs(models=ML_MODELS, feature_sets=FEATURE_SETS, formulations=("level", "delta")):
    out = [{"name": b, "kind": "baseline", "model": b} for b in BASELINES]
    for m in models:
        for fs in feature_sets:
            for f in formulations:
                out.append({"name": f"{m}|{fs}|{f}", "kind": "ml", "model": m, "feature_set": fs, "formulation": f})
    return out


def config(**kw) -> R.Config:
    return R.Config(min_minutes=360, min_minutes_target=360, first_test_year=2, last_test_year=3, **kw)


def main():
    panel = pd.read_parquet(PROCESSED / "design_b_blocks.parquet")
    pred = R.run_all(panel, config(), specs())
    pred.to_parquet(PROCESSED / "predictions_design_b.parquet", index=False)
    print("predictions", pred.shape)


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    main()
