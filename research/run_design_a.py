"""Design A experiments (next-season performance, Barcelona-match panel).  Run: python -m research.run_design_a

Outputs (reports/tables/, data/research/processed/): predictions_design_a*.parquet and metric tables.
Development period (model/feature-set selection) = test transitions 2009/10->2010/11 .. 2014/15->2015/16.
Confirmation period (reported, never used for selection)  = 2015/16->2016/17 .. 2019/20->2020/21.
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from research.config import PROCESSED, REPORTS
from research.evaluation import rolling as R
from research.evaluation import report as RP

TABLES = REPORTS / "tables"
DEV_MAX_TEST_YEAR = 2014
FEATURE_SETS = ["pi", "pi+profile", "pi+profile+agepos", "pi+profile+agepos+exposure",
                "pi+profile+agepos+exposure+context", "pi+profile+agepos+exposure+context+history"]
ML_MODELS = ["ridge", "enet", "rf", "lgbm"]
BASELINES = ["mean", "mean_by_pos", "persistence", "persistence_cal", "age_pos",
             "mcda_weighted_sum", "mcda_topsis", "mcda_vikor", "equal_weight"]


def specs(formulations=("level", "delta"), models=ML_MODELS, feature_sets=FEATURE_SETS) -> list[dict]:
    out = [{"name": b, "kind": "baseline", "model": b} for b in BASELINES]
    for m in models:
        for fs in feature_sets:
            for f in formulations:
                out.append({"name": f"{m}|{fs}|{f}", "kind": "ml", "model": m, "feature_set": fs, "formulation": f})
    return out


def load_panel() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / "design_a_player_seasons.parquet")


def tag_period(pred: pd.DataFrame) -> pd.DataFrame:
    pred = pred.copy()
    pred["period"] = pred["test_year"].map(lambda y: "development" if y <= DEV_MAX_TEST_YEAR else "confirmation")
    return pred


def main(variants: str = "main"):
    TABLES.mkdir(parents=True, exist_ok=True)
    panel = load_panel()
    cfg = R.Config()
    pred = tag_period(R.run_all(panel, cfg, specs()))
    pred.to_parquet(PROCESSED / "predictions_design_a.parquet", index=False)
    print("main predictions", pred.shape)
    return pred


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    main()
