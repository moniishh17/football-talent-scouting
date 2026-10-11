"""Summarise predictions_sensitivity.parquet: RMSE of each model relative to the variant's own persistence_cal baseline."""
import numpy as np
import pandas as pd

from research.config import PROCESSED, REPORTS
from research.evaluation import metrics as M


def main():
    p = pd.read_parquet(PROCESSED / "predictions_sensitivity.parquet")
    rows = []
    for (d, v), g in p.groupby(["design", "variant"]):
        base = g[g.name == "persistence_cal"]
        for n, h in g.groupby("name"):
            m = h.merge(base[["player_id", "test_year", "yhat"]], on=["player_id", "test_year"], suffixes=("", "_b"))
            y = m.pi_next.to_numpy(); e1 = m.yhat.to_numpy(); e0 = m.yhat_b.to_numpy()
            r, lo, hi = M.cluster_bootstrap(lambda i: M.rmse(y[i], e1[i]) / M.rmse(y[i], e0[i]), m.player_id.to_numpy(), 500, 0)
            rows.append({"design": d, "variant": v, "model": n, "n": len(m), "rmse": M.rmse(y, e1), "rmse_persistence_cal": M.rmse(y, e0),
                         "ratio_to_persistence_cal": r, "ratio_lo": lo, "ratio_hi": hi})
    t = pd.DataFrame(rows)
    t.to_csv(REPORTS / "tables" / "sensitivity.csv", index=False)
    pd.set_option("display.width", 250, "display.max_colwidth", 60)
    print(t[t.model != "persistence_cal"].round(3).to_string())


if __name__ == "__main__":
    main()
