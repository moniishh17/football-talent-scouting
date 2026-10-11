"""Data access for the dashboard: reads precomputed research outputs; scores history with the frozen model."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
TABLES = REPORTS / "tables"
PROC = ROOT / "data" / "research" / "processed"

MODEL_LABELS = {"persistence": "Persistence (raw)", "persistence_cal": "Persistence (calibrated)", "mean": "Train mean",
                "mean_by_pos": "Mean by position", "age_pos": "Age/position ridge", "mcda_weighted_sum": "Legacy WSM",
                "mcda_topsis": "Legacy TOPSIS", "mcda_vikor": "Legacy VIKOR", "equal_weight": "Equal-weight legacy"}


def pretty(name: str) -> str:
    if name in MODEL_LABELS:
        return MODEL_LABELS[name]
    parts = name.split("|")
    fam = {"ridge": "Ridge", "enet": "Elastic Net", "rf": "Random Forest", "lgbm": "LightGBM"}.get(parts[0], parts[0])
    fs = parts[1] if len(parts) > 1 else ""
    fs = (fs.replace("pi+profile+agepos+exposure+context+league+history", "full + league + history")
            .replace("pi+profile+agepos+exposure+context+history", "full + history")
            .replace("pi+profile+agepos+exposure+context", "profile+age/pos+exposure+context")
            .replace("pi+profile+agepos+exposure", "profile+age/pos+exposure")
            .replace("pi+profile+agepos", "profile+age/pos").replace("pi+profile", "profile").replace("pi", "PI only"))
    form = f" ({parts[2]})" if len(parts) > 2 else ""
    return f"{fam} — {fs}{form}"


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def selected() -> dict:
    return json.loads((TABLES / "selected_configs.json").read_text())


def main_table(design: str) -> pd.DataFrame:
    return read_csv(TABLES / f"{design}_main.csv")


def backtest(design: str) -> pd.DataFrame:
    """Out-of-sample predictions (every row predicted by a model trained only on earlier data)."""
    if design == "A":
        p = pd.read_parquet(PROC / "predictions_design_a.parquet")
    elif design == "B":
        p = pd.read_parquet(PROC / "predictions_design_b.parquet")
        p["period"] = np.where(p.test_year <= 2, "development", "confirmation")
    else:
        p = pd.read_parquet(PROC / "predictions_valuation.parquet").rename(columns={"y": "pi_next"})
    p["model"] = p["name"].map(pretty)
    return p


def candidates() -> pd.DataFrame:
    a = read_csv(REPORTS / "candidates_in_domain_la_liga_2020_21.csv")
    b = read_csv(REPORTS / "candidates_out_of_domain.csv")
    a["scope"] = "In-domain: La Liga 2020/21 (Barcelona)"
    if len(b):
        b["scope"] = b["competition_name"] + " " + b["season_name"] + " (out-of-domain, unvalidated)"
    df = pd.concat([a, b], ignore_index=True)
    df["interval_width_80"] = df.pred_hi_80 - df.pred_lo_80
    return df


def history(player_id: int) -> pd.DataFrame:
    """Seasonal performance-index history for a Barcelona-panel player (frozen reference; description only)."""
    from research import inference as I
    from research.targets import performance_index as P
    panel = pd.read_parquet(PROC / "design_a_player_seasons.parquet")
    frozen = I.load_frozen()
    d = panel[(panel.player_id == player_id) & panel.position_group.isin(P.OUTFIELD)].copy()
    if d.empty:
        return d
    d = P.performance_index(d, frozen["ref"])
    return d.sort_values("season_year")


def momentum(cands: pd.DataFrame) -> pd.Series:
    """PI change vs the previous season for in-domain players (NaN when not observed >= 450 min)."""
    try:
        from research import inference as I
        from research.targets import performance_index as P
        panel = pd.read_parquet(PROC / "design_a_player_seasons.parquet")
        panel = panel[panel.position_group.isin(P.OUTFIELD) & (panel.minutes >= 450)]
        d = P.performance_index(panel, I.load_frozen()["ref"])[["player_id", "season_year", "pi"]]
        prev = d.assign(season_year=d.season_year + 1).rename(columns={"pi": "pi_prev"})
        latest = d[d.season_year == d.season_year.max()].merge(prev, on=["player_id", "season_year"], how="left")
        m = latest.set_index("player_id").eval("pi - pi_prev")
        return cands["player_id"].map(m)
    except Exception:
        return pd.Series(np.nan, index=cands.index)
