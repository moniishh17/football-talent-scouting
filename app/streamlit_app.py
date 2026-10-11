"""ScoutIQ Research dashboard.  Run from talent-scouting/:  streamlit run app/streamlit_app.py

Shows precomputed research outputs and scores with the frozen model. Everything displayed is a MODEL OUTPUT for research
review; none of it is evidence that a prediction is correct (see reports/05_results.md).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app import data as D

st.set_page_config(page_title="ScoutIQ Research", page_icon="⚽", layout="wide")
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#898781"
CAVEAT = ("Model outputs for research review — not proof that any prediction is correct, and not claims about future stardom. "
          "Based on Barcelona-centric StatsBomb Open Data; intervals are wide (≈ ±0.4 on an index whose player-to-player SD is ≈ 0.6).")


@st.cache_data(show_spinner=False)
def load_candidates():
    c = D.candidates()
    c["momentum"] = D.momentum(c)
    return c


@st.cache_data(show_spinner=False)
def load_backtest(design):
    return D.backtest(design)


@st.cache_data(show_spinner=False)
def load_history(pid):
    return D.history(int(pid))


@st.cache_data(show_spinner=False)
def component_zscores(scope_is_indomain: bool, pid: int):
    """z-scores of the profile metrics for a candidate (in-domain: frozen reference on the panel)."""
    from research import inference as I
    from research.targets import performance_index as P
    frozen = I.load_frozen()
    if scope_is_indomain:
        panel = pd.read_parquet(D.PROC / "design_a_player_seasons.parquet")
        rows = panel[(panel.season_year == frozen["meta"]["latest_scouting_season_start_year"]) & (panel.player_id == pid)]
    else:
        from research.score_latest import OUT_OF_DOMAIN, other_panel
        rows = pd.concat([other_panel(*k) for k, _, _ in OUT_OF_DOMAIN])
        rows = rows[rows.player_id == pid]
    s = I.score(frozen, rows)
    if s.empty:
        return None
    r = s.iloc[0]
    comp = P.COMPONENTS[r.position_group]
    return pd.DataFrame({"component": [I.COMPONENT_LABELS[m] for m in comp], "contribution_sd": [sg * r[f"z_{m}"] for m, sg in comp.items()]})


def flags(r) -> str:
    out = []
    if r.minutes < 900:
        out.append("low exposure (<900 min)")
    if not bool(r.age_available):
        out.append("age unknown")
    if r.n_prior_seasons_observed == 0:
        out.append("no prior season observed")
    if str(r.scope).find("out-of-domain") >= 0:
        out.append("out-of-domain")
    return "; ".join(out)


st.sidebar.title("⚽ ScoutIQ Research")
page = st.sidebar.radio("Page", ["Overview", "Players to watch", "Player detail", "Model predictions (backtest)", "Data & method"])
st.sidebar.caption(CAVEAT)

cands = load_candidates()
cands["flags"] = [flags(r) for r in cands.itertuples()]

# ------------------------------------------------------------------------------------------------ Overview
if page == "Overview":
    st.title("Football talent-development prediction — research dashboard")
    st.warning(CAVEAT)
    a, b = D.main_table("A"), D.main_table("B")
    sel = D.selected()

    def row(df, name):
        return df[df.name == name].iloc[0]

    ra, rp = row(a, sel["A"]["overall"]), row(a, "persistence_cal")
    rb, rpb = row(b, sel["B"]["overall"]), row(b, "persistence_cal")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Design A (next season, Barça) — best model RMSE", f"{ra.confirm_rmse:.3f}", f"{ra.confirm_rmse - rp.confirm_rmse:+.3f} vs persistence", delta_color="off")
    c2.metric("Persistence baseline RMSE", f"{rp.confirm_rmse:.3f}")
    c3.metric("Design B (next block, 4 leagues) — best model RMSE", f"{rb.confirm_rmse:.3f}", f"{rb.confirm_rmse - rpb.confirm_rmse:+.3f} vs persistence", delta_color="inverse")
    c4.metric("Legacy WSM RMSE (Design A)", f"{row(a, 'mcda_weighted_sum').confirm_rmse:.3f}")
    st.markdown("""
**What the research found**
- Next-season performance (Barcelona players, n = 143 held-out pairs): performance is highly persistent; **no model demonstrably beats calibrated persistence**. Age, position, team context and exposure add nothing demonstrable.
- Short-horizon performance across four leagues (n ≈ 1,400): richer models **do** beat persistence by ≈ 0.035 RMSE (the gain comes from the full stat profile and earlier-window history).
- The original WSM/TOPSIS/VIKOR rankers are far worse at predicting the new performance index; which ranker looks best depends on how "performance" is defined.
- Season-to-season *change* is mostly measurement noise, so 'breakout' labels are not reliable and are not shown here.
- Predicted "development" below is mostly regression to the mean: low-index players are predicted to rise, high-index players to fall.
""")
    st.caption("Source: reports/05_results.md. Confirmation-period RMSE shown (never used for model selection).")

# ------------------------------------------------------------------------------------------------ Players to watch
elif page == "Players to watch":
    st.title("Players to look out for")
    st.info("Shortlist from the latest seasons available in StatsBomb Open Data. Use it to decide whom to *look at*, not whom to buy. " + CAVEAT)
    f1, f2, f3, f4 = st.columns(4)
    scopes = f1.multiselect("Season / competition", sorted(cands.scope.unique()), default=[s for s in cands.scope.unique() if s.startswith("In-domain")])
    pos = f2.multiselect("Position", ["DEF", "MID", "FWD"], default=["DEF", "MID", "FWD"])
    min_min = f3.slider("Minimum minutes", 450, 3000, 900, 50)
    age_rng = f4.slider("Age range (unknown ages kept)", 16, 40, (16, 40))
    crit = st.radio("Rank by", [
        "Highest predicted next-season index",
        "Highest predicted change vs cohort (mostly mean reversion)",
        "Recent momentum (index change vs previous season, Barcelona only)",
        "Young and strong (age ≤ 23, current index > 0)"], horizontal=False)
    d = cands[cands.scope.isin(scopes) & cands.position_group.isin(pos) & (cands.minutes >= min_min)
              & (cands.age.isna() | cands.age.between(*age_rng))].copy()
    if crit.startswith("Highest predicted next"):
        d = d.sort_values("pred_next_pi", ascending=False)
    elif crit.startswith("Highest predicted change"):
        d = d.sort_values("pred_change_vs_cohort_mean", ascending=False)
    elif crit.startswith("Recent momentum"):
        d = d[d.momentum.notna()].sort_values("momentum", ascending=False)
    else:
        d = d[(d.age <= 23) & (d.pi > 0)].sort_values("pi", ascending=False)
    if d.empty:
        st.warning("No players match these filters.")
    else:
        d["rank"] = np.arange(1, len(d) + 1)
        show = d[["rank", "player_name", "team", "scope", "position_group", "age", "minutes", "pi", "momentum", "pred_next_pi", "pred_lo_80", "pred_hi_80",
                  "pred_change_vs_cohort_mean", "flags"]].rename(columns={
            "player_name": "Player", "team": "Team", "scope": "Season", "position_group": "Pos", "age": "Age", "minutes": "Minutes",
            "pi": "Current index", "momentum": "Momentum", "pred_next_pi": "Pred. next index", "pred_lo_80": "80% low", "pred_hi_80": "80% high",
            "pred_change_vs_cohort_mean": "Pred. change vs cohort", "flags": "Data-quality flags"})
        st.dataframe(show, hide_index=True, use_container_width=True, column_config={
            "Age": st.column_config.NumberColumn(format="%.1f"), "Minutes": st.column_config.NumberColumn(format="%d"),
            "Current index": st.column_config.NumberColumn(format="%.2f", help="Position-standardised performance index (z); 0 = average of the Barcelona reference"),
            "Momentum": st.column_config.NumberColumn(format="%+.2f"), "Pred. next index": st.column_config.NumberColumn(format="%.2f"),
            "80% low": st.column_config.NumberColumn(format="%.2f"), "80% high": st.column_config.NumberColumn(format="%.2f"),
            "Pred. change vs cohort": st.column_config.NumberColumn(format="%+.2f")})
        st.download_button("Download shortlist (CSV)", d.to_csv(index=False).encode(), "scoutiq_watchlist.csv", "text/csv")
        top = d.head(15).iloc[::-1]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=top.pred_next_pi, y=top.player_name, mode="markers", marker=dict(color=BLUE, size=9), name="Predicted next index",
                                 error_x=dict(type="data", symmetric=False, array=top.pred_hi_80 - top.pred_next_pi, arrayminus=top.pred_next_pi - top.pred_lo_80, color=BLUE)))
        fig.add_trace(go.Scatter(x=top.pi, y=top.player_name, mode="markers", marker=dict(color=ORANGE, size=9, symbol="diamond"), name="Current index"))
        fig.update_layout(height=max(320, 28 * len(top) + 120), margin=dict(l=10, r=10, t=30, b=10), xaxis_title="Performance index (z) — bars: 80% prediction interval",
                          title="Top 15 of the current ranking: current vs predicted next index", legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Intervals overlap heavily across most players: ranks within a few places of each other are not distinguishable. "
                   "'Momentum' is the index change vs the previous season (only for Barcelona 2020/21 players observed ≥ 450 min the season before) and is dominated by measurement noise.")

# ------------------------------------------------------------------------------------------------ Player detail
elif page == "Player detail":
    st.title("Player detail")
    scope = st.selectbox("Season / competition", sorted(cands.scope.unique(), key=lambda x: (not x.startswith("In-domain"), x)))
    sub = cands[cands.scope == scope].sort_values("player_name")
    name = st.selectbox("Player", sub.player_name)
    r = sub[sub.player_name == name].iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current index", f"{r.pi:.2f}")
    c2.metric("Predicted next-season index", f"{r.pred_next_pi:.2f}", f"80%: {r.pred_lo_80:.2f} to {r.pred_hi_80:.2f}", delta_color="off")
    c3.metric("Rank in cohort (current → predicted)", f"{int(r.rank_current_index)} → {int(r.rank_pred_next_index)}", f"of {int(r.cohort_n)} (all positions; ranks are within {r.position_group})", delta_color="off")
    c4.metric("Minutes / age", f"{int(r.minutes)} / {'n/a' if pd.isna(r.age) else f'{r.age:.1f}'}")
    st.write(f"**{r.player_name}** — {r.team}, {r.position_group} ({r.primary_role}), {r.scope}")
    st.write(r.explanation)
    if r["flags"]:
        st.warning("Data-quality flags: " + r["flags"])
    left, right = st.columns(2)
    with left:
        h = load_history(r.player_id) if r.scope.startswith("In-domain") else pd.DataFrame()
        if len(h):
            h = h[h.minutes >= 450]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=h.season_name, y=h.pi, mode="lines+markers", name="Observed index (≥ 450 min)", line=dict(color=BLUE)))
            last = h.season_name.iloc[-1]
            nxt = f"{int(last[:4]) + 1}/{int(last[5:]) + 1}" if "/" in last else "next"
            fig.add_trace(go.Scatter(x=[last, nxt], y=[r.pi, r.pred_next_pi], mode="lines+markers", name="Prediction", line=dict(color=ORANGE, dash="dash"),
                                     error_y=dict(type="data", array=[0, r.pred_hi_80 - r.pred_next_pi], arrayminus=[0, r.pred_next_pi - r.pred_lo_80])))
            fig.update_layout(title="Index history and prediction (80% interval)", height=360, margin=dict(l=10, r=10, t=40, b=10), yaxis_title="index (z)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("No multi-season history in the training panel for this scope.")
    with right:
        try:
            z = component_zscores(r.scope.startswith("In-domain"), int(r.player_id))
        except Exception as exc:  # cache missing etc.
            z = None
            st.caption(f"Component breakdown unavailable ({exc.__class__.__name__}).")
        if z is not None:
            fig = px.bar(z.sort_values("contribution_sd"), x="contribution_sd", y="component", orientation="h", color_discrete_sequence=[BLUE])
            fig.update_layout(title="What drives the current index (sd vs reference, sign-adjusted)", height=360, margin=dict(l=10, r=10, t=40, b=10), xaxis_title="standard deviations")
            st.plotly_chart(fig, use_container_width=True)
    if "pred_log_value_change_12m_ESTIMATE" in r and pd.notna(r.get("pred_log_value_change_12m_ESTIMATE")):
        st.subheader("Market value (estimate — weak evidence)")
        st.write(f"Valuation at scouting date: €{r.value_t_eur:,.0f}. Model estimate of 12-month change: {100 * (np.exp(r.pred_log_value_change_12m_ESTIMATE) - 1):+.0f}% "
                 f"(80% range €{r.pred_value_lo80_eur_ESTIMATE:,.0f} – €{r.pred_value_hi80_eur_ESTIMATE:,.0f}). The research found no demonstrated gain over a simple age/position/value baseline "
                 "(confirmation interval includes 0); valuations are crowd opinions, not transfer fees.")

# ------------------------------------------------------------------------------------------------ Model predictions
elif page == "Model predictions (backtest)":
    st.title("Model predictions (out-of-sample backtest)")
    st.caption("Every prediction below was made by a model trained only on data available before the scouting date (rolling origin). "
               "Development period = used to choose models; confirmation period = held-out reporting.")
    tab = st.tabs(["Design A — next season (Barcelona)", "Design B — next block (4 leagues)", "Valuation — 12-month log value change"])
    for t, design in zip(tab, "ABV"):
        with t:
            main = D.main_table(design)
            main["Model"] = main.name.map(D.pretty)
            lb = main[["Model", "selected_as", "dev_rmse", "confirm_rmse", "confirm_rmse_lo", "confirm_rmse_hi", "confirm_spearman" if "confirm_spearman" in main else "all_spearman"]].rename(
                columns={"selected_as": "Selected as", "dev_rmse": "Dev RMSE", "confirm_rmse": "Confirm RMSE", "confirm_rmse_lo": "CI low", "confirm_rmse_hi": "CI high",
                         "confirm_spearman": "Spearman (confirm)", "all_spearman": "Spearman (all)"})
            st.subheader("Leaderboard (lower RMSE is better; overlapping CIs ≠ a difference)")
            st.dataframe(lb.sort_values("Confirm RMSE"), hide_index=True, use_container_width=True,
                         column_config={c: st.column_config.NumberColumn(format="%.3f") for c in lb.columns if c not in ("Model", "Selected as")})
            bt = load_backtest(design)
            models = sorted(bt.model.unique())
            default = [D.pretty(D.selected()[design]["overall"]), D.pretty("persistence_cal")]
            chosen = st.multiselect("Compare models", models, default=[m for m in default if m in models], key=f"m{design}")
            c1, c2 = st.columns(2)
            years = sorted(bt.test_year.unique())
            yr = c1.selectbox("Test fold (scouting season / block)", ["All"] + years, key=f"y{design}")
            per = c2.selectbox("Period", ["all", "development", "confirmation"], key=f"p{design}")
            d = bt[bt.model.isin(chosen)]
            if yr != "All":
                d = d[d.test_year == yr]
            if per != "all":
                d = d[d.period == per]
            if d.empty:
                st.info("Select at least one model.")
                continue
            lab = "observed 12-month log value change" if design == "V" else "observed next-period index"
            fig = px.scatter(d, x="yhat", y="pi_next", color="model", hover_data=["player_name", "position_group", "test_year"], opacity=0.6,
                             labels={"yhat": "predicted", "pi_next": lab}, color_discrete_sequence=[BLUE, ORANGE, AQUA, "#e87ba4"])
            lo, hi = float(min(d.yhat.min(), d.pi_next.min())), float(max(d.yhat.max(), d.pi_next.max()))
            fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color=GREY, dash="dash"), name="perfect"))
            fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)
            piv = d.pivot_table(index=["player_name", "position_group", "test_year"], columns="model", values="yhat").reset_index()
            obs = d.drop_duplicates(["player_name", "test_year"])[["player_name", "test_year", "pi_next"]].rename(columns={"pi_next": "observed"})
            piv = piv.merge(obs, on=["player_name", "test_year"], how="left")
            st.subheader("Player-level predictions vs outcome")
            st.dataframe(piv.sort_values(["test_year", "observed"], ascending=[False, False]), hide_index=True, use_container_width=True,
                         column_config={c: st.column_config.NumberColumn(format="%.2f") for c in piv.columns if c not in ("player_name", "position_group", "test_year")})
    st.caption("Design A uses ~143 held-out pairs from ~60 Barcelona players; Design B has two test folds of ≈ 690 pairs over a ≈ 2-month horizon. See reports/05_results.md for limits.")

# ------------------------------------------------------------------------------------------------ Data & method
else:
    st.title("Data & method")
    st.markdown("""
- **Source:** StatsBomb Open Data (credit StatsBomb). Club coverage is mostly single-club (Barcelona in La Liga 2004/05–2020/21; PSG, Leverkusen elsewhere); only the 2015/16 leagues are complete. No 2025-era men's club data exists.
- **Target:** a position-standardised *performance index* (equal-weight z-scores of position-specific per-90 metrics; minutes are not part of it). Goalkeepers are excluded.
- **Age** comes from a documented name + club crosswalk to a Transfermarkt snapshot (≈ 84 % of players linked).
- **Validation:** rolling-origin by season (Design A) or matchweek block (Design B); models chosen on the development period only.
- **Prediction intervals:** split-conformal, calibrated on earlier folds; empirical coverage ≈ 81 % at nominal 80 % for Design A (under-covers in Design B).
- **Out-of-domain** rows (Ligue 1 2022/23 PSG, Bundesliga 2023/24 Leverkusen) apply a model trained on Barcelona seasons and are unvalidated.
""")
    for title, f in (("Data availability (competition-seasons with ≥ 5 matches)", D.REPORTS.parent / "data" / "research" / "audit" / "competition_seasons.csv"),):
        df = D.read_csv(f)
        if len(df):
            st.subheader(title)
            st.dataframe(df[df.matches >= 5], hide_index=True, use_container_width=True)
    c1, c2 = st.columns(2)
    for col, img in ((c1, "figures/audit/01_coverage_by_competition_season.png"), (c2, "figures/models/m1_confirmation_rmse_forest.png")):
        p = ROOT / img
        if p.exists():
            col.image(str(p), use_container_width=True)
    st.markdown("Full write-up: `reports/05_results.md`; limitations: §6.")
