"""EDA figures and tables (no model fitting). Run: python -m research.analysis.eda

Descriptive z-scores here use a reference fitted on the full panel -- acceptable for description only; all modelling code
re-fits the reference inside each training window.
Figures -> figures/audit, figures/eda ; tables -> reports/tables.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

from research.analysis import style as S
from research.config import AUDIT, FIGURES, PROCESSED, REPORTS
from research.features import player_season as ps
from research.features import panel as PN
from research.targets import performance_index as P

S.apply()
TABLES = REPORTS / "tables"
DIV = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#eb6834"])


def load():
    a = pd.read_parquet(PROCESSED / "design_a_player_seasons.parquet")
    b = pd.read_parquet(PROCESSED / "design_b_blocks.parquet")
    refa = P.fit_reference(a, 450)
    refb = P.fit_reference(b, 360)
    return a, b, P.performance_index(a, refa), P.performance_index(b, refb)


def fig_audit():
    cs = pd.read_csv(AUDIT / "competition_seasons.csv")
    cs = cs[cs.matches >= 5].copy()
    cs["label"] = cs.competition + " " + cs.season
    cs["kind"] = np.where(cs.matches >= 300, "complete league season", "matches of one club / partial")
    cs = cs.sort_values(["competition", "season"], ascending=[False, False])
    fig, ax = plt.subplots(figsize=(7, 0.22 * len(cs) + 1))
    colors = cs.kind.map({"complete league season": S.SERIES[0], "matches of one club / partial": S.SERIES[1]})
    ax.barh(cs.label, cs.matches, color=colors, height=0.7)
    ax.set_xscale("log")
    ax.set_xlabel("matches available in StatsBomb Open Data (log scale)")
    ax.set_title("Open-data coverage is mostly single-club subsets")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=S.SERIES[0], label="complete league season"),
                       Patch(color=S.SERIES[1], label="matches of one club / partial")], loc="upper right")
    ax.tick_params(axis="y", labelsize=6)
    S.save(fig, FIGURES / "audit" / "01_coverage_by_competition_season.png",
           "Figure A1. Matches per male club competition-season with >= 5 matches. Only 2015/16 leagues are complete.")
    ps_ = pd.read_csv(AUDIT / "player_seasons_lineup.csv")
    ll = ps_[(ps_.competition == "La Liga") & (ps_.minutes >= 450)].copy()
    ll["y"] = ll.season.str[:4].astype(int)
    rows = []
    for y in sorted(ll.y.unique()):
        x, z = ll[ll.y == y], ll[ll.y == y + 1]
        if len(z):
            rows.append((f"{y % 100:02d}/{(y + 1) % 100:02d}->{(y + 1) % 100:02d}/{(y + 2) % 100:02d}", len(set(x.player_id) & set(z.player_id)), y))
    d = pd.DataFrame(rows, columns=["t", "linked", "y"])
    fig, ax = plt.subplots(figsize=(7, 3))
    cols = [S.SERIES[1] if y in (2014, 2015) else S.SERIES[0] for y in d.y]
    ax.bar(d.t, d.linked, color=cols)
    for i, v in enumerate(d.linked):
        ax.text(i, v + 0.3, str(v), ha="center", fontsize=7, color=S.INK2)
    ax.set_ylabel("players >= 450 min in both seasons")
    ax.set_title("La Liga player continuity across consecutive seasons")
    plt.xticks(rotation=60)
    S.save(fig, FIGURES / "audit" / "02_la_liga_player_continuity.png",
           "Figure A2. Linked players per season transition (StatsBomb player_id). Orange = transitions touching the single complete season (2015/16), "
           "where linked players are Barcelona players only.")


def fig_minutes(a, b):
    fig, axs = plt.subplots(1, 2, figsize=(8, 3))
    axs[0].hist(a.minutes, bins=40, color=S.SERIES[0])
    axs[0].axvline(450, color=S.INK2, ls="--", lw=1)
    axs[0].set_title("Design A: Barcelona player-seasons")
    axs[0].set_xlabel("minutes in season")
    axs[1].hist(b.minutes, bins=40, color=S.SERIES[2])
    axs[1].axvline(360, color=S.INK2, ls="--", lw=1)
    axs[1].set_title("Design B: player-blocks (~9-10 matchweeks)")
    axs[1].set_xlabel("minutes in block")
    S.save(fig, FIGURES / "eda" / "03_minutes_distribution.png",
           "Figure E1. Exposure distributions; dashed line = primary eligibility threshold (450 / 360 min).")


def fig_missing(a, b):
    feats = ["age", "pass_completion", "shot_accuracy", "xg_per_shot", "dribble_success", "avg_pass_length"]
    rows = []
    for tag, d, t in (("Design A (>=450)", a, 450), ("Design B (>=360)", b, 360)):
        x = d[(d.minutes >= t) & (d.position_group != "GK")]
        for f in feats:
            rows.append({"design": tag, "feature": f, "missing": 100 * x[f].isna().mean()})
    m = pd.DataFrame(rows)
    m.to_csv(TABLES / "missingness_outfield.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 3))
    w = 0.38
    for i, (tag, g) in enumerate(m.groupby("design")):
        ax.bar(np.arange(len(feats)) + i * w, g.missing, w, label=tag, color=S.SERIES[i])
    ax.set_xticks(np.arange(len(feats)) + w / 2)
    ax.set_xticklabels(feats, rotation=25)
    ax.set_ylabel("% missing (outfield)")
    ax.set_title("Missingness of derived features")
    ax.legend()
    S.save(fig, FIGURES / "eda" / "04_missingness.png",
           "Figure E2. Share of eligible outfield player-seasons/blocks with an undefined value. Age is missing when no tier-A Transfermarkt link exists.")


def fig_features(db):
    x = db[(db.minutes >= 360) & (db.position_group != "GK")]
    metrics = ["npxg_p90", "xa_p90", "passes_progressive_p90", "carries_progressive_p90", "def_actions_p90", "pass_completion"]
    fig, axs = plt.subplots(2, 3, figsize=(9, 5))
    for ax, m in zip(axs.ravel(), metrics):
        data = [x[x.position_group == g][m].dropna() for g in ("DEF", "MID", "FWD")]
        bp = ax.boxplot(data, tick_labels=["DEF", "MID", "FWD"], patch_artist=True, showfliers=False, widths=0.6)
        for patch, g in zip(bp["boxes"], ("DEF", "MID", "FWD")):
            patch.set_facecolor(S.POS_COLOR[g]); patch.set_alpha(0.8)
        for med in bp["medians"]:
            med.set_color(S.INK)
        ax.set_title(m, fontsize=9)
    S.save(fig, FIGURES / "eda" / "05_feature_distributions_by_position.png",
           "Figure E3. Distribution of selected per-90 features by position group (Design B blocks, >= 360 min, 4 leagues).")
    prof = [m for m in P.PROFILE]
    corr = x[prof].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr, cmap=DIV, vmin=-1, vmax=1)
    ax.set_xticks(range(len(prof))); ax.set_xticklabels(prof, rotation=75, ha="right", fontsize=7)
    ax.set_yticks(range(len(prof))); ax.set_yticklabels(prof, fontsize=7)
    ax.grid(False)
    for i in range(len(prof)):
        for j in range(len(prof)):
            ax.text(j, i, f"{corr.iloc[i, j]:.1f}", ha="center", va="center", fontsize=5, color=S.INK)
    fig.colorbar(im, shrink=0.7)
    ax.set_title("Spearman correlation of profile features (outfield)")
    S.save(fig, FIGURES / "eda" / "06_feature_correlations.png",
           "Figure E4. Many volume features co-vary (positively with team involvement); pooled across positions so partly reflects role differences.")


def fig_age(a, b, da, db):
    x = db[(db.minutes >= 360) & db.age.notna() & db.pi.notna()]
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.4))
    axs[0].scatter(x.age, x.pi, s=5, alpha=0.25, color=S.SERIES[0])
    bins = np.arange(16, 40, 2)
    x = x.assign(ab=pd.cut(x.age, bins))
    g = x.groupby("ab", observed=True).pi.agg(["mean", "sem", "size"])
    mid = [i.mid for i in g.index]
    axs[0].errorbar(mid, g["mean"], yerr=1.96 * g["sem"], color=S.INK, lw=1.5, marker="o", ms=3)
    axs[0].set_xlabel("age at season end"); axs[0].set_ylabel("performance index (z)")
    axs[0].set_title("Age vs performance (Design B)")
    # development: delta between blocks vs age (noisy), design B pairs
    y = db[(db.minutes >= 360) & db.age.notna() & db.pi.notna()][["player_id", "block", "pi", "age", "position_group"]]
    nxt = y.assign(block=y.block - 1).rename(columns={"pi": "pi_next"})[["player_id", "block", "pi_next"]]
    j = y.merge(nxt, on=["player_id", "block"])
    j["delta"] = j.pi_next - j.pi
    j["ab"] = pd.cut(j.age, bins)
    g2 = j.groupby("ab", observed=True).delta.agg(["mean", "sem", "size"])
    g2 = g2[g2["size"] >= 15]
    axs[1].errorbar([i.mid for i in g2.index], g2["mean"], yerr=1.96 * g2["sem"], color=S.SERIES[1], marker="o", ms=3, lw=1.5)
    axs[1].axhline(0, color=S.MUTED, lw=0.8)
    axs[1].set_xlabel("age"); axs[1].set_ylabel("mean change in index to next block")
    axs[1].set_title("Development vs age (within season)")
    S.save(fig, FIGURES / "eda" / "07_age_vs_performance_and_development.png",
           "Figure E5. Left: index vs age with binned means (95% CI). Right: mean block-to-block change by age (bins with >= 15 players); "
           "differences are small relative to noise and reflect regression to the mean as much as development. Not causal.")
    a2 = da[(da.minutes >= 450) & da.age.notna() & da.pi.notna() & (da.position_group != "GK")]
    a2.groupby(pd.cut(a2.age, [16, 22, 25, 28, 31, 40]), observed=True).pi.agg(["mean", "std", "size"]).to_csv(TABLES / "age_profile_design_a.csv")


def fig_roles(db):
    x = db[(db.minutes >= 360) & (db.position_group != "GK")]
    zc = [f"z_{m}" for m in P.PROFILE]
    h = x.groupby("primary_role")[zc].mean().loc[["CB", "FB", "DM", "CM", "AM", "WM", "W", "CF"]]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    im = ax.imshow(h.to_numpy(), cmap=DIV, vmin=-1.5, vmax=1.5, aspect="auto")
    ax.set_yticks(range(len(h))); ax.set_yticklabels(h.index)
    ax.set_xticks(range(len(zc))); ax.set_xticklabels([c[2:] for c in zc], rotation=70, ha="right", fontsize=7)
    ax.grid(False)
    for i in range(h.shape[0]):
        for j in range(h.shape[1]):
            ax.text(j, i, f"{h.iat[i, j]:.1f}", ha="center", va="center", fontsize=6)
    fig.colorbar(im, shrink=0.8)
    ax.set_title("Role profiles: mean z-score within position group")
    S.save(fig, FIGURES / "eda" / "08_role_profiles.png",
           "Figure E6. Mean z-score (standardised within DEF/MID/FWD) by finer role. Roles differ systematically, so position-group references leave residual role effects.")


def fig_noise(db):
    x = db[(db.position_group == "FWD") & (db.minutes > 30)]
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.scatter(x.minutes, x.npxg_p90, s=6, alpha=0.4, color=S.SERIES[0])
    ax.set_xlabel("minutes in block"); ax.set_ylabel("non-penalty xG per 90 (forwards)")
    ax.set_title("Per-90 rates are noisy at low exposure")
    S.save(fig, FIGURES / "eda" / "09_exposure_and_rate_noise.png",
           "Figure E7. Funnel shape: dispersion of the per-90 rate shrinks with minutes, motivating exposure thresholds and shrinkage.")


def fig_targets(da):
    x = da[(da.minutes >= 450) & da.pi.notna()]
    nxt = x[["player_id", "season_year", "pi", "minutes", "position_group"]].copy()
    nxt["season_year"] -= 1
    j = x.merge(nxt.rename(columns={"pi": "pi_next", "minutes": "minutes_next", "position_group": "pos_next"})[["player_id", "season_year", "pi_next", "minutes_next"]], on=["player_id", "season_year"])
    j["delta"] = j.pi_next - j.pi
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.4))
    for g, c in S.POS_COLOR.items():
        s = j[j.position_group == g]
        if len(s):
            axs[0].scatter(s.pi, s.pi_next, s=14, color=c, alpha=0.8, label=g)
    lim = [j.pi.min() - 0.1, j.pi.max() + 0.1]
    axs[0].plot(lim, lim, color=S.MUTED, lw=1, ls="--")
    axs[0].set_xlabel("index at t"); axs[0].set_ylabel("index at t+1"); axs[0].legend(); axs[0].set_title("Persistence (Design A)")
    axs[1].hist(j.delta, bins=25, color=S.SERIES[1])
    q = j.delta.quantile(0.8)
    axs[1].axvline(q, color=S.INK, ls="--", lw=1)
    axs[1].set_xlabel("change in index (t+1 minus t)"); axs[1].set_title("Change score (dashed: top 20%)", fontsize=10)
    counts = j.assign(b=(j.delta >= j.groupby("position_group").delta.transform(lambda s: s.quantile(0.8))).astype(int)).groupby("position_group").b.agg(["sum", "size"])
    axs[2].bar(counts.index, counts["size"] - counts["sum"], color=S.MUTED, label="not breakout")
    axs[2].bar(counts.index, counts["sum"], bottom=counts["size"] - counts["sum"], color=S.SERIES[1], label="breakout (top 20%)")
    axs[2].legend(); axs[2].set_title("Breakout class balance", fontsize=10)
    S.save(fig, FIGURES / "eda" / "10_target_distributions.png",
           "Figure E8. Next-season index vs current (r~0.83), distribution of the change score, and class balance of the exploratory breakout label (pooled quantile for display).")
    j.to_csv(PROCESSED / "design_a_pairs_descriptive.csv", index=False)
    return j


def split_half(a, ref):
    pm, tm = ps.load_player_match([(PN.LALIGA, s) for s in PN.LALIGA_SEASONS])
    b = set(tm[tm.team_id == PN.BARCA_TEAM_ID].match_id)
    pm = pm[pm.match_id.isin(b) & (pm.team_id == PN.BARCA_TEAM_ID)]
    tm = tm[tm.match_id.isin(b)]
    pm["half"] = np.where(pm.match_week % 2 == 0, "E", "O")
    d = ps.add_rates(ps.aggregate(pm, tm, "half")).merge(PN.season_meta(), on=["competition_id", "season_id"])
    d = P.performance_index(d, ref)
    w = d[(d.minutes >= 225) & d.pi.notna()].pivot_table(index=["player_id", "season_year"], columns="half", values="pi").dropna()
    r = np.corrcoef(w.E, w.O)[0, 1]
    rel = 2 * r / (1 + r)
    sd = a.pipe(lambda x: P.performance_index(x, ref)).query("minutes>=450").pi.std()
    sem = sd * np.sqrt(1 - rel)
    return w, {"n": len(w), "split_half_r": r, "spearman_brown_reliability": rel, "sd_pi": sd, "sem": sem,
               "noise_sd_of_change": sem * np.sqrt(2)}


def fig_reliability(a, da, j):
    ref = P.fit_reference(a, 450)
    w, st = split_half(a, ref)
    obs = j.delta.std()
    st.update({"observed_sd_of_change": obs, "signal_share_of_change_variance": max(0.0, 1 - st["noise_sd_of_change"] ** 2 / obs ** 2)})
    pd.DataFrame([st]).to_csv(TABLES / "reliability_design_a.csv", index=False)
    fig, ax = plt.subplots(figsize=(4.4, 4))
    ax.scatter(w.O, w.E, s=10, alpha=0.6, color=S.SERIES[0])
    lim = [w.values.min() - 0.1, w.values.max() + 0.1]
    ax.plot(lim, lim, ls="--", color=S.MUTED, lw=1)
    ax.set_xlabel("index, odd matchweeks"); ax.set_ylabel("index, even matchweeks")
    ax.set_title(f"Split-half r = {st['split_half_r']:.2f}")
    S.save(fig, FIGURES / "eda" / "11_split_half_reliability.png",
           f"Figure E9. Within-season odd/even split (n={st['n']}). Spearman-Brown reliability {st['spearman_brown_reliability']:.2f}; "
           f"implied noise SD of a season-to-season change {st['noise_sd_of_change']:.2f} vs observed SD {obs:.2f}.")
    return st


def fig_attrition(da):
    x = da[(da.minutes >= 450) & da.pi.notna() & (da.season_year < da.season_year.max())].copy()
    k = da[da.minutes >= 450][["player_id", "season_year"]].assign(has_next=1)
    k["season_year"] -= 1
    x = x.merge(k, on=["player_id", "season_year"], how="left")
    x["retained"] = x.has_next.fillna(0).astype(int)
    fig, ax = plt.subplots(figsize=(5.5, 3.3))
    ax.hist([x[x.retained == 1].pi, x[x.retained == 0].pi], bins=18, stacked=False, color=[S.SERIES[0], S.SERIES[1]],
            label=[f"observed next season (n={int(x.retained.sum())})", f"no next-season observation (n={int((1 - x.retained).sum())})"])
    ax.set_xlabel("index at t"); ax.legend(); ax.set_title("Who is missing next season?")
    S.save(fig, FIGURES / "eda" / "12_attrition_by_current_index.png",
           "Figure E10. Outcome observed only for players who stay at Barcelona with >= 450 min: missing-not-at-random selection.")
    stt = stats.mannwhitneyu(x[x.retained == 1].pi, x[x.retained == 0].pi)
    pd.DataFrame([{"n_retained": int(x.retained.sum()), "n_missing": int((1 - x.retained).sum()),
                   "mean_pi_retained": x[x.retained == 1].pi.mean(), "mean_pi_missing": x[x.retained == 0].pi.mean(),
                   "mannwhitney_p": stt.pvalue}]).to_csv(TABLES / "attrition_design_a.csv", index=False)


def fig_confounding(da):
    x = da[(da.minutes >= 450) & da.pi.notna()]
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.2))
    g = x.groupby("season_year").team_poss.first()
    axs[0].plot(g.index, g.values, marker="o", color=S.SERIES[0]); axs[0].set_title("Barcelona pass share"); axs[0].set_xlabel("season start")
    axs[1].scatter(x.minutes, x.pi, s=10, alpha=0.6, color=S.SERIES[0])
    rho = stats.spearmanr(x.minutes, x.pi)[0]
    axs[1].set_title(f"Index vs minutes (rho={rho:.2f})"); axs[1].set_xlabel("minutes")
    axs[2].boxplot([x[x.position_group == g].pi for g in ("DEF", "MID", "FWD")], tick_labels=["DEF", "MID", "FWD"])
    axs[2].set_title("Index by position (z within group)")
    S.save(fig, FIGURES / "eda" / "13_confounders.png",
           "Figure E11. Potential confounders: team style drifts over time; minutes correlate with the index (starters are better and more exposed); position groups are standardised separately.")


def fig_value(a):
    v = PN.load_valuations()
    x = a[(a.minutes >= 450) & (a.position_group != "GK") & a.tm_id.notna()]
    x = PN.attach_valuation(x, v).dropna(subset=["value_t", "value_future"])
    x["y"] = np.log(x.value_future / x.value_t)
    fig, axs = plt.subplots(1, 2, figsize=(8.5, 3.2))
    axs[0].hist(np.log10(x.value_t), bins=20, color=S.SERIES[2]); axs[0].set_title("log10 market value at scouting date (EUR)")
    axs[1].hist(x.y, bins=20, color=S.SERIES[1]); axs[1].set_title("12-month change in log value")
    S.save(fig, FIGURES / "eda" / "14_market_value_distributions.png",
           f"Figure E12. Transfermarkt valuation labels linked to Design A (n={len(x)}). Crowd valuations are a proxy, not ability or transfer profit.")


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    a, b, da, db = load()
    fig_audit(); fig_minutes(a, b); fig_missing(a, b); fig_features(db); fig_age(a, b, da, db); fig_roles(db); fig_noise(db)
    j = fig_targets(da)
    st = fig_reliability(a, da, j)
    fig_attrition(da); fig_confounding(da); fig_value(a)
    print(st)


if __name__ == "__main__":
    main()
