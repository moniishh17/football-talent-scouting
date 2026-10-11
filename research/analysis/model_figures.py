"""Model/evaluation figures (figures/models). Run after results.py, uncertainty_errors.py, sensitivity_table.py.
Run: python -m research.analysis.model_figures
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from research.analysis import results as RS
from research.analysis import style as S
from research.analysis import uncertainty_errors as UE
from research.config import FIGURES, PROCESSED, REPORTS
from research.evaluation import metrics as M

S.apply()
T = REPORTS / "tables"
OUT = FIGURES / "models"


def short(n: str) -> str:
    n = n.replace("pi+profile+agepos+exposure+context+league+history", "full+history").replace("pi+profile+agepos+exposure+context+history", "full+history")
    n = n.replace("pi+profile+agepos+exposure+context", "+context").replace("|level", "").replace("|delta", " (delta)")
    return n.replace("mcda_weighted_sum", "MCDA WSM").replace("mcda_topsis", "MCDA TOPSIS").replace("mcda_vikor", "MCDA VIKOR").replace("_", " ")


def kind_color(n: str):
    if "|" in n:
        return S.SERIES[0]
    if n.startswith("mcda") or n == "equal_weight":
        return S.SERIES[1]
    return S.MUTED


def fig_forest():
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.4), gridspec_kw={"wspace": 1.1})
    for ax, d, title in zip(axs, "ABV", ("Design A: next season (Barcelona panel)", "Design B: next block, 4 leagues", "Valuation: 12-month log value change")):
        m = pd.read_csv(T / f"{d}_main.csv")
        m = m.iloc[::-1].reset_index(drop=True)
        y = np.arange(len(m))
        ax.hlines(y, m.confirm_rmse_lo, m.confirm_rmse_hi, color=[kind_color(n) for n in m.name], lw=2)
        ax.scatter(m.confirm_rmse, y, color=[kind_color(n) for n in m.name], zorder=3, s=22)
        ax.set_yticks(y); ax.set_yticklabels([short(n) for n in m.name], fontsize=6.5)
        ax.set_title(title, fontsize=9.5); ax.set_xlabel("confirmation-period RMSE (95% CI)")
    S.save(fig, OUT / "m1_confirmation_rmse_forest.png",
           "Figure M1. Held-out confirmation-period RMSE. Grey = simple baselines, orange = MCDA rankers (legacy), blue = supervised models selected on the development period. "
           "Overlapping intervals are not evidence of a difference.")


def fig_ablation():
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, d, title in zip(axs, "ABV", ("Design A", "Design B", "Valuation target")):
        a = pd.read_csv(T / f"{d}_ablation.csv")
        steps = list(dict.fromkeys(a.step))
        for i, (fam, g) in enumerate(a.groupby("family")):
            x = np.arange(len(steps)) + (i - 1) * 0.12
            g = g.set_index("step").reindex(steps)
            ax.errorbar(x, g.all_rmse, yerr=[g.all_rmse - g.all_lo, g.all_hi - g.all_rmse], marker="o", ms=3, lw=1.2, color=S.SERIES[i], label=fam, capsize=0)
        ax.set_xticks(range(len(steps))); ax.set_xticklabels(steps, rotation=40, ha="right", fontsize=7)
        ax.set_title(title); ax.set_ylabel("pooled RMSE")
        ax.legend()
    S.save(fig, OUT / "m2_ablation_feature_sets.png",
           "Figure M2. Cumulative feature-set ablation (pooled held-out RMSE, 95% CI). Lower is better. Includes development and confirmation periods; "
           "see tables for paired differences and confirmation-only results.")


def fig_folds():
    f = pd.read_csv(T / "A_folds.csv")
    sel = json.loads((T / "selected_configs.json").read_text())["A"]["overall"]
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    for n, c, lab in (("mean", S.MUTED, "train mean"), ("mcda_weighted_sum", S.SERIES[1], "MCDA WSM"), ("persistence_cal", S.SERIES[0], "calibrated persistence"), (sel, S.SERIES[2], "selected: " + short(sel))):
        g = f[f.name == n].sort_values("test_year")
        ax.plot(g.test_year + 1, g.rmse, marker="o", ms=3.5, color=c, label=lab, lw=1.5, ls="--" if n == sel else "-")
    ax.axvline(2015.5, color=S.INK2, lw=0.8, ls=":"); ax.text(2015.6, ax.get_ylim()[1] * 0.97, "confirmation →", fontsize=7, color=S.INK2, va="top")
    ax.set_xlabel("target season start year"); ax.set_ylabel("RMSE (fold)")
    ax.legend(fontsize=7); ax.set_title("Design A: per-fold error (folds have 10-17 players)")
    S.save(fig, OUT / "m3_per_fold_design_a.png", "Figure M3. Fold-level RMSE; folds are small (see tables) and noisy.")


def fig_pred_vs_actual():
    sel = json.loads((T / "selected_configs.json").read_text())
    fig, axs = plt.subplots(1, 2, figsize=(9, 4))
    for ax, d in zip(axs, "AB"):
        p = UE.attach_panel_cols(RS.load(d), d)
        g = p[p.name == sel[d]["overall"]]
        for pos, c in S.POS_COLOR.items():
            s = g[g.position_group == pos]
            if len(s):
                ax.scatter(s.yhat, s.pi_next, s=8 if d == "B" else 16, alpha=0.5 if d == "B" else 0.8, color=c, label=pos)
        lim = [min(g.yhat.min(), g.pi_next.min()) - 0.1, max(g.yhat.max(), g.pi_next.max()) + 0.1]
        ax.plot(lim, lim, color=S.INK2, lw=1, ls="--")
        ax.set_xlabel("predicted index"); ax.set_ylabel("observed index")
        ax.set_title(f"Design {d}: {short(sel[d]['overall'])}", fontsize=9)
        ax.legend()
    S.save(fig, OUT / "m4_predicted_vs_observed.png", "Figure M4. Out-of-sample predictions of the development-selected model (pooled test folds).")


def fig_subgroups():
    e = pd.read_csv(T / "error_by_subgroup.csv")
    sel = json.loads((T / "selected_configs.json").read_text())
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={"wspace": 0.9})
    for ax, d in zip(axs, "AB"):
        g = e[(e.design == d) & (e.name == sel[d]["overall"]) & (e.dimension.isin(["position_group", "age_band", "minutes_band", "coverage_level"]))]
        g = g.iloc[::-1].reset_index(drop=True)
        lab = [f"{r.dimension.split('_')[0]}: {r.group} (n={r.n})" for r in g.itertuples()]
        cols = [S.MUTED if r.small_sample else S.SERIES[0] for r in g.itertuples()]
        ax.barh(lab, g.rmse, color=cols, height=0.65)
        ax.set_xlabel("RMSE"); ax.set_title(f"Design {d}: error by subgroup", fontsize=10)
        ax.tick_params(axis="y", labelsize=7)
    S.save(fig, OUT / "m5_error_by_subgroup.png", "Figure M5. RMSE by subgroup for the selected model; grey bars have n < 20 and should not be interpreted.")


def fig_ranking():
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, d, ks in zip(axs, "AB", ((3, 5), (10, 25))):
        r = pd.read_csv(T / f"{d}_ranking.csv")
        sel = json.loads((T / "selected_configs.json").read_text())[d]["overall"]
        names = ["mean", "equal_weight", "mcda_weighted_sum", "mcda_topsis", "mcda_vikor", "persistence_cal", sel]
        w = 0.38
        for i, k in enumerate(ks):
            g = r[r.k == k].set_index("name").reindex(names)
            ax.bar(np.arange(len(names)) + i * w, g.shortlist_lift, w, label=f"top-{k}", color=S.SERIES[i])
        ax.set_xticks(np.arange(len(names)) + w / 2); ax.set_xticklabels([short(n) for n in names], rotation=35, ha="right", fontsize=7)
        ax.set_ylabel("mean realised index of shortlist minus cohort mean")
        ax.set_title(f"Design {d}: shortlist outcome lift"); ax.legend()
    S.save(fig, OUT / "m6_shortlist_lift.png", "Figure M6. Outcome quality of top-k shortlists: realised next-period index of the k highest-scored candidates relative to the cohort mean, averaged over folds.")


def fig_breakout():
    from research.run_breakout import summarize
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, d in zip(axs, "ab"):
        pr = pd.read_parquet(PROCESSED / f"predictions_breakout_{d}.parquet")
        s = summarize(pr).iloc[::-1].reset_index(drop=True)
        y = np.arange(len(s))
        ax.hlines(y, s.pr_auc_lo, s.pr_auc_hi, color=S.SERIES[0], lw=2)
        ax.scatter(s.pr_auc, y, color=S.SERIES[0], s=18, zorder=3)
        ax.axvline(s.prevalence.iloc[0], color=S.SERIES[1], ls="--", lw=1.2)
        ax.text(s.prevalence.iloc[0], len(s) - 0.2, " base rate", color=S.SERIES[1], fontsize=7)
        ax.set_yticks(y); ax.set_yticklabels([short(n).replace("pi+profile+agepos", "feat") for n in s.name], fontsize=6)
        ax.set_xlabel("PR-AUC (95% CI)"); ax.set_title(f"Breakout (top-20% change), Design {d.upper()}", fontsize=9.5)
    S.save(fig, OUT / "m7_breakout_pr_auc.png",
           "Figure M7. Exploratory breakout classification. The no-model heuristic 'low current index' (regression to the mean) is as good as the learned classifiers.")
    pr = pd.read_parquet(PROCESSED / "predictions_breakout_b.parquet")
    s = pr[pr.is_prob & (pr.name != "prevalence")]
    best = s.groupby("name").apply(lambda g: M.pr_auc(g.y, g.p), include_groups=False).idxmax()
    g = pr[pr.name == best]
    ct = pd.DataFrame(M.calibration_table(g.y, g.p, 6))
    fig, ax = plt.subplots(figsize=(4.4, 4))
    ax.plot([0, 0.6], [0, 0.6], color=S.MUTED, ls="--", lw=1)
    ax.plot(ct.mean_pred, ct.obs_rate, marker="o", color=S.SERIES[0])
    ax.set_xlabel("mean predicted probability"); ax.set_ylabel("observed breakout rate"); ax.set_title("Calibration (Design B, best PR-AUC)", fontsize=9.5)
    S.save(fig, OUT / "m8_breakout_calibration.png", f"Figure M8. Reliability diagram (quantile bins) for {short(best)}; Brier skill vs base rate is small (see tables).")


def fig_coverage():
    c = pd.read_csv(T / "uncertainty_coverage.csv")
    fig, ax = plt.subplots(figsize=(6, 3.6))
    g = c[c.grouping == "global"]
    labels = [f"{r.design}: {short(r.name)[:22]}" for r in g.itertuples()]
    x = np.arange(len(g))
    ax.scatter(x, g.empirical_coverage, color=[S.SERIES[0] if n == 0.8 else S.SERIES[1] for n in g.nominal], zorder=3)
    for xi, (nom, emp) in enumerate(zip(g.nominal, g.empirical_coverage)):
        ax.hlines(nom, xi - 0.3, xi + 0.3, color=S.INK2, lw=1)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=6.5)
    ax.set_ylabel("empirical coverage (dash = nominal)"); ax.set_title("Split-conformal interval coverage (time-respecting)", fontsize=9.5)
    S.save(fig, OUT / "m9_interval_coverage.png", "Figure M9. Blue = nominal 80%, orange = nominal 90%. Design A is close to nominal; Design B under-covers (calibrated on one earlier block).")


def fig_sensitivity():
    s = pd.read_csv(T / "sensitivity.csv")
    s = s[~s.model.isin(["persistence_cal", "mean", "mcda_weighted_sum"])]
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for ax, d in zip(axs, "AB"):
        g = s[s.design == d]
        models = list(dict.fromkeys(g.model))
        variants = list(dict.fromkeys(g.variant))
        for i, m in enumerate(models):
            h = g[g.model == m].set_index("variant").reindex(variants)
            x = np.arange(len(variants)) + (i - (len(models) - 1) / 2) * 0.16
            ax.errorbar(x, h.ratio_to_persistence_cal, yerr=[h.ratio_to_persistence_cal - h.ratio_lo, h.ratio_hi - h.ratio_to_persistence_cal], fmt="o", ms=3, color=S.SERIES[i % 4], label=short(m)[:38], lw=1)
        ax.axhline(1, color=S.INK2, lw=0.8)
        ax.set_xticks(range(len(variants))); ax.set_xticklabels(variants, rotation=35, ha="right", fontsize=7)
        ax.set_ylabel("RMSE / RMSE of calibrated persistence"); ax.set_title(f"Design {d}: sensitivity (<1 beats persistence)", fontsize=9.5)
        ax.legend(fontsize=6)
    S.save(fig, OUT / "m10_sensitivity.png", "Figure M10. Ratio of model RMSE to the variant's own persistence baseline under changed exposure thresholds, shrinkage and position definitions.")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for f in (fig_forest, fig_ablation, fig_folds, fig_pred_vs_actual, fig_subgroups, fig_ranking, fig_breakout, fig_coverage, fig_sensitivity):
        try:
            f()
            print("ok", f.__name__)
        except Exception as exc:  # report but continue
            print("FAILED", f.__name__, repr(exc))


if __name__ == "__main__":
    main()
