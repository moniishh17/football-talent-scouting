"""Empirical-Bayes shrinkage of noisy per-90 rates and proportions toward position-group baselines.

Poisson rates: counts c over exposure e (units of 90 min) with gamma prior Gamma(alpha, beta):
    mu = sum(c)/sum(e)              (pooled rate in the fitting sample)
    tau2 = max(Var(c/e) - mean(mu/e), floor)   (between-player variance after removing Poisson sampling noise)
    beta = mu / tau2 (prior worth of beta 90-minute periods), alpha = mu * beta
    posterior mean = (c + alpha) / (e + beta)
Proportions (successes s of n): beta-binomial analogue with prior mean p0 and prior size kappa.

`fit` uses only the rows it is given (training data); `apply` is a pure function of those parameters.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BETA_BOUNDS = (0.25, 60.0)       # prior worth between 22 and 5400 minutes
KAPPA_BOUNDS = (5.0, 2000.0)


def fit(train: pd.DataFrame, group_col: str, poisson: dict[str, str], binom: dict[str, tuple[str, str]],
        min_minutes: float = 90.0) -> dict:
    """poisson: {rate_name: count_col}; binom: {name: (success_col, trials_col)}."""
    params: dict = {"poisson": {}, "binom": {}}
    for grp, g in train[train["minutes"] >= min_minutes].groupby(group_col):
        e = g["minutes"].to_numpy() / 90.0
        for name, col in poisson.items():
            c = g[col].to_numpy()
            mu = c.sum() / e.sum()
            if mu <= 0:
                params["poisson"][(grp, name)] = (0.0, BETA_BOUNDS[1])
                continue
            rates = c / e
            tau2 = rates.var(ddof=1) - np.mean(mu / e) if len(g) > 2 else 0.0
            tau2 = max(tau2, 1e-6)
            beta = float(np.clip(mu / tau2, *BETA_BOUNDS))
            params["poisson"][(grp, name)] = (mu * beta, beta)
        for name, (scol, ncol) in binom.items():
            s, n = g[scol].to_numpy(), g[ncol].to_numpy()
            ok = n > 0
            if ok.sum() < 3 or n[ok].sum() == 0:
                params["binom"][(grp, name)] = (0.5, KAPPA_BOUNDS[1])
                continue
            p0 = s[ok].sum() / n[ok].sum()
            p = s[ok] / n[ok]
            var_b = p.var(ddof=1) - np.mean(p0 * (1 - p0) / n[ok])
            kappa = (p0 * (1 - p0) / max(var_b, 1e-9)) - 1.0 if p0 not in (0.0, 1.0) else KAPPA_BOUNDS[1]
            params["binom"][(grp, name)] = (p0, float(np.clip(kappa, *KAPPA_BOUNDS)))
    return params


def apply(df: pd.DataFrame, params: dict, group_col: str, poisson: dict[str, str],
          binom: dict[str, tuple[str, str]], suffix: str = "_shr") -> pd.DataFrame:
    df = df.copy()
    e = df["minutes"].to_numpy() / 90.0
    for name, col in poisson.items():
        out = np.full(len(df), np.nan)
        for grp in df[group_col].unique():
            a_b = params["poisson"].get((grp, name))
            if a_b is None:
                continue
            m = (df[group_col] == grp).to_numpy()
            out[m] = (df.loc[m, col].to_numpy() + a_b[0]) / (e[m] + a_b[1])
        df[f"{name}{suffix}"] = out
    for name, (scol, ncol) in binom.items():
        out = np.full(len(df), np.nan)
        for grp in df[group_col].unique():
            pk = params["binom"].get((grp, name))
            if pk is None:
                continue
            m = (df[group_col] == grp).to_numpy()
            s, n = df.loc[m, scol].to_numpy(), df.loc[m, ncol].to_numpy()
            out[m] = (s + pk[1] * pk[0]) / (n + pk[1])
        df[f"{name}{suffix}"] = out
    return df
