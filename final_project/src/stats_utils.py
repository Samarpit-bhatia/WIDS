"""
Statistical utilities inspired by Week-2 content:
- Augmented Dickey-Fuller (ADF) stationarity test (diagnostics)
- Hurst exponent (trend vs mean reversion)
- Half-life of mean reversion (holding period intuition)

These are used as:
- optional diagnostics in artifacts/diagnostics
- regime features in the allocator (e.g., become more defensive when H < 0.5)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from statsmodels.tsa.stattools import adfuller
from scipy import stats


@dataclass
class AdfResult:
    test_stat: float
    p_value: float
    n_lags: int
    n_obs: int


def adf_test(series: pd.Series, maxlag: Optional[int] = None) -> AdfResult:
    s = series.dropna().astype(float)
    res = adfuller(s.values, maxlag=maxlag, autolag="AIC")
    return AdfResult(test_stat=float(res[0]), p_value=float(res[1]), n_lags=int(res[2]), n_obs=int(res[3]))


def hurst_exponent(series: pd.Series) -> float:
    """
    Simple Hurst estimate via variance of lagged differences:
        tau(l) = sqrt(var(x_{t+l} - x_t))
        H = slope of log(tau) vs log(l)
    """
    x = series.dropna().astype(float).values
    if len(x) < 200:
        return float("nan")

    lags = np.array([2, 5, 10, 20, 50, 100], dtype=int)
    taus = []
    for lag in lags:
        if lag >= len(x):
            continue
        diff = x[lag:] - x[:-lag]
        taus.append(np.sqrt(np.var(diff)))
    if len(taus) < 3:
        return float("nan")

    taus = np.array(taus)
    lags = lags[: len(taus)]
    slope, _, _, _, _ = stats.linregress(np.log(lags), np.log(taus))
    return float(slope)


def mean_reversion_halflife(series: pd.Series) -> float:
    """
    Half-life estimate using regression:
        Δx_t = α + φ x_{t-1} + ε_t
    Half-life = -ln(2) / φ
    """
    x = series.dropna().astype(float)
    if len(x) < 200:
        return float("nan")

    x_lag = x.shift(1).dropna()
    dx = (x - x.shift(1)).dropna()

    # align
    idx = x_lag.index.intersection(dx.index)
    x_lag = x_lag.loc[idx]
    dx = dx.loc[idx]

    if len(x_lag) < 50:
        return float("nan")

    X = np.vstack([np.ones(len(x_lag)), x_lag.values]).T
    y = dx.values
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    phi = beta[1]
    if phi >= 0:
        return float("inf")
    return float(-np.log(2.0) / phi)
