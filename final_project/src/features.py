"""
Feature engineering (causal) for multi-asset allocation.

We compute:
- returns (log)
- rolling vol
- momentum (fast/slow)
- z-scores
- kalman trend quality:
    tq_t = slope_t / sqrt(var_slope_t + eps)

Additionally, we compute:
- per-asset Hurst and half-life on rolling windows (optional)
  used for diagnostics + regime gating.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .kalman import fit_local_trend
from .stats_utils import hurst_exponent, mean_reversion_halflife


@dataclass
class KalmanPanel:
    level: pd.DataFrame
    slope: pd.DataFrame
    var_slope: pd.DataFrame
    trend_quality: pd.DataFrame


def rolling_zscore(x: pd.Series, window: int) -> pd.Series:
    mu = x.rolling(window).mean()
    sd = x.rolling(window).std()
    return (x - mu) / sd


def compute_kalman_panel(
    prices: pd.DataFrame,
    q_level: float,
    q_slope: float,
    r_obs: float,
    use_pykalman: bool,
    eps: float = 1e-12,
) -> KalmanPanel:
    levels = {}
    slopes = {}
    var_slopes = {}
    tqs = {}

    for col in prices.columns:
        y = prices[col].values
        out = fit_local_trend(y, q_level=q_level, q_slope=q_slope, r_obs=r_obs, use_pykalman=use_pykalman)
        levels[col] = pd.Series(out.level, index=prices.index)
        slopes[col] = pd.Series(out.slope, index=prices.index)
        var_slope = out.cov[:, 1, 1]
        var_slopes[col] = pd.Series(var_slope, index=prices.index)
        tqs[col] = (slopes[col] / prices[col]) / np.sqrt(var_slopes[col] + eps)


    level_df = pd.DataFrame(levels)
    slope_df = pd.DataFrame(slopes)
    var_slope_df = pd.DataFrame(var_slopes)
    tq_df = pd.DataFrame(tqs)

    return KalmanPanel(level=level_df, slope=slope_df, var_slope=var_slope_df, trend_quality=tq_df)


def compute_features(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    kalman: KalmanPanel,
    vol_window: int,
    mom_fast: int,
    mom_slow: int,
    z_window: int,
) -> Dict[str, pd.DataFrame]:
    feats: Dict[str, pd.DataFrame] = {}

    # Rolling vol on returns
    feats["vol"] = returns.rolling(vol_window).std()

    # Momentum: fast/slow on prices (log)
    logp = np.log(prices)
    feats["mom_fast"] = logp.diff(mom_fast)
    feats["mom_slow"] = logp.diff(mom_slow)

    # Z-score on returns (mean-reversion indicator)
    z = {}
    for c in returns.columns:
        z[c] = rolling_zscore(returns[c], z_window)
    feats["z_ret"] = pd.DataFrame(z)

    # Kalman trend quality
    feats["tq"] = kalman.trend_quality

    return feats


def rolling_regime_diagnostics(
    series: pd.Series,
    window: int,
) -> pd.DataFrame:
    """
    Rolling Hurst + half-life diagnostics for one series.
    Used for interpretability; not required but adds rigor.
    """
    hs = []
    hl = []
    idx = []

    for i in range(window, len(series)):
        seg = series.iloc[i - window : i]
        hs.append(hurst_exponent(seg))
        hl.append(mean_reversion_halflife(seg))
        idx.append(series.index[i])

    return pd.DataFrame({"hurst": hs, "halflife": hl}, index=pd.to_datetime(idx))
