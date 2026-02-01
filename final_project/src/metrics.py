"""
Performance metrics required by the project:
- PnL and turnover
- annualized return / vol
- Sharpe ratio
- max drawdown
- benchmark vs Nifty buy-and-hold

We also include:
- Calmar ratio
- downside deviation + Sortino
- turnover statistics (avg, p95)

All computed from daily returns (post costs).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


def annualized_return(r: pd.Series, trading_days: int = 252) -> float:
    r = r.dropna()
    if len(r) == 0:
        return float("nan")
    return float((1.0 + r).prod() ** (trading_days / len(r)) - 1.0)


def annualized_vol(r: pd.Series, trading_days: int = 252) -> float:
    r = r.dropna()
    if len(r) == 0:
        return float("nan")
    return float(r.std(ddof=1) * np.sqrt(trading_days))


def sharpe(r: pd.Series, trading_days: int = 252, rf_annual: float = 0.0) -> float:
    r = r.dropna()
    if len(r) == 0:
        return float("nan")
    rf_daily = rf_annual / trading_days
    excess = r - rf_daily
    sd = excess.std(ddof=1)
    if sd == 0:
        return float("nan")
    return float(excess.mean() / sd * np.sqrt(trading_days))


def downside_dev(r: pd.Series, trading_days: int = 252) -> float:
    r = r.dropna()
    neg = r[r < 0]
    if len(neg) == 0:
        return 0.0
    return float(neg.std(ddof=1) * np.sqrt(trading_days))


def sortino(r: pd.Series, trading_days: int = 252, rf_annual: float = 0.0) -> float:
    r = r.dropna()
    if len(r) == 0:
        return float("nan")
    rf_daily = rf_annual / trading_days
    excess = r - rf_daily
    dd = downside_dev(excess, trading_days=trading_days)
    if dd == 0:
        return float("nan")
    return float(excess.mean() / (dd / np.sqrt(trading_days)) * np.sqrt(trading_days))


def max_drawdown(equity: pd.Series) -> float:
    equity = equity.dropna()
    if len(equity) == 0:
        return float("nan")
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def calmar(r: pd.Series, equity: pd.Series, trading_days: int = 252) -> float:
    ar = annualized_return(r, trading_days=trading_days)
    mdd = abs(max_drawdown(equity))
    if mdd == 0:
        return float("nan")
    return float(ar / mdd)


def summarize(
    daily: pd.DataFrame,
    trading_days: int = 252,
) -> Dict[str, float]:
    r = daily["port_ret"]
    eq = daily["equity"]

    out = {
        "AnnReturn": annualized_return(r, trading_days=trading_days),
        "AnnVol": annualized_vol(r, trading_days=trading_days),
        "Sharpe": sharpe(r, trading_days=trading_days),
        "Sortino": sortino(r, trading_days=trading_days),
        "MaxDrawdown": max_drawdown(eq),
        "Calmar": calmar(r, eq, trading_days=trading_days),
        "AvgDailyTurnover": float(daily["turnover"].mean()),
        "P95Turnover": float(daily["turnover"].quantile(0.95)),
        "TotalCost": float(daily["cost"].sum()),
        "FinalEquity": float(eq.iloc[-1]) if len(eq) else float("nan"),
    }
    return out
