"""
Portfolio allocator.

Core design:
- Inputs at time t:
    - per-asset trend_quality_t (Kalman slope / uncertainty)
    - realized vol_t
    - market regime proxy (e.g., Nifty TQ + vol)
- Output:
    - weights_t over assets + cash, summing to 1

Key ideas:
1) Trend Quality: prefer assets with strong positive trend quality.
2) Risk Budgeting: allocate inversely to vol (risk parity tilt).
3) Regime Gate: when risk-off, increase cash + defensive (gold).
4) Turnover Control: partial adjustment + rebalance band to reduce costs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class AllocatorState:
    prev_weights: pd.Series


def _clip_weights(w: pd.Series, max_weight: float, min_cash: float, cash_symbol: str) -> pd.Series:
    w2 = w.copy()
    # cap risky weights
    risky = [c for c in w2.index if c != cash_symbol]
    w2[risky] = w2[risky].clip(lower=0.0, upper=max_weight)

    # enforce min cash
    if cash_symbol in w2.index:
        w2[cash_symbol] = max(w2[cash_symbol], min_cash)

    # renormalize
    s = float(w2.sum())
    if s <= 0:
        w2[cash_symbol] = 1.0
        for c in risky:
            w2[c] = 0.0
        return w2
    return w2 / s


def compute_target_weights(
    date: pd.Timestamp,
    tq_row: pd.Series,
    vol_row: pd.Series,
    cash_symbol: str,
    trend_clip: float,
    min_signal_strength: float,
    cash_boost_when_riskoff: float,
    max_weight: float,
    min_cash: float,
) -> pd.Series:
    # Clean inputs
    tq = tq_row.clip(lower=-trend_clip, upper=trend_clip).fillna(0.0)
    vol = vol_row.replace(0.0, np.nan).fillna(np.nan)

    # Risk-on assets: positive trend-quality
    pos = tq.clip(lower=0.0)
    strength = float(pos.sum())

    # Base signal -> risk adjusted by inverse vol
    inv_vol = (1.0 / vol).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    raw = pos * inv_vol

    if raw.sum() <= 0:
        w = pd.Series(0.0, index=tq.index)
        w[cash_symbol] = 1.0
        return _clip_weights(w, max_weight=max_weight, min_cash=min_cash, cash_symbol=cash_symbol)

    risky_w = raw / raw.sum()

    # Regime proxy: if signal weak, shift to cash
    riskoff = strength < min_signal_strength

    cash_w = 1.0 - float(risky_w.sum())
    if riskoff:
        cash_w = min(1.0, cash_w + cash_boost_when_riskoff)

    # scale risky down to make room for cash
    scale = max(0.0, 1.0 - cash_w)
    risky_w = risky_w * scale

    w = risky_w.copy()
    w[cash_symbol] = cash_w
    return _clip_weights(w, max_weight=max_weight, min_cash=min_cash, cash_symbol=cash_symbol)


def apply_turnover_control(
    target_w: pd.Series,
    prev_w: pd.Series,
    rebalance_band: float,
    partial_adjust_alpha: float,
) -> Tuple[pd.Series, float]:
    """
    - If turnover below band: keep prev weights (no trade)
    - Else: partially move towards target:
        w = (1-a)*prev + a*target
    Returns: new_weights, turnover
    """
    prev = prev_w.fillna(0.0).reindex(target_w.index).fillna(0.0)
    target = target_w.fillna(0.0)

    turnover = float((target - prev).abs().sum())
    if turnover < rebalance_band:
        return prev, 0.0

    a = float(np.clip(partial_adjust_alpha, 0.0, 1.0))
    w = (1.0 - a) * prev + a * target
    # renormalize defensively
    s = float(w.sum())
    if s > 0:
        w = w / s
    return w, float((w - prev).abs().sum())
