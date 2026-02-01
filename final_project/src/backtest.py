"""
Backtesting engine.

Project requirements:
- no lookahead: decisions at time i use info up to i
- execution at i+1
- transaction costs: 0.1% * turnover

We:
1) compute daily weights_t from features_t (date t)
2) apply weights_t to next-day returns_{t+1}
3) charge cost on day t based on weight change
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .allocator import AllocatorState, compute_target_weights, apply_turnover_control


@dataclass
class BacktestResult:
    daily: pd.DataFrame
    weights: pd.DataFrame


def run_backtest(
    returns: pd.DataFrame,
    tq: pd.DataFrame,
    vol: pd.DataFrame,
    cash_symbol: str,
    transaction_cost: float,
    # strategy knobs
    trend_clip: float,
    min_signal_strength: float,
    cash_boost_when_riskoff: float,
    rebalance_band: float,
    partial_adjust_alpha: float,
    max_weight: float,
    min_cash: float,
) -> BacktestResult:
    idx = returns.index.intersection(tq.index).intersection(vol.index)
    idx = idx.sort_values()

    assets = [c for c in returns.columns]
    if cash_symbol not in assets:
        assets = assets + [cash_symbol]

    # Ensure cash return is zero
    rets = returns.reindex(idx)[assets].copy()
    if cash_symbol in rets.columns:
        rets[cash_symbol] = 0.0

    tq2 = tq.reindex(idx)[assets].fillna(0.0)
    vol2 = vol.reindex(idx)[assets].fillna(np.nan)

    weights = []
    turnovers = []
    costs = []
    port_rets = []

    prev_w = pd.Series(0.0, index=assets)
    prev_w[cash_symbol] = 1.0
    state = AllocatorState(prev_weights=prev_w)

    # next-day returns for execution
    next_rets = rets.shift(-1)

    for t in idx:
        target = compute_target_weights(
            date=t,
            tq_row=tq2.loc[t],
            vol_row=vol2.loc[t],
            cash_symbol=cash_symbol,
            trend_clip=trend_clip,
            min_signal_strength=min_signal_strength,
            cash_boost_when_riskoff=cash_boost_when_riskoff,
            max_weight=max_weight,
            min_cash=min_cash,
        )

        w_new, turnover = apply_turnover_control(
            target_w=target,
            prev_w=state.prev_weights,
            rebalance_band=rebalance_band,
            partial_adjust_alpha=partial_adjust_alpha,
        )

        cost = transaction_cost * turnover
        # apply weights at t to returns at t+1 (causal)
        r_next = next_rets.loc[t].fillna(0.0)
        port_r = float((w_new * r_next).sum()) - cost

        weights.append(w_new.rename(t))
        turnovers.append(turnover)
        costs.append(cost)
        port_rets.append(port_r)

        state = AllocatorState(prev_weights=w_new)

    w_df = pd.DataFrame(weights)
    daily = pd.DataFrame(
        {"port_ret": port_rets, "turnover": turnovers, "cost": costs},
        index=idx
    ).dropna()

    daily["equity"] = (1.0 + daily["port_ret"]).cumprod()
    return BacktestResult(daily=daily, weights=w_df)
