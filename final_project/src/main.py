"""
Command-line entrypoint.

This script wires everything together:
1) load data (prices)
2) compute returns
3) Kalman filter each asset (level, slope, uncertainty)
4) compute features (vol, tq, etc.)
5) run causal backtest with transaction costs
6) compute benchmark (Nifty buy & hold)
7) compute metrics + save plots + config snapshot

Run:
    python -m src.main --out artifacts

This produces reviewer-friendly artifacts.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict

import numpy as np
import pandas as pd

from .config import default_config, RunConfig
from .data_loader import add_cash, build_price_panel, compute_returns
from .features import compute_kalman_panel, compute_features
from .backtest import run_backtest
from .rl_allocator import RLConfig, train_q_policy, rollout_policy
from .metrics import summarize
from .plots import ensure_dir, plot_equity, plot_weights, plot_diagnostics_per_asset
from .logging_utils import JsonlLogger


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--start", type=str, default=None)
    p.add_argument("--end", type=str, default=None)
    p.add_argument("--out", type=str, default="artifacts")
    p.add_argument("--allocator", type=str, default="rule", choices=["rule","rl"])
    p.add_argument("--data_mode", type=str, default=None, choices=["yfinance", "csv"])
    p.add_argument("--csv_dir", type=str, default=None)
    return p.parse_args()


def nifty_benchmark_equity(nifty_returns: pd.Series) -> pd.Series:
    eq = (1.0 + nifty_returns.dropna()).cumprod()
    eq = eq / float(eq.iloc[0])
    return eq



def backtest_external_weights(weights: pd.DataFrame, returns: pd.DataFrame, cash_symbol: str, tc: float) -> pd.DataFrame:
    # Align
    idx = weights.index.intersection(returns.index).sort_values()
    w = weights.reindex(idx).fillna(0.0)
    r = returns.reindex(idx).fillna(0.0)
    if cash_symbol in r.columns:
        r[cash_symbol] = 0.0

    # execute at t+1
    next_r = r.shift(-1)
    w_prev = w.shift(1).fillna(0.0)
    turnover = (w - w_prev).abs().sum(axis=1)
    cost = tc * turnover
    port_ret = (w * next_r).sum(axis=1) - cost
    out = pd.DataFrame({"port_ret": port_ret, "turnover": turnover, "cost": cost}, index=idx).dropna()
    out["equity"] = (1.0 + out["port_ret"]).cumprod()
    return out


def main():
    args = parse_args()
    cfg = default_config()

    # Override config from CLI (keep simple)
    start = args.start or cfg.data.start
    end = args.end or cfg.data.end
    data_mode = args.data_mode or cfg.data.data_mode
    csv_dir = args.csv_dir or cfg.data.csv_dir

    out_dir = args.out
    ensure_dir(out_dir)
    logger = JsonlLogger(out_dir)

    logger.log("run_start", {"start": start, "end": end, "data_mode": data_mode})

    prices = build_price_panel(
        tickers=cfg.universe.tickers,
        start=start,
        end=end,
        field=cfg.data.price_field,
        data_mode=data_mode,
        csv_dir=csv_dir,
        ffill_limit=cfg.data.ffill_limit,
    )
    prices = add_cash(prices, cash_symbol=cfg.universe.cash_symbol)

    returns = compute_returns(prices.drop(columns=[cfg.universe.cash_symbol]), kind="log")
    # add cash return = 0
    returns[cfg.universe.cash_symbol] = 0.0
    returns = returns.dropna()

    logger.log("data_loaded", {"n_days": int(len(prices)), "assets": list(prices.columns)})

    kal = compute_kalman_panel(
        prices=prices,
        q_level=cfg.kalman.q_level,
        q_slope=cfg.kalman.q_slope,
        r_obs=cfg.kalman.r_obs,
        use_pykalman=cfg.kalman.use_pykalman,
    )

    feats = compute_features(
        prices=prices,
        returns=returns,
        kalman=kal,
        vol_window=cfg.feat.vol_window,
        mom_fast=cfg.feat.mom_fast,
        mom_slow=cfg.feat.mom_slow,
        z_window=cfg.feat.z_window,
    )

    vol = feats["vol"]
    tq = feats["tq"]

    if args.allocator == "rule":
        bt = run_backtest(
            returns=returns,
            tq=tq,
            vol=vol,
            cash_symbol=cfg.universe.cash_symbol,
            transaction_cost=cfg.bt.transaction_cost,
            trend_clip=cfg.strat.trend_clip,
            min_signal_strength=cfg.strat.min_signal_strength,
            cash_boost_when_riskoff=cfg.strat.cash_boost_when_riskoff,
            rebalance_band=cfg.strat.rebalance_band,
            partial_adjust_alpha=cfg.strat.partial_adjust_alpha,
            max_weight=cfg.strat.max_weight,
            min_cash=cfg.strat.min_cash,
        )
        daily = bt.daily
        weights = bt.weights
    else:
        # RL training window (simple, reviewer-friendly): train on 2015–2021, validate implicitly via later years
        rl_cfg = RLConfig()
        assets = list(returns.columns)
        if cfg.universe.cash_symbol not in assets:
            assets.append(cfg.universe.cash_symbol)

        actions = rl_cfg.build_default_actions(assets=assets, cash_symbol=cfg.universe.cash_symbol)

        idx_all = tq.index.intersection(vol.index).intersection(returns.index).sort_values()
        train_end = pd.Timestamp("2021-12-31")
        train_mask = idx_all <= train_end
        train_end_idx = int(train_mask.sum())

        policy = train_q_policy(
            tq=tq,
            vol=vol,
            returns=returns,
            assets=assets,
            cash_symbol=cfg.universe.cash_symbol,
            actions=actions,
            tc=cfg.bt.transaction_cost,
            cfg=rl_cfg,
            seed=42,
            start_idx=0,
            end_idx=train_end_idx,
        )

        weights = rollout_policy(
            policy=policy,
            tq=tq,
            vol=vol,
            assets=assets,
            cash_symbol=cfg.universe.cash_symbol,
            actions=actions,
            start_date=idx_all.min(),
            end_date=idx_all.max(),
            cfg=rl_cfg,
        )
        daily = backtest_external_weights(weights, returns, cfg.universe.cash_symbol, cfg.bt.transaction_cost)

    # Benchmark vs Nifty buy-hold
    if "NIFTY" in returns.columns:
        bench_eq = nifty_benchmark_equity(returns["NIFTY"])
        bench_eq = bench_eq.reindex(daily.index).ffill().dropna()
    else:
        bench_eq = pd.Series(index=bt.daily.index, data=np.nan)

    metrics = summarize(daily, trading_days=cfg.bt.trading_days)

    # Save artifacts
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg.to_dict(), f, indent=2)

    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    daily.to_csv(os.path.join(out_dir, "daily.csv"))
    weights.to_csv(os.path.join(out_dir, "weights.csv"))

    plot_equity(daily, bench_eq, os.path.join(out_dir, "equity_curve.png"))
    plot_weights(weights, os.path.join(out_dir, "weights.png"))
    plot_diagnostics_per_asset(
        prices=prices.drop(columns=[cfg.universe.cash_symbol]),
        tq=tq.drop(columns=[cfg.universe.cash_symbol]),
        slopes=kal.slope.drop(columns=[cfg.universe.cash_symbol]),
        out_dir=os.path.join(out_dir, "diagnostics"),
    )

    logger.log("run_complete", {"metrics": metrics, "allocator": args.allocator})
    print("Done. Metrics:")
    for k, v in metrics.items():
        print(f"{k:>16s}: {v:.4f}" if isinstance(v, float) else f"{k:>16s}: {v}")


if __name__ == "__main__":
    main()
