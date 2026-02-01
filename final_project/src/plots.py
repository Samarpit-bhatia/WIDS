"""
Plotting utilities.

We keep plotting separate so strategy/backtest stays testable.

Outputs:
- equity curve vs benchmark
- stacked area plot for weights
- diagnostics per asset (trend quality, slope, etc.)
"""
from __future__ import annotations

import os
from typing import Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def plot_equity(daily: pd.DataFrame, bench_equity: pd.Series, out_path: str):
    plt.figure()
    plt.plot(daily.index, daily["equity"].values, label="Strategy")
    plt.plot(bench_equity.index, bench_equity.values, label="Nifty Buy&Hold")
    plt.xlabel("Date")
    plt.ylabel("Equity (normalized)")
    plt.title("Equity Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_weights(weights: pd.DataFrame, out_path: str, max_cols: int = 8):
    # Stackplot expects numpy arrays
    cols = list(weights.columns)[:max_cols]
    x = weights.index
    ys = [weights[c].fillna(0.0).values for c in cols]

    plt.figure(figsize=(10, 5))
    plt.stackplot(x, ys, labels=cols)
    plt.xlabel("Date")
    plt.ylabel("Weight")
    plt.title("Portfolio Weights Over Time")
    plt.legend(loc="upper left", ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_diagnostics_per_asset(
    prices: pd.DataFrame,
    tq: pd.DataFrame,
    slopes: pd.DataFrame,
    out_dir: str
):
    ensure_dir(out_dir)
    for col in prices.columns:
        fig = plt.figure(figsize=(10, 6))
        ax1 = fig.add_subplot(2, 1, 1)
        ax1.plot(prices.index, prices[col].values)
        ax1.set_title(f"{col}: Price")

        ax2 = fig.add_subplot(2, 1, 2)
        ax2.plot(tq.index, tq[col].values, label="TrendQuality")
        ax2.plot(slopes.index, slopes[col].values, label="Slope")
        ax2.axhline(0.0)
        ax2.legend()
        ax2.set_title(f"{col}: Kalman Trend Signals")

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{col}_diagnostic.png"), dpi=160)
        plt.close()
