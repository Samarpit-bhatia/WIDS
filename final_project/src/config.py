"""
Configuration layer.

We keep most knobs in a single dataclass so:
- experiments are reproducible (saved to artifacts/)
- reviewers can see what was changed
- the system remains modular (swap model/strategy without rewriting backtest)

The project statement requires:
- strict causality (decide at i, execute at i+1)
- 0.1% transaction costs on allocation changes
- multi-asset: BTC, Nifty, Gold, Cash

This repo adds:
- regime gating (risk-on / risk-off)
- turnover-aware smoothing
- walk-forward parameter selection (optional)
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional


@dataclass(frozen=True)
class UniverseConfig:
    tickers: Dict[str, str]
    cash_symbol: str = "CASH"


@dataclass(frozen=True)
class DataConfig:
    start: str = "2015-01-01"
    end: str = "2025-01-01"
    price_field: str = "Close"
    data_mode: str = "yfinance"  # yfinance | csv
    csv_dir: str = "data"
    ffill_limit: int = 5


@dataclass(frozen=True)
class KalmanConfig:
    """
    Two-state local-trend model per asset:
        state x_t = [level, slope]
        y_t = level + obs_noise

    We expose Q and R for both:
    - fixed values (simple)
    - or auto-calibration from data (optional)

    Using pykalman if available for numerical stability, else our numpy implementation.
    """
    q_level: float = 1e-4
    q_slope: float = 1e-5
    r_obs: float = 1e-3
    use_pykalman: bool = True


@dataclass(frozen=True)
class FeatureConfig:
    """
    Borrowing ideas from Week-2:
    - stationarity checks (ADF) for diagnostics
    - Hurst exponent + half-life for regime selection
    - rolling vol, momentum, z-scores

    Features are only computed from past & present data (causal).
    """
    vol_window: int = 20
    mom_fast: int = 10
    mom_slow: int = 60
    z_window: int = 60
    hurst_window: int = 252
    halflife_window: int = 252


@dataclass(frozen=True)
class StrategyConfig:
    """
    Portfolio strategy = Trend Quality + Regime Gate + Risk Budgeting + Turnover Control.

    - Trend quality uses Kalman slope / sqrt(var_slope)
    - Regime gate uses market condition proxies (Nifty trend quality + vol)
    - Risk budgeting uses inverse-vol weights across assets with positive trend
    - Turnover control uses partial adjustment + rebalance band
    """
    trend_clip: float = 3.0
    min_signal_strength: float = 0.25
    cash_boost_when_riskoff: float = 0.40
    rebalance_band: float = 0.15        # total abs weight change required to trade
    partial_adjust_alpha: float = 0.35  # 0->never move, 1->jump to target

    max_weight: float = 0.70
    min_cash: float = 0.05


@dataclass(frozen=True)
class BacktestConfig:
    """
    Causality:
    - Decide weights at date t using information up to t
    - Apply weights to returns at t+1
    Costs:
    - 0.1% cost on allocation changes (L1 turnover)
    """
    transaction_cost: float = 0.001
    trading_days: int = 252


@dataclass(frozen=True)
class RunConfig:
    universe: UniverseConfig
    data: DataConfig
    kalman: KalmanConfig
    feat: FeatureConfig
    strat: StrategyConfig
    bt: BacktestConfig

    def to_dict(self):
        return asdict(self)


def default_config() -> RunConfig:
    # Asset universe per the project statement:
    # - Bitcoin (BTC-USD)
    # - Nifty 50 (^NSEI)
    # - Gold (GLD as practical proxy via Yahoo)
    tickers = {
        "BTC": "BTC-USD",
        "NIFTY": "^NSEI",
        "GOLD": "GLD",
    }
    return RunConfig(
        universe=UniverseConfig(tickers=tickers, cash_symbol="CASH"),
        data=DataConfig(),
        kalman=KalmanConfig(),
        feat=FeatureConfig(),
        strat=StrategyConfig(),
        bt=BacktestConfig(),
    )
