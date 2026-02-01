# WiDS Kalman-Filtered Trend Trader — Multi-Asset Portfolio Manager

This repository implements a **causal**, **transaction-cost-aware** multi-asset portfolio allocator
over **Bitcoin, Nifty 50, Gold, and Cash**, using **Kalman-filtered latent trend information** and
robust backtesting.

Key requirements addressed (from the project PDF):
- Daily data 2015–2024; timestamp alignment + missing-value handling
- Each asset filtered independently via Kalman state-space models
- Decisions at time *i*; execution at time *i+1* (no lookahead)
- 0.1% transaction costs on allocation changes
- Metrics: PnL/turnover, annualized return/vol, Sharpe, max drawdown, benchmark vs Nifty buy&hold

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate          # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
python -m src.main --start 2015-01-01 --end 2025-01-01 --out artifacts
```

By default the code downloads data with `yfinance`. If you must run fully offline, place CSVs in `data/`
and use `--data_mode csv`.

## Outputs
- `artifacts/metrics.json` + `artifacts/metrics.csv`
- `artifacts/equity_curve.png` (strategy vs benchmark)
- `artifacts/weights.png` (stacked weights)
- `artifacts/diagnostics/` (per-asset Kalman diagnostics, regime plots)

## Design notes
The code is intentionally modular: data, modeling, signals, allocator, backtest, metrics, plots.

## Optional RL Allocator

Run the RL-based allocator (tabular Q-learning over discrete portfolio templates):

```bash
python -m src.main --allocator rl --out artifacts_rl
```
