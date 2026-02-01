"""
Data loading and alignment.

Requirements (project PDF):
- daily data 2015–2024 for BTC, Nifty, Gold, Cash
- align timestamps across assets
- handle missing values
- ensure consistent preprocessing

We support:
- yfinance download (default)
- csv mode (offline / pre-downloaded)

Notes on CASH:
- modeled as an explicit column with zero returns
"""
from __future__ import annotations

import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except Exception:
    yf = None

def _download_close(ticker: str, start: str, end: str, field: str = "Close") -> pd.Series:
    if yf is None:
        raise RuntimeError("yfinance not installed/available. Use --data_mode csv instead.")

    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)

    if df.empty:
        raise ValueError(f"No data returned for ticker={ticker}")

    if field not in df.columns.get_level_values(0) if isinstance(df.columns, pd.MultiIndex) else df.columns:
        raise ValueError(f"Missing field={field} in yfinance data for ticker={ticker}")

    # --- Robust extraction: handle both single-level and MultiIndex columns ---
    sel = df[field]

    if isinstance(sel, pd.DataFrame):
        # common case: MultiIndex columns -> sel has 1 column
        if sel.shape[1] == 1:
            s = sel.iloc[:, 0]
        else:
            # if multiple columns exist, try to pick the one matching ticker
            if ticker in sel.columns:
                s = sel[ticker]
            else:
                s = sel.iloc[:, 0]
    else:
        s = sel

    s = s.rename(ticker)
    s.index = pd.to_datetime(s.index)
    return s



def _load_csv_close(csv_path: str, date_col: str = "Date", price_col: str = "Close") -> pd.Series:
    df = pd.read_csv(csv_path)
    if date_col not in df.columns or price_col not in df.columns:
        raise ValueError(f"CSV must contain columns {date_col}, {price_col}. Got={list(df.columns)}")
    s = pd.Series(df[price_col].values, index=pd.to_datetime(df[date_col].values))
    s = s.sort_index()
    name = os.path.splitext(os.path.basename(csv_path))[0]
    return s.rename(name)


def build_price_panel(
    tickers: Dict[str, str],
    start: str,
    end: str,
    field: str,
    data_mode: str,
    csv_dir: str,
    ffill_limit: int,
) -> pd.DataFrame:
    series = []
    if data_mode == "yfinance":
        for sym, tkr in tickers.items():
            series.append(_download_close(tkr, start, end, field=field).rename(sym))
    elif data_mode == "csv":
        for sym, tkr in tickers.items():
            # Expect user to name files by sym or ticker; attempt both
            candidates = [
                os.path.join(csv_dir, f"{sym}.csv"),
                os.path.join(csv_dir, f"{tkr}.csv"),
            ]
            found = None
            for c in candidates:
                if os.path.exists(c):
                    found = c
                    break
            if found is None:
                raise FileNotFoundError(f"Could not find CSV for {sym} in {csv_dir}. Tried: {candidates}")
            series.append(_load_csv_close(found).rename(sym))
    else:
        raise ValueError(f"Unknown data_mode={data_mode}")

    prices = pd.concat(series, axis=1).sort_index()

    # Forward-fill small gaps; drop remaining NAs to enforce consistent panel
    prices = prices.ffill(limit=ffill_limit).dropna()

    return prices


def compute_returns(prices: pd.DataFrame, kind: str = "log") -> pd.DataFrame:
    if kind not in {"log", "simple"}:
        raise ValueError("kind must be 'log' or 'simple'")
    if kind == "simple":
        rets = prices.pct_change()
    else:
        rets = np.log(prices / prices.shift(1))
    return rets.dropna()


def add_cash(prices: pd.DataFrame, cash_symbol: str = "CASH") -> pd.DataFrame:
    prices2 = prices.copy()
    if cash_symbol in prices2.columns:
        return prices2
    # "Price" for cash is irrelevant; set to 1 constant (return 0)
    prices2[cash_symbol] = 1.0
    return prices2
