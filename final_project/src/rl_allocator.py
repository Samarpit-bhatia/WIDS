"""
RL Allocator (optional) — Week-3 flavored extension.

Goal:
- Combine Week-2 Kalman trend extraction with a Week-3 style RL decision layer.
- Keep everything **causal** and **transaction-cost-aware**.
- Avoid heavy compute: use a lightweight **tabular Q-learning** policy over a small discrete action set.

High-level design
-----------------
State (at day t) is built only from information available up to t:
- Per-asset Kalman Trend Quality (TQ): slope / sqrt(var_slope)
- Per-asset realized volatility (rolling window)
- Market regime proxy (e.g., Nifty TQ sign)
- Previous portfolio weights bucket (coarse)

Action is one of K discrete allocation templates over {BTC, NIFTY, GOLD, CASH}.
Example templates:
- risk_on: [0.35, 0.45, 0.15, 0.05]
- defensive: [0.10, 0.25, 0.35, 0.30]
- cash_heavy: [0.05, 0.10, 0.15, 0.70]
(and a few more)

Reward uses next-day return (execution at t+1) minus turnover penalty:
    r_{t+1} = (w_t · ret_{t+1}) - tc * turnover(w_t, w_{t-1}) - lambda_dd * dd_penalty

This module is intentionally compact and interpretable so a reviewer can verify:
- causality (uses ret_{t+1} for reward only)
- explicit cost penalty
- clear mapping state -> action -> weights
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


@dataclass
class RLConfig:
    # Discretization
    tq_bins: Tuple[float, ...] = (-1.0, -0.25, 0.25, 1.0)  # 5 buckets
    vol_bins: Tuple[float, ...] = (0.01, 0.02, 0.04, 0.08)  # depends on asset scale (log-returns)
    # Q-learning
    gamma: float = 0.95
    alpha: float = 0.10
    eps_start: float = 0.80
    eps_end: float = 0.05
    eps_decay_days: int = 800
    # Reward shaping
    turnover_penalty: float = 0.15      # multiplied by tc * turnover
    drawdown_penalty: float = 0.0      # optional (keep 0 for simplicity)
    max_btc_weight: float = 0.30
    # Action set
    action_templates: Optional[List[Dict[str, float]]] = None

    def build_default_actions(self, assets: List[str], cash_symbol: str) -> List[pd.Series]:
        """
        Build a small action set of interpretable portfolios.
        All portfolios must sum to 1 and contain all assets.
        """
        a = assets.copy()
        if cash_symbol not in a:
            a.append(cash_symbol)

        templates = [
            # Risk-on variants
            {"BTC": 0.35, "NIFTY": 0.45, "GOLD": 0.15, cash_symbol: 0.05},
            {"BTC": 0.45, "NIFTY": 0.35, "GOLD": 0.15, cash_symbol: 0.05},
            {"BTC": 0.25, "NIFTY": 0.55, "GOLD": 0.15, cash_symbol: 0.05},

            # Balanced
            {"BTC": 0.25, "NIFTY": 0.40, "GOLD": 0.25, cash_symbol: 0.10},
            {"BTC": 0.20, "NIFTY": 0.35, "GOLD": 0.30, cash_symbol: 0.15},

            # Defensive / risk-off
            {"BTC": 0.10, "NIFTY": 0.25, "GOLD": 0.35, cash_symbol: 0.30},
            {"BTC": 0.05, "NIFTY": 0.15, "GOLD": 0.35, cash_symbol: 0.45},
            {"BTC": 0.05, "NIFTY": 0.10, "GOLD": 0.15, cash_symbol: 0.70},
        ]

        actions = []
        for tpl in templates:
            w = pd.Series(0.0, index=a, dtype=float)
            for k, v in tpl.items():
                if k in w.index:
                    w[k] = float(v)
            # renormalize defensively
            s = float(w.sum())
            if s <= 0:
                w[cash_symbol] = 1.0
            else:
                w = w / s
            actions.append(w)
        return actions


def _bucketize(x: float, bins: Tuple[float, ...]) -> int:
    # returns bucket in [0, len(bins)]
    for i, b in enumerate(bins):
        if x < b:
            return i
    return len(bins)


def build_state_key(
    tq_row: pd.Series,
    vol_row: pd.Series,
    assets: List[str],
    tq_bins: Tuple[float, ...],
    vol_bins: Tuple[float, ...],
    cash_symbol: str,
    prev_action: int,
) -> Tuple[int, ...]:
    """
    State is a small tuple of discrete buckets to keep Q-table manageable.
    """
    key: List[int] = []
    for a in assets:
        if a == cash_symbol:
            continue
        key.append(_bucketize(float(tq_row.get(a, 0.0)), tq_bins))
    for a in assets:
        if a == cash_symbol:
            continue
        key.append(_bucketize(float(vol_row.get(a, 0.0)), vol_bins))

    # Regime proxy: sign bucket of NIFTY tq
    nifty_tq = float(tq_row.get("NIFTY", 0.0))
    key.append(0 if nifty_tq < -0.25 else (1 if nifty_tq < 0.25 else 2))

    # Previous action index (encodes inertia / churn)
    key.append(int(prev_action))
    return tuple(key)


class TabularQPolicy:
    def __init__(self, n_actions: int, alpha: float, gamma: float):
        self.n_actions = int(n_actions)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.Q: Dict[Tuple[int, ...], np.ndarray] = {}

    def _ensure(self, s: Tuple[int, ...]) -> np.ndarray:
        if s not in self.Q:
            self.Q[s] = np.zeros(self.n_actions, dtype=float)
        return self.Q[s]

    def act(self, s: Tuple[int, ...], eps: float, rng: np.random.Generator) -> int:
        q = self._ensure(s)
        if rng.random() < eps:
            return int(rng.integers(0, self.n_actions))
        return int(np.argmax(q))

    def update(self, s: Tuple[int, ...], a: int, r: float, s2: Tuple[int, ...]):
        q = self._ensure(s)
        q2 = self._ensure(s2)
        td_target = float(r) + self.gamma * float(np.max(q2))
        td_error = td_target - float(q[a])
        q[a] = float(q[a]) + self.alpha * td_error


def epsilon_schedule(day_idx: int, eps_start: float, eps_end: float, decay_days: int) -> float:
    if decay_days <= 0:
        return float(eps_end)
    frac = min(1.0, max(0.0, day_idx / decay_days))
    # exponential-ish decay
    eps = eps_start * (1.0 - frac) + eps_end * frac
    return float(eps)


def train_q_policy(
    tq: pd.DataFrame,
    vol: pd.DataFrame,
    returns: pd.DataFrame,
    assets: List[str],
    cash_symbol: str,
    actions: List[pd.Series],
    tc: float,
    cfg: RLConfig,
    seed: int = 42,
    start_idx: int = 0,
    end_idx: Optional[int] = None,
) -> TabularQPolicy:
    """
    Train on a contiguous window [start_idx, end_idx) using single-pass online Q-learning.
    Reward uses next-day returns -> still causal for action selection.
    """
    rng = np.random.default_rng(seed)
    n_actions = len(actions)
    policy = TabularQPolicy(n_actions=n_actions, alpha=cfg.alpha, gamma=cfg.gamma)

    idx = tq.index.intersection(vol.index).intersection(returns.index).sort_values()
    if end_idx is None:
        end_idx = len(idx) - 2  # need t+1
    end_idx = min(end_idx, len(idx) - 2)

    prev_action = 0
    prev_w = actions[prev_action].reindex(assets).fillna(0.0)

    for k in range(start_idx, end_idx):
        t = idx[k]
        t2 = idx[k + 1]

        s = build_state_key(
            tq_row=tq.loc[t],
            vol_row=vol.loc[t],
            assets=assets,
            tq_bins=cfg.tq_bins,
            vol_bins=cfg.vol_bins,
            cash_symbol=cash_symbol,
            prev_action=prev_action,
        )

        eps = epsilon_schedule(k, cfg.eps_start, cfg.eps_end, cfg.eps_decay_days)
        a = policy.act(s, eps=eps, rng=rng)
        w = actions[a].reindex(assets).fillna(0.0)
        # ---- Risk control: cap BTC exposure ----
        if "BTC" in w.index:
            w["BTC"] = min(float(w["BTC"]), float(cfg.max_btc_weight))
            s = float(w.sum())
            if s > 0:
                w = w / s

        # turnover cost at decision time t
        turnover = float((w - prev_w).abs().sum())
        cost = tc * turnover * cfg.turnover_penalty

        # reward: realized at t+1
        r_next = returns.loc[t2].reindex(assets).fillna(0.0)
        reward = float((w * r_next).sum()) - cost

        # next state uses info at t2
        s2 = build_state_key(
            tq_row=tq.loc[t2],
            vol_row=vol.loc[t2],
            assets=assets,
            tq_bins=cfg.tq_bins,
            vol_bins=cfg.vol_bins,
            cash_symbol=cash_symbol,
            prev_action=a,
        )

        policy.update(s, a, reward, s2)

        prev_action = a
        prev_w = w

    return policy


def rollout_policy(
    policy: TabularQPolicy,
    tq: pd.DataFrame,
    vol: pd.DataFrame,
    assets: List[str],
    cash_symbol: str,
    actions: List[pd.Series],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    cfg: RLConfig,
) -> pd.DataFrame:
    """
    Deterministic rollout (eps=0) to produce weights over a date range.
    """
    idx = tq.index.intersection(vol.index).sort_values()
    idx = idx[(idx >= start_date) & (idx <= end_date)]
    if len(idx) == 0:
        raise ValueError("No dates in rollout range.")

    prev_action = 0
    w_rows = []

    for t in idx:
        s = build_state_key(
            tq_row=tq.loc[t],
            vol_row=vol.loc[t],
            assets=assets,
            tq_bins=cfg.tq_bins,
            vol_bins=cfg.vol_bins,
            cash_symbol=cash_symbol,
            prev_action=prev_action,
        )
        a = policy.act(s, eps=0.0, rng=np.random.default_rng(0))
        w = actions[a].reindex(assets).fillna(0.0)
        # ---- Risk control: cap BTC exposure ----
        if "BTC" in w.index:
            w["BTC"] = min(float(w["BTC"]), float(cfg.max_btc_weight))
            s = float(w.sum())
            if s > 0:
                w = w / s
        w_rows.append(w.rename(t))
        prev_action = a

    return pd.DataFrame(w_rows)
