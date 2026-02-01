"""
Kalman filtering models used for trading.

Week-2 materials describe two standard finance uses:
1) KalmanFilterAverage: latent fair value (random walk)
2) KalmanFilterTrend: time-varying trend slope (alpha, beta)

We implement a robust two-state local-trend model:
    state = [level, slope]
and expose:
- numpy implementation (no dependency)
- pykalman implementation (if installed)

Outputs:
- level_t (filtered fair value / smoothed price)
- slope_t (instantaneous trend / velocity)
- covariance of state => uncertainty of slope

We filter each asset independently (project requirement).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

try:
    from pykalman import KalmanFilter as PyKalman
except Exception:
    PyKalman = None


@dataclass
class KalmanOutput:
    level: np.ndarray
    slope: np.ndarray
    cov: np.ndarray  # shape [T, 2, 2]


class LocalTrendKalmanNumpy:
    def __init__(self, q_level: float, q_slope: float, r_obs: float):
        self.F = np.array([[1.0, 1.0],
                           [0.0, 1.0]], dtype=float)
        self.H = np.array([[1.0, 0.0]], dtype=float)
        self.Q = np.array([[q_level, 0.0],
                           [0.0, q_slope]], dtype=float)
        self.R = np.array([[r_obs]], dtype=float)

    def filter(self, y: np.ndarray) -> KalmanOutput:
        y = np.asarray(y, dtype=float)
        n = len(y)
        x = np.zeros((n, 2), dtype=float)
        P = np.zeros((n, 2, 2), dtype=float)

        x_prev = np.array([y[0], 0.0], dtype=float)
        P_prev = np.eye(2, dtype=float)

        I = np.eye(2, dtype=float)

        for t in range(n):
            # Predict
            x_pred = self.F @ x_prev
            P_pred = self.F @ P_prev @ self.F.T + self.Q

            # Update
            z = np.array([[y[t]]], dtype=float)
            innov = z - (self.H @ x_pred).reshape(1, 1)
            S = self.H @ P_pred @ self.H.T + self.R
            K = (P_pred @ self.H.T) @ np.linalg.inv(S)

            x_new = x_pred + (K @ innov).flatten()
            P_new = (I - K @ self.H) @ P_pred

            x[t] = x_new
            P[t] = P_new
            x_prev, P_prev = x_new, P_new

        return KalmanOutput(level=x[:, 0], slope=x[:, 1], cov=P)


class LocalTrendKalmanPy:
    """
    pykalman-based implementation for numerical stability.
    """
    def __init__(self, q_level: float, q_slope: float, r_obs: float):
        if PyKalman is None:
            raise RuntimeError("pykalman not installed. Install it or set use_pykalman=False.")
        self.q_level = q_level
        self.q_slope = q_slope
        self.r_obs = r_obs

        self.F = np.array([[1.0, 1.0],
                           [0.0, 1.0]], dtype=float)
        self.H = np.array([[1.0, 0.0]], dtype=float)

        self.Q = np.array([[q_level, 0.0],
                           [0.0, q_slope]], dtype=float)
        self.R = np.array([[r_obs]], dtype=float)

    def filter(self, y: np.ndarray) -> KalmanOutput:
        y = np.asarray(y, dtype=float)
        kf = PyKalman(
            transition_matrices=self.F,
            observation_matrices=self.H,
            transition_covariance=self.Q,
            observation_covariance=self.R,
            initial_state_mean=np.array([y[0], 0.0], dtype=float),
            initial_state_covariance=np.eye(2, dtype=float),
        )
        state_means, state_covs = kf.filter(y.reshape(-1, 1))
        state_means = np.asarray(state_means)
        state_covs = np.asarray(state_covs)
        return KalmanOutput(level=state_means[:, 0], slope=state_means[:, 1], cov=state_covs)


def fit_local_trend(
    y: np.ndarray,
    q_level: float,
    q_slope: float,
    r_obs: float,
    use_pykalman: bool = True,
) -> KalmanOutput:
    if use_pykalman and PyKalman is not None:
        return LocalTrendKalmanPy(q_level, q_slope, r_obs).filter(y)
    return LocalTrendKalmanNumpy(q_level, q_slope, r_obs).filter(y)
