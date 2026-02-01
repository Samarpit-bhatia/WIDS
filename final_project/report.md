# WiDS Kalman-Filtered Trend Trader — Final Project Report (Template)

## 1. Objective
Build a **multi-asset** portfolio manager that allocates across **Bitcoin, Nifty 50, Gold, and Cash** using **Kalman-filtered trend information** with strict causality and explicit transaction costs.

## 2. Data
- Daily prices (2015–2024) aligned across all assets.
- Missing data: forward-fill small gaps and drop remaining NaNs.
- Cash modeled explicitly with zero returns.

## 3. Kalman Filtering
Per-asset **local-trend** state-space model:

- State: \(x_t = [\ell_t, b_t]^T\) where \(\ell_t\) is level and \(b_t\) is slope (trend).
- Transition:
  \[
  x_t = \begin{bmatrix}1 & 1\\ 0 & 1\end{bmatrix} x_{t-1} + w_t, \quad w_t \sim \mathcal{N}(0, Q)
  \]
- Observation:
  \[
  y_t = \begin{bmatrix}1 & 0\end{bmatrix} x_t + v_t, \quad v_t \sim \mathcal{N}(0, R)
  \]

Trend quality:
\[
TQ_t = \frac{b_t}{\sqrt{\mathrm{Var}(b_t) + \epsilon}}
\]

## 4. Strategy Logic
- Prefer assets with positive **trend quality**.
- Risk-budgeting via inverse realized volatility.
- Regime gating: if total signal strength is weak, shift capital to cash (and reduce churn).
- Turnover control: partial adjustment + rebalance band.

## 5. Backtesting Protocol (Causality & Costs)
- Decide weights at time \(t\) using data up to \(t\).
- Execute at time \(t+1\).
- Transaction cost: \(0.1\% \times \sum_j |w_{j,t} - w_{j,t-1}|\).

## 6. Results
Include:
- Equity curve vs Nifty buy-and-hold.
- Weight dynamics + turnover.
- Metrics table: ann return, ann vol, Sharpe, max drawdown, total costs.

## 7. Discussion & Improvements
- Parameter calibration / walk-forward validation.
- Alternative trend definitions (stability, acceleration).
- Add drawdown control and tail-risk measures (CVaR).
- Hybrid RL allocator (optional) with churn penalty.
