# WIDS
Kalman Filtered Trend Trader

WiDS – Assignments 1 & 2

Overview

This repository contains my work for WiDS Assignment 1 and Assignment 2, focused on building strong foundations in statistical learning, regression theory, and time-series modeling, and extending them into a Kalman-filtered machine learning trading strategy for financial markets.

The project progresses from core regression theory and bias analysis to a full end-to-end trading system that models non-stationary market dynamics using state-space models and rolling machine learning predictions.

Assignment 1: Statistical Learning Foundations
1. Linear Regression Theory

Implemented and derived multiple linear regression from first principles, including:

Formal specification of the linear model and assumptions

Derivation of the OLS estimator from the MSE objective

Conditions for invertibility of 
𝑋
𝑇
𝑋
X
T
X and the impact of multicollinearity

Geometric interpretation of regression as an orthogonal projection

Batch gradient descent derivation and update rules

Key concepts learned

Bias–variance decomposition

Multicollinearity and numerical instability

Projection matrices and leverage

Interpretation of residual diagnostics

2. Salary Prediction & Bias Detection

Built a regression-based salary prediction model with a strong emphasis on responsible AI:

Performed extensive EDA on demographic and skill-based features

Encoded categorical variables and handled missing data

Trained and evaluated an OLS regression model

Computed fairness metrics including:

Demographic Parity Difference

Equal Opportunity Difference

Disparate Impact Ratio

Used SHAP to interpret feature contributions and detect bias

Key concepts learned

Model fairness and bias quantification

Trade-offs between accuracy and fairness

Residual-based bias diagnostics

Explainable ML using SHAP

Assignment 2: Kalman Filtered Trading Strategy
Problem Statement

Financial markets exhibit non-stationary behavior, where relationships between price, momentum, volatility, and volume evolve over time. This assignment focuses on designing a Kalman-filtered trading system that adapts to such changes and evaluates performance through realistic backtesting.

Step-by-Step Implementation
1. Data Collection

Downloaded daily MSFT stock data (2015–2024) from Yahoo Finance

Used adjusted prices to account for corporate actions

2. Feature Engineering

Constructed a comprehensive feature set including:

Log returns and lagged returns

Moving averages (5, 20, 60)

Momentum and Rate of Change indicators

Rolling volatility measures

Volume-based features and z-scores

This resulted in 35+ engineered features designed to capture short-term and medium-term market dynamics.

3. Kalman Filter Model

Formulated a state-space model where:

Latent state represents time-varying regression coefficients

Observation equation links returns to market features

State transition allows parameters to evolve via a random walk

Used a Kalman Filter to recursively estimate latent parameters and visualize their evolution over time.

Key insight: Kalman filtering enables adaptive modeling in non-stationary environments where static regression fails.

4. Machine Learning Integration

Used Kalman-filtered states as inputs to a Ridge regression model

Predicted the future price ratio of MSFT

Employed rolling walk-forward training to avoid look-ahead bias and ensure realistic deployment

5. Trading Signal Design

Generated buy signals when predicted price ratios exceeded dynamic thresholds

Enforced causal decision-making using only past and present information

Implemented risk controls such as:

Maximum exposure constraints

Threshold-based entry and exit rules

Transaction cost modeling

6. Strategy Simulation & Backtesting

Simulated daily positions and PnL over the historical period

Accumulated returns to generate equity curves

Benchmarked performance against Buy & Hold MSFT

Evaluation metrics included

Cumulative return

Sharpe ratio

Maximum drawdown

Win/loss ratio

Average trade return

Results Summary

The strategy demonstrated significant drawdown reduction compared to Buy & Hold

Achieved a high win/loss ratio, indicating effective short-term timing

Underperformed Buy & Hold in cumulative returns, highlighting the trade-off between:

Risk control and

Long-term compounding in strong bull markets

This behavior is consistent with a short-horizon, adaptive timing strategy rather than a passive investment approach.

Key Concepts Learned Across Both Assignments

Regression geometry and optimization

Bias–variance trade-offs

Fairness and interpretability in ML

State-space modeling and Kalman Filters

Non-stationarity in financial time series

Causal backtesting and signal leakage prevention

Risk-aware trading system design

Technologies Used

Python, NumPy, Pandas

Scikit-learn

Matplotlib

Yahoo Finance (yfinance)

SHAP

Conclusion

This project demonstrates a full progression from statistical learning theory to a practical, adaptive trading system, emphasizing correctness, interpretability, and realistic evaluation. The work highlights both the strengths and limitations of Kalman-filtered ML models in financial markets and provides a strong foundation for more advanced quantitative research.