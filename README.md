# Forecasting US Stock Closing Dynamics

> Predicting short-term price movements during the Nasdaq Closing Cross auction using machine learning on high-frequency order book data — achieving a **16.6% improvement** over the linear baseline with LightGBM.

---

## Context

The Nasdaq Closing Cross determines the official closing price for listed equities. It's one of the most volatile and information-dense phases of the trading day, directly impacting index calculation, portfolio valuation, and algorithmic execution.

The challenge: predict short-horizon price movements in the final minutes of trading using high-frequency auction and order book snapshots — a noisy, fast-evolving environment.

**Dataset**: [Optiver — Trading at the Close](https://www.kaggle.com/competitions/optiver-trading-at-the-close) (Kaggle)

## Feature Engineering

We designed features grounded in market microstructure theory:

| Type | Feature | What it captures |
|------|---------|-----------------|
| Core | Bid/Ask Price, WAP, Reference Price | Instantaneous market state |
| Engineered | Directional Imbalance Pressure | Net buying/selling pressure (size × direction) |
| Engineered | Bid–Ask Spread | Liquidity and market uncertainty |
| Engineered | Auction Price Divergence | Difference between far and near prices — uncertainty in price formation |
| Dynamic | Imbalance Size Diff, Matched Size Diff | Short-term momentum in order flow |
| Dynamic | Lagged variables | Temporal evolution of auction conditions (LightGBM only) |
| Cross-sectional | Global Imbalance Mean | Market-wide pressure across all stocks at a given time |

## Models Benchmarked

| Model | MAE | RMSE | Role |
|-------|-----|------|------|
| **LightGBM** | **5.26** | **7.63** | ✅ Best — captures nonlinear patterns + temporal dynamics |
| Ridge Regression | 6.31 | 9.29 | Stable linear baseline, handles multicollinearity |
| Random Forest | 6.45 | 9.85 | Nonlinear but marginal gain over Ridge |

## Key Findings

- **Price variables dominate**: Bid, Ask, and WAP are the top predictors across all models — the market state at time *t* is the best predictor of *t+1*
- **Engineered microstructure features provide the edge**: Imbalance Pressure and Spread ranked among the most important non-price features in LightGBM
- **Temporal dynamics matter**: Lag-based and first-difference features capture how rapid changes in auction conditions influence price — this is where LightGBM pulls ahead
- **Random Forest ≈ Ridge**: Nonlinear effects added limited predictive power on their own, suggesting that the real gains come from combining gradient boosting with temporal feature engineering

## Tech Stack

`Python` · `LightGBM` · `Scikit-learn` · `Pandas` · `NumPy` · `Matplotlib` · `Seaborn`

## Team

- Naël Arnoux
- **Marta Shkreli**
- Wei Luo

MSc in Data Science & Business Analytics — ESSEC & CentraleSupélec (2025–2026)
