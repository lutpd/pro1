# Robust Portfolio Engineering

This project implements and compares three advanced portfolio optimization methods with a focus on navigating market crises like the 2022 correlation breakdown. The analysis demonstrates how different approaches perform during periods of market stress when traditional correlations fail.

## Project Structure

```
robust-portfolio-engineering/
├── data/                    # Local data storage
├── src/                    # Source Python modules
│   ├── data.py             # Data loading and preprocessing
│   └── engine.py           # Portfolio optimization engine
├── notebooks/              # Jupyter notebooks
│   └── analysis.ipynb      # Main analysis notebook
├── requirements.txt        # Project dependencies
└── README.md              # This file
```

## Key Features

### 1. Data Module (`src/data.py`)
- `DataLoader` class for fetching historical data from Yahoo Finance
- Log return calculation
- Local data storage to avoid repeated downloads

### 2. Portfolio Engine (`src/engine.py`)
- **Ledoit-Wolf Shrinkage**: De-noising of covariance matrices
- **Hierarchical Risk Parity (HRP)**: 
  - Tree clustering using scipy linkage
  - Matrix seriation (quasi-diagonalization)
  - Recursive bisection allocation
- **CVaR Optimization**: Minimizes expected shortfall at 95% confidence
- **Mean-Variance Optimization (MVO)**: Standard approach for comparison

### 3. Analysis Notebook (`notebooks/analysis.ipynb`)
- Analysis of the 2022 correlation crisis (especially SPY vs TLT)
- Dendrogram visualization for HRP clustering
- Cumulative returns comparison
- Drawdown analysis
- Performance metrics comparison

## Assets Analyzed

- **SPY**: S&P 500 ETF
- **TLT**: 20+ Year Treasury Bond ETF
- **GLD**: Gold ETF
- **BTC-USD**: Bitcoin (using Yahoo Finance data)

## Getting Started

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the analysis notebook:
```bash
cd notebooks
jupyter notebook analysis.ipynb
```

## Case Study: Navigating the 2022 Correlation Crisis

The 2022 market environment was characterized by a breakdown of traditional diversification relationships, particularly between stocks (SPY) and bonds (TLT). This project demonstrates:

- How correlations changed during the crisis period
- Performance comparison of different optimization methods
- Risk characteristics during market stress
- The robustness of HRP during correlation breakdowns

## Models Compared

1. **Mean-Variance Optimization (MVO)**: Traditional approach maximizing Sharpe ratio
2. **Hierarchical Risk Parity (HRP)**: Robust to correlation matrix instability
3. **CVaR Optimization**: Focuses on tail risk minimization

## Mathematical Background

### Hierarchical Risk Parity (HRP)
HRP addresses the weaknesses of traditional quadratic optimizers by:
- Not requiring matrix inversion
- Providing exact solutions (no quadratic optimization)
- Being robust to random matrix effects

### Ledoit-Wolf Shrinkage
The covariance matrix is estimated using shrinkage to reduce noise:
- Shrinks sample covariance toward a target matrix
- Improves out-of-sample performance
- Reduces estimation error

### CVaR Optimization
Conditional Value at Risk (CVaR) represents the expected loss beyond Value at Risk (VaR):
- Focuses on tail risk beyond a certain confidence level
- Provides more information about the tail of the distribution
- More robust to distributional assumptions