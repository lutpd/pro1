# Strategic Asset Allocation: Multi-Asset Portfolio Optimization

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Modern Portfolio Theory](https://img.shields.io/badge/Finance-MPT-green.svg)](https://en.wikipedia.org/wiki/Modern_portfolio_theory)

A professional-grade Python implementation of **Modern Portfolio Theory (MPT)** for multi-asset portfolio optimization. This project constructs optimal portfolios across Equities, Bonds, Gold, and Cryptocurrency using the Markowitz Efficient Frontier methodology.

![Efficient Frontier](https://via.placeholder.com/800x400/1a1a2e/ffffff?text=Efficient+Frontier+Visualization)

---

## 📊 Key Performance Indicators (KPIs)

| Metric | Equal Weight | Max Sharpe | Min Volatility |
|--------|:------------:|:----------:|:--------------:|
| **Annualized Return** | 17.31% | 24.10% | 8.78% |
| **Annualized Volatility** | 20.61% | 25.17% | 12.03% |
| **Sharpe Ratio** | 0.60 | 0.76 | 0.31 |

### Value-Add of Optimization

| Improvement Metric | Value |
|--------------------|:-----:|
| **Sharpe Improvement** (MSR vs EW) | +27% |
| **Volatility Reduction** (MinVol vs EW) | -42% |
| **BTC Constraint Compliance** | ✅ ≤15% |

> **Note**: Results based on 2020-01-01 to 2024-12-31 data. Max Sharpe optimizes for risk-adjusted return (higher Sharpe), while Min Volatility minimizes risk.

---

## 🎯 Project Objectives

1. **Monte Carlo Simulation**: Generate 10,000 random portfolio allocations to explore the risk-return spectrum
2. **Mean-Variance Optimization**: Find Maximum Sharpe Ratio (MSR) and Minimum Volatility portfolios using `scipy.optimize`
3. **Benchmark Comparison**: Compare optimized portfolios against an Equal-Weight (1/N) benchmark
4. **Constraint Modeling**: Implement institutional-grade sector constraints (BTC ≤ 15%)

---

## 🏗️ Project Structure

```
├── data/                   # Saved CSV data files
│   ├── prices.csv          # Historical price data
│   ├── monte_carlo_results.csv
│   └── portfolio_weights.csv
├── src/
│   └── engine.py           # Core optimization engine (PortfolioOptimizer class)
├── notebook.ipynb          # Research notebook with analysis & visualizations
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## 💹 Asset Universe

| Ticker | Asset Class | Description | Constraint |
|--------|-------------|-------------|------------|
| **SPY** | Equities | S&P 500 ETF | None |
| **TLT** | Fixed Income | 20+ Year Treasury Bond | None |
| **GLD** | Commodities | Gold ETF | None |
| **BTC-USD** | Cryptocurrency | Bitcoin | Max 15% |

---

## 📐 Mathematical Foundation

### Portfolio Return
$$R_p = \sum_{i=1}^{n} w_i \cdot r_i = \mathbf{w}^T \boldsymbol{\mu}$$

### Portfolio Volatility
$$\sigma_p = \sqrt{\mathbf{w}^T \Sigma \mathbf{w}}$$

### Sharpe Ratio
$$SR = \frac{R_p - R_f}{\sigma_p}$$

Where:
- $\mathbf{w}$ = vector of portfolio weights
- $\boldsymbol{\mu}$ = vector of expected returns
- $\Sigma$ = covariance matrix
- $R_f$ = risk-free rate

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/portfolio-optimization.git
cd portfolio-optimization

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Usage

#### Option 1: Run the Jupyter Notebook (Recommended)

```bash
jupyter notebook notebook.ipynb
```

#### Option 2: Use the Engine Directly

```python
from src.engine import PortfolioOptimizer

# Initialize optimizer
optimizer = PortfolioOptimizer(
    tickers=['SPY', 'TLT', 'GLD', 'BTC-USD'],
    start_date='2020-01-01',
    end_date='2024-12-31',
    risk_free_rate=0.05,
    max_weights={'BTC-USD': 0.15}
)

# Fetch data and calculate metrics
optimizer.fetch_data()
optimizer.calculate_returns()
optimizer.get_metrics()

# Find optimal portfolios
msr = optimizer.optimize_sharpe()
print(f"Max Sharpe Portfolio: {msr}")

min_vol = optimizer.optimize_min_volatility()
print(f"Min Volatility Portfolio: {min_vol}")
```

#### Option 3: Run the Engine Script

```bash
cd src
python engine.py
```

---

## 📈 Visualizations

The notebook produces the following visualizations:

1. **Normalized Price Series** - Track asset performance over time
2. **Correlation Heatmap** - Visualize asset relationships
3. **Monte Carlo Cloud** - 10,000 simulated portfolios colored by Sharpe Ratio
4. **Efficient Frontier** - Optimal risk-return trade-off curve
5. **Weight Comparison** - Bar chart of EW vs Optimized allocations
6. **Cumulative Returns** - Performance comparison over time
7. **KPI Dashboard** - Summary metrics comparison

---

## 🔧 Technical Features

- **Type Hints**: Full typing for better code clarity
- **Docstrings**: Comprehensive documentation
- **PEP8 Compliant**: Clean, readable code
- **Modular Design**: Reusable `PortfolioOptimizer` class
- **Reproducibility**: Seeded random number generation

---

## 📚 Dependencies

```
pandas>=1.5.0
numpy>=1.21.0
scipy>=1.9.0
matplotlib>=3.5.0
yfinance>=0.2.0
plotly>=5.10.0
jupyter>=1.0.0
```

---

## 🎓 Educational Context

### Why This Project Stands Out

1. **Benchmark Comparison**: Most projects show only optimized portfolios. We compare against Equal-Weight to prove value-add.

2. **Constraint Modeling**: Real-world portfolios have constraints. Our BTC 15% limit demonstrates regulatory compliance awareness.

3. **Modern Asset Mix**: Including Bitcoin alongside traditional assets shows understanding of alternative investments and diversification.

4. **Monte Carlo Simulation**: 10,000 iterations is industry standard for exploring the portfolio space.

---

## ⚠️ Disclaimer

This project is for **educational purposes only** and does not constitute investment advice. Past performance does not guarantee future results. Always consult a qualified financial advisor before making investment decisions.

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 👤 Author

**Quantitative Portfolio Manager**

*Strategic Asset Allocation & Risk Management*
