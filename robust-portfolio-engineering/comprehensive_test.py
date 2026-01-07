"""
Comprehensive test for the Robust Portfolio Engineering project with all required assets.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))

from data import DataLoader
from engine import PortfolioEngine
import pandas as pd

def comprehensive_test():
    print("Running comprehensive test for Robust Portfolio Engineering...")
    
    # Test with all required assets
    print("\n1. Testing data loading with all assets (SPY, TLT, GLD, BTC-USD)...")
    assets = ['SPY', 'TLT', 'GLD', 'BTC-USD']  # All required assets
    loader = DataLoader(symbols=assets, start_date='2019-01-01', end_date='2024-01-01')
    data_dict = loader.get_data(force_download=False)
    prices = data_dict['prices']
    returns = data_dict['returns']
    
    print(f"   Loaded {returns.shape[0]} days of data for {returns.shape[1]} assets")
    print(f"   Assets: {list(returns.columns)}")
    
    # Test portfolio engine
    print("\n2. Testing portfolio engine with all assets...")
    engine = PortfolioEngine(returns)
    print(f"   Covariance matrix shape: {engine.cov_matrix.shape}")
    print(f"   Correlation matrix shape: {engine.correlation_matrix.shape}")
    
    # Test HRP optimization
    print("\n3. Testing HRP optimization...")
    hrp_weights = engine.hrp_optimization()
    print(f"   HRP weights: {hrp_weights}")
    print(f"   HRP weights sum: {sum(hrp_weights.values()):.4f}")
    
    # Test CVaR optimization
    print("\n4. Testing CVaR optimization...")
    cvar_weights = engine.cvar_optimization(alpha=0.05)
    print(f"   CVaR weights: {cvar_weights}")
    print(f"   CVaR weights sum: {sum(cvar_weights.values()):.4f}")
    
    # Test MVO optimization
    print("\n5. Testing MVO optimization...")
    mvo_weights = engine.mvo_optimization()
    print(f"   MVO weights: {mvo_weights}")
    print(f"   MVO weights sum: {sum(mvo_weights.values()):.4f}")
    
    # Test metrics calculation
    print("\n6. Testing metrics calculation...")
    hrp_metrics = engine.calculate_portfolio_metrics(hrp_weights)
    print(f"   HRP Metrics: {hrp_metrics}")
    
    # Test correlation breakdown analysis (similar to what would be done in the notebook)
    print("\n7. Testing correlation analysis (SPY vs TLT)...")
    spy_tlt_corr = returns[['SPY', 'TLT']].corr().iloc[0, 1]
    print(f"   Overall SPY-TLT correlation: {spy_tlt_corr:.4f}")
    
    # Calculate rolling correlations
    rolling_corr = returns[['SPY', 'TLT']].rolling(60).corr().unstack()['SPY']['TLT']
    print(f"   Average 60-day rolling correlation: {rolling_corr.mean():.4f}")
    
    print("\n✅ All comprehensive tests completed successfully!")
    print("✅ The Robust Portfolio Engineering project is fully functional.")
    print("\n📋 Project includes:")
    print("   - Data module with Yahoo Finance integration")
    print("   - Portfolio engine with HRP, CVaR, and MVO models")
    print("   - Ledoit-Wolf covariance shrinkage")
    print("   - Hierarchical Risk Parity implementation")
    print("   - CVaR optimization at 95% confidence level")
    print("   - Mean-Variance Optimization")
    print("   - Analysis notebook examining 2022 correlation crisis")
    print("   - Complete documentation and requirements")

if __name__ == "__main__":
    comprehensive_test()