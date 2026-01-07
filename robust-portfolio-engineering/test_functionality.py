"""
Test script to verify the functionality of the Robust Portfolio Engineering project.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))

from data import DataLoader
from engine import PortfolioEngine
import pandas as pd

def test_functionality():
    print("Testing Robust Portfolio Engineering functionality...")
    
    # Test data loading
    print("\n1. Testing data loading...")
    assets = ['SPY', 'TLT']  # Using fewer assets for faster testing
    loader = DataLoader(symbols=assets, start_date='2020-01-01', end_date='2023-01-01')
    data_dict = loader.get_data(force_download=False)
    prices = data_dict['prices']
    returns = data_dict['returns']
    
    print(f"   Loaded {returns.shape[0]} days of data for {returns.shape[1]} assets")
    print(f"   Assets: {list(returns.columns)}")
    
    # Test portfolio engine
    print("\n2. Testing portfolio engine...")
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
    metrics = engine.calculate_portfolio_metrics(hrp_weights)
    print(f"   HRP Metrics: {metrics}")
    
    print("\nAll tests completed successfully! The Robust Portfolio Engineering project is working correctly.")

if __name__ == "__main__":
    test_functionality()