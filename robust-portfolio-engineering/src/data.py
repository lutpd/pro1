"""
Data loading and preprocessing module for portfolio engineering.
Implements data fetching, log return calculation, and local storage.
"""
import yfinance as yf
import pandas as pd
import numpy as np
import os
from typing import List, Dict, Optional


class DataLoader:
    """
    A class for loading and preprocessing financial data.
    
    This class handles fetching historical price data from Yahoo Finance,
    calculating log returns, and saving/loading data locally to avoid
    repeated downloads.
    """
    
    def __init__(self, symbols: List[str], start_date: str = "2019-01-01", 
                 end_date: str = "2024-01-01"):
        """
        Initialize the DataLoader with symbols and date range.
        
        Args:
            symbols: List of ticker symbols to download
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
        """
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.data_path = "data/"
        
    def fetch_data(self) -> pd.DataFrame:
        """
        Fetch adjusted close prices for all symbols from Yahoo Finance.
        
        Returns:
            DataFrame with dates as index and symbols as columns
        """
        print(f"Fetching data for {self.symbols} from {self.start_date} to {self.end_date}")
        data = yf.download(self.symbols, start=self.start_date, end=self.end_date, 
                          progress=False, group_by='ticker', auto_adjust=True)
        
        # Handle case where only one symbol is provided
        if len(self.symbols) == 1:
            df = data[self.symbols[0]].copy()[['Close']]
            df.columns = self.symbols
        else:
            # Extract close prices - data structure is different when multiple symbols are provided
            if isinstance(data.columns, pd.MultiIndex):
                # Multiple symbols case: columns are MultiIndex with ('Ticker', 'Price')
                df = pd.DataFrame(index=data.index)
                for symbol in self.symbols:
                    df[symbol] = data[(symbol, 'Close')]
            else:
                # Single symbol case: column is just 'Close'
                df = data[['Close']].copy()
                df.columns = self.symbols
        
        return df.dropna()
    
    def calculate_log_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate log returns from price data.
        
        Args:
            prices: DataFrame with price data
            
        Returns:
            DataFrame with log returns
        """
        return np.log(prices / prices.shift(1)).dropna()
    
    def save_data(self, data: pd.DataFrame, filename: str) -> None:
        """
        Save data to local CSV file.
        
        Args:
            data: DataFrame to save
            filename: Name of the file to save to (without path)
        """
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
        
        filepath = os.path.join(self.data_path, filename)
        data.to_csv(filepath)
        print(f"Data saved to {filepath}")
    
    def load_data(self, filename: str) -> Optional[pd.DataFrame]:
        """
        Load data from local CSV file.
        
        Args:
            filename: Name of the file to load (without path)
            
        Returns:
            DataFrame with the loaded data or None if file doesn't exist
        """
        filepath = os.path.join(self.data_path, filename)
        if os.path.exists(filepath):
            print(f"Loading data from {filepath}")
            return pd.read_csv(filepath, index_col=0, parse_dates=True)
        else:
            print(f"File {filepath} does not exist.")
            return None
    
    def get_data(self, force_download: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Get price and return data, with option to force re-download.
        
        Args:
            force_download: If True, re-download data even if local file exists
            
        Returns:
            Dictionary with 'prices' and 'returns' DataFrames
        """
        prices_filename = f"prices_{'_'.join(self.symbols)}.csv"
        
        if not force_download:
            prices = self.load_data(prices_filename)
        else:
            prices = None
            
        if prices is None:
            prices = self.fetch_data()
            self.save_data(prices, prices_filename)
        
        returns = self.calculate_log_returns(prices)
        
        return {
            'prices': prices,
            'returns': returns
        }