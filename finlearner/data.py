import yfinance as yf
import pandas as pd
from typing import Optional, List, Union

class DataLoader:
    """
    A robust data loader for fetching financial data.
    """
    @staticmethod
    def download_data(ticker: Union[str, List[str]], start: str, end: str) -> pd.DataFrame:
        """
        Downloads data from Yahoo Finance.
        
        Args:
            ticker: Single ticker string or list of tickers.
            start: Start date 'YYYY-MM-DD'.
            end: End date 'YYYY-MM-DD'.
        
        Returns:
            pd.DataFrame: Adjusted Close prices and other data.
        """
        print(f"Fetching data for {ticker}...")
        data = yf.download(ticker, start=start, end=end, progress=False)
        if data.empty:
            raise ValueError(f"No data found for {ticker}. Check ticker symbol or date range.")
        
        # Flatten multi-level columns (yfinance returns MultiIndex for single tickers)
        if isinstance(data.columns, pd.MultiIndex):
            # For single ticker, take the first level only
            if isinstance(ticker, str):
                data.columns = data.columns.get_level_values(0)
            else:
                # For multiple tickers, keep as-is or flatten appropriately
                pass
        
        return data

    @staticmethod
    def download_options_chain(ticker: str, expiration: str = None) -> dict:
        """
        Downloads options chain data from Yahoo Finance.
        
        Args:
            ticker: Stock ticker symbol.
            expiration: Optional specific expiration date 'YYYY-MM-DD'.
                       If None, uses nearest available expiration.
        
        Returns:
            dict: {
                'calls': pd.DataFrame with call options data,
                'puts': pd.DataFrame with put options data,
                'underlying_price': float current stock price,
                'expiration': str selected expiration date,
                'available_expirations': list of all available expiration dates
            }
        
        Note:
            Options chain data is a current snapshot only.
            Historical options data requires paid data sources.
        """
        print(f"Fetching options chain for {ticker}...")
        stock = yf.Ticker(ticker)
        
        # Get available expiration dates
        available_expirations = stock.options
        if not available_expirations:
            raise ValueError(f"No options data available for {ticker}.")
        
        # Select expiration
        if expiration:
            if expiration not in available_expirations:
                raise ValueError(f"Expiration {expiration} not available. Choose from: {available_expirations}")
            selected_exp = expiration
        else:
            selected_exp = available_expirations[0]  # Nearest expiration
        
        # Fetch options chain
        chain = stock.option_chain(selected_exp)
        
        # Get current underlying price
        try:
            underlying_price = stock.info.get('regularMarketPrice') or stock.info.get('currentPrice')
            if underlying_price is None:
                # Fallback to last close from history
                hist = stock.history(period='1d')
                underlying_price = hist['Close'].iloc[-1] if not hist.empty else None
        except Exception:
            underlying_price = None
        
        return {
            'calls': chain.calls,
            'puts': chain.puts,
            'underlying_price': underlying_price,
            'expiration': selected_exp,
            'available_expirations': list(available_expirations)
        }