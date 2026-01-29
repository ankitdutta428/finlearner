"""
Shared pytest fixtures for finlearner tests.
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch


@pytest.fixture
def sample_ohlcv_data():
    """Creates a realistic dummy OHLCV dataframe with 100 days of stock data."""
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    
    # Generate realistic price movements
    base_price = 150.0
    returns = np.random.normal(0.001, 0.02, 100)
    close_prices = base_price * np.cumprod(1 + returns)
    
    data = pd.DataFrame({
        'Open': close_prices * np.random.uniform(0.99, 1.01, 100),
        'High': close_prices * np.random.uniform(1.01, 1.03, 100),
        'Low': close_prices * np.random.uniform(0.97, 0.99, 100),
        'Close': close_prices,
        'Adj Close': close_prices,
        'Volume': np.random.randint(1000000, 10000000, 100)
    }, index=dates)
    
    return data


@pytest.fixture
def sample_multi_ticker_data():
    """Creates OHLCV data for multiple tickers (for portfolio tests)."""
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    tickers = ['AAPL', 'GOOG', 'MSFT']
    
    multi_data = {}
    for i, ticker in enumerate(tickers):
        base_price = 100 + i * 50
        returns = np.random.normal(0.001, 0.02, 100)
        close_prices = base_price * np.cumprod(1 + returns)
        multi_data[ticker] = close_prices
    
    return pd.DataFrame(multi_data, index=dates)


@pytest.fixture
def sample_predictions_csv(tmp_path):
    """Creates a temporary CSV file for utils testing."""
    csv_path = tmp_path / "predictions.csv"
    data = pd.DataFrame({
        'Close': [100.0, 105.0, 110.0, 95.0, 100.0],
        'Predicted': [101.0, 103.0, 115.0, 90.0, 102.0]
    })
    data.to_csv(csv_path, index=False)
    return str(csv_path)


@pytest.fixture
def mock_yfinance_download():
    """Fixture to mock yfinance download function."""
    with patch('yfinance.download') as mock_download:
        yield mock_download
