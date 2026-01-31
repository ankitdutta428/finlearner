"""
FinLearner Demo: Data Loading
Run: python examples/examples-python/01_data_loading.py
"""
import warnings
warnings.filterwarnings('ignore')

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from finlearner import DataLoader
import matplotlib.pyplot as plt

print("=" * 60)
print("[*] FinLearner Demo: Data Loading")
print("=" * 60)

# Download stock data
ticker = 'AAPL'
print(f"\n[>] Downloading {ticker} data...")
df = DataLoader.download_data(ticker, start='2023-01-01', end='2024-01-01')

print(f"[+] Downloaded {len(df)} days of data")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nFirst 5 rows:")
print(df.head())

print(f"\nLast 5 rows:")
print(df.tail())

print(f"\n[*] Basic Statistics:")
print(df['Close'].describe())

# Quick plot
plt.figure(figsize=(12, 4))
plt.plot(df['Close'], label=ticker)
plt.title(f'{ticker} Closing Prices')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.legend()
plt.tight_layout()
plt.savefig('data_loading_plot.png', dpi=100)
print(f"\n[+] Plot saved to: data_loading_plot.png")

print("\n" + "=" * 60)
print("Demo Complete!")
print("=" * 60)
