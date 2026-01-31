"""
FinLearner Demo: Technical Indicators
Run: python examples/examples-python/02_technical_indicators.py
"""
import warnings
warnings.filterwarnings('ignore')

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from finlearner import DataLoader, TechnicalIndicators
import matplotlib.pyplot as plt

print("=" * 60)
print("[*] FinLearner Demo: Technical Indicators")
print("=" * 60)

# Download data
ticker = 'AAPL'
print(f"\n[>] Downloading {ticker} data...")
df = DataLoader.download_data(ticker, start='2023-01-01', end='2024-01-01')
print(f"[+] Downloaded {len(df)} days")

# Add all indicators
print("\n[>] Adding technical indicators...")
ti = TechnicalIndicators(df.copy())
df_indicators = ti.add_all()

print(f"\n[+] Added indicators:")
new_cols = [c for c in df_indicators.columns if c not in df.columns]
for col in new_cols:
    print(f"   - {col}")

print(f"\nSample (last 5 rows):")
print(df_indicators[['Close', 'MA20', 'RSI', 'MACD']].tail())

# Plot
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# Price + Moving Averages & Bollinger Bands
axes[0].plot(df_indicators['Close'], label='Close', alpha=0.8)
axes[0].plot(df_indicators['MA20'], label='MA20', linestyle='--')
axes[0].plot(df_indicators['BB_Upper'], label='BB Upper', linestyle=':', color='gray')
axes[0].plot(df_indicators['BB_Lower'], label='BB Lower', linestyle=':', color='gray')
axes[0].fill_between(df_indicators.index, df_indicators['BB_Lower'], df_indicators['BB_Upper'], alpha=0.1)
axes[0].set_title('Price with Bollinger Bands')
axes[0].legend()
axes[0].set_ylabel('Price')

# RSI
axes[1].plot(df_indicators['RSI'], label='RSI', color='purple')
axes[1].axhline(70, color='red', linestyle='--', alpha=0.5, label='Overbought')
axes[1].axhline(30, color='green', linestyle='--', alpha=0.5, label='Oversold')
axes[1].set_title('RSI')
axes[1].legend()
axes[1].set_ylabel('RSI')

# MACD
axes[2].plot(df_indicators['MACD'], label='MACD', color='blue')
axes[2].plot(df_indicators['MACD_Signal'], label='Signal', color='orange')
macd_hist = df_indicators['MACD'] - df_indicators['MACD_Signal']
axes[2].bar(df_indicators.index, macd_hist, alpha=0.3, color='gray', label='Histogram')
axes[2].set_title('MACD')
axes[2].legend()
axes[2].set_ylabel('MACD')

plt.tight_layout()
plt.savefig('technical_indicators_plot.png', dpi=100)
print(f"\n[+] Plot saved to: technical_indicators_plot.png")

print("\n" + "=" * 60)
print("Demo Complete!")
print("=" * 60)
