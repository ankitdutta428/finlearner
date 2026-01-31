"""
FinLearner Demo: Deep Learning Models
Run: python examples/examples-python/03_deep_learning.py
"""
import warnings
warnings.filterwarnings('ignore')

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import matplotlib.pyplot as plt
from finlearner import (
    DataLoader,
    TimeSeriesPredictor,
    GRUPredictor,
    CNNLSTMPredictor,
    TransformerPredictor,
    EnsemblePredictor
)

print("=" * 60)
print("[*] FinLearner Demo: Deep Learning Models")
print("=" * 60)

# Download data
ticker = 'AAPL'
print(f"\n[>] Downloading {ticker} data...")
df = DataLoader.download_data(ticker, start='2023-01-01', end='2024-01-01')
print(f"[+] Downloaded {len(df)} days")

EPOCHS = 3  # Quick training for demo

# 1. LSTM
print("\n" + "-" * 40)
print("[1] Training LSTM Model...")
lstm = TimeSeriesPredictor(lookback_days=30)
lstm.fit(df, epochs=EPOCHS)
lstm_pred = lstm.predict(df)
print(f"   [+] LSTM predictions: {lstm_pred.shape}")

# 2. GRU
print("\n[2] Training GRU Model...")
gru = GRUPredictor(lookback_days=30, units=32)
gru.fit(df, epochs=EPOCHS)
gru_pred = gru.predict(df)
print(f"   [+] GRU predictions: {gru_pred.shape}")

# 3. CNN-LSTM
print("\n[3] Training CNN-LSTM Model...")
cnn_lstm = CNNLSTMPredictor(lookback_days=30, filters=16)
cnn_lstm.fit(df, epochs=EPOCHS)
cnn_pred = cnn_lstm.predict(df)
print(f"   [+] CNN-LSTM predictions: {cnn_pred.shape}")

# 4. Transformer
print("\n[4] Training Transformer Model...")
transformer = TransformerPredictor(lookback_days=30, num_heads=2, d_model=32)
transformer.fit(df, epochs=EPOCHS)
trans_pred = transformer.predict(df)
print(f"   [+] Transformer predictions: {trans_pred.shape}")

# 5. Ensemble
print("\n[5] Training Ensemble Model...")
ensemble = EnsemblePredictor(lookback_days=30)
ensemble.fit(df, epochs=EPOCHS)
ens_pred = ensemble.predict(df)
print(f"   [+] Ensemble predictions: {ens_pred.shape}")

# Compare predictions
print("\n" + "-" * 40)
print("[*] Model Comparison")

# Align lengths
min_len = min(len(lstm_pred), len(gru_pred), len(cnn_pred), len(trans_pred), len(ens_pred))
actual = df['Close'].values[-min_len:]

# Helper format
def fmt_mae(pred, actual):
    p = pred[-len(actual):].flatten()
    mae = np.mean(np.abs(p - actual))
    return f"{mae:.2f}"

print(f"\nMean Absolute Error:")
print(f"  LSTM:        {fmt_mae(lstm_pred, actual)}")
print(f"  GRU:         {fmt_mae(gru_pred, actual)}")
print(f"  CNN-LSTM:    {fmt_mae(cnn_pred, actual)}")
print(f"  Transformer: {fmt_mae(trans_pred, actual)}")
print(f"  Ensemble:    {fmt_mae(ens_pred, actual)}")

# Plot comparison
plt.figure(figsize=(14, 6))
plt.plot(actual, label='Actual', linewidth=2, color='black')
plt.plot(lstm_pred[-len(actual):].flatten(), label='LSTM', alpha=0.5)
plt.plot(gru_pred[-len(actual):].flatten(), label='GRU', alpha=0.5)
plt.plot(cnn_pred[-len(actual):].flatten(), label='CNN-LSTM', alpha=0.5)
plt.plot(trans_pred[-len(actual):].flatten(), label='Transformer', alpha=0.5)
plt.plot(ens_pred[-len(actual):].flatten(), label='Ensemble', alpha=0.5)
plt.title('Deep Learning Model Comparison')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.tight_layout()
plt.savefig('deep_learning_comparison.png', dpi=100)
print(f"\n[+] Plot saved to: deep_learning_comparison.png")

print("\n" + "=" * 60)
print("Demo Complete!")
print("=" * 60)
