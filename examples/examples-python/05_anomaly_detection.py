"""
FinLearner Demo: VAE Anomaly Detection
Run: python examples/examples-python/05_anomaly_detection.py
"""
import warnings
warnings.filterwarnings('ignore')

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import matplotlib.pyplot as plt
from finlearner import DataLoader, VAEAnomalyDetector

print("=" * 60)
print("[*] FinLearner Demo: VAE Anomaly Detection")
print("=" * 60)

# Download data
ticker = 'AAPL'
print(f"\n[>] Downloading {ticker} data...")
df = DataLoader.download_data(ticker, start='2023-01-01', end='2024-01-01')
print(f"[+] Downloaded {len(df)} days")

# Create VAE
print("\n[>] Building VAE Anomaly Detector...")
vae = VAEAnomalyDetector(
    lookback_days=20,
    latent_dim=8,
    hidden_dims=(64, 32)
)

# Train
print("[>] Training VAE (this may take a moment)...")
vae.fit(df, epochs=30, batch_size=16, verbose=0)
print(f"[+] Training complete!")
print(f"   Reconstruction threshold: {vae.reconstruction_threshold:.6f}")

# Detect anomalies
print("\n[>] Detecting anomalies...")
anomaly_scores = vae.detect_anomalies(df)
anomaly_df = vae.get_anomalies(df, percentile=95)

num_anomalies = anomaly_df['Is_Anomaly'].sum()
print(f"[+] Detected {num_anomalies} anomalies ({num_anomalies/len(anomaly_df)*100:.1f}%)")

# Show anomalies
print("\n[*] Top 10 Anomalies (highest scores):")
top_anomalies = anomaly_df[anomaly_df['Is_Anomaly']].nlargest(10, 'Anomaly_Score')
print(top_anomalies[['Close', 'Anomaly_Score']])

# Get latent representation
print("\n[>] Extracting latent representation...")
latent = vae.get_latent_representation(df)
print(f"[+] Latent shape: {latent.shape}")

# Plot
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Price with anomalies
ax1 = axes[0, 0]
ax1.plot(anomaly_df['Close'], label='Price', color='steelblue')
anomalies = anomaly_df[anomaly_df['Is_Anomaly']]
ax1.scatter(anomalies.index, anomalies['Close'], color='red', s=50, label='Anomaly', zorder=5)
ax1.set_title('Price with Detected Anomalies')
ax1.legend()

# 2. Anomaly scores
ax2 = axes[0, 1]
ax2.plot(anomaly_df.index, anomaly_df['Anomaly_Score'], color='orange')
ax2.axhline(vae.reconstruction_threshold, color='red', linestyle='--', label='Threshold')
ax2.set_title('Anomaly Scores')
ax2.legend()

# 3. Score distribution
ax3 = axes[1, 0]
ax3.hist(anomaly_df['Anomaly_Score'], bins=50, color='steelblue', alpha=0.7, edgecolor='black')
ax3.axvline(vae.reconstruction_threshold, color='red', linestyle='--', label='Threshold')
ax3.set_title('Anomaly Score Distribution')
ax3.set_xlabel('Score')
ax3.set_ylabel('Frequency')
ax3.legend()

# 4. Latent space (first 2 dims)
ax4 = axes[1, 1]
scatter = ax4.scatter(latent[:, 0], latent[:, 1], c=range(len(latent)), cmap='viridis', alpha=0.6, s=20)
plt.colorbar(scatter, ax=ax4, label='Time Index')
ax4.set_xlabel('Latent Dim 1')
ax4.set_ylabel('Latent Dim 2')
ax4.set_title('VAE Latent Space')

plt.tight_layout()
plt.savefig('anomaly_detection_plot.png', dpi=100)
print(f"\n[+] Plot saved to: anomaly_detection_plot.png")

print("\n" + "=" * 60)
print("Demo Complete!")
print("=" * 60)
