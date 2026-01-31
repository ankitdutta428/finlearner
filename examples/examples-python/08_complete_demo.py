"""
FinLearner Demo: Complete Demo - All Modules
Run: python examples/examples-python/08_complete_demo.py
"""
import warnings
warnings.filterwarnings('ignore')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow info messages

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import matplotlib.pyplot as plt

print("=" * 60)
print("[*] FinLearner Complete Demo - All Modules")
print("=" * 60)

# Import modules
print("\n[>] Importing FinLearner modules...")
from finlearner import (
    DataLoader,
    TechnicalIndicators,
    GradientBoostPredictor,
    VAEAnomalyDetector,
    RiskMetrics,
    PortfolioOptimizer,
    BlackLittermanOptimizer,
    RiskParityOptimizer,
)
print("[+] All modules imported!")

# ========================================
# 1. DATA LOADING
# ========================================
print("\n" + "=" * 50)
print("[1] DATA LOADING")
print("=" * 50)

ticker = 'AAPL'
df = DataLoader.download_data(ticker, start='2023-01-01', end='2024-01-01')
print(f"[+] Downloaded {len(df)} days of {ticker} data")
print(f"    Columns: {df.columns.tolist()}")

# ========================================
# 2. TECHNICAL INDICATORS
# ========================================
print("\n" + "=" * 50)
print("[2] TECHNICAL INDICATORS")
print("=" * 50)

ti = TechnicalIndicators(df.copy())
df_tech = ti.add_all()
print(f"[+] Added {len(df_tech.columns) - len(df.columns)} indicators")
print(f"    RSI (latest): {df_tech['RSI'].iloc[-1]:.2f}")
print(f"    MACD (latest): {df_tech['MACD'].iloc[-1]:.4f}")
print(f"    All indicators: {[c for c in df_tech.columns if c not in df.columns]}")

# ========================================
# 3. GRADIENT BOOSTING
# ========================================
print("\n" + "=" * 50)
print("[3] GRADIENT BOOSTING (XGBoost)")
print("=" * 50)

xgb = GradientBoostPredictor(backend='xgboost', n_estimators=50)
xgb.fit(df, verbose=False)
xgb_pred = xgb.predict(df)
print(f"[+] XGBoost trained - predictions shape: {xgb_pred.shape}")

importance = xgb.feature_importance().head(5)
print(f"    Top 5 features:")
for _, row in importance.iterrows():
    print(f"      - {row['feature']}: {row['importance']:.4f}")

# ========================================
# 4. ANOMALY DETECTION
# ========================================
print("\n" + "=" * 50)
print("[4] ANOMALY DETECTION (VAE)")
print("=" * 50)

vae = VAEAnomalyDetector(lookback_days=20, latent_dim=4)
vae.fit(df, epochs=20, verbose=0)
anomaly_df = vae.get_anomalies(df)
n_anomalies = anomaly_df['Is_Anomaly'].sum()
print(f"[+] VAE trained - threshold: {vae.reconstruction_threshold:.6f}")
print(f"    Anomalies detected: {n_anomalies} ({n_anomalies/len(anomaly_df)*100:.1f}%)")

# ========================================
# 5. RISK METRICS
# ========================================
print("\n" + "=" * 50)
print("[5] RISK METRICS")
print("=" * 50)

risk = RiskMetrics.from_prices(df['Close'])
var_95 = risk.historical_var(0.95)
cvar_95 = risk.cvar(0.95)
max_dd, dd_series = risk.max_drawdown()  # Returns tuple

print(f"[+] Risk calculated:")
print(f"    VaR (95%):    {var_95:.2%}")
print(f"    CVaR (95%):   {cvar_95:.2%}")
print(f"    Max Drawdown: {max_dd:.2%}")
print(f"    Calmar Ratio: {risk.calmar_ratio():.2f}")

# ========================================
# 6. PORTFOLIO OPTIMIZATION
# ========================================
print("\n" + "=" * 50)
print("[6] PORTFOLIO OPTIMIZATION")
print("=" * 50)

tickers = ['AAPL', 'GOOG', 'MSFT']

# Markowitz
print("[>] Running Markowitz...")
mk = PortfolioOptimizer(tickers=tickers, start='2023-01-01', end='2024-01-01')
_, mk_alloc, _ = mk.optimize(num_portfolios=1000)
print(f"    Max Sharpe weights: {dict(zip(mk_alloc.index, mk_alloc['allocation'].round(2)))}")

# Black-Litterman
print("[>] Running Black-Litterman...")
bl = BlackLittermanOptimizer(tickers=tickers, start='2023-01-01', end='2024-01-01')
bl_alloc, bl_metrics = bl.optimize(views={'AAPL': 0.15})
print(f"    B-L weights: {dict(zip(bl_alloc.index, bl_alloc['Weight'].round(2)))}")

# Risk Parity
print("[>] Running Risk Parity...")
rp = RiskParityOptimizer(tickers=tickers, start='2023-01-01', end='2024-01-01')
rp_alloc, rp_metrics = rp.optimize()
print(f"    Risk Parity weights: {dict(zip(rp_alloc.index, rp_alloc['Weight'].round(2)))}")

# ========================================
# SUMMARY
# ========================================
print("\n" + "=" * 60)
print("[*] DEMO SUMMARY")
print("=" * 60)
print(f"""
[+] Data Loading:       {len(df)} days loaded for {ticker}
[+] Technical:          {len(df_tech.columns) - len(df.columns)} indicators added
[+] Gradient Boosting:  XGBoost trained ({xgb_pred.shape[0]} predictions)
[+] Anomaly Detection:  {n_anomalies} anomalies found
[+] Risk Metrics:       VaR={var_95:.2%}, MaxDD={max_dd:.2%}
[+] Portfolio:          3 optimization strategies tested

All FinLearner modules working correctly!
""")

print("=" * 60)
print("Demo Complete!")
print("=" * 60)
