"""
FinLearner Demo: Gradient Boosting (XGBoost / LightGBM)
Run: python examples/examples-python/04_gradient_boosting.py
"""
import warnings
warnings.filterwarnings('ignore')

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import matplotlib.pyplot as plt
from finlearner import DataLoader, GradientBoostPredictor

print("=" * 60)
print("[*] FinLearner Demo: Gradient Boosting")
print("=" * 60)

# Download data
ticker = 'AAPL'
print(f"\n[>] Downloading {ticker} data...")
df = DataLoader.download_data(ticker, start='2023-01-01', end='2024-01-01')
print(f"[+] Downloaded {len(df)} days")

# 1. XGBoost
print("\n" + "-" * 40)
print("[1] Training XGBoost Model...")
xgb = GradientBoostPredictor(
    backend='xgboost',
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1
)
xgb.fit(df, verbose=False)
xgb_pred = xgb.predict(df)
print(f"   [+] XGBoost predictions: {xgb_pred.shape}")

# Feature importance
print("\n   [*] Top 10 Features (XGBoost):")
importance = xgb.feature_importance()
for i, row in importance.head(10).iterrows():
    print(f"      {row['feature']:20s} : {row['importance']:.4f}")

# 2. LightGBM
print("\n" + "-" * 40)
print("[2] Training LightGBM Model...")
lgb = GradientBoostPredictor(
    backend='lightgbm',
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1
)
lgb.fit(df, verbose=False)
lgb_pred = lgb.predict(df)
print(f"   [+] LightGBM predictions: {lgb_pred.shape}")

# Compare predictions
print("\n" + "-" * 40)
print("[*] Model Comparison")

# Calculate MAE
def calc_mae(pred, actual):
    min_len = min(len(pred), len(actual))
    return np.mean(np.abs(pred[:min_len].flatten() - actual[-min_len:]))

actual = df['Close'].values
print(f"\nMean Absolute Error:")
print(f"  XGBoost:  {calc_mae(xgb_pred, actual):.2f}")
print(f"  LightGBM: {calc_mae(lgb_pred, actual):.2f}")

# Plot feature importance
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# XGBoost
top_xgb = importance.head(10)
axes[0].barh(top_xgb['feature'], top_xgb['importance'], color='steelblue')
axes[0].set_xlabel('Importance')
axes[0].set_title('XGBoost Feature Importance')
axes[0].invert_yaxis()

# LightGBM
lgb_importance = lgb.feature_importance()
top_lgb = lgb_importance.head(10)
axes[1].barh(top_lgb['feature'], top_lgb['importance'], color='forestgreen')
axes[1].set_xlabel('Importance')
axes[1].set_title('LightGBM Feature Importance')
axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig('gradient_boosting_importance.png', dpi=100)
print(f"\n[+] Plot saved to: gradient_boosting_importance.png")

print("\n" + "=" * 60)
print("Demo Complete!")
print("=" * 60)
