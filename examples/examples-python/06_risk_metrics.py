"""
FinLearner Demo: Risk Metrics
Run: python examples/examples-python/06_risk_metrics.py
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
    RiskMetrics,
    historical_var,
    parametric_var,
    monte_carlo_var,
    cvar,
    max_drawdown
)

print("=" * 60)
print("[*] FinLearner Demo: Risk Metrics")
print("=" * 60)

# Download data
ticker = 'AAPL'
print(f"\n[>] Downloading {ticker} data...")
df = DataLoader.download_data(ticker, start='2023-01-01', end='2024-01-01')
print(f"[+] Downloaded {len(df)} days")

# Create RiskMetrics
print("\n[>] Calculating risk metrics...")
risk = RiskMetrics.from_prices(df['Close'])

# Summary
print("\n" + "-" * 40)
print("[*] Risk Summary (95% confidence)")
print("-" * 40)
summary = risk.summary(confidence=0.95)
print(summary)

# Value at Risk
print("\n" + "-" * 40)
print("[1] Value at Risk (VaR)")
print("-" * 40)
print("\n95% Confidence:")
var_hist = risk.historical_var(0.95)
var_param = risk.parametric_var(0.95)
var_mc = risk.monte_carlo_var(0.95, simulations=10000)
var_cf = risk.cornish_fisher_var(0.95)

print(f"  Historical:     {var_hist:.4f} ({var_hist*100:.2f}%)")
print(f"  Parametric:     {var_param:.4f} ({var_param*100:.2f}%)")
print(f"  Monte Carlo:    {var_mc:.4f} ({var_mc*100:.2f}%)")
print(f"  Cornish-Fisher: {var_cf:.4f} ({var_cf*100:.2f}%)")

print("\n99% Confidence:")
print(f"  Historical:     {risk.historical_var(0.99):.4f}")
print(f"  Parametric:     {risk.parametric_var(0.99):.4f}")

# CVaR
print("\n" + "-" * 40)
print("[2] Conditional VaR (Expected Shortfall)")
print("-" * 40)
cvar_hist = risk.cvar(0.95, method='historical')
cvar_param = risk.cvar(0.95, method='parametric')
print(f"  Historical CVaR (95%): {cvar_hist:.4f} ({cvar_hist*100:.2f}%)")
print(f"  Parametric CVaR (95%): {cvar_param:.4f} ({cvar_param*100:.2f}%)")

# Drawdown
print("\n" + "-" * 40)
print("[3] Drawdown Metrics")
print("-" * 40)
max_dd, dd_series = risk.max_drawdown()
calmar = risk.calmar_ratio()
print(f"  Max Drawdown:  {max_dd:.4f} ({max_dd*100:.2f}%)")
print(f"  Calmar Ratio:  {calmar:.4f}")

# Standalone functions
print("\n" + "-" * 40)
print("[4] Standalone Functions")
print("-" * 40)
returns = df['Close'].pct_change().dropna()
print(f"  historical_var(returns, 0.95): {historical_var(returns, 0.95):.4f}")
print(f"  parametric_var(returns, 0.95): {parametric_var(returns, 0.95):.4f}")
print(f"  monte_carlo_var(returns, 0.95): {monte_carlo_var(returns, 0.95):.4f}")
print(f"  cvar(returns, 0.95):           {cvar(returns, 0.95):.4f}")
# Fixed: unpack tuple and pass prices, not returns
func_max_dd, _ = max_drawdown(df['Close'])
print(f"  max_drawdown(prices):          {func_max_dd:.4f}")

# Plot
cumulative = (1 + returns).cumprod()

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Cumulative returns
axes[0, 0].plot(cumulative, color='steelblue')
axes[0, 0].set_title('Cumulative Returns')
axes[0, 0].set_ylabel('Growth Factor')

# 2. Drawdown over time
axes[0, 1].fill_between(dd_series.index, dd_series.values, 0, color='red', alpha=0.3)
axes[0, 1].plot(dd_series, color='red')
axes[0, 1].set_title('Drawdown Over Time')
axes[0, 1].set_ylabel('Drawdown')

# 3. Returns distribution with VaR
axes[1, 0].hist(returns, bins=50, color='steelblue', alpha=0.7, edgecolor='black', density=True)
axes[1, 0].axvline(-var_hist, color='red', linestyle='--', label=f'VaR 95% = {var_hist:.2%}')
axes[1, 0].axvline(-cvar_hist, color='darkred', linestyle=':', label=f'CVaR 95% = {cvar_hist:.2%}')
axes[1, 0].set_title('Returns Distribution with VaR/CVaR')
axes[1, 0].set_xlabel('Daily Return')
axes[1, 0].legend()

# 4. VaR comparison
methods = ['Historical', 'Parametric', 'Monte Carlo', 'Cornish-Fisher']
values = [var_hist, var_param, var_mc, var_cf]
colors = ['steelblue', 'forestgreen', 'orange', 'purple']
axes[1, 1].bar(methods, values, color=colors, alpha=0.8)
axes[1, 1].set_title('VaR Method Comparison (95%)')
axes[1, 1].set_ylabel('VaR')

plt.tight_layout()
plt.savefig('risk_metrics_plot.png', dpi=100)
print(f"\n[+] Plot saved to: risk_metrics_plot.png")

print("\n" + "=" * 60)
print("Demo Complete!")
print("=" * 60)
