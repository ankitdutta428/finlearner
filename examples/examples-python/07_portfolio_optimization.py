"""
FinLearner Demo: Portfolio Optimization
Run: python examples/examples-python/07_portfolio_optimization.py
"""
import warnings
warnings.filterwarnings('ignore')

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import matplotlib.pyplot as plt
from finlearner import (
    PortfolioOptimizer,
    BlackLittermanOptimizer,
    RiskParityOptimizer
)

print("=" * 60)
print("[*] FinLearner Demo: Portfolio Optimization")
print("=" * 60)

# Portfolio of tech stocks
tickers = ['AAPL', 'GOOG', 'MSFT', 'AMZN', 'META']
start = '2023-01-01'
end = '2024-01-01'

# 1. Markowitz Mean-Variance
print("\n" + "-" * 40)
print("[1] Markowitz Mean-Variance Optimization")
print("-" * 40)

print(f"\n[>] Downloading data for: {tickers}")
markowitz = PortfolioOptimizer(tickers=tickers, start=start, end=end)

print("[>] Running Monte Carlo optimization...")
_, allocations, metrics = markowitz.optimize(num_portfolios=5000)

print("\n[*] Optimal Portfolio (Max Sharpe):")
print(allocations)

print(f"\n[*] Metrics (std, return, sharpe):")
print(metrics)

# 2. Black-Litterman
print("\n" + "-" * 40)
print("[2] Black-Litterman Optimization (with Views)")
print("-" * 40)

bl = BlackLittermanOptimizer(tickers=tickers, start=start, end=end)

# Define views: AAPL will return 20%, MSFT 15%
views = {'AAPL': 0.20, 'MSFT': 0.15}
print(f"\n[>] Investor Views:")
for ticker, view in views.items():
    print(f"   {ticker}: {view:.0%} expected return")

bl_alloc, bl_metrics = bl.optimize(views=views)

print("\n[*] Black-Litterman Portfolio:")
print(bl_alloc)

print(f"\n[*] Metrics:")
print(f"  Expected Return: {bl_metrics['expected_return']:.2%}")
print(f"  Volatility:      {bl_metrics['volatility']:.2%}")
print(f"  Sharpe Ratio:    {bl_metrics['sharpe_ratio']:.2f}")

# 3. Risk Parity
print("\n" + "-" * 40)
print("[3] Risk Parity Optimization")
print("-" * 40)

rp = RiskParityOptimizer(tickers=tickers, start=start, end=end)

print("\n[>] Optimizing for equal risk contribution...")
rp_alloc, rp_metrics = rp.optimize()

print("\n[*] Risk Parity Portfolio:")
print(rp_alloc)

print("\n[*] Metrics:")
print(rp_metrics)

# Compare allocations
print("\n" + "-" * 40)
print("[*] Allocation Comparison")
print("-" * 40)

print("\n         Markowitz | Black-Litterman | Risk Parity")
print("         " + "-" * 45)
for ticker in tickers:
    mk_w = allocations.loc[ticker, 'allocation'] if ticker in allocations.index else 0
    bl_w = bl_alloc.loc[ticker, 'Weight'] if ticker in bl_alloc.index else 0
    rp_w = rp_alloc.loc[ticker, 'Weight'] if ticker in rp_alloc.index else 0
    print(f"{ticker:5s}:   {mk_w:8.1%}    |    {bl_w:8.1%}     |   {rp_w:8.1%}")

# Plot
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Markowitz
axes[0].pie(allocations['allocation'], labels=allocations.index, autopct='%1.1f%%', 
            colors=plt.cm.Set3.colors[:len(tickers)])
axes[0].set_title('Markowitz\n(Max Sharpe)')

# Black-Litterman
axes[1].pie(bl_alloc['Weight'], labels=bl_alloc.index, autopct='%1.1f%%',
            colors=plt.cm.Set3.colors[:len(tickers)])
axes[1].set_title('Black-Litterman\n(with Views)')

# Risk Parity
axes[2].pie(rp_alloc['Weight'], labels=rp_alloc.index, autopct='%1.1f%%',
            colors=plt.cm.Set3.colors[:len(tickers)])
axes[2].set_title('Risk Parity\n(Equal Risk)')

plt.suptitle('Portfolio Allocation Comparison', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('portfolio_optimization_plot.png', dpi=100)
print(f"\n[+] Plot saved to: portfolio_optimization_plot.png")

print("\n" + "=" * 60)
print("Demo Complete!")
print("=" * 60)
