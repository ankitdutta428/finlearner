"""
Options Trading Demo with Real Options Chain Data

This example demonstrates how to:
1. Fetch real options chain data using DataLoader.download_options_chain()
2. Analyze available strikes and expirations
3. Use actual bid/ask prices for options pricing
4. Compare market prices to theoretical model prices
5. Simulate an options trading strategy using real chain data

Note: yfinance provides CURRENT SNAPSHOT only, not historical data.
For true backtesting, a paid data source (CBOE, Polygon.io) is needed.
"""

import numpy as np
import pandas as pd
from datetime import datetime
from finlearner import DataLoader, BlackScholesMerton, BinomialTreePricing


def analyze_options_chain(ticker: str = 'TCS.NS'):
    """
    Fetch and analyze a real options chain.
    """
    print(f"🚀 Fetching Options Chain for {ticker}...\n")
    
    # 1. Fetch Options Chain
    # ----------------------
    try:
        chain = DataLoader.download_options_chain(ticker)
    except Exception as e:
        print(f"❌ Error fetching options chain: {e}")
        return
    
    calls = chain['calls']
    puts = chain['puts']
    spot = chain['underlying_price']
    expiration = chain['expiration']
    all_expirations = chain['available_expirations']
    
    print(f"✅ Successfully fetched options chain!")
    print(f"   Underlying Price: ${spot:.2f}")
    print(f"   Selected Expiration: {expiration}")
    print(f"   Available Expirations: {len(all_expirations)} dates")
    print(f"   Total Calls: {len(calls)}")
    print(f"   Total Puts: {len(puts)}")
    
    # 2. Analyze Call Options
    # -----------------------
    print("\n" + "="*60)
    print("📊 CALL OPTIONS ANALYSIS")
    print("="*60)
    
    # Find ATM options (closest to spot price)
    calls['distance_from_spot'] = abs(calls['strike'] - spot)
    atm_call = calls.loc[calls['distance_from_spot'].idxmin()]
    
    print(f"\n🎯 ATM Call Option (Strike ${atm_call['strike']:.2f}):")
    print(f"   Last Price:      ${atm_call['lastPrice']:.2f}")
    print(f"   Bid:             ${atm_call['bid']:.2f}")
    print(f"   Ask:             ${atm_call['ask']:.2f}")
    print(f"   Implied Vol:     {atm_call['impliedVolatility']*100:.1f}%")
    print(f"   Volume:          {atm_call['volume']}")
    print(f"   Open Interest:   {atm_call['openInterest']}")
    
    # 3. Compare Market vs Model Prices
    # ----------------------------------
    print("\n" + "="*60)
    print("🔬 MARKET vs MODEL PRICE COMPARISON")
    print("="*60)
    
    # Calculate time to expiry
    exp_date = datetime.strptime(expiration, '%Y-%m-%d')
    today = datetime.now()
    T = max((exp_date - today).days, 1) / 365.0
    
    # Use market IV for pricing
    iv = atm_call['impliedVolatility']
    r = 0.045  # Risk-free rate (approximate)
    
    # Black-Scholes Price
    bsm = BlackScholesMerton(spot, atm_call['strike'], T, r, iv)
    bsm_call_price = bsm.price('call')
    bsm_put_price = bsm.price('put')
    
    # Binomial Tree Price
    binom = BinomialTreePricing(spot, atm_call['strike'], T, r, iv, N=100, option_style='american')
    binom_call_price = binom.price('call')
    binom_put_price = binom.price('put')
    
    print(f"\nATM Strike: ${atm_call['strike']:.2f} | Spot: ${spot:.2f} | T: {T*365:.0f} days | IV: {iv*100:.1f}%")
    print("-" * 60)
    print(f"{'Pricing Method':<25} {'Call Price':<15} {'Put Price':<15}")
    print("-" * 60)
    print(f"{'Market (Mid)':<25} ${(atm_call['bid']+atm_call['ask'])/2:.2f}{'':>10} N/A")
    print(f"{'Black-Scholes':<25} ${bsm_call_price:.2f}{'':>10} ${bsm_put_price:.2f}")
    print(f"{'Binomial Tree (N=100)':<25} ${binom_call_price:.2f}{'':>10} ${binom_put_price:.2f}")
    
    # 4. Display Option Chain Table
    # ------------------------------
    print("\n" + "="*60)
    print("📋 TOP CALLS BY VOLUME")
    print("="*60)
    
    display_cols = ['strike', 'lastPrice', 'bid', 'ask', 'volume', 'openInterest', 'impliedVolatility']
    top_calls = calls.nlargest(10, 'volume')[display_cols].copy()
    top_calls['impliedVolatility'] = top_calls['impliedVolatility'].apply(lambda x: f"{x*100:.1f}%")
    print(top_calls.to_string(index=False))
    
    return chain


def simulate_options_strategy(ticker: str = 'TCS.NS'):
    """
    Simulate an options trading strategy using real chain data.
    This is a forward-looking simulation using current prices.
    """
    print("\n" + "="*60)
    print("💹 SIMULATED OPTIONS STRATEGY")
    print("="*60)
    
    try:
        chain = DataLoader.download_options_chain(ticker)
    except Exception as e:
        print(f"❌ Error: {e}")
        return
    
    calls = chain['calls']
    puts = chain['puts']
    spot = chain['underlying_price']
    expiration = chain['expiration']
    
    # Strategy: Bull Call Spread
    # Buy ATM Call, Sell OTM Call
    
    calls_sorted = calls.sort_values('strike')
    atm_idx = (calls_sorted['strike'] - spot).abs().idxmin()
    atm_call = calls_sorted.loc[atm_idx]
    
    # Find next strike up (OTM)
    otm_candidates = calls_sorted[calls_sorted['strike'] > atm_call['strike']]
    if len(otm_candidates) == 0:
        print("❌ Cannot find OTM strike for spread")
        return
    
    otm_call = otm_candidates.iloc[0]
    
    # Calculate spread
    long_cost = atm_call['ask']  # Pay ask to buy
    short_credit = otm_call['bid']  # Receive bid to sell
    net_debit = long_cost - short_credit
    max_profit = otm_call['strike'] - atm_call['strike'] - net_debit
    max_loss = net_debit
    breakeven = atm_call['strike'] + net_debit
    
    print(f"\n📈 Bull Call Spread on {ticker}")
    print(f"   Underlying: ${spot:.2f}")
    print(f"   Expiration: {expiration}")
    print("-" * 40)
    print(f"   BUY  {atm_call['strike']:.0f} Call @ ${long_cost:.2f}")
    print(f"   SELL {otm_call['strike']:.0f} Call @ ${short_credit:.2f}")
    print("-" * 40)
    print(f"   Net Debit:    ${net_debit:.2f} per share (${net_debit*100:.2f} per contract)")
    print(f"   Max Profit:   ${max_profit:.2f} per share (${max_profit*100:.2f} per contract)")
    print(f"   Max Loss:     ${max_loss:.2f} per share (${max_loss*100:.2f} per contract)")
    print(f"   Breakeven:    ${breakeven:.2f}")
    print(f"   Risk/Reward:  1:{max_profit/max_loss:.2f}")
    
    # Strategy Greeks
    exp_date = datetime.strptime(expiration, '%Y-%m-%d')
    T = max((exp_date - datetime.now()).days, 1) / 365.0
    r = 0.045
    
    long_pricer = BlackScholesMerton(spot, atm_call['strike'], T, r, atm_call['impliedVolatility'])
    short_pricer = BlackScholesMerton(spot, otm_call['strike'], T, r, otm_call['impliedVolatility'])
    
    long_greeks = long_pricer.greeks('call')
    short_greeks = short_pricer.greeks('call')
    
    net_delta = long_greeks['delta'] - short_greeks['delta']
    net_gamma = long_greeks['gamma'] - short_greeks['gamma']
    net_vega = long_greeks['vega'] - short_greeks['vega']
    
    print(f"\n📊 Net Greeks:")
    print(f"   Delta: {net_delta:.4f}")
    print(f"   Gamma: {net_gamma:.6f}")
    print(f"   Vega:  {net_vega:.4f}")


def run_comprehensive_demo():
    """
    Run the full options chain demo.
    """
    print("="*60)
    print("🎯 FINLEARNER - OPTIONS CHAIN DATA DEMO")
    print("="*60)
    print("""
This demo shows how to work with REAL options chain data:
- Fetching live options chains from Yahoo Finance
- Analyzing strikes, volumes, and implied volatility
- Comparing market prices to theoretical model prices
- Building and analyzing options strategies

NOTE: This uses CURRENT market data (snapshot).
For historical backtesting, a paid data source is required.
""")
    
    ticker = 'TCS.NS'
    
    # Part 1: Analyze the chain
    chain = analyze_options_chain(ticker)
    
    if chain is not None:
        # Part 2: Simulate a strategy
        simulate_options_strategy(ticker)
    
    print("\n" + "="*60)
    print("✅ Demo Complete!")
    print("="*60)


if __name__ == "__main__":
    run_comprehensive_demo()
