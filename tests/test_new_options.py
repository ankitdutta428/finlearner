import pytest
import numpy as np
from finlearner.options import BinomialTreePricing, MonteCarloPricing, BlackScholesMerton

def test_binomial_pricing_convergence():
    """Verify Binomial pricing converges to Black-Scholes for European options."""
    S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
    
    # BS Price
    bs = BlackScholesMerton(S, K, T, r, sigma)
    bs_price = bs.price('call')
    
    # Binomial Price (High N for accuracy)
    bn = BinomialTreePricing(S, K, T, r, sigma, N=500, option_style='european')
    bn_price = bn.price('call')
    
    assert np.isclose(bs_price, bn_price, rtol=1e-2)

def test_american_option_value():
    """Verify American Put is worth more than European Put when early exercise is optimal."""
    S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
    
    # Deep ITM Put might be exercised early? 
    # Usually American Call on non-div stock = European Call.
    # American Put can be > European Put.
    
    bn_eu = BinomialTreePricing(S, K, T, r, sigma, N=100, option_style='european')
    bn_am = BinomialTreePricing(S, K, T, r, sigma, N=100, option_style='american')
    
    price_eu = bn_eu.price('put')
    price_am = bn_am.price('put')
    
    assert price_am >= price_eu

def test_monte_carlo_pricing():
    """Verify Monte Carlo is reasonably close to Black-Scholes."""
    S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
    
    bs = BlackScholesMerton(S, K, T, r, sigma)
    bs_price = bs.price('call')
    
    mc = MonteCarloPricing(S, K, T, r, sigma, iterations=50000)
    mc_price = mc.price_european('call')
    
    # MC has variance, loose tolerance
    assert np.isclose(bs_price, mc_price, rtol=5e-2)
