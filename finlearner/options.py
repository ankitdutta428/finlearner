import numpy as np
from scipy.stats import norm

class BlackScholesMerton:
    """
    Black-Scholes-Merton Model for European Option Pricing.
    Supports continuous dividend yield.
    """
    def __init__(self, S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0):
        """
        Args:
            S: Spot price
            K: Strike price
            T: Time to maturity (years)
            r: Risk-free rate
            sigma: Volatility
            q: Continuous dividend yield (0 for non-dividend paying)
        """
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.q = q

    def _d1_d2(self):
        d1 = (np.log(self.S / self.K) + (self.r - self.q + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)
        return d1, d2

    def price(self, option_type: str = 'call') -> float:
        d1, d2 = self._d1_d2()
        if option_type == 'call':
            price = (self.S * np.exp(-self.q * self.T) * norm.cdf(d1)) - \
                    (self.K * np.exp(-self.r * self.T) * norm.cdf(d2))
        else:
            price = (self.K * np.exp(-self.r * self.T) * norm.cdf(-d2)) - \
                    (self.S * np.exp(-self.q * self.T) * norm.cdf(-d1))
        return price

    def greeks(self, option_type: str = 'call') -> dict:
        """Calculates Delta, Gamma, Vega, Theta, Rho"""
        d1, d2 = self._d1_d2()
        
        # Common terms
        pdf_d1 = norm.pdf(d1)
        cdf_d1 = norm.cdf(d1) if option_type == 'call' else norm.cdf(-d1)
        
        # Delta
        if option_type == 'call':
            delta = np.exp(-self.q * self.T) * norm.cdf(d1)
        else:
            delta = np.exp(-self.q * self.T) * (norm.cdf(d1) - 1)
            
        # Gamma (Same for Call and Put)
        gamma = (np.exp(-self.q * self.T) * pdf_d1) / (self.S * self.sigma * np.sqrt(self.T))
        
        # Vega (Same for Call and Put)
        vega = self.S * np.exp(-self.q * self.T) * pdf_d1 * np.sqrt(self.T) / 100 # Scaled
        
        return {'delta': delta, 'gamma': gamma, 'vega': vega}

class BinomialTreePricing:
    """
    Binomial Tree Model for Option Pricing.
    Supports American and European options.
    """
    def __init__(self, S: float, K: float, T: float, r: float, sigma: float, N: int = 100, option_style: str = 'european'):
        """
        Args:
            S: Spot price
            K: Strike price
            T: Time to maturity (years)
            r: Risk-free rate
            sigma: Volatility
            N: Number of time steps
            option_style: 'european' or 'american'
        """
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.N = N
        self.option_style = option_style.lower()

    def price(self, option_type: str = 'call') -> float:
        dt = self.T / self.N
        u = np.exp(self.sigma * np.sqrt(dt))
        d = 1 / u
        p = (np.exp(self.r * dt) - d) / (u - d)
        
        # Initialize asset prices at maturity
        asset_prices = np.zeros(self.N + 1)
        for i in range(self.N + 1):
            asset_prices[i] = self.S * (u ** (self.N - i)) * (d ** i)
            
        # Initialize option values at maturity
        option_values = np.zeros(self.N + 1)
        if option_type == 'call':
            option_values = np.maximum(asset_prices - self.K, 0)
        else:
            option_values = np.maximum(self.K - asset_prices, 0)
            
        # Step back through tree
        for j in range(self.N - 1, -1, -1):
            for i in range(j + 1):
                option_values[i] = np.exp(-self.r * dt) * (p * option_values[i] + (1 - p) * option_values[i+1])
                
                if self.option_style == 'american':
                    # Check for early exercise
                    # Recompute asset price at this node
                    current_spot = self.S * (u ** (j - i)) * (d ** i)
                    if option_type == 'call':
                        intrinsic = max(current_spot - self.K, 0)
                    else:
                        intrinsic = max(self.K - current_spot, 0)
                    option_values[i] = max(option_values[i], intrinsic)
                    
        return option_values[0]

class MonteCarloPricing:
    """
    Monte Carlo Simulation for Option Pricing.
    Useful for path-dependent options or complex payoffs.
    """
    def __init__(self, S: float, K: float, T: float, r: float, sigma: float, iterations: int = 10000):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.iterations = iterations
        
    def price_european(self, option_type: str = 'call') -> float:
        """Standard European Option Pricing via MC."""
        z = np.random.standard_normal(self.iterations)
        ST = self.S * np.exp((self.r - 0.5 * self.sigma ** 2) * self.T + self.sigma * np.sqrt(self.T) * z)
        
        if option_type == 'call':
            payoffs = np.maximum(ST - self.K, 0)
        else:
            payoffs = np.maximum(self.K - ST, 0)
            
        return np.exp(-self.r * self.T) * np.mean(payoffs)

    def price_asian(self, option_type: str = 'call', steps: int = 252) -> float:
        """
        Arithmetic Asian Option Pricing (Average Price).
        """
        dt = self.T / steps
        paths = np.zeros((self.iterations, steps + 1))
        paths[:, 0] = self.S
        
        for t in range(1, steps + 1):
            z = np.random.standard_normal(self.iterations)
            paths[:, t] = paths[:, t-1] * np.exp((self.r - 0.5 * self.sigma**2) * dt + self.sigma * np.sqrt(dt) * z)
            
        average_prices = np.mean(paths[:, 1:], axis=1) # Exclude initial price usually
        
        if option_type == 'call':
            payoffs = np.maximum(average_prices - self.K, 0)
        else:
            payoffs = np.maximum(self.K - average_prices, 0)
            
        return np.exp(-self.r * self.T) * np.mean(payoffs)
