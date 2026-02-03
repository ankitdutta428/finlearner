import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from finlearner.models import TimeSeriesPredictor, TFTPredictor, NBeatsPredictor, GPUConstraintError
from finlearner.agent import Agent

def create_sample_data(days=500):
    """Create a dummy stock price dataframe."""
    dates = pd.date_range(end=datetime.now(), periods=days)
    # Generate random walk
    prices = [100]
    for _ in range(days-1):
        change = np.random.normal(0, 1)
        prices.append(prices[-1] + change)
    
    df = pd.DataFrame(data={'Close': prices}, index=dates)
    return df

def run_demo():
    print("=== FinLearner Agent & Advanced Models Demo ===\n")
    
    # 1. Load Data
    print("1. Generating Sample Data...")
    df = create_sample_data()
    print(f"   Data Shape: {df.shape}")
    
    # 2. Try Loading Advanced Models (Expect Failure on <32GB GPU)
    print("\n2. Attempting to load TFT Model (Requires >32GB VRAM)...")
    try:
        tft_model = TFTPredictor(lookback_days=60)
        tft_model.fit(df) # This triggers the check
        print("   SUCCESS: TFT Model loaded and trained.")
        active_model = tft_model
    except GPUConstraintError as e:
        print(f"   FAILED (Expected): {e}")
        print("   -> Fallback to Standard LSTM Model.")
        active_model = TimeSeriesPredictor(lookback_days=60)
        active_model.fit(df, epochs=5) # Train briefly
        
    # 3. Initialize Agent
    print("\n3. Initializing Trading Agent...")
    agent = Agent(model=active_model, initial_balance=10000, strategy='threshold', threshold=0.01)
    
    # 4. Run Simulation
    print(f"4. Running Simulation with {active_model.__class__.__name__}...")
    history = agent.simulate(df)
    
    if history:
        print(f"\nSimulation Steps Executed: {len(history)}")
        trades = [h for h in history if h.action != 'HOLD']
        print(f"Total Trades: {len(trades)}")
        
        # Simple plot
        final_val = history[-1].portfolio_value
        print(f"Final Value: ${final_val:.2f}")
    else:
        print("No history generated.")

if __name__ == "__main__":
    run_demo()
