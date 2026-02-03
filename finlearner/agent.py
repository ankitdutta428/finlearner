from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from dataclasses import dataclass

@dataclass
class TradeRecord:
    step: int
    action: str  # 'BUY', 'SELL', 'HOLD'
    price: float
    confidence: float
    portfolio_value: float

class Agent:
    """
    Trading Agent that can use any predictor model from finlearner.models.
    """
    def __init__(self, model, initial_balance: float = 10000.0, strategy: str = 'threshold', threshold: float = 0.01):
        """
        Initialize the Agent.

        Args:
            model: A trained predictor model (e.g., TimeSeriesPredictor, TFTPredictor, etc.)
                   Must have a predict(df) -> np.ndarray method.
            initial_balance: Starting cash balance.
            strategy: Trading strategy ('threshold', 'trend_following').
            threshold: Threshold for buy/sell decisions (percentage change).
        """
        self.model = model
        self.balance = initial_balance
        self.holdings = 0.0
        self.strategy = strategy
        self.threshold = threshold
        self.history: List[TradeRecord] = []
        
    def act(self, current_price: float, predicted_price: float, step: int) -> str:
        """
        Decide on an action based on current and predicted price.
        """
        predicted_change = (predicted_price - current_price) / current_price
        
        action = 'HOLD'
        
        if self.strategy == 'threshold':
            if predicted_change > self.threshold:
                if self.balance > 0:
                    self._buy(current_price, step)
                    action = 'BUY'
            elif predicted_change < -self.threshold:
                if self.holdings > 0:
                    self._sell(current_price, step)
                    action = 'SELL'
                    
        return action

    def _buy(self, price: float, step: int):
        """Execute buy order (all-in)."""
        if self.balance > 0:
            units = self.balance / price
            self.holdings += units
            self.balance = 0
            self._log_trade(step, 'BUY', price)

    def _sell(self, price: float, step: int):
        """Execute sell order (sell-all)."""
        if self.holdings > 0:
            cash = self.holdings * price
            self.balance += cash
            self.holdings = 0
            self._log_trade(step, 'SELL', price)

    def _log_trade(self, step: int, action: str, price: float):
        value = self.get_portfolio_value(price)
        self.history.append(TradeRecord(step, action, price, 0.0, value))

    def get_portfolio_value(self, current_price: float) -> float:
        return self.balance + (self.holdings * current_price)

    def simulate(self, df: pd.DataFrame):
        """
        Run a simulation over the provided dataframe.
        
        Args:
            df: DataFrame containing 'Close' prices.
        """
        prices = df['Close'].values
        predictions = self.model.predict(df)
        
        # Adjust lengths. The model predicts for the *next* step based on *lookback* steps.
        # If lookback is 60, prediction[0] corresponds to day 61 (predicting day 61 price).
        # We need to align this with the actual prices.
        
        # Depending on the model, lengths might vary slightly.
        # We will iterate through the valid range where we have both a current price,
        # a prediction for the next step, and the actual next step price (to verify accuracy if needed, 
        # but for trading we act before knowing the next price).
        
        # Assuming predictions align with the end of the dataframe.
        # e.g. predictions[-1] is the prediction for tomorrow (which is not in df).
        # OR predictions corresponds to the valid 'y' entries in training.
        
        # Let's assume standard behavior:
        # predict(df) returns predictions for indices [lookback, len(df)].
        # So predictions[0] is the predicted value for df.iloc[lookback].
        # The decision to buy/sell at step `i` (where we are at `lookback-1` trying to predict `lookback`)
        # should be based on comparing `predictions[0]` with `df.iloc[lookback-1]`.
        
        lookback = self.model.lookback_days
        
        # Ensure we have enough data
        if len(predictions) == 0:
            print("No predictions generated. Simulation aborted.")
            return

        # Aligning:
        # At day `i`, we observe price `prices[i]`.
        # We have a prediction for day `i+1`: `predictions[i - (lookback - 1)]`? No let's match indices.
        
        # Valid indices for which we have predictions:
        start_idx = lookback
        
        # We iterate from start_idx to the end.
        # For each `i` in that range, `predictions[i - start_idx]` is the prediction for `prices[i]`.
        # The decision must be made at `i-1`.
        
        pred_idx_offset = 0 # predictions starts from 0
        
        print(f"Starting simulation. Initial Balance: ${self.balance:.2f}")
        
        for i in range(start_idx, len(prices)):
            current_price = prices[i-1] # Price available at decision time
            predicted_price = predictions[i - start_idx] # Prediction for NOW (price[i])
            
            # Wait, if we use standard `predict` from `models.py`:
            # predict() reconstructs x_test from the whole dataframe.
            # `X_test` entries are [i-lookback : i].
            # Prediction is for `i`.
            # So `predictions[k]` corresponds to `prices[lookback + k]`.
            
            # Act based on prediction for *today* (or tomorrow)?
            # Usually: At close of day i-1, we predict close of day i.
            # If predicted_close > current_close, we buy at current_close (assuming we can).
            
            self.act(current_price, predicted_price, i)
            
        final_value = self.get_portfolio_value(prices[-1])
        print(f"Simulation Complete. Final Portfolio Value: ${final_value:.2f}")
        return self.history
