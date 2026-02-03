import numpy as np
import pandas as pd
from typing import Callable, Union, List, Any, Dict
from dataclasses import dataclass

@dataclass
class BacktestResult:
    """Stores the results of a backtest."""
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    volatility: float
    trades: int
    equity_curve: pd.Series
    trade_history: List[Dict]

class BacktestEngine:
    """
    A flexible backtesting engine for financial strategies.
    Supports both internal FinLearner models and custom user functions.
    """
    def __init__(self, initial_capital: float = 10000.0, commission_rate: float = 0.001):
        """
        Args:
            initial_capital: Starting cash.
            commission_rate: Transaction fee as a percentage of trade value (e.g. 0.001 = 0.1%).
        """
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.strategy = None
        self.lookback_days = 0 
        
    def add_strategy(self, strategy: Union[Callable, Any], lookback_days: int = 0):
        """
        Adds a strategy to the engine.
        
        Args:
            strategy: Can be a function `def strategy(data) -> 'BUY'|'SELL'|'HOLD'` 
                      OR a class instance with a `.predict(df)` method.
            lookback_days: Required historical window size for the strategy to function.
        """
        self.strategy = strategy
        # If strategy is a model object with lookback_days attribute, use it
        if hasattr(strategy, 'lookback_days'):
            self.lookback_days = strategy.lookback_days
        else:
            self.lookback_days = lookback_days
            
    def run(self, df: pd.DataFrame, price_col: str = 'Close') -> BacktestResult:
        """
        Runs the backtest simulation.
        
        Args:
            df: DataFrame containing price history (must contain `price_col`).
        """
        if self.strategy is None:
            raise ValueError("No strategy added. Use add_strategy() first.")

        capital = self.initial_capital
        position = 0.0 # Number of shares/units
        equity_curve = []
        trade_history = []
        
        prices = df[price_col].values
        dates = df.index
        
        # Start loop
        # We process step-by-step or vectorized where possible
        return self._run_event_driven(df, price_col)

    def _run_event_driven(self, df, price_col):
        """Internal event-driven loop."""
        capital = self.initial_capital
        position = 0
        equity = []
        trades = []
        
        prices = df[price_col].values
        
        # Check if strategy is a predictor (returns float) or logic (returns str)
        is_predictor = hasattr(self.strategy, 'predict') and not callable(self.strategy)
        
        # If it's a predictor, we need a decision rule.
        # Default Rule: If Pred > Current * 1.01 -> BUY.
        
        predictions = None
        if is_predictor:
            print("Generating predictions for entire series (vectorized)...")
            # This is much faster
            try:
                raw_preds = self.strategy.predict(df)
                # Align predictions
                predictions = raw_preds
            except Exception as e:
                print(f"Model prediction failed: {e}")
                return None

        for i in range(self.lookback_days, len(df)):
            price = prices[i] # Today's Close
            
            action = 'HOLD'
            
            if is_predictor:
                # Logic for predictor models
                pred_idx = i - self.lookback_days
                if pred_idx < len(predictions):
                    curr_pred = predictions[pred_idx]
                    
                    # Logic: 0.5% threshold
                    if curr_pred > price * 1.005: 
                         action = 'BUY'
                    elif curr_pred < price * 0.995:
                         action = 'SELL'
            else:
                # Callable function strategy
                step_df = df.iloc[:i+1]
                try:
                    action = self.strategy(step_df)
                except Exception:
                    pass
            
            # Execute
            if action == 'BUY' and capital > price:
                # Buy Max using 99% of capital to account for commission/slippage
                shares_to_buy = (capital * 0.99) // price 
                if shares_to_buy > 0:
                    cost = shares_to_buy * price
                    comm = max(1.0, cost * self.commission_rate)
                    
                    if capital >= (cost + comm):
                        capital -= (cost + comm)
                        position += shares_to_buy
                        trades.append({'step': i, 'action': 'BUY', 'price': price, 'cost': cost+comm})
                    else:
                        # Debug info for rejected trades
                        # print(f"DEBUG: Trade rejected. Need {cost+comm:.2f}, Have {capital:.2f}")
                        pass
            
            elif action == 'SELL' and position > 0:
                revenue = position * price
                comm = max(1.0, revenue * self.commission_rate)
                capital += (revenue - comm)
                position = 0
                trades.append({'step': i, 'action': 'SELL', 'price': price, 'revenue': revenue-comm})
                
            curr_equity = capital + (position * price)
            equity.append(curr_equity)
            
        # Calculate Metrics
        equity_curve = pd.Series(equity, index=df.index[self.lookback_days:])
        
        if len(equity) > 0:
            total_ret = (equity[-1] - self.initial_capital) / self.initial_capital
            
            # Sharpe
            daily_returns = equity_curve.pct_change().dropna()
            if daily_returns.std() > 0:
                sharpe = (daily_returns.mean() / daily_returns.std()) * (252**0.5)
            else:
                sharpe = 0.0
                
            # Drawdown
            rolling_max = equity_curve.cummax()
            drawdown = (equity_curve - rolling_max) / rolling_max
            max_dd = drawdown.min()
            
            ann_ret = (1 + total_ret) ** (252 / len(df)) - 1
            
            return BacktestResult(
                total_return=total_ret,
                annualized_return=ann_ret,
                sharpe_ratio=sharpe,
                max_drawdown=max_dd,
                volatility=daily_returns.std() * (252**0.5),
                trades=len(trades),
                equity_curve=equity_curve,
                trade_history=trades
            )
        else:
            return BacktestResult(0,0,0,0,0,0,pd.Series(),[])
