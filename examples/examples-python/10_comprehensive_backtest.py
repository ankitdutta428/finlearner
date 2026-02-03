"""
Comprehensive Backtest Demo with Real Market Data

This example demonstrates:
1. Loading REAL stock data from Yahoo Finance (TCS.NS)
2. Testing multiple deep learning models:
   - LSTM (TimeSeriesPredictor)
   - GRU (GRUPredictor)
   - CNN-LSTM Hybrid (CNNLSTMPredictor)
   - Transformer with Attention (TransformerPredictor)
   - Ensemble Model (EnsemblePredictor)
3. Custom strategy backtesting with the BacktestEngine
4. Model comparison and performance metrics
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from finlearner import (
    BacktestEngine, 
    DataLoader,
    TimeSeriesPredictor,
    GRUPredictor,
    CNNLSTMPredictor,
    TransformerPredictor,
    EnsemblePredictor
)


def load_real_data(ticker: str = 'TCS.NS', start: str = '2022-01-01', end: str = '2024-01-01'):
    """
    Load real market data from Yahoo Finance.
    """
    print(f"📊 Loading real data for {ticker}...")
    try:
        df = DataLoader.download_data(ticker, start=start, end=end)
        print(f"✅ Loaded {len(df)} trading days of data")
        print(f"   Date Range: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
        print(f"   Price Range: ₹{df['Close'].min():.2f} - ₹{df['Close'].max():.2f}")
        return df
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None


def run_custom_strategy_backtest(df: pd.DataFrame):
    """
    SCENARIO A: Backtest a custom Python strategy function.
    """
    print("\n" + "="*70)
    print("📈 SCENARIO A: Custom Strategy Backtest (Golden Cross)")
    print("="*70)
    
    def golden_cross_strategy(data: pd.DataFrame) -> str:
        """
        Simple Moving Average Crossover Strategy.
        Buy when SMA_20 > SMA_50
        Sell when SMA_20 < SMA_50
        """
        if len(data) < 50:
            return 'HOLD'
            
        sma_20 = data['Close'].rolling(window=20).mean().iloc[-1]
        sma_50 = data['Close'].rolling(window=50).mean().iloc[-1]
        
        if sma_20 > sma_50:
            return 'BUY'
        elif sma_20 < sma_50:
            return 'SELL'
        return 'HOLD'

    engine = BacktestEngine(initial_capital=100000, commission_rate=0.001)
    engine.add_strategy(golden_cross_strategy, lookback_days=50)
    
    result = engine.run(df)
    
    print(f"\n📊 Golden Cross Strategy Results:")
    print(f"   Initial Capital: ₹100,000")
    print(f"   Final Capital:   ₹{result.equity_curve.iloc[-1]:,.2f}")
    print(f"   Total Return:    {result.total_return*100:.2f}%")
    print(f"   Sharpe Ratio:    {result.sharpe_ratio:.2f}")
    print(f"   Max Drawdown:    {result.max_drawdown*100:.2f}%")
    print(f"   Trades Executed: {result.trades}")
    
    return result


def train_and_backtest_model(model, model_name: str, df: pd.DataFrame, epochs: int = 3):
    """
    Train a model and run backtest.
    """
    print(f"\n{'='*70}")
    print(f"🤖 Training {model_name}...")
    print(f"{'='*70}")
    
    try:
        # Train the model
        model.fit(df, epochs=epochs, batch_size=32)
        
        # Run backtest
        engine = BacktestEngine(initial_capital=100000, commission_rate=0.001)
        engine.add_strategy(model)
        result = engine.run(df)
        
        if result:
            print(f"\n📊 {model_name} Backtest Results:")
            print(f"   Initial Capital: ₹100,000")
            print(f"   Final Capital:   ₹{result.equity_curve.iloc[-1]:,.2f}")
            print(f"   Total Return:    {result.total_return*100:.2f}%")
            print(f"   Sharpe Ratio:    {result.sharpe_ratio:.2f}")
            print(f"   Max Drawdown:    {result.max_drawdown*100:.2f}%")
            print(f"   Trades Executed: {result.trades}")
            return result
        else:
            print(f"❌ Backtest failed for {model_name}")
            return None
            
    except Exception as e:
        print(f"❌ Error with {model_name}: {e}")
        return None


def run_all_models_comparison(df: pd.DataFrame):
    """
    Run all deep learning models and compare their performance.
    """
    print("\n" + "="*70)
    print("🏆 MULTI-MODEL COMPARISON")
    print("="*70)
    print("""
Testing the following models on TCS.NS:
1. LSTM (TimeSeriesPredictor)
2. GRU (GRUPredictor)
3. CNN-LSTM Hybrid (CNNLSTMPredictor)
4. Transformer with Self-Attention (TransformerPredictor)
""")
    
    lookback = 60
    epochs = 3  # Reduced for demo speed
    
    results = {}
    
    # 1. LSTM Model
    lstm = TimeSeriesPredictor(lookback_days=lookback)
    lstm_result = train_and_backtest_model(lstm, "LSTM", df, epochs)
    if lstm_result:
        results['LSTM'] = lstm_result
    
    # 2. GRU Model
    gru = GRUPredictor(lookback_days=lookback)
    gru_result = train_and_backtest_model(gru, "GRU", df, epochs)
    if gru_result:
        results['GRU'] = gru_result
    
    # 3. CNN-LSTM Hybrid
    cnn_lstm = CNNLSTMPredictor(lookback_days=lookback, filters=64, kernel_size=3)
    cnn_lstm_result = train_and_backtest_model(cnn_lstm, "CNN-LSTM Hybrid", df, epochs)
    if cnn_lstm_result:
        results['CNN-LSTM'] = cnn_lstm_result
    
    # 4. Transformer (Attention Model)
    transformer = TransformerPredictor(lookback_days=lookback, d_model=64, num_heads=4, num_blocks=2)
    transformer_result = train_and_backtest_model(transformer, "Transformer (Attention)", df, epochs)
    if transformer_result:
        results['Transformer'] = transformer_result
    
    return results


def print_comparison_table(results: dict, custom_result):
    """
    Print a comparison table of all strategies.
    """
    print("\n" + "="*70)
    print("📊 FINAL COMPARISON TABLE")
    print("="*70)
    
    print(f"\n{'Strategy':<25} {'Return %':<12} {'Sharpe':<10} {'Max DD %':<12} {'Trades':<10}")
    print("-" * 70)
    
    # Custom strategy
    print(f"{'Golden Cross':<25} {custom_result.total_return*100:>8.2f}%   {custom_result.sharpe_ratio:>8.2f}   {custom_result.max_drawdown*100:>8.2f}%   {custom_result.trades:>6}")
    
    # ML Models
    for name, result in results.items():
        print(f"{name:<25} {result.total_return*100:>8.2f}%   {result.sharpe_ratio:>8.2f}   {result.max_drawdown*100:>8.2f}%   {result.trades:>6}")
    
    # Find best performer
    if results:
        all_results = {'Golden Cross': custom_result, **results}
        best = max(all_results.items(), key=lambda x: x[1].total_return)
        print(f"\n🏆 Best Performer: {best[0]} with {best[1].total_return*100:.2f}% return")


def run_demo():
    """
    Main demo function.
    """
    print("="*70)
    print("🚀 FINLEARNER - COMPREHENSIVE BACKTEST DEMO")
    print("="*70)
    print("""
This demo uses REAL market data from TCS.NS (Tata Consultancy Services)
to backtest multiple deep learning models and trading strategies.
""")
    
    # Load Real Data
    df = load_real_data('TCS.NS', start='2022-01-01', end='2024-01-01')
    
    if df is None:
        print("Failed to load data. Exiting.")
        return
    
    # Run Custom Strategy
    custom_result = run_custom_strategy_backtest(df)
    
    # Run All ML Models
    ml_results = run_all_models_comparison(df)
    
    # Print Comparison
    print_comparison_table(ml_results, custom_result)
    
    print("\n" + "="*70)
    print("✅ Demo Complete!")
    print("="*70)


if __name__ == "__main__":
    run_demo()
