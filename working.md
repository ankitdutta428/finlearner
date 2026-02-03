# File Descriptions for FinLearner

This document provides a detailed overview of the files in the `finlearner` repository, explaining the purpose and functionality of each.

## 📦 `finlearner/` (Core Package)

The main library code containing all financial models, data processors, and utilities.

### Core Modules

*   **`__init__.py`**
    *   **Purpose**: Exports the public API of the package, making models and tools easily importable. Defines `__all__` for cleaner namespace management.
    *   **Exports**: `DataLoader`, `TechnicalIndicators`, Predictors (LSTM, GRU, etc.), Risk Metrics, Anomaly Detectors, etc.

*   **`agent.py`**
    *   **Purpose**: Implements a trading agent for backtesting and simulation.
    *   **Key Classes**:
        *   `Agent`: Simulates trading decisions (Buy/Sell/Hold) based on model predictions. Supports strategies like threshold-based trading.
        *   `TradeRecord`: Dataclass for logging trade history.

*   **`anomaly.py`**
    *   **Purpose**: Anomaly detection logic using Variational Autoencoders (VAE).
    *   **Key Classes**:
        *   `VAEAnomalyDetector`: Uses a VAE to learn normal market patterns and flag deviations (anomalies) based on reconstruction error.

*   **`data.py`**
    *   **Purpose**: Data loading and preprocessing utilities.
    *   **Key Classes**:
        *   `DataLoader`: Handles loading data from CSVs (e.g., Yahoo Finance exports) and basic preprocessing like converting dates and sorting.

*   **`ml_models.py`**
    *   **Purpose**: Tree-based machine learning models for forecasting (non-deep learning).
    *   **Key Classes**:
        *   `GradientBoostPredictor`: A wrapper around XGBoost and LightGBM for time series prediction.

*   **`models.py`**
    *   **Purpose**: Deep learning models for time-series forecasting. Currently focuses on TensorFlow/Keras implementations.
    *   **Key Classes**:
        *   `TimeSeriesPredictor`: Standard LSTM implementation.
        *   `GRUPredictor`: Gated Recurrent Unit implementation (faster/lighter than LSTM).
        *   `CNNLSTMPredictor`: Hybrid model using 1D Convolutions for feature extraction + LSTM for temporal logic.
        *   `TransformerPredictor`: Transformer-based architecture using self-attention (Keras implementation).
        *   `EnsemblePredictor`: Combines predictions from LSTM, GRU, and Attention models via weighted averaging.
        *   `TFTPredictor` / `NBeatsPredictor`: Placeholders/Wrappers for Hugging Face Time Series Transformer models.

*   **`options.py`**
    *   **Purpose**: Quantitative finance models for options pricing.
    *   **Key Classes**:
        *   `BlackScholesMerton`: Implements the Black-Scholes formula for pricing European Call/Put options and calculating Greeks (Delta, Gamma, Vega).

*   **`pinn.py`**
    *   **Purpose**: Physics-Informed Neural Networks (PINNs) for solving financial PDEs.
    *   **Key Classes**:
        *   `BlackScholesPINN`: A TensorFlow model capable of solving the Black-Scholes Partial Differential Equation directly using physics constraints (PDE residuals) in the loss function.

*   **`plotting.py`**
    *   **Purpose**: Visualization tools for model performance and market data.
    *   **Key Classes**:
        *   `Plotter`: Static methods for plotting training history, price predictions vs actuals, anomalies, and correlation matrices.

*   **`portfolio.py`**
    *   **Purpose**: Portfolio optimization and allocation algorithms.
    *   **Key Classes**:
        *   `PortfolioOptimizer`: Efficient Frontier and Sharpe Ratio optimization (Markowitz Mean-Variance).
        *   `BlackLittermanOptimizer`: Implements Black-Litterman model incorporating market views.
        *   `RiskParityOptimizer`: Allocates assets to equalize risk contributions (Hierarchical Risk Parity).

*   **`risk.py`**
    *   **Purpose**: Financial risk measurement and management tools.
    *   **Key Classes/Functions**:
        *   `RiskMetrics`: Class containing methods for VaR (Value at Risk) and CVaR (Conditional VaR).
        *   `historical_var`, `parametric_var`, `monte_carlo_var`: Standalone functions for different VaR calculation methods.
        *   `max_drawdown`: Calculates the maximum loss from a peak.

*   **`technical.py`**
    *   **Purpose**: Technical analysis indicators calculation.
    *   **Key Classes**:
        *   `TechnicalIndicators`: Computes RSI, MACD, Bollinger Bands, Moving Averages (SMA/EMA), etc.

*   **`utils.py`**
    *   **Purpose**: General helper functions.
    *   **Functions**: `check_val` (validation utility), etc.

---

## 📂 `examples/` (Usage & Demos)

Scripts and notebooks demonstrating how to use the library.

### `examples-python/`
*   **`01_data_loading.py`**: demonstrates how to use `DataLoader`.
*   **`02_technical_indicators.py`**: Shows how to compute RSI, MACD, etc.
*   **`03_deep_learning.py`**: Demo of training and predicting with LSTM/GRU models.
*   **`04_gradient_boosting.py`**: Demo of using XGBoost/LightGBM.
*   **`05_anomaly_detection.py`**: Shows how to train a VAE to detect market anomalies.
*   **`06_risk_metrics.py`**: improved calculation examples for VaR and Drawdowns.
*   **`07_portfolio_optimization.py`**: Example of optimizing a portfolio of assets.
*   **`08_complete_demo.py`**: Data pipeline combining multiple features.
*   **`09_advanced_models_agent.py`**: Demo of the Trading Agent.

### `notebooks/`
*   **`finlearner_demo.ipynb`**: A comprehensive Jupyter Notebook tutorial covering the end-to-end workflow of the library.

---

## 🧪 `tests/` (Quality Assurance)

Unit tests using `pytest` to ensure correctness.

*   **`conftest.py`**: Pytest configuration and fixtures.
*   **`test_anomaly.py`**: Tests for VAE anomaly detection.
*   **`test_data.py`**: Tests for data loading and sanity checks.
*   **`test_ml_models.py`**: Tests for Gradient Boosting wrappers.
*   **`test_models.py`**: Tests for Deep Learning models (shapes, outputs).
*   **`test_options.py`**: Tests for Black-Scholes pricing accuracy.
*   **`test_pinn.py`**: Tests for Physics-Informed Neural Network convergence.
*   **`test_plotting.py`**: Tests for plotting functions (ensuring no errors during render).
*   **`test_portfolio.py`**: Tests for portfolio optimization mathematics.
*   **`test_risk.py`**: Tests for VaR, CVaR and other risk calculations.
*   **`test_technical.py`**: Verification of technical indicator values against known benchmarks.
*   **`test_utils.py`**: Tests for utility functions.

---
