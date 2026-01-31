# FinLearner Examples

This directory contains demo scripts and notebooks for the FinLearner library.

## 📁 Structure

```
examples/
├── examples-python/     # CLI-runnable Python scripts
│   ├── 01_data_loading.py
│   ├── 02_technical_indicators.py
│   ├── 03_deep_learning.py
│   ├── 04_gradient_boosting.py
│   ├── 05_anomaly_detection.py
│   ├── 06_risk_metrics.py
│   ├── 07_portfolio_optimization.py
│   └── 08_complete_demo.py
└── notebooks/           # Jupyter notebooks
    └── finlearner_demo.ipynb
```

## 🐍 Running Python Examples

Run individual demos from the project root:

```bash
# Data loading
python examples/examples-python/01_data_loading.py

# Technical indicators
python examples/examples-python/02_technical_indicators.py

# Deep learning (LSTM, GRU, Transformer, etc.)
python examples/examples-python/03_deep_learning.py

# Gradient Boosting (XGBoost, LightGBM)
python examples/examples-python/04_gradient_boosting.py

# VAE Anomaly Detection
python examples/examples-python/05_anomaly_detection.py

# Risk Metrics (VaR, CVaR, Drawdown)
python examples/examples-python/06_risk_metrics.py

# Portfolio Optimization (Markowitz, Black-Litterman, Risk Parity)
python examples/examples-python/07_portfolio_optimization.py

# Complete demo (all modules)
python examples/examples-python/08_complete_demo.py
```

## 📓 Running Notebooks

```bash
cd examples/notebooks
jupyter notebook finlearner_demo.ipynb
```

## 📊 Output

Each script saves visualization plots to the current directory:
- `data_loading_plot.png`
- `technical_indicators_plot.png`
- `deep_learning_comparison.png`
- `gradient_boosting_importance.png`
- `anomaly_detection_plot.png`
- `risk_metrics_plot.png`
- `portfolio_optimization_plot.png`
