# Implementation Plan - Options & Backtesting

## Goal Description
Enhance `finlearner` with advanced options pricing models (Binomial, Monte Carlo) and a flexible `BacktestEngine` that can simulate trading strategies using both internal pre-trained models and arbitrary user-defined Python functions.

## User Review Required
> [!NOTE]
> The `Agent` class in `agent.py` will be marked as legacy/deprecated in favor of the new `BacktestEngine` in `backtest.py`, though I will keep `Agent` for backward compatibility or refactor it to use `BacktestEngine` internally if feasible.

## Proposed Changes

### finlearner
#### [MODIFY] [options.py](file:///c:/Users/user/OneDrive/Desktop/finlearner/finlearner/options.py)
- Add `BinomialTreePricing` class for American/European options.
- Add `MonteCarloPricing` class for path-dependent options (Asian) or complex payoffs.

#### [NEW] [backtest.py](file:///c:/Users/user/OneDrive/Desktop/finlearner/finlearner/backtest.py)
- Create `BacktestEngine` class.
- Support `add_strategy(strategy_func_or_class)`.
- Support `run(data)`.
- return `BacktestResult` object with metrics (Sharpe, Returns, Drawdown) and equity curve.

#### [MODIFY] [__init__.py](file:///c:/Users/user/OneDrive/Desktop/finlearner/finlearner/__init__.py)
- Export new options classes.
- Export `BacktestEngine`.

### examples/examples-python
#### [NEW] [10_comprehensive_backtest.py](file:///c:/Users/user/OneDrive/Desktop/finlearner/examples/examples-python/10_comprehensive_backtest.py)
- Demonstrate backtesting with a standard `LSTM` model from `finlearner`.
- Demonstrate backtesting with a simple "Golden Cross" SMA python function.
- Compare results.

## Verification Plan

### Automated Tests
- Create `tests/test_backtest.py` to verify engine logic (entry/exit/profit calc).
- Update `tests/test_options.py` to test new pricing models against known benchmarks (e.g. comparing Binomial with large N to Black-Scholes).

### Manual Verification
- Run `10_comprehensive_backtest.py` and inspect console output and potential plots.
