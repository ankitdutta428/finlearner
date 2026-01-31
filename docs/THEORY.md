<p align="center">
  <img src="assets/mainlogo.svg" alt="FinLearner Logo" width="300">
</p>

<h1 align="center">FinLearner: Theoretical Foundations</h1>

<p align="center">
  <em>A comprehensive guide to the mathematics and theory behind FinLearner's modules</em>
</p>

---

## Table of Contents

1. [Time Series Prediction Models](#1-time-series-prediction-models)
   - [LSTM Networks](#11-lstm-long-short-term-memory)
   - [GRU Networks](#12-gru-gated-recurrent-unit)
   - [CNN-LSTM Hybrid](#13-cnn-lstm-hybrid)
   - [Transformer Architecture](#14-transformer-architecture)
   - [Ensemble Methods](#15-ensemble-methods)
2. [Gradient Boosting Models](#2-gradient-boosting-models)
   - [XGBoost](#21-xgboost)
   - [LightGBM](#22-lightgbm)
3. [Anomaly Detection](#3-anomaly-detection)
   - [Variational Autoencoders](#31-variational-autoencoders-vae)
4. [Risk Metrics](#4-risk-metrics)
   - [Value at Risk (VaR)](#41-value-at-risk-var)
   - [Conditional VaR (CVaR)](#42-conditional-var-cvar--expected-shortfall)
   - [Maximum Drawdown](#43-maximum-drawdown)
5. [Portfolio Optimization](#5-portfolio-optimization)
   - [Markowitz Mean-Variance](#51-markowitz-mean-variance-optimization)
   - [Black-Litterman Model](#52-black-litterman-model)
   - [Risk Parity](#53-risk-parity)
6. [Options Pricing](#6-options-pricing)
   - [Black-Scholes-Merton](#61-black-scholes-merton-model)
   - [Physics-Informed Neural Networks](#62-physics-informed-neural-networks-pinn)
7. [Technical Analysis](#7-technical-analysis)

---

## 1. Time Series Prediction Models

Financial time series prediction involves forecasting future asset prices based on historical patterns. FinLearner implements several state-of-the-art deep learning architectures.

### 1.1 LSTM (Long Short-Term Memory)

**Theory**: LSTMs are a type of Recurrent Neural Network (RNN) designed to learn long-term dependencies by addressing the vanishing gradient problem.

**Architecture**:

```
┌─────────────────────────────────────────────────────────┐
│                    LSTM Cell                             │
│  ┌──────┐   ┌──────┐   ┌──────┐   ┌──────┐              │
│  │ fₜ   │   │ iₜ   │   │ C̃ₜ   │   │ oₜ   │              │
│  │Forget│   │Input │   │Candi-│   │Output│              │
│  │ Gate │   │ Gate │   │ date │   │ Gate │              │
│  └──────┘   └──────┘   └──────┘   └──────┘              │
└─────────────────────────────────────────────────────────┘
```

**Mathematical Formulation**:

| Gate | Equation |
|------|----------|
| **Forget Gate** | $f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$ |
| **Input Gate** | $i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$ |
| **Candidate** | $\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$ |
| **Cell State** | $C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$ |
| **Output Gate** | $o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$ |
| **Hidden State** | $h_t = o_t \odot \tanh(C_t)$ |

Where:
- $\sigma$ = Sigmoid function
- $\odot$ = Element-wise multiplication (Hadamard product)
- $W$ = Weight matrices, $b$ = Bias vectors

---

### 1.2 GRU (Gated Recurrent Unit)

**Theory**: GRU simplifies LSTM by combining the forget and input gates into a single "update gate" and merging the cell state with the hidden state.

**Advantages over LSTM**:
- Fewer parameters (faster training)
- Comparable performance on many tasks
- Better for smaller datasets

**Mathematical Formulation**:

| Gate | Equation |
|------|----------|
| **Update Gate** | $z_t = \sigma(W_z \cdot [h_{t-1}, x_t] + b_z)$ |
| **Reset Gate** | $r_t = \sigma(W_r \cdot [h_{t-1}, x_t] + b_r)$ |
| **Candidate** | $\tilde{h}_t = \tanh(W_h \cdot [r_t \odot h_{t-1}, x_t] + b_h)$ |
| **Hidden State** | $h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$ |

---

### 1.3 CNN-LSTM Hybrid

**Theory**: Combines Convolutional Neural Networks (for local pattern extraction) with LSTMs (for temporal sequence learning).

**Architecture Flow**:
```
Input → [Conv1D → Conv1D → MaxPool → Dropout] → [LSTM → LSTM] → Dense → Output
         ↑                                        ↑
    Feature Extraction                    Sequence Learning
```

**1D Convolution Operation**:

$$y[n] = \sum_{k=0}^{K-1} x[n+k] \cdot w[k]$$

Where:
- $K$ = Kernel size
- $w$ = Learnable filter weights
- $x$ = Input sequence

**Why it works for finance**:
- CNN captures local patterns (candlestick formations, short-term trends)
- LSTM captures long-range dependencies (seasonality, cycles)

---

### 1.4 Transformer Architecture

**Theory**: Uses self-attention mechanisms to capture relationships between all positions in a sequence simultaneously, without the sequential constraints of RNNs.

**Self-Attention Mechanism**:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Where:
- $Q$ = Query matrix (what we're looking for)
- $K$ = Key matrix (what we have)
- $V$ = Value matrix (the actual content)
- $d_k$ = Dimension of keys (scaling factor)

**Multi-Head Attention**:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$

$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

**Positional Encoding** (for temporal awareness):

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

**Advantages for Financial Data**:
- Parallel processing (faster training)
- Captures global context in one step
- No vanishing gradient problem

---

### 1.5 Ensemble Methods

**Theory**: Combines predictions from multiple models to reduce variance and improve robustness.

**Weighted Average Ensemble**:

$$\hat{y}_{ensemble} = \sum_{i=1}^{n} w_i \cdot \hat{y}_i$$

Where:
- $\sum w_i = 1$ (weights sum to 1)
- $\hat{y}_i$ = Prediction from model $i$

**FinLearner's Ensemble**:
- LSTM (weight: 0.4) - Strong sequential learning
- GRU (weight: 0.3) - Faster, complementary
- Attention (weight: 0.3) - Global context

**Bias-Variance Tradeoff**:

$$\text{MSE} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error}$$

Ensembles reduce variance while maintaining low bias.

---

## 2. Gradient Boosting Models

### 2.1 XGBoost

**Theory**: Extreme Gradient Boosting builds an ensemble of decision trees sequentially, where each tree corrects the errors of the previous ensemble.

**Objective Function**:

$$\mathcal{L}(\phi) = \sum_{i} l(y_i, \hat{y}_i) + \sum_{k} \Omega(f_k)$$

Where:
- $l$ = Differentiable loss function (e.g., MSE)
- $\Omega(f) = \gamma T + \frac{1}{2}\lambda||w||^2$ = Regularization term
- $T$ = Number of leaves
- $w$ = Leaf weights

**Gradient Boosting Update**:

$$\hat{y}_i^{(t)} = \hat{y}_i^{(t-1)} + \eta \cdot f_t(x_i)$$

Where $\eta$ is the learning rate and $f_t$ is the new tree.

**Second-Order Taylor Expansion**:

$$\mathcal{L}^{(t)} \approx \sum_{i=1}^{n} \left[ g_i f_t(x_i) + \frac{1}{2} h_i f_t^2(x_i) \right] + \Omega(f_t)$$

Where:
- $g_i = \frac{\partial l(y_i, \hat{y}_i^{(t-1)})}{\partial \hat{y}_i^{(t-1)}}$ (First derivative)
- $h_i = \frac{\partial^2 l(y_i, \hat{y}_i^{(t-1)})}{\partial (\hat{y}_i^{(t-1)})^2}$ (Second derivative)

### 2.2 LightGBM

**Key Innovations**:

1. **Gradient-based One-Side Sampling (GOSS)**: Keeps instances with large gradients, randomly samples those with small gradients
2. **Exclusive Feature Bundling (EFB)**: Bundles mutually exclusive features
3. **Leaf-wise tree growth**: Grows the leaf with max delta loss

---

## 3. Anomaly Detection

### 3.1 Variational Autoencoders (VAE)

**Theory**: VAEs learn a probabilistic latent representation of normal price patterns. Anomalies produce high reconstruction error.

**Architecture**:
```
Input → Encoder → μ, σ → Sampling → Decoder → Reconstruction
           ↓                                        ↓
     Latent Space                          Reconstruction Error
```

**Loss Function (ELBO)**:

$$\mathcal{L}(\theta, \phi) = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{KL}(q_\phi(z|x) || p(z))$$

| Term | Meaning |
|------|---------|
| **Reconstruction Loss** | $\mathbb{E}[\log p_\theta(x\|z)]$ — How well can we reconstruct the input? |
| **KL Divergence** | $D_{KL}$ — How close is the learned distribution to the prior? |

**Reparameterization Trick**:

$$z = \mu + \sigma \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

This allows backpropagation through the sampling operation.

**Anomaly Score**:

$$\text{Anomaly Score}(x) = \|x - \hat{x}\|^2$$

High reconstruction error → Anomalous pattern

---

## 4. Risk Metrics

### 4.1 Value at Risk (VaR)

**Definition**: The maximum expected loss over a given time horizon at a specified confidence level.

**Interpretation**: "With 95% confidence, we will not lose more than VaR in one day."

#### Historical VaR

$$\text{VaR}_\alpha = -\text{Percentile}(R, \alpha)$$

Where $R$ = historical returns and $\alpha$ = significance level (e.g., 0.05)

#### Parametric (Gaussian) VaR

$$\text{VaR}_\alpha = -(\mu + z_\alpha \cdot \sigma)$$

Where:
- $\mu$ = Mean return
- $\sigma$ = Standard deviation
- $z_\alpha$ = Z-score for confidence level (e.g., -1.645 for 95%)

#### Monte Carlo VaR

1. Simulate $N$ return paths: $R_i \sim \mathcal{N}(\mu, \sigma^2)$
2. Calculate cumulative returns for each path
3. $\text{VaR} = -\text{Percentile(cumulative returns}, \alpha)$

#### Cornish-Fisher VaR

Adjusts for skewness ($S$) and excess kurtosis ($K$):

$$z_{CF} = z + \frac{(z^2-1)S}{6} + \frac{(z^3-3z)K}{24} - \frac{(2z^3-5z)S^2}{36}$$

---

### 4.2 Conditional VaR (CVaR) / Expected Shortfall

**Definition**: The expected loss given that the loss exceeds VaR.

$$\text{CVaR}_\alpha = \mathbb{E}[L | L > \text{VaR}_\alpha]$$

**Parametric Formula**:

$$\text{ES}_\alpha = \mu - \sigma \cdot \frac{\phi(z_\alpha)}{\alpha}$$

Where $\phi$ is the standard normal PDF.

**Why CVaR > VaR**:
- VaR doesn't tell you how bad losses can get beyond the threshold
- CVaR captures tail risk
- CVaR is coherent (satisfies subadditivity)

---

### 4.3 Maximum Drawdown

**Definition**: Largest peak-to-trough decline before a new peak.

$$\text{Drawdown}(t) = \frac{P(t) - P^{max}_{[0,t]}}{P^{max}_{[0,t]}}$$

$$\text{MDD} = \max_{t \in [0,T]} |\text{Drawdown}(t)|$$

**Calmar Ratio**:

$$\text{Calmar} = \frac{\text{Annualized Return}}{\text{Maximum Drawdown}}$$

---

## 5. Portfolio Optimization

### 5.1 Markowitz Mean-Variance Optimization

**Theory**: Modern Portfolio Theory (MPT) finds the optimal portfolio by maximizing return for a given level of risk.

**Portfolio Return**:

$$\mu_p = \sum_{i=1}^{n} w_i \mu_i = \mathbf{w}^T \boldsymbol{\mu}$$

**Portfolio Variance**:

$$\sigma_p^2 = \sum_{i=1}^{n} \sum_{j=1}^{n} w_i w_j \sigma_{ij} = \mathbf{w}^T \boldsymbol{\Sigma} \mathbf{w}$$

**Optimization Problem**:

$$\max_{\mathbf{w}} \frac{\mu_p - r_f}{\sigma_p} \quad \text{(Sharpe Ratio)}$$

Subject to $\sum w_i = 1$ and $w_i \geq 0$

**Efficient Frontier**: The set of all optimal portfolios offering the highest expected return for each level of risk.

---

### 5.2 Black-Litterman Model

**Problem with Markowitz**: Highly sensitive to expected return estimates.

**Solution**: Combine market equilibrium (CAPM) with investor views.

**Key Components**:

| Symbol | Meaning |
|--------|---------|
| $\Pi$ | Equilibrium excess returns (implied by market) |
| $P$ | Pick matrix (which assets the views are about) |
| $Q$ | View returns (what we expect) |
| $\Omega$ | Uncertainty in views |
| $\tau$ | Scaling factor for prior uncertainty |

**Equilibrium Returns** (Reverse Optimization):

$$\Pi = \delta \cdot \Sigma \cdot w_{mkt}$$

Where $\delta$ is the risk aversion coefficient.

**Posterior Returns** (Black-Litterman Formula):

$$E[R] = [(\tau\Sigma)^{-1} + P^T\Omega^{-1}P]^{-1}[(\tau\Sigma)^{-1}\Pi + P^T\Omega^{-1}Q]$$

**Intuition**: Blend prior (market equilibrium) with likelihood (investor views) to get posterior expected returns.

---

### 5.3 Risk Parity

**Theory**: Allocate risk equally across assets rather than capital.

**Risk Contribution of Asset $i$**:

$$RC_i = w_i \cdot \frac{\partial \sigma_p}{\partial w_i} = w_i \cdot \frac{(\Sigma \mathbf{w})_i}{\sigma_p}$$

**Objective**: Make all $RC_i$ equal.

$$RC_i = \frac{\sigma_p}{n} \quad \forall i$$

**Optimization**:

$$\min_{\mathbf{w}} \sum_{i=1}^{n} \left( \frac{RC_i}{\sigma_p} - \frac{1}{n} \right)^2$$

**Why Risk Parity?**
- Traditional portfolios are often dominated by equity risk
- Risk parity creates more balanced risk exposure
- More stable across different market regimes

---

## 6. Options Pricing

### 6.1 Black-Scholes-Merton Model

**Assumptions**:
1. Lognormal stock price distribution
2. No dividends (or continuous dividend yield)
3. Constant volatility and interest rate
4. Frictionless markets
5. European-style options

**Black-Scholes PDE**:

$$\frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + rS\frac{\partial V}{\partial S} - rV = 0$$

**Call Option Price**:

$$C = S_0 e^{-qT} N(d_1) - K e^{-rT} N(d_2)$$

**Put Option Price**:

$$P = K e^{-rT} N(-d_2) - S_0 e^{-qT} N(-d_1)$$

Where:

$$d_1 = \frac{\ln(S_0/K) + (r - q + \sigma^2/2)T}{\sigma\sqrt{T}}$$

$$d_2 = d_1 - \sigma\sqrt{T}$$

**The Greeks**:

| Greek | Formula | Measures |
|-------|---------|----------|
| **Delta** ($\Delta$) | $\frac{\partial V}{\partial S}$ | Price sensitivity |
| **Gamma** ($\Gamma$) | $\frac{\partial^2 V}{\partial S^2}$ | Delta sensitivity |
| **Theta** ($\Theta$) | $\frac{\partial V}{\partial t}$ | Time decay |
| **Vega** ($\mathcal{V}$) | $\frac{\partial V}{\partial \sigma}$ | Volatility sensitivity |
| **Rho** ($\rho$) | $\frac{\partial V}{\partial r}$ | Interest rate sensitivity |

---

### 6.2 Physics-Informed Neural Networks (PINN)

**Theory**: Embed the Black-Scholes PDE directly into the neural network's loss function.

**Loss Function**:

$$\mathcal{L} = \mathcal{L}_{data} + \mathcal{L}_{physics}$$

$$\mathcal{L}_{data} = \frac{1}{N}\sum_{i=1}^{N}|V_{pred}(t_i, S_i) - V_{true}(t_i, S_i)|^2$$

$$\mathcal{L}_{physics} = \frac{1}{M}\sum_{j=1}^{M}|f(t_j, S_j)|^2$$

Where $f$ is the PDE residual:

$$f = \frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + rS\frac{\partial V}{\partial S} - rV$$

**Advantages**:
- Works with sparse data
- Enforces physical constraints
- Generalizes to complex boundary conditions

---

## 7. Technical Analysis

### Momentum Indicators

**RSI (Relative Strength Index)**:

$$RSI = 100 - \frac{100}{1 + RS}$$

$$RS = \frac{\text{Average Gain}}{\text{Average Loss}}$$

**MACD**:

$$MACD = EMA_{12} - EMA_{26}$$
$$Signal = EMA_9(MACD)$$

### Volatility Indicators

**Bollinger Bands**:

$$\text{Upper} = SMA_{20} + 2\sigma_{20}$$
$$\text{Lower} = SMA_{20} - 2\sigma_{20}$$

**ATR (Average True Range)**:

$$TR = \max(H-L, |H-C_{prev}|, |L-C_{prev}|)$$
$$ATR = SMA_{14}(TR)$$

### Trend Indicators

**Ichimoku Cloud**:
- Tenkan-sen: $(9\text{-period high} + 9\text{-period low}) / 2$
- Kijun-sen: $(26\text{-period high} + 26\text{-period low}) / 2$
- Senkou Span A: $(Tenkan + Kijun) / 2$, shifted 26 periods ahead
- Senkou Span B: $(52\text{-period high} + 52\text{-period low}) / 2$, shifted 26 periods ahead

---

## References

1. Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*
2. Cho, K., et al. (2014). Learning Phrase Representations using RNN Encoder-Decoder
3. Vaswani, A., et al. (2017). Attention Is All You Need. *NeurIPS*
4. Kingma, D. P., & Welling, M. (2014). Auto-Encoding Variational Bayes. *ICLR*
5. Markowitz, H. (1952). Portfolio Selection. *The Journal of Finance*
6. Black, F., & Litterman, R. (1992). Global Portfolio Optimization. *Financial Analysts Journal*
7. Black, F., & Scholes, M. (1973). The Pricing of Options and Corporate Liabilities. *JPE*
8. Raissi, M., et al. (2019). Physics-Informed Neural Networks. *Journal of Computational Physics*

---

<p align="center">
  <b>FinLearner</b> — State-of-the-art Financial Analysis & Deep Learning
  <br>
  <a href="https://github.com/ankitdutta428/finlearner">⭐ Star on GitHub</a>
</p>
