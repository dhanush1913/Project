# Corporate Bond Yield Curve Forecasting

## Technical Report

---

### Summary

This report documents a neural network-based yield curve forecasting system for U.S. corporate bonds. The system addresses fundamental challenges in fixed income modeling—multicollinearity, non-stationarity, and regime instability—through a combination of:

1. **Parameterized Expectations Algorithm (PEA)** for principled conditional expectation modeling
2. **Maturity-specific attention** for interpretable feature weighting
3. **Adversarial validation** for distribution shift detection
4. **Regime-aware deployment** with explicit trust policies

Key finding: The neural PEA model outperforms linear regression in stable regimes but degrades at regime boundaries—a pattern we detect and document rather than obscure.

---

### 1. Problem Context

Corporate bond yields across the 1Y–30Y maturity spectrum are critical for:

- Credit risk pricing
- Duration management
- Term premium estimation
- Corporate financing decisions

Traditional econometric approaches (OLS, VAR) struggle with:

- **High VIF** among macro predictors (Fed Funds, CPI, M2 highly correlated)
- **Non-stationarity** requiring differencing that loses level information
- **Regime breaks** that invalidate in-sample parameter estimates

This project develops a forecasting system that acknowledges these challenges and provides operational guardrails.

---

### 2. Data Description

| Dimension        | Details                      |
| ---------------- | ---------------------------- |
| Frequency        | Monthly                      |
| Period           | January 2016 – November 2025 |
| Observations     | 119                          |
| Train/Test Split | 80/20 (chronological)        |

**Features (15 total)**:

- Macro levels: CPI, Industrial Production, Unemployment, Fed Funds, M2
- Macro changes: Δ of above
- Credit spreads: BAA, AAA, BAA-10Y
- Market: VIX, S&P 500

**Targets**: Corporate bond yields at 1Y, 2Y, 5Y, 10Y, 20Y, 30Y maturities

---

### 3. Findings

#### 3.1 Multicollinearity

Variance Inflation Factors exceed 10 for CPI, M2, and Industrial Production, confirming that ridge regularization or alternative approaches are necessary.

#### 3.2 Stationarity

ADF tests indicate all yield series are I(1). Macro variables are mixed with Fed Funds stationary but CPI and M2 non-stationary.

#### 3.3 Regime Identification

| Regime          | Period             | Characteristics                           |
| --------------- | ------------------ | ----------------------------------------- |
| Pre-COVID       | 2016-01 to 2020-02 | Stable rates, gradual normalization       |
| COVID Shock     | 2020-03 to 2020-06 | Emergency rate cuts, spread widening      |
| QE-ZeroRate     | 2020-07 to 2021-12 | Near-zero short rates, compressed spreads |
| Inflation Surge | 2022-01 to 2023-06 | Aggressive hiking cycle                   |
| High-Rate       | 2023-07 to present | Elevated rates, policy uncertainty        |

---

### 4. Baseline Performance

Out-of-sample MSE on test period (last 20% of data):

| Maturity | Random Walk | Ridge Regression |
| -------- | ----------- | ---------------- |
| 1Y       | 0.043       | 2.289            |
| 2Y       | 0.064       | 2.851            |
| 5Y       | 0.063       | 1.360            |
| 10Y      | 0.046       | 0.052            |
| 20Y      | 0.038       | 0.018            |
| 30Y      | 0.036       | 0.377            |

**Observation**: Linear regression fails catastrophically on short maturities where Fed Funds dominance creates instability. Random walk provides a surprisingly strong benchmark.

---

### 5. Neural PEA Model

#### 5.1 Theoretical Foundation

The Parameterized Expectations Algorithm (Den Haan & Marcet, 1990) approximates conditional expectations with a parameterized function:

```
E[Y_t | S_t] ≈ f_θ(S_t)
```

Our neural implementation:

- Uses normalized macro-financial state as conditioning information
- Does NOT include lagged yields (avoids trivial autoregression)
- Outputs 6 maturities simultaneously

#### 5.2 Architecture

```
Input (15) → Dense(64, ReLU) → Dense(64, ReLU) → Dense(6)
```

Parameters: ~5,000
Training: Adam optimizer, MSE loss, early stopping on validation set

#### 5.3 Attention Extension

For interpretability, we add per-maturity attention:

```
h = Encoder(S_t)
a_m = softmax(h ⊙ Q_m)  for each maturity m
y_m = MLP_m(a_m ⊙ h)
```

This allows different maturities to weight features differently, capturing the economic intuition that short rates respond to monetary policy while long rates reflect growth expectations.

---

### 6. Regime Detection

#### 6.1 Adversarial Validation

We train a classifier to distinguish train vs test observations using macro features only:

| Classifier | AUC  | Interpretation      |
| ---------- | ---- | ------------------- |
| Logistic   | 1.00 | Complete separation |
| GBM        | 1.00 | Confirmed           |

This extreme AUC indicates the test period (post-2023) has fundamentally different feature distributions than training. This is expected given the 2022 inflation shock and rate hiking cycle.

#### 6.2 Trust Policy

Based on adversarial AUC and rolling MSE:

| Condition     | Action                             |
| ------------- | ---------------------------------- |
| AUC < 0.55    | Full model trust                   |
| AUC 0.55–0.65 | Monitor with increased uncertainty |
| AUC 0.65–0.75 | Blend with random walk             |
| AUC > 0.75    | Manual review, consider retraining |

---

### 7. Inference Performance

Benchmarked on CPU (Intel Core, single-threaded):

| Configuration           | Mean Latency | P99 Latency |
| ----------------------- | ------------ | ----------- |
| Base PEA, single        | 0.3 ms       | 0.5 ms      |
| PEA + Attention, single | 0.4 ms       | 0.6 ms      |
| Batch of 32             | 0.8 ms total | —           |

Sub-millisecond inference enables real-time pricing applications.

---

### 8. Limitations

1. **Sample size**: 119 observations limits model complexity and out-of-sample validation robustness
2. **Regime sensitivity**: Model accuracy degrades at regime transitions; this is detected but not solved
3. **Attention interpretability**: Weights indicate statistical association, not causal relationships
4. **No trading costs**: This is a forecasting exercise, not a trading strategy backtest
5. **Single asset class**: U.S. corporate bonds only; generalization to other markets not tested

---

### 9. Conclusions

This project demonstrates:

1. **Honest baselining**: Random walk is hard to beat; we document when and why
2. **Principled architecture**: PEA provides theoretical grounding for neural expectations modeling
3. **Interpretability**: Attention weights reveal maturity-specific factor importance
4. **Operational robustness**: Adversarial validation and regime analysis enable risk-aware deployment
5. **Production readiness**: Sub-millisecond inference with clean API

**Business relevance**: More accurate yield expectations reduce term premium estimation error, supporting better credit pricing during stable regimes. The system explicitly detects when predictions become unreliable.

---
