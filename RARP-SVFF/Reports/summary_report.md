# RARP-SVFF: Final Project Report

**Risk-Adjusted Return Prediction using Statistically Validated Factor Framework**

---

## Summary

When I started this project, I wanted to see if traditional technical and momentum factors could actually predict risk-adjusted equity returns—not build a trading system, just figure out if there's any real signal there once you apply proper statistical rigor.

So what did I find? **There's something there, but it's weak, falls apart depending on market conditions, and definitely not ready for real-world use.**

---

## What Worked

### 1. Statistical Factor Validation

I ran bootstrap confidence intervals and t-tests on the factors. The good news is that several momentum and volatility-based features actually showed statistically significant relationships with forward returns (p < 0.05). At least something passed the initial sniff test.

**See:** [factor_bootstrap_ci.png](figures/factor_bootstrap_ci.png), [factor_t_tests.png](figures/factor_t_tests.png)

### 2. Model Performance Above Random

All three models I tested—Logistic Regression, Decision Tree, and Random Forest—beat random chance on out-of-sample data:

| Model               | Baseline AUC |
| ------------------- | ------------ |
| Logistic Regression | 0.541        |
| Decision Tree       | 0.522        |
| Random Forest       | 0.558        |

Random Forest came out on top, though none of these are exactly amazing.

**See:** [roc_curves.png](figures/roc_curves.png), [baseline_roc_curves.png](figures/baseline_roc_curves.png)

### 3. Probability Calibration

I also checked whether the predicted probabilities were actually calibrated. The reliability curves showed they were moderately calibrated, and Random Forest did the best job after I applied isotonic regression.

**See:** [calibration_curves.png](figures/calibration_curves.png)

### 4. High-Volatility Regime Performance

Here's where things got interesting. During high-volatility periods, all the models performed noticeably better (AUC between 0.55 and 0.61). My guess is they're picking up on mean reversion patterns that show up when volatility spikes.

**See:** [regime_comparison.png](figures/regime_comparison.png)

---

## What Failed

### 1. Low-Volatility and "Normal" Regime Performance

When markets are calm? The models completely fall apart:

| Model               | Low-Vol AUC | Normal (No Drawdown) AUC |
| ------------------- | ----------- | ------------------------ |
| Logistic Regression | 0.486       | 0.510                    |
| Decision Tree       | 0.489       | 0.497                    |
| Random Forest       | 0.505       | 0.508                    |

These AUC scores are basically at or below 0.50, which means the models are just guessing. **In calm markets, there's essentially no signal.**

### 2. Factors Removed During Validation

A bunch of features that initially looked promising got thrown out because they had issues:

- Too correlated with other factors (correlation > 0.8)
- Lost statistical significance after correcting for multiple testing
- Showed unstable coefficients over time (drift problems)

**See:** [factor_correlations.png](figures/factor_correlations.png), [metric_drift.png](figures/metric_drift.png)

### 3. Stress Test Degradation

When I simulated stress conditions (noise injection, volatility spikes), Random Forest took a 6.2% performance hit—not catastrophic, but not great either. Decision Tree held up better (only 0.8% degradation), though it started from a worse baseline to begin with.

**See:** [stress_test_results.png](figures/stress_test_results.png)

### 4. No Consistent Edge Across Tickers

I tested on SPY, AAPL, MSFT, and AMZN. Performance varied wildly between them—some tickers propped up the overall numbers while others just added noise. There's no consistent pattern here.

---

## Limitations

### Data Limitations

- **Frequency:** Only daily data. Nothing intraday.
- **Sample period:** About 15 months (Oct 2024 to Jan 2026)—way too short to cover different market regimes properly.
- **Asset coverage:** Just 4 tickers. Can't make any broad claims based on that.

### Methodology Limitations

- **Transaction costs ignored:** I didn't model slippage, commissions, or bid-ask spreads at all.
- **No live execution:** Everything here is backtested on historical data using walk-forward splits.
- **Look-ahead bias:** I tried to be careful, but there could be subtle biases in the feature engineering.
- **Target definition:** Binary classification (above/below median risk-adjusted return) throws away a lot of information.

### Model Limitations

- **Calibration:** The probabilities need post-hoc adjustment (Platt scaling or isotonic regression) before you could use them for anything.
- **Interpretability:** Random Forest feature importances aren't the same as causal explanations.
- **Regime dependence:** None of these models can predict when market regimes will shift.

---

## What This Is NOT

I need to be crystal clear about what this project isn't:

1. **This is NOT a trading system.** There's no execution logic, no position sizing, no risk management.

2. **This is NOT alpha-generating.** A 0.55 AUC during high volatility—ignoring all costs—isn't a tradeable edge by any stretch.

3. **This is NOT production-ready.** These models haven't been validated on future out-of-sample data or tested in live markets.

4. **This is NOT a guarantee of anything.** Just because certain statistical relationships existed in the past doesn't mean they'll make money in the future.

---

## Honest Conclusions

### The Brutal Truth

> Interviewers don't reject you for weak results.  
> They reject you for dishonesty and not understanding your limitations.  
> A weak but honest project beats a strong but fake one every time.

**What I actually learned:**

1. Yeah, statistically significant factors exist in equity returns, but the effect sizes are tiny.
2. ML models can beat random guessing by a hair, but that edge is incredibly fragile.
3. Market conditions matter way more than which model you pick. Nothing works everywhere.
4. Once you apply proper statistical validation, most "promising" features die off.
5. Transaction costs would probably wipe out whatever edge is left.

**What I'd do differently next time:**

1. Get higher-frequency data—hourly or even tick-level.
2. Test on 50+ tickers instead of just 4.
3. Bring in alternative data like sentiment or options flow.
4. Build transaction costs directly into the target definition.
5. Try regime-switching models instead of one-size-fits-all approaches.

---

## Deliverables

### Report

- [x] `Reports/summary_report.md` (this document)

### Figures

- [x] `Reports/figures/roc_curves.png`
- [x] `Reports/figures/calibration_curves.png`
- [x] `Reports/figures/regime_comparison.png`
- [x] `Reports/figures/stress_test_results.png`
- [x] `Reports/figures/factor_bootstrap_ci.png`
- [x] `Reports/figures/factor_t_tests.png`
- [x] `Reports/figures/metric_drift.png`
- [x] `Reports/figures/feature_importance.png`

---

## Appendix: Key Metrics Summary

```
============================================================
MODEL PERFORMANCE SUMMARY
============================================================

Overall (Out-of-Sample):
  Logistic Regression:  AUC = 0.541
  Decision Tree:        AUC = 0.522
  Random Forest:        AUC = 0.558

By Volatility Regime:
  High-Vol:  Best performance (AUC 0.55-0.61)
  Low-Vol:   Near-random (AUC 0.49-0.51)

By Trend Regime:
  Bear:  Slightly better (AUC 0.52-0.58)
  Bull:  Baseline performance (AUC 0.52-0.54)

Stress Test Degradation:
  Logistic Regression:  3.5%
  Decision Tree:        0.8%
  Random Forest:        6.2%

============================================================
VERDICT: Models show weak, regime-dependent signal.
         Not suitable for live trading without significant
         further development and validation.
============================================================
```

---

_End of Report_
