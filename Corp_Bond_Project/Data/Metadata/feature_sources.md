# Feature Sources Documentation

## Data Source Selection Rationale

### Treasury Yields as Benchmark

We use **Treasury yields** (DGS series) rather than corporate yields because:

- Public availability at consistent maturities (1Y-30Y)
- High liquidity ensures accurate pricing
- Serves as risk-free benchmark for spread calculations

> **Proxy Assumption**: Treasury yields used as corporate yield proxies due to limited public long-maturity corporate data. The credit spread variables (BAA, AAA) capture the corporate premium.

---

### FRED as Primary Source

**Why FRED?**

- Free, reliable API access
- Consistent frequency harmonization
- Long historical coverage (most series back to 1960s+)
- Daily data available for aggregation to monthly

**Known Biases:**

1. **Survivorship in indices**: Credit indices only include surviving issuers
2. **Composition drift**: Index constituents change over time
3. **Vintage effects**: Historical data subject to revision

---

## Variable-Specific Notes

### Macro Factors

| Variable              | Why Chosen                                    | Alternative Considered                     |
| --------------------- | --------------------------------------------- | ------------------------------------------ |
| CPI                   | Headline inflation matters for nominal yields | Core CPI (excludes food/energy volatility) |
| Industrial Production | Real activity proxy; timely release           | GDP (quarterly, lagged)                    |
| Unemployment          | Labor market slack → Fed reaction function    | Initial claims (weekly noise)              |
| Fed Funds             | Direct policy rate                            | 2Y Treasury (market expectation)           |
| M2                    | Liquidity conditions; QE transmission         | Monetary base (narrower)                   |

### Credit Spreads

| Variable   | Why Chosen                                     | Alternative Considered            |
| ---------- | ---------------------------------------------- | --------------------------------- |
| BAA Spread | Lowest investment grade; most credit-sensitive | BBB (S&P equivalent)              |
| AAA        | Top-tier benchmark; minimal default risk       | Treasury (no credit risk)         |
| BAA-10Y    | Direct borrowing cost measure                  | OAS (option-adjusted, not public) |

### Market Indicators

| Variable | Why Chosen                          | Alternative Considered                  |
| -------- | ----------------------------------- | --------------------------------------- |
| VIX      | Standard fear gauge; mean-reverting | MOVE (bond volatility, short history)   |
| S&P 500  | Equity market state; wealth effects | Total return index (not available free) |

---

## Known Data Limitations

1. **30Y Treasury Gap**: DGS30 discontinued 2002-2006; may have missing values
2. **M2 Redefinition (2020)**: Savings deposits added; structural break possible
3. **VIX History**: Only available from 1990; limits backtest period
4. **Zero Lower Bound**: Fed Funds stuck at 0% (2008-2015, 2020-2022); nonlinear dynamics

---

## Frequency Handling

All series converted to **monthly (end-of-month)** using:

```python
series.resample('ME').last()
```

This ensures:

- No look-ahead bias (using month-end values)
- Consistent alignment across sources
- Matches typical model training frequency
