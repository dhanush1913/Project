# Data Dictionary

Complete documentation for all variables used in the corporate bond yield prediction project.

---

## Bond Yields (bond_yields.csv)

| Variable | Source     | Frequency     | Units | Economic Role                              | Known Issues                            |
| -------- | ---------- | ------------- | ----- | ------------------------------------------ | --------------------------------------- |
| `y1`     | FRED DGS1  | Daily→Monthly | %     | 1-year Treasury yield; short-end benchmark | Sensitive to Fed policy                 |
| `y2`     | FRED DGS2  | Daily→Monthly | %     | 2-year Treasury yield; policy expectations | Forward guidance sensitive              |
| `y5`     | FRED DGS5  | Daily→Monthly | %     | 5-year Treasury yield; intermediate term   | Inflation expectations embedded         |
| `y10`    | FRED DGS10 | Daily→Monthly | %     | 10-year Treasury yield; benchmark rate     | Most liquid; flight-to-quality asset    |
| `y20`    | FRED DGS20 | Daily→Monthly | %     | 20-year Treasury yield; long duration      | Less liquid than 10Y or 30Y             |
| `y30`    | FRED DGS30 | Daily→Monthly | %     | 30-year Treasury yield; term premium proxy | Discontinuous (1977-2002, 2006-present) |

---

## Macro Factors (macro_factors.csv)

| Variable   | Source        | Frequency      | Units               | Economic Role                                   | Known Issues                                      |
| ---------- | ------------- | -------------- | ------------------- | ----------------------------------------------- | ------------------------------------------------- |
| `cpi`      | FRED CPIAUCSL | Monthly        | Index (1982-84=100) | CPI for All Urban Consumers; headline inflation | Lagged release (~2 weeks); seasonal adjustments   |
| `indpro`   | FRED INDPRO   | Monthly        | Index (2017=100)    | Industrial Production Index; real output proxy  | Subject to revision                               |
| `unrate`   | FRED UNRATE   | Monthly        | %                   | Unemployment rate; labor market slack           | Household survey; doesn't capture underemployment |
| `fedfunds` | FRED FEDFUNDS | Daily→Monthly  | %                   | Effective Federal Funds Rate; policy rate       | Zero lower bound 2008-2015, 2020-2022             |
| `m2`       | FRED M2SL     | Weekly→Monthly | Billion $           | M2 money supply; liquidity conditions           | Definition changed 2020 (added savings deposits)  |

---

## Credit Spreads (credit_spreads.csv)

| Variable | Source      | Frequency     | Units | Economic Role                                             | Known Issues                                    |
| -------- | ----------- | ------------- | ----- | --------------------------------------------------------- | ----------------------------------------------- |
| `baa`    | FRED BAA    | Daily→Monthly | %     | Moody's Baa Corporate Bond Yield; investment-grade credit | Lowest investment grade; credit cycle indicator |
| `aaa`    | FRED AAA    | Daily→Monthly | %     | Moody's Aaa Corporate Bond Yield; top-tier credit         | Liquidity premium minimal                       |
| `baa10y` | FRED BAA10Y | Daily→Monthly | %     | Baa minus 10Y Treasury spread; credit risk premium        | Direct measure of corporate borrowing cost      |

---

## Market Indicators (market_indicators.csv)

| Variable | Source      | Frequency     | Units        | Economic Role                                   | Known Issues                                |
| -------- | ----------- | ------------- | ------------ | ----------------------------------------------- | ------------------------------------------- |
| `vix`    | FRED VIXCLS | Daily→Monthly | Index points | CBOE Volatility Index; equity market fear gauge | Mean-reverting; spikes during crises        |
| `sp500`  | FRED SP500  | Daily→Monthly | Index level  | S&P 500 index; equity market performance        | Not total return; doesn't include dividends |

---

## Processed Data

| File                      | Contents                           | Notes                                            |
| ------------------------- | ---------------------------------- | ------------------------------------------------ |
| `aligned_panel.csv`       | All variables inner-joined on date | No missing values; base table for modeling       |
| `maturity_targets.csv`    | y1, y2, y5, y10, y20, y30 only     | Prediction targets; separated to prevent leakage |
| `normalized_features.csv` | All non-target variables           | Placeholder; normalization applied in Step 3     |
