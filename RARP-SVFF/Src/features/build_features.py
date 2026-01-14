import os
import pandas as pd

try:
    from features.volatility import rolling_volatility, realized_volatility
    from features.momentum import cumulative_returns, rate_of_change
    from features.beta import rolling_beta, rolling_correlation, excess_return
    from features.moving_averages import price_vs_ma, ma_crossover
except ModuleNotFoundError:
    from volatility import rolling_volatility, realized_volatility
    from momentum import cumulative_returns, rate_of_change
    from beta import rolling_beta, rolling_correlation, excess_return
    from moving_averages import price_vs_ma, ma_crossover

INTERIM_DIR = "Data/Interim"
PROCESSED_DIR = "Data/Processed"


def build_all_features(input_path=None, output_path=None):
    if input_path is None:
        input_path = os.path.join(INTERIM_DIR, "cleaned_prices.csv")
    if output_path is None:
        output_path = os.path.join(PROCESSED_DIR, "features.csv")
    
    print("=" * 60)
    print("STEP 3: FEATURE CONSTRUCTION")
    print("=" * 60)
    print()
    
    # Load cleaned data
    print("Loading cleaned data...")
    df = pd.read_csv(input_path, parse_dates=["date"])
    print(f"  Loaded {len(df):,} rows")
    print()
    
    # Build features
    print("Building features...")

    # 1. Volatility features
    print("  [1/7] Volatility features...")
    df = rolling_volatility(df, windows=[5, 21, 63])
    df = realized_volatility(df, window=21)
    # 2. Momentum features
    print("  [2/7] Momentum features...")
    df = cumulative_returns(df, windows=[5, 21, 63, 126])
    df = rate_of_change(df, window=21)
    # 3. Beta features
    print("  [3/7] Beta features...")
    df = rolling_beta(df, window=63)
    # 4. Correlation
    print("  [4/7] Market correlation...")
    df = rolling_correlation(df, window=63)
    # 5. Excess return
    print("  [5/7] Excess return...")
    df = excess_return(df)
    # 6. Moving average features
    print("  [6/7] Moving average features...")
    df = price_vs_ma(df, windows=[21, 50, 200])
    # 7. MA crossover
    print("  [7/7] MA crossover signal...")
    df = ma_crossover(df, short=21, long=50)
    print()
    
    # Drop rows with NaN (from rolling calculations)
    before = len(df)
    df = df.dropna()
    after = len(df)
    print(f"Dropped {before - after:,} rows with NaN (from rolling windows)")
    print(f"Final dataset: {after:,} rows")
    print()
    
    # Sort and reset index
    df.sort_values(["ticker", "date"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # Save
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")
    print()
    
    # Feature summary
    print("-" * 60)
    print("FEATURE SUMMARY")
    print("-" * 60)
    
    feature_cols = [c for c in df.columns if c not in ["date", "ticker"]]
    print(f"Total features: {len(feature_cols)}")
    print()
    print("Features created:")
    for col in feature_cols:
        print(f"  - {col}")
    print()
    
    print("=" * 60)
    print("STEP 3 COMPLETE")
    print("=" * 60)
    
    return df


# Feature documentation
FEATURE_DOCS = """
FEATURE DEFINITIONS
===================

VOLATILITY FEATURES:
- volatility_5d: 5-day rolling std of log returns (1 week risk)
- volatility_21d: 21-day rolling std (1 month risk)
- volatility_63d: 63-day rolling std (1 quarter risk)
- realized_vol: Annualized volatility (sqrt(252) * 21d std)

MOMENTUM FEATURES:
- momentum_5d: 5-day cumulative log return
- momentum_21d: 21-day cumulative log return
- momentum_63d: 63-day cumulative log return
- momentum_126d: 126-day cumulative log return
- roc: 21-day rate of change (price momentum)

RISK FEATURES:
- beta: 63-day rolling beta vs market
- market_corr: 63-day rolling correlation with market
- excess_return: Daily return minus risk-free rate

TREND FEATURES:
- ma_21d/50d/200d: Moving averages
- price_to_ma_21d/50d/200d: Price relative to MA
- ma_crossover: +1 if 21-day MA > 50-day MA, else -1

All features use PAST DATA ONLY (no look-ahead bias).
"""


if __name__ == "__main__":
    features = build_all_features()
    print(FEATURE_DOCS)
