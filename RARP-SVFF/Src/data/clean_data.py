import os
import numpy as np
import pandas as pd

try:
    from data.load_data import load_stock_prices, load_market_index, load_risk_free_rate
except ModuleNotFoundError:
    from load_data import load_stock_prices, load_market_index, load_risk_free_rate

RAW_DIR = "Data/Raw"
INTERIM_DIR = "Data/Interim"


# =============================================================================
# CLEANING FUNCTIONS
# =============================================================================

def clean_stock_prices(df):

    df = df.copy()
    df.sort_values(["ticker", "date"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df["adj_close"] = df.groupby("ticker")["adj_close"].ffill()
    before = len(df)
    df.dropna(subset=["adj_close"], inplace=True)
    after = len(df)
    
    if before != after:
        print(f"  Dropped {before - after} rows with NaN adj_close")
    return df


def clean_market_index(df):
    df = df.copy()
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    df["adj_close"] = df["adj_close"].ffill()
    df.dropna(subset=["adj_close"], inplace=True)
    return df

def clean_risk_free_rate(df):
    df = df.copy()
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    before_nan = df["risk_free_rate"].isna().sum()
    df["risk_free_rate"] = df["risk_free_rate"].ffill()
    df["risk_free_rate"] = df["risk_free_rate"].bfill()
    after_nan = df["risk_free_rate"].isna().sum()
    print(f"  Risk-free rate: filled {before_nan} NaN values")
    return df


# =============================================================================
# LOG RETURNS
# =============================================================================

def compute_log_returns(df, price_col="adj_close", group_col="ticker"):
    df = df.copy()
    if group_col:
        df.sort_values([group_col, "date"], inplace=True)
    else:
        df.sort_values("date", inplace=True)
    if group_col:
        df["log_return"] = df.groupby(group_col)[price_col].transform(
            lambda x: np.log(x / x.shift(1))
        )
    else:
        df["log_return"] = np.log(df[price_col] / df[price_col].shift(1))
    return df


def compute_market_log_return(df):
    df = df.copy()
    df.sort_values("date", inplace=True)
    df["log_return"] = np.log(df["adj_close"] / df["adj_close"].shift(1))
    return df


# =============================================================================
# ALIGNMENT
# =============================================================================

def align_datasets(stocks_df, market_df, rf_df):

    stock_dates = set(stocks_df["date"].unique())
    market_dates = set(market_df["date"].unique())
    common_dates = stock_dates & market_dates
    print(f"  Common dates between stocks and market: {len(common_dates)}")
    stocks_df = stocks_df[stocks_df["date"].isin(common_dates)].copy()
    market_df = market_df[market_df["date"].isin(common_dates)].copy()
    rf_df = rf_df[rf_df["date"].isin(common_dates)].copy()
    return stocks_df, market_df, rf_df


# =============================================================================
# MAIN CLEANING PIPELINE
# =============================================================================

def clean_all_data(raw_dir=RAW_DIR, output_dir=INTERIM_DIR):
    print("=" * 60)
    print("STEP 2: DATA CLEANING & TIME-SERIES CONSISTENCY")
    print("=" * 60)
    print()
    os.makedirs(output_dir, exist_ok=True)
    print("Loading raw data...")
    stocks = load_stock_prices(raw_dir)
    market = load_market_index(raw_dir)
    rf = load_risk_free_rate(raw_dir)
    print(f"  Loaded {len(stocks)} stock rows")
    print(f"  Loaded {len(market)} market rows")
    print(f"  Loaded {len(rf)} rf rate rows")
    print()
    
    print("Cleaning data...")
    stocks = clean_stock_prices(stocks)
    market = clean_market_index(market)
    rf = clean_risk_free_rate(rf)
    print()

    print("Computing log returns...")
    stocks = compute_log_returns(stocks, "adj_close", "ticker")
    market = compute_market_log_return(market)
    print(f"  Stock log returns computed")
    print(f"  Market log returns computed")
    print()
    
    print("Aligning datasets to common dates...")
    stocks, market, rf = align_datasets(stocks, market, rf)
    print()
    
    market_returns = market[["date", "log_return"]].rename(
        columns={"log_return": "market_log_return"}
    )
    stocks = stocks.merge(market_returns, on="date", how="left")
    rf["rf_daily"] = rf["risk_free_rate"] / 100 / 252
    rf_daily = rf[["date", "rf_daily"]]
    stocks = stocks.merge(rf_daily, on="date", how="left")
    
    stocks["rf_daily"] = stocks["rf_daily"].ffill().bfill()
    stocks = stocks.dropna(subset=["log_return"])
    stocks.sort_values(["ticker", "date"], inplace=True)
    stocks.reset_index(drop=True, inplace=True)
    
    final_cols = [
        "date", "ticker", "adj_close", 
        "log_return", "market_log_return", "rf_daily"
    ]
    stocks = stocks[final_cols]
    
    output_path = os.path.join(output_dir, "cleaned_prices.csv")
    stocks.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")
    print(f"  Final shape: {stocks.shape}")
    print()
    
    print("-" * 60)
    print("CLEANING SUMMARY")
    print("-" * 60)
    print(f"Tickers: {stocks['ticker'].nunique()}")
    print(f"Date range: {stocks['date'].min().date()} to {stocks['date'].max().date()}")
    print(f"Total rows: {len(stocks):,}")
    print()
    print("Log return statistics:")
    print(stocks.groupby("ticker")["log_return"].describe().round(4))
    print()
    print("=" * 60)
    print("STEP 2 COMPLETE")
    print("=" * 60)

    return stocks

if __name__ == "__main__":
    cleaned = clean_all_data()
