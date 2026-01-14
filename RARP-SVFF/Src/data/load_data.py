import os
import pandas as pd
from datetime import datetime

RAW_DIR = "Data/Raw"

# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_stock_prices(data_dir=RAW_DIR):
    path = os.path.join(data_dir, "stock_prices.csv")

    df = pd.read_csv(path, parse_dates=["date"])
    expected = ["date", "ticker", "open", "high", "low", 
                "close", "adj_close", "volume"]
    _check_columns(df, expected, "stock_prices")
    df.sort_values(["ticker", "date"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    return df


def load_market_index(data_dir=RAW_DIR):
    path = os.path.join(data_dir, "market_index.csv")
    df = pd.read_csv(path, parse_dates=["date"])
    expected = ["date", "close", "adj_close"]
    _check_columns(df, expected, "market_index")
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def load_risk_free_rate(data_dir=RAW_DIR):
    path = os.path.join(data_dir, "risk_free_rate.csv")
    df = pd.read_csv(path, parse_dates=["date"])
    expected = ["date", "risk_free_rate"]
    _check_columns(df, expected, "risk_free_rate")
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def _check_columns(df, expected, name):
    missing = set(expected) - set(df.columns)
    if missing:
        raise ValueError(f"{name}: Missing columns {missing}")

# =============================================================================
# INTEGRITY CHECKS
# =============================================================================

def check_date_range(df, name, date_col="date"):
    start = df[date_col].min()
    end = df[date_col].max()
    n_days = df[date_col].nunique()

    print(f"[{name}]")
    print(f"  Start: {start.date()}")
    print(f"  End:   {end.date()}")
    print(f"  Trading days: {n_days}")
    print()
    
    return start, end


def check_missing_values(df, name):
    missing = df.isnull().sum()
    total = len(df)
    print(f"[{name}] Missing Values:")
    for col in df.columns:
        n_miss = missing[col]
        if n_miss > 0:
            pct = (n_miss / total) * 100
            print(f"  {col}: {n_miss} ({pct:.2f}%)")
    
    if missing.sum() == 0:
        print("  None - data is complete")
    print()

def check_price_columns(df, name):
    if "close" not in df.columns or "adj_close" not in df.columns:
        print(f"[{name}] Cannot check price columns - missing close/adj_close")
        return
    
    diff = (df["close"] - df["adj_close"]).abs()
    max_diff = diff.max()
    has_diff = max_diff > 0.01  # Allow tiny floating point errors
    
    if has_diff:
        print(f"[{name}] Price Column Check:")
        print(f"  Close and Adj Close DIFFER (max diff: {max_diff:.4f})")
        print("  -> Corporate actions (splits/dividends) are present")
        print("  -> MUST use Adj Close for return calculations")
    else:
        print(f"[{name}] Close and Adj Close are identical")
    print()


def check_risk_free_rate_units(df):
    if "risk_free_rate" not in df.columns:
        print("[WARNING] No risk_free_rate column found")
        return
    
    valid = df["risk_free_rate"].dropna()
    print("[Risk-Free Rate Unit Check]")
    print(f"  Min: {valid.min():.4f}")
    print(f"  Max: {valid.max():.4f}")
    print(f"  Mean: {valid.mean():.4f}")
    
    if valid.max() < 10 and valid.min() > -1:
        print("  -> Appears to be ANNUALIZED PERCENTAGE (DTB3 standard)")
        print("  -> To get daily rate: divide by 252 and by 100")
    else:
        print("  [WARNING] Unexpected range - verify units manually")
    print()


def get_trading_dates(df, date_col="date"):
    return set(df[date_col].dt.date)

def check_date_alignment(stocks_df, market_df, rf_df):
    print("[Date Alignment Check]")
    stock_dates = get_trading_dates(stocks_df)
    market_dates = get_trading_dates(market_df)
    rf_dates = get_trading_dates(rf_df)
    common = stock_dates & market_dates & rf_dates
    only_stocks = stock_dates - market_dates - rf_dates
    only_market = market_dates - stock_dates - rf_dates
    only_rf = rf_dates - stock_dates - market_dates
    
    print(f"  Stock trading days: {len(stock_dates)}")
    print(f"  Market trading days: {len(market_dates)}")
    print(f"  Risk-free rate days: {len(rf_dates)}")
    print(f"  Common to all: {len(common)}")
    print()
    
    if only_stocks:
        print(f"  [WARNING] {len(only_stocks)} dates only in stocks")
    if only_market:
        print(f"  [WARNING] {len(only_market)} dates only in market")
    if only_rf:
        print(f"  [INFO] {len(only_rf)} dates only in rf_rate (includes weekends/holidays)")
    
    return common, stock_dates, market_dates, rf_dates


def check_tickers(df):
    print("[Ticker Coverage]")
    tickers = df["ticker"].unique()
    print(f"  Total tickers: {len(tickers)}")
    print(f"  Tickers: {list(tickers)}")
    print()
    for t in sorted(tickers):
        sub = df[df["ticker"] == t]
        start = sub["date"].min().date()
        end = sub["date"].max().date()
        n = len(sub)
        print(f"  {t}: {start} to {end} ({n} days)")
    print()
    
    return list(tickers)


def check_survivorship_bias(df, tickers):
    print("[Survivorship Bias Warning]")
    print("  Current dataset includes only: " + ", ".join(tickers))
    print("  These are all current market leaders that survived.")
    print("  [CAUTION] Delisted/failed stocks are NOT included.")
    print("  -> Backtest results may be optimistically biased")
    print()


def check_missing_trading_days(df, date_col="date"):
    dates = df[date_col].drop_duplicates().sort_values()
    gaps = dates.diff().dt.days
    large_gaps = gaps[gaps > 5]
    
    print(f"[Trading Day Gaps]")
    if len(large_gaps) > 0:
        print(f"  Found {len(large_gaps)} gaps > 5 calendar days:")
        for idx in large_gaps.index[:5]:  # Show first 5
            gap_date = dates.loc[idx]
            gap_size = gaps.loc[idx]
            print(f"    {gap_date.date()}: {int(gap_size)} day gap")
        if len(large_gaps) > 5:
            print(f"    ... and {len(large_gaps) - 5} more")
    else:
        print("  No suspicious gaps found")
    print()


# =============================================================================
# MAIN VALIDATION
# =============================================================================

def run_all_checks(data_dir=RAW_DIR):
    print("=" * 60)
    print("DATA INGESTION & INTEGRITY VERIFICATION")
    print("Step 1 - Loading and Validating Raw Data")
    print("=" * 60)
    print()
    
    # Load all datasets
    print("Loading datasets...")
    stocks = load_stock_prices(data_dir)
    market = load_market_index(data_dir)
    rf = load_risk_free_rate(data_dir)
    print("All datasets loaded successfully.\n")
    
    print("-" * 60)
    print("DATE RANGE CHECKS")
    print("-" * 60)
    check_date_range(stocks, "Stock Prices")
    check_date_range(market, "Market Index")
    check_date_range(rf, "Risk-Free Rate")
    
    print("-" * 60)
    print("MISSING VALUE CHECKS")
    print("-" * 60)
    check_missing_values(stocks, "Stock Prices")
    check_missing_values(market, "Market Index")
    check_missing_values(rf, "Risk-Free Rate")
    
    print("-" * 60)
    print("PRICE COLUMN CHECKS")
    print("-" * 60)
    check_price_columns(stocks, "Stock Prices")
    check_price_columns(market, "Market Index")
    
    print("-" * 60)
    print("RISK-FREE RATE VALIDATION")
    print("-" * 60)
    check_risk_free_rate_units(rf)
    
    print("-" * 60)
    print("DATE ALIGNMENT")
    print("-" * 60)
    check_date_alignment(stocks, market, rf)
    
    print("-" * 60)
    print("TICKER ANALYSIS")
    print("-" * 60)
    tickers = check_tickers(stocks)
    check_survivorship_bias(stocks, tickers)
    
    print("-" * 60)
    print("TRADING DAY GAPS")
    print("-" * 60)
    check_missing_trading_days(stocks)
    
    print("=" * 60)
    print("INTEGRITY CHECK COMPLETE")
    print("=" * 60)
    
    return stocks, market, rf

if __name__ == "__main__":
    stocks, market, rf = run_all_checks()
