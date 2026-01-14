import os
import pandas as pd
import yfinance as yf
from pandas_datareader import data as pdr
from datetime import datetime

TICKERS = ["AAPL", "MSFT", "AMZN", "GOOG", "META"]
MARKET = "^GSPC"
START = "2019-01-01"
END = datetime.today().strftime("%Y-%m-%d")
RAW_DIR = "Data/Raw"


def setup():
    os.makedirs(RAW_DIR, exist_ok=True)


def get_stocks():
    print("Fetching stock prices...")
    
    frames = []
    for t in TICKERS:
        df = yf.download(t, start=START, end=END,
                         auto_adjust=False, progress=False)
        
        if df.empty:
            raise ValueError(f"No data for {t}")
        
        df = df.reset_index()
        df["ticker"] = t
        df = df[["Date", "ticker", "Open", "High", 
                 "Low", "Close", "Adj Close", "Volume"]]
        df.columns = ["date", "ticker", "open", "high",
                      "low", "close", "adj_close", "volume"]
        frames.append(df)
    
    result = pd.concat(frames, ignore_index=True)
    result.sort_values(["ticker", "date"], inplace=True)
    result.to_csv(f"{RAW_DIR}/stock_prices.csv", index=False)
    print("-> Saved stock_prices.csv")


def get_market():
    print("Fetching market index...")
    
    df = yf.download(MARKET, start=START, end=END,
                     auto_adjust=False, progress=False)
    
    if df.empty:
        raise ValueError("No market data found")
    df = df.reset_index()
    df = df[["Date", "Close", "Adj Close"]]
    df.columns = ["date", "close", "adj_close"]
    df.sort_values("date", inplace=True)
    df.to_csv(f"{RAW_DIR}/market_index.csv", index=False)
    print("-> Saved market_index.csv")


def get_rf_rate():
    print("Fetching risk-free rate...")
    
    df = pdr.DataReader("DTB3", "fred", START, END)
    
    if df.empty:
        raise ValueError("No risk-free rate data")
    
    df = df.reset_index()
    df.columns = ["date", "risk_free_rate"]
    df.sort_values("date", inplace=True)
    df.to_csv(f"{RAW_DIR}/risk_free_rate.csv", index=False)
    print("-> Saved risk_free_rate.csv")

def main():
    setup()
    get_stocks()
    get_market()
    get_rf_rate()
    print("\nDone! Raw data ready for cleaning.")

if __name__ == "__main__":
    main()
