import numpy as np
import pandas as pd


def load_volume_data(raw_dir="Data/Raw"):
    path = f"{raw_dir}/stock_prices.csv"
    df = pd.read_csv(path, parse_dates=["date"])
    return df[["date", "ticker", "volume"]]

def relative_volume(df, window=21):

    df = df.copy()
    if "volume" not in df.columns:
        return df  # Skip if no volume data
    
    avg_vol = df.groupby("ticker")["volume"].transform(
        lambda x: x.rolling(window=window, min_periods=window).mean()
    )
    df["relative_volume"] = df["volume"] / avg_vol
    return df


def volume_trend(df, window=21):
    df = df.copy()
    if "volume" not in df.columns:
        return df
    df["volume_trend"] = df.groupby("ticker")["volume"].transform(
        lambda x: x.pct_change(periods=window)
    )
    return df
