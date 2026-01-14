import numpy as np
import pandas as pd


def cumulative_returns(df, windows=[5, 21, 63, 126]):
    df = df.copy()
    
    for w in windows:
        col_name = f"momentum_{w}d"
        df[col_name] = df.groupby("ticker")["log_return"].transform(
            lambda x: x.rolling(window=w, min_periods=w).sum()
        )
    return df


def rate_of_change(df, window=21):
    df = df.copy()
    
    df["roc"] = df.groupby("ticker")["adj_close"].transform(
        lambda x: x.pct_change(periods=window)
    )
    
    return df
