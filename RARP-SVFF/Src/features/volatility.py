import numpy as np
import pandas as pd


def rolling_volatility(df, windows=[5, 21, 63]):
    df = df.copy()
    for w in windows:
        col_name = f"volatility_{w}d"
        df[col_name] = df.groupby("ticker")["log_return"].transform(
            lambda x: x.rolling(window=w, min_periods=w).std()
        )
    
    return df


def realized_volatility(df, window=21):
    df = df.copy()
    
    df["realized_vol"] = df.groupby("ticker")["log_return"].transform(
        lambda x: x.rolling(window=window, min_periods=window).std() * np.sqrt(252)
    )
    return df
