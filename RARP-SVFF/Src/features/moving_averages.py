import numpy as np
import pandas as pd


def price_vs_ma(df, windows=[21, 50, 200]):
    df = df.copy()
    
    for w in windows:
        ma_col = f"ma_{w}d"
        ratio_col = f"price_to_ma_{w}d"
        
        df[ma_col] = df.groupby("ticker")["adj_close"].transform(
            lambda x: x.rolling(window=w, min_periods=w).mean()
        )
        df[ratio_col] = df["adj_close"] / df[ma_col]
    return df

def ma_crossover(df, short=21, long=50):
    df = df.copy()
    short_ma = df.groupby("ticker")["adj_close"].transform(
        lambda x: x.rolling(window=short, min_periods=short).mean()
    )
    long_ma = df.groupby("ticker")["adj_close"].transform(
        lambda x: x.rolling(window=long, min_periods=long).mean()
    )
    df["ma_crossover"] = np.where(short_ma > long_ma, 1, -1)
    return df
