import numpy as np
import pandas as pd


def rolling_beta(df, window=63, min_periods=21):
    df = df.copy()
    
    def calc_beta(group):
        stock_ret = group["log_return"]
        market_ret = group["market_log_return"]
        cov = stock_ret.rolling(window=window, min_periods=min_periods).cov(market_ret)
        var = market_ret.rolling(window=window, min_periods=min_periods).var()
        return cov / var
    
    df["beta"] = df.groupby("ticker", group_keys=False).apply(calc_beta).reset_index(level=0, drop=True)
    return df


def rolling_correlation(df, window=63, min_periods=21):
    df = df.copy()
    
    def calc_corr(group):
        stock_ret = group["log_return"]
        market_ret = group["market_log_return"]
        
        return stock_ret.rolling(window=window, min_periods=min_periods).corr(market_ret)
    df["market_corr"] = df.groupby("ticker", group_keys=False).apply(calc_corr).reset_index(level=0, drop=True)
    return df

def excess_return(df):
    df = df.copy()
    df["excess_return"] = df["log_return"] - df["rf_daily"]
    return df
