import pandas as pd
import numpy as np


def time_series_split(df: pd.DataFrame, n_splits: int = 5, test_size: float = 0.2):
    df = df.sort_values('date').reset_index(drop=True)
    n = len(df)
    test_len = int(n * test_size)
    fold_size = (n - test_len) // n_splits
    for i in range(n_splits):
        train_end = fold_size * (i + 1)
        test_start = train_end
        test_end = min(test_start + test_len, n)
        
        if test_end <= test_start:
            break 
        train_idx = df.index[:train_end].tolist()
        test_idx = df.index[test_start:test_end].tolist()
        
        yield train_idx, test_idx


def train_test_split_time(df: pd.DataFrame, train_ratio: float = 0.8):
    df = df.sort_values('date').reset_index(drop=True)
    split_idx = int(len(df) * train_ratio)
    
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()
