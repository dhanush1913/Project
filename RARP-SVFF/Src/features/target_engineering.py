import numpy as np
import pandas as pd

def compute_forward_sharpe(df: pd.DataFrame, window: int = 21, rf_daily: float = 0.0) -> pd.Series:

    def _calc_sharpe(group):
        returns = group['log_return']
        
        fwd_mean = returns.shift(-window + 1).rolling(window).mean()
        fwd_std = returns.shift(-window + 1).rolling(window).std()
        sharpe = (fwd_mean - rf_daily) / fwd_std * np.sqrt(252)
        return sharpe
    return df.groupby('ticker', group_keys=False).apply(_calc_sharpe, include_groups=False)


def create_binary_target(sharpe: pd.Series, threshold: str = 'median') -> pd.Series:
    if threshold == 'median':
        thresh = sharpe.median()
    elif threshold == 'zero':
        thresh = 0.0
    else:
        thresh = float(threshold)
    return (sharpe > thresh).astype(int)

def build_modeling_dataset(df: pd.DataFrame, forward_window: int = 21) -> pd.DataFrame:

    result = df.copy()
    result['fwd_sharpe'] = compute_forward_sharpe(result, window=forward_window)
    result['target'] = create_binary_target(result['fwd_sharpe'])
    result = result.dropna(subset=['target']).reset_index(drop=True)
    return result
