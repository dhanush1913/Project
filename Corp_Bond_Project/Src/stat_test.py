import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.outliers_influence import variance_inflation_factor

def correlation_matrix(df: pd.DataFrame, method: str = 'pearson') -> pd.DataFrame:
    return df.corr(method=method)

def compute_vif(df: pd.DataFrame) -> pd.DataFrame:
    X = df.select_dtypes(include=[np.number]).dropna()
    vif_data = pd.DataFrame({'feature': X.columns,
                             'VIF': [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]})
    return vif_data.sort_values('VIF', ascending=False).reset_index(drop=True)

def adf_test(series: pd.Series, name: str = '') -> dict:
    result = adfuller(series.dropna(), autolag='AIC')
    return {'variable': name, 'adf_stat': result[0], 'p_value': result[1],
            'lags_used': result[2], 'stationary': result[1] < 0.05}

def kpss_test(series: pd.Series, name: str = '') -> dict:
    result = kpss(series.dropna(), regression='c', nlags='auto')
    return {'variable': name, 'kpss_stat': result[0], 'p_value': result[1],
            'stationary': result[1] >= 0.05}

def stationarity_summary(df: pd.DataFrame) -> pd.DataFrame:
    results = [adf_test(df[col], col) for col in df.select_dtypes(include=[np.number]).columns]
    return pd.DataFrame(results)

def rolling_stats(series: pd.Series, window: int = 36) -> pd.DataFrame:
    return pd.DataFrame({'rolling_mean': series.rolling(window).mean(),
                         'rolling_std': series.rolling(window).std()})
