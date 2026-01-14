import numpy as np
def correlation_ci(r: float, n: int, confidence: float = 0.95) -> tuple:
    from scipy import stats
    if abs(r) >= 1 or n < 4:
        return (np.nan, np.nan)
    z = np.arctanh(r)  # Fisher z-transform
    se = 1 / np.sqrt(n - 3)
    z_crit = stats.norm.ppf((1 + confidence) / 2)
    z_lo, z_hi = z - z_crit * se, z + z_crit * se
    return (np.tanh(z_lo), np.tanh(z_hi))


def mean_ci(x: np.ndarray, confidence: float = 0.95) -> tuple:
    from scipy import stats
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 2:
        return (np.nan, np.nan)
    mean = np.mean(x)
    se = np.std(x, ddof=1) / np.sqrt(n)
    t_crit = stats.t.ppf((1 + confidence) / 2, n - 1)
    return (mean - t_crit * se, mean + t_crit * se)
