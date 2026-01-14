import numpy as np
import pandas as pd

def bootstrap_correlation(x: np.ndarray, y: np.ndarray, n_boot: int = 1000, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    n = len(x)
    
    if n < 30:
        return {"stable": False, "reason": "insufficient_data"}
    
    correlations = np.zeros(n_boot)
    
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        correlations[i] = np.corrcoef(x[idx], y[idx])[0, 1]
    
    positive_frac = np.mean(correlations > 0)
    sign_stable = positive_frac > 0.9 or positive_frac < 0.1
    
    return {
        "mean_corr": np.mean(correlations),
        "std_corr": np.std(correlations),
        "ci_lower": np.percentile(correlations, 2.5),
        "ci_upper": np.percentile(correlations, 97.5),
        "positive_frac": positive_frac,
        "sign_stable": sign_stable,
        "stable": sign_stable  # Primary stability flag
    }


def bootstrap_all_features(df: pd.DataFrame, feature_cols: list, target_col: str, 
                           n_boot: int = 1000, seed: int = 42) -> pd.DataFrame:
    results = []
    target = df[target_col].values
    
    for col in feature_cols:
        res = bootstrap_correlation(df[col].values, target, n_boot, seed)
        res["feature"] = col
        results.append(res)
    
    out = pd.DataFrame(results)
    cols = ["feature", "mean_corr", "std_corr", "ci_lower", "ci_upper", "positive_frac", "stable"]
    return out[[c for c in cols if c in out.columns]]
