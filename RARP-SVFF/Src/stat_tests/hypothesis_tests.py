import numpy as np
import pandas as pd
from scipy import stats
def test_factor_significance(feature: np.ndarray, target: np.ndarray, alpha: float = 0.05) -> dict:
    mask = ~(np.isnan(feature) | np.isnan(target))
    x, y = feature[mask], target[mask]
    if len(x) < 30:
        return {"significant": False, "reason": "insufficient_data"}
    r, pval = stats.pearsonr(x, y)
    t_stat = r * np.sqrt((len(x) - 2) / (1 - r**2)) if abs(r) < 1 else 0
    return {
        "correlation": r,
        "t_statistic": t_stat,
        "p_value": pval,
        "significant": pval < alpha,
        "n_samples": len(x)
    }

def run_factor_tests(df: pd.DataFrame, feature_cols: list, target_col: str, alpha: float = 0.05) -> pd.DataFrame:
    results = []
    target = df[target_col].values
    for col in feature_cols:
        res = test_factor_significance(df[col].values, target, alpha)
        res["feature"] = col
        results.append(res)
    out = pd.DataFrame(results)
    out = out[["feature", "correlation", "t_statistic", "p_value", "significant", "n_samples"]]
    return out.sort_values("p_value")
