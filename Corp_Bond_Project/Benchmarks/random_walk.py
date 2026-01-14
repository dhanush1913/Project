
import pandas as pd
import numpy as np
from pathlib import Path

TARGETS = ['y1', 'y2', 'y5', 'y10', 'y20', 'y30']

def random_walk_forecast(y_series):
    """Naive forecast: ŷ_t = y_{t-1}"""
    return y_series.shift(1)

def compute_metrics(y_true, y_pred):
    """Compute MSE and RMSE for valid (non-NaN) predictions."""
    mask = ~(y_true.isna() | y_pred.isna())
    mse = ((y_true[mask] - y_pred[mask]) ** 2).mean()
    return {'mse': mse, 'rmse': np.sqrt(mse), 'n': mask.sum()}

def evaluate_random_walk(targets_df, test_start_idx):
    """Evaluate random walk on test set only."""
    results = []
    test_df = targets_df.iloc[test_start_idx:].copy()
    
    for mat in TARGETS:
        y_true = test_df[mat]
        y_pred = random_walk_forecast(targets_df[mat]).iloc[test_start_idx:]
        metrics = compute_metrics(y_true, y_pred)
        results.append({'maturity': mat, 'model': 'random_walk', **metrics})
    
    return pd.DataFrame(results)

def run_baseline(data_path=None, test_ratio=0.2):
    """Main entry point for random walk evaluation."""
    base = Path(__file__).parent.parent / "Data" / "Processed"
    data_path = data_path or base / "maturity_targets.csv"
    
    targets = pd.read_csv(data_path, parse_dates=['date'])
    n = len(targets)
    test_start = int(n * (1 - test_ratio))
    
    print(f"Data: {n} obs | Test starts: idx {test_start} ({targets['date'].iloc[test_start].strftime('%Y-%m')})")
    
    results = evaluate_random_walk(targets, test_start)
    avg_mse = results['mse'].mean()
    
    print(f"\nRandom Walk Results (Test Set):")
    print(results[['maturity', 'mse', 'rmse']].to_string(index=False))
    print(f"\nAggregate MSE: {avg_mse:.6f}")
    
    return results, targets

if __name__ == "__main__":
    results, _ = run_baseline()
