
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from pathlib import Path

TARGETS = ['y1', 'y2', 'y5', 'y10', 'y20', 'y30']

def train_and_evaluate(X_train, y_train, X_test, y_test, alpha=1.0):
    """Train Ridge regression (regularized to handle multicollinearity)."""
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = ((y_test - y_pred) ** 2).mean()
    return {'mse': mse, 'rmse': np.sqrt(mse), 'coefs': model.coef_}

def evaluate_linear_regression(features_df, targets_df, train_end_idx, test_start_idx):
    """Train on train set, evaluate on test set (separate model per maturity)."""
    feature_cols = [c for c in features_df.columns if c != 'date']
    
    X_train = features_df[feature_cols].iloc[:train_end_idx].values
    X_test = features_df[feature_cols].iloc[test_start_idx:].values
    
    results, coef_dict = [], {}
    for mat in TARGETS:
        y_train = targets_df[mat].iloc[:train_end_idx].values
        y_test = targets_df[mat].iloc[test_start_idx:].values
        
        out = train_and_evaluate(X_train, y_train, X_test, y_test)
        results.append({'maturity': mat, 'model': 'linear_reg', 'mse': out['mse'], 'rmse': out['rmse']})
        coef_dict[mat] = dict(zip(feature_cols, out['coefs']))
    
    return pd.DataFrame(results), pd.DataFrame(coef_dict).T

def run_baseline(data_path=None, train_ratio=0.7, val_ratio=0.15):
    """Main entry point: chronological split, no shuffling."""
    base = Path(__file__).parent.parent / "Data" / "Processed"
    
    features = pd.read_csv(data_path or base / "normalized_features.csv", parse_dates=['date'])
    targets = pd.read_csv(base / "maturity_targets.csv", parse_dates=['date'])
    
    n = len(features)
    train_end = int(n * train_ratio)
    test_start = int(n * (train_ratio + val_ratio))
    
    print(f"Split: Train 0-{train_end} | Val {train_end}-{test_start} | Test {test_start}-{n}")
    print(f"Test period: {targets['date'].iloc[test_start].strftime('%Y-%m')} to {targets['date'].iloc[-1].strftime('%Y-%m')}")
    
    results, coefs = evaluate_linear_regression(features, targets, train_end, test_start)
    
    print(f"\nLinear Regression Results (Test Set):")
    print(results[['maturity', 'mse', 'rmse']].to_string(index=False))
    print(f"\nAggregate MSE: {results['mse'].mean():.6f}")
    
    return results, coefs

if __name__ == "__main__":
    results, coefs = run_baseline()
    print(f"\nCoefficient signs (shows instability):")
    print((coefs > 0).astype(int).replace({0: '-', 1: '+'}))
