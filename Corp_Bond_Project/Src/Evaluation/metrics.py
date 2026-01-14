import numpy as np
import pandas as pd


def mse(y_true, y_pred, axis=None):
    return np.mean((y_true - y_pred) ** 2, axis=axis)

def rmse(y_true, y_pred, axis=None):
    return np.sqrt(mse(y_true, y_pred, axis=axis))

def mae(y_true, y_pred, axis=None):
    return np.mean(np.abs(y_true - y_pred), axis=axis)

def mape(y_true, y_pred, axis=None, epsilon=1e-8):
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon)), axis=axis) * 100

def compute_maturity_metrics(y_true_df, y_pred_df, maturities=None):

    maturities = maturities or ['y1', 'y2', 'y5', 'y10', 'y20', 'y30']
    results = []
    
    for mat in maturities:
        if mat in y_true_df.columns and mat in y_pred_df.columns:
            y_t, y_p = y_true_df[mat].values, y_pred_df[mat].values
            results.append({
                'maturity': mat,
                'mse': mse(y_t, y_p),
                'rmse': rmse(y_t, y_p),
                'mae': mae(y_t, y_p)
            })
    
    return pd.DataFrame(results)
