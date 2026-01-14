import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def compute_metric_drift(df, y_col, prob_col, date_col='date', n_periods=5, metric_fn=roc_auc_score):
    df = df.sort_values(date_col).reset_index(drop=True)
    n = len(df)
    period_size = n // n_periods
    
    results = []
    for i in range(n_periods):
        start_idx = i * period_size
        end_idx = (i + 1) * period_size if i < n_periods - 1 else n
        period_df = df.iloc[start_idx:end_idx]
        if len(period_df) < 10:
            continue
        y_true = period_df[y_col].values
        y_prob = period_df[prob_col].values
        
        if len(np.unique(y_true)) < 2:
            continue
            
        metric = metric_fn(y_true, y_prob)
        results.append({
            'period': i + 1,
            'start_date': period_df[date_col].iloc[0],
            'end_date': period_df[date_col].iloc[-1],
            'n_samples': len(period_df),
            'metric': metric
        })
    return pd.DataFrame(results)


def compute_feature_importance_drift(model_list, feature_names):
    results = []
    for period_name, model in model_list:
        if hasattr(model, 'feature_importances_'):
            imp = model.feature_importances_
        elif hasattr(model, 'coef_'):
            imp = np.abs(model.coef_).flatten()
        else:
            continue

        for feat, val in zip(feature_names, imp):
            results.append({
                'period': period_name,
                'feature': feat,
                'importance': val
            })
    
    return pd.DataFrame(results)


def detect_performance_degradation(metrics_by_period, threshold=0.05):
    if len(metrics_by_period) < 3:
        return False, "Not enough periods"
    
    recent = metrics_by_period['metric'].iloc[-2:].mean()
    earlier = metrics_by_period['metric'].iloc[:-2].mean()
    degraded = (earlier - recent) > threshold
    msg = f"Earlier: {earlier:.3f}, Recent: {recent:.3f}, Diff: {earlier - recent:.3f}"
    return degraded, msg


def stability_summary(df, y_col, prob_col, date_col='date'):
    drift_df = compute_metric_drift(df, y_col, prob_col, date_col)
    
    if len(drift_df) == 0:
        return {'status': 'insufficient_data'}
    
    metrics = drift_df['metric'].values
    degraded, msg = detect_performance_degradation(drift_df)
    
    return {
        'mean_metric': np.mean(metrics),
        'std_metric': np.std(metrics),
        'min_metric': np.min(metrics),
        'max_metric': np.max(metrics),
        'trend': 'degrading' if degraded else 'stable',
        'details': msg,
        'periods': drift_df.to_dict('records')
    }

# ============================================================================
# REGIME ANALYSIS
# ============================================================================

def classify_volatility_regime(df, return_col='log_return', window=21, threshold_pct=50):
    vol = df[return_col].rolling(window=window, min_periods=window//2).std()
    threshold = vol.quantile(threshold_pct / 100)
    return np.where(vol >= threshold, 'high_vol', 'low_vol')


def classify_trend_regime(df, return_col='log_return', window=63):
    cum_ret = df[return_col].rolling(window=window, min_periods=window//2).sum()
    return np.where(cum_ret >= 0, 'bull', 'bear')


def classify_drawdown_regime(df, return_col='log_return', threshold=-0.1):
    cum = df[return_col].cumsum()
    peak = cum.expanding().max()
    drawdown = cum - peak
    return np.where(drawdown <= threshold, 'drawdown', 'normal')


def evaluate_by_regime(df, y_col, prob_col, regime_col, metric_fn=roc_auc_score):
    results = []
    for regime in df[regime_col].dropna().unique():
        mask = df[regime_col] == regime
        subset = df[mask]
        
        if len(subset) < 20 or subset[y_col].nunique() < 2:
            continue
        
        metric = metric_fn(subset[y_col], subset[prob_col])
        results.append({
            'regime': regime,
            'n_samples': len(subset),
            'metric': metric,
            'positive_rate': subset[y_col].mean()
        })
    
    return pd.DataFrame(results)

# ============================================================================
# STRESS TESTING
# ============================================================================

def simulate_volatility_spike(y_prob, spike_factor=2.0):
    mean_prob = y_prob.mean()
    stressed = mean_prob + (y_prob - mean_prob) * spike_factor
    return np.clip(stressed, 0, 1)


def simulate_correlation_breakdown(y_prob, noise_level=0.2):
    noise = np.random.normal(0, noise_level, len(y_prob))
    return np.clip(y_prob + noise, 0, 1)


def stress_test_model(y_true, y_prob, metric_fn=roc_auc_score):
    baseline = metric_fn(y_true, y_prob)
    results = [{'scenario': 'baseline', 'metric': baseline}]
    # Volatility spike
    stressed = simulate_volatility_spike(y_prob, spike_factor=1.5)
    results.append({'scenario': 'vol_spike_1.5x', 'metric': metric_fn(y_true, stressed)})
    stressed = simulate_volatility_spike(y_prob, spike_factor=2.0)
    results.append({'scenario': 'vol_spike_2x', 'metric': metric_fn(y_true, stressed)})
    # Correlation breakdown
    np.random.seed(42)
    stressed = simulate_correlation_breakdown(y_prob, noise_level=0.1)
    results.append({'scenario': 'noise_10%', 'metric': metric_fn(y_true, stressed)})
    stressed = simulate_correlation_breakdown(y_prob, noise_level=0.2)
    results.append({'scenario': 'noise_20%', 'metric': metric_fn(y_true, stressed)})
    return pd.DataFrame(results)

