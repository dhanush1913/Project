"""Evaluation module for model assessment, calibration, and stability analysis."""

from .metrics import (
    compute_roc_auc,
    compute_roc_curve,
    compute_brier_score,
    compute_precision_recall,
    compute_average_precision,
    compute_classification_metrics,
    compute_all_metrics
)

from .calibration import (
    compute_calibration_curve,
    compute_calibration_error,
    platt_scaling,
    isotonic_regression,
    calibrate_classifier,
    assess_calibration
)

from .stability_analysis import (
    compute_metric_drift,
    compute_feature_importance_drift,
    detect_performance_degradation,
    stability_summary,
    classify_volatility_regime,
    classify_trend_regime,
    classify_drawdown_regime,
    evaluate_by_regime,
    simulate_volatility_spike,
    simulate_correlation_breakdown,
    stress_test_model
)
