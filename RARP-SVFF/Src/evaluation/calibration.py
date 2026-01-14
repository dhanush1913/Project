import numpy as np
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def compute_calibration_curve(y_true, y_prob, n_bins=10):
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='uniform')
    return prob_pred, prob_true


def compute_calibration_error(y_true, y_prob, n_bins=10):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    ece = 0.0
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() == 0:
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        ece += mask.sum() * np.abs(bin_acc - bin_conf)
    
    return ece / len(y_true)


def platt_scaling(y_true, y_prob):
    y_prob_2d = y_prob.reshape(-1, 1)
    calibrator = LogisticRegression(max_iter=1000)
    calibrator.fit(y_prob_2d, y_true)
    return calibrator.predict_proba(y_prob_2d)[:, 1]


def isotonic_regression(y_true, y_prob):
    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(y_prob, y_true)
    return calibrator.predict(y_prob)


def calibrate_classifier(model, X_train, y_train, method='isotonic'):
    calibrated = CalibratedClassifierCV(model, method=method, cv=3)
    calibrated.fit(X_train, y_train)
    return calibrated


def assess_calibration(y_true, y_prob, model_name="Model"):
    prob_pred, prob_true = compute_calibration_curve(y_true, y_prob)
    ece = compute_calibration_error(y_true, y_prob)
    
    return {
        'model': model_name,
        'ece': ece,
        'prob_pred': prob_pred,
        'prob_true': prob_true,
        'mean_predicted': np.mean(y_prob),
        'actual_positive_rate': np.mean(y_true)
    }
