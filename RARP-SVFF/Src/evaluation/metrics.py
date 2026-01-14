import numpy as np
from sklearn.metrics import (
    roc_auc_score, roc_curve, brier_score_loss,
    precision_recall_curve, average_precision_score,
    accuracy_score, precision_score, recall_score, f1_score
)

def compute_roc_auc(y_true, y_prob):
    return roc_auc_score(y_true, y_prob)

def compute_roc_curve(y_true, y_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    return fpr, tpr, thresholds

def compute_brier_score(y_true, y_prob):
    return brier_score_loss(y_true, y_prob)

def compute_precision_recall(y_true, y_prob):
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    return precision, recall, thresholds

def compute_average_precision(y_true, y_prob):
    return average_precision_score(y_true, y_prob)

def compute_classification_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0)
    }

def compute_all_metrics(y_true, y_prob):
    return {
        'roc_auc': compute_roc_auc(y_true, y_prob),
        'brier_score': compute_brier_score(y_true, y_prob),
        'avg_precision': compute_average_precision(y_true, y_prob),
        **compute_classification_metrics(y_true, y_prob)
    }
