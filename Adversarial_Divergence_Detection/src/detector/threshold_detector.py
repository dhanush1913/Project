import numpy as np
import yaml
from pathlib import Path

class ThresholdDetector:
    
    def __init__(self, weights=None, threshold=None, config_path=None):
        if config_path is not None:
            self.load_config(config_path)
        else:
            self.weights = np.array(weights) if weights is not None else np.ones(6) / 6
            self.threshold = threshold if threshold is not None else 0.0
        
        self.weights = self.weights.astype(np.float32)
    
    def compute_score(self, features):
        return np.dot(features, self.weights)
    
    def predict(self, features):
        scores = self.compute_score(features)
        return scores > self.threshold
    
    def predict_with_score(self, features):
        score = self.compute_score(features)
        prediction = score > self.threshold
        
        if np.isscalar(score):
            return {'prediction': bool(prediction), 'score': float(score)}
        else:
            return {'prediction': prediction, 'score': score}
    
    def save_config(self, config_path):
        config = {
            'weights': self.weights.tolist(),
            'threshold': float(self.threshold),
            'feature_names': [
                'kl_global', 'js_global', 'wasserstein_global',
                'kl_pixel_mean', 'kl_pixel_max', 'wasserstein_pixel_mean'
            ]
        }
        
        Path(config_path).parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    
    def load_config(self, config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.weights = np.array(config['weights'], dtype=np.float32)
        self.threshold = float(config['threshold'])

def calibrate_weights(features_clean, features_adv, method='roc_auc'):
    if method == 'uniform':
        return np.ones(features_clean.shape[1]) / features_clean.shape[1]
    from sklearn.metrics import roc_auc_score
    
    y_clean = np.zeros(len(features_clean))
    y_adv = np.ones(len(features_adv))
    y_true = np.concatenate([y_clean, y_adv])
    
    aucs = []
    for i in range(features_clean.shape[1]):
        feat_values = np.concatenate([features_clean[:, i], features_adv[:, i]])
        try:
            auc = roc_auc_score(y_true, feat_values)
            aucs.append(max(auc, 1 - auc))  # Handle inverse correlation
        except:
            aucs.append(0.5)  # No discriminative power
    
    aucs = np.array(aucs)
    weights = np.maximum(aucs - 0.5, 0)
    weights = weights / (weights.sum() + 1e-8)
    
    return weights.astype(np.float32)

def calibrate_threshold(scores_clean, scores_adv, fpr_penalty=2.0, target_fpr=0.01):
    from sklearn.metrics import roc_curve
    
    y_true = np.concatenate([np.zeros(len(scores_clean)), np.ones(len(scores_adv))])
    y_score = np.concatenate([scores_clean, scores_adv])
    
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    objective = tpr - fpr_penalty * fpr
    idx_max = np.argmax(objective)
    threshold_cost = thresholds[idx_max]
    
    idx_target = np.argmin(np.abs(fpr - target_fpr))
    threshold_target = thresholds[idx_target]
    return float(threshold_cost)
