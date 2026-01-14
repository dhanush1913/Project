import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

FEATURE_COLS = ['cpi', 'indpro', 'unrate', 'fedfunds', 'm2', 'baa', 'aaa', 'baa10y', 'vix', 'sp500']

class AdversarialValidator:
    """Detects train/test distribution shift via binary classification."""
    
    def __init__(self, train_ratio=0.8, classifier='logistic'):
        self.train_ratio = train_ratio
        self.classifier = classifier
        self.auc = None
        self.fpr, self.tpr = None, None
    
    def _normalize(self, df, train_mean, train_std):
        """Z-score normalize using train stats only."""
        return (df - train_mean) / (train_std + 1e-8)
    
    def _get_classifier(self):
        """Return classifier (simple models detect real drift, not noise)."""
        if self.classifier == 'logistic':
            return LogisticRegression(max_iter=500, random_state=42)
        return GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)
    
    def run_validation(self, df, feature_cols=None, n_splits=5):
        """
        Core adversarial validation: classify train(0) vs test(1) periods.
        
        Returns:
            dict with 'auc', 'fpr', 'tpr', 'interpretation'
        """
        feature_cols = feature_cols or [c for c in FEATURE_COLS if c in df.columns]
        X = df[feature_cols].copy()

        n_train = int(len(df) * self.train_ratio)
        y = np.array([0]*n_train + [1]*(len(df)-n_train))
        
        train_mean, train_std = X.iloc[:n_train].mean(), X.iloc[:n_train].std()
        X = self._normalize(X, train_mean, train_std).values
        
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        aucs, all_probs = [], np.zeros(len(y))
        
        for train_idx, val_idx in skf.split(X, y):
            clf = self._get_classifier()
            clf.fit(X[train_idx], y[train_idx])
            probs = clf.predict_proba(X[val_idx])[:, 1]
            aucs.append(roc_auc_score(y[val_idx], probs))
            all_probs[val_idx] = probs
        
        self.auc = np.mean(aucs)
        self.fpr, self.tpr, _ = roc_curve(y, all_probs)
        
        if self.auc < 0.55:
            interp = "No distribution shift detected - model should generalize well"
        elif self.auc < 0.65:
            interp = "Minor drift - monitor performance closely"
        elif self.auc < 0.75:
            interp = "Meaningful drift - consider regime-specific adjustments"
        else:
            interp = "DANGER ZONE - severe regime break, predictions unreliable"
        
        return {'auc': self.auc, 'fpr': self.fpr, 'tpr': self.tpr, 
                'cv_aucs': aucs, 'interpretation': interp}

def run_adversarial_validation(data_path=None, output_path=None):
    base = Path(__file__).parent.parent.parent
    data_path = data_path or base / "Data" / "Processed" / "aligned_panel.csv"
    output_path = output_path or base / "Experiments" / "Results" / "adversarial_auc.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(data_path, parse_dates=['date'])
    validator = AdversarialValidator(train_ratio=0.8)
    
    results = []
    for clf_name in ['logistic', 'gbm']:
        validator.classifier = clf_name
        res = validator.run_validation(df)
        results.append({
            'classifier': clf_name, 'auc': res['auc'],
            'cv_std': np.std(res['cv_aucs']), 'interpretation': res['interpretation']
        })
        print(f"{clf_name.upper()}: AUC = {res['auc']:.3f} | {res['interpretation']}")
    
    pd.DataFrame(results).to_csv(output_path, index=False)
    print(f"\n✓ Saved to {output_path}")
    return results


if __name__ == "__main__":
    print("=" * 60)
    print("STEP 7A: Adversarial Validation - Distribution Shift Detection")
    print("=" * 60)
    run_adversarial_validation()
