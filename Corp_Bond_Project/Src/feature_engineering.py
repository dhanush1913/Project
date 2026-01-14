import pandas as pd
import numpy as np
from pathlib import Path

MACRO_LEVELS = ['cpi', 'indpro', 'unrate', 'fedfunds', 'm2']
CREDIT_RISK = ['baa', 'aaa', 'baa10y', 'vix']
MARKET = ['sp500']
TARGETS = ['y1', 'y2', 'y5', 'y10', 'y20', 'y30']

class FeatureEngineer:
    def __init__(self, train_ratio=0.8):
        self.train_ratio = train_ratio
        self.norm_params = {}
        
    def _add_changes(self, df, cols):
        for col in cols:
            if col in df.columns:
                df[f'd_{col}'] = df[col].diff()
        return df
    
    def _compute_norm_params(self, train_df, cols):
        self.norm_params = {col: {'mean': train_df[col].mean(), 'std': train_df[col].std()} 
                           for col in cols if col in train_df.columns}
    
    def _apply_zscore(self, df, cols):
        result = df.copy()
        for col in cols:
            if col in self.norm_params and col in result.columns:
                p = self.norm_params[col]
                result[col] = (result[col] - p['mean']) / (p['std'] + 1e-8)
        return result
    
    def build_features(self, panel):
        df = panel.copy()
        
        df = self._add_changes(df, MACRO_LEVELS)
        df = df.iloc[1:].reset_index(drop=True)
        level_cols = MACRO_LEVELS + CREDIT_RISK + MARKET
        change_cols = [f'd_{c}' for c in MACRO_LEVELS if f'd_{c}' in df.columns]
        feature_cols = level_cols + change_cols
        n_train = int(len(df) * self.train_ratio)
        train_df = df.iloc[:n_train]
        self._compute_norm_params(train_df, feature_cols)
        features = self._apply_zscore(df[['date'] + feature_cols] if 'date' in df.columns 
                                      else df[feature_cols], feature_cols)
        targets = df[['date'] + TARGETS] if 'date' in df.columns else df[TARGETS]
        return features, targets, feature_cols
    
    def get_feature_groups(self):
        return {
            'Macro Levels': MACRO_LEVELS,
            'Credit & Risk': CREDIT_RISK,
            'Market': MARKET,
            'Macro Changes': [f'd_{c}' for c in MACRO_LEVELS]
        }

def run_pipeline(input_path=None, output_dir=None):
    base = Path(__file__).parent.parent / "Data"
    input_path = input_path or base / "Processed" / "aligned_panel.csv"
    output_dir = output_dir or base / "Processed"
    
    panel = pd.read_csv(input_path, parse_dates=['date'])
    print(f"Loaded {len(panel)} rows from {input_path.name}")
    fe = FeatureEngineer(train_ratio=0.8)
    features, targets, feature_cols = fe.build_features(panel)
    features.to_csv(output_dir / "normalized_features.csv", index=False)
    targets.to_csv(output_dir / "maturity_targets.csv", index=False)
    print(f"✓ normalized_features.csv: {features.shape[1]-1} features × {len(features)} rows")
    print(f"✓ maturity_targets.csv: {len(TARGETS)} maturities × {len(targets)} rows")
    print(f"\nFeature groups: {fe.get_feature_groups()}")
    return fe, features, targets

if __name__ == "__main__":
    print("=" * 50)
    print("STEP 3: Feature Engineering Pipeline")
    print("=" * 50)
    run_pipeline()
