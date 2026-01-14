import pandas as pd
import numpy as np
from pathlib import Path
import torch
import warnings
warnings.filterwarnings('ignore')

REGIME_DEFS = {
    'Pre-COVID': ('2016-01-01', '2020-02-29'),
    'COVID Shock': ('2020-03-01', '2020-06-30'),
    'QE-ZeroRate': ('2020-07-01', '2021-12-31'),
    'Inflation Surge': ('2022-01-01', '2023-06-30'),
    'High-Rate': ('2023-07-01', '2025-12-31')
}

MATURITIES = ['y1', 'y2', 'y5', 'y10', 'y20', 'y30']

class RegimeAnalyzer:    
    def __init__(self, regime_defs=None):
        self.regimes = regime_defs or REGIME_DEFS
        self.results = None
    
    def assign_regimes(self, df):
        df = df.copy()
        df['regime'] = 'Unknown'
        for regime, (start, end) in self.regimes.items():
            mask = (df['date'] >= start) & (df['date'] <= end)
            df.loc[mask, 'regime'] = regime
        return df
    
    def compute_mse(self, y_true, y_pred):
        return {mat: np.mean((y_true[mat] - y_pred[mat])**2) 
                for mat in MATURITIES if mat in y_true.columns}
    
    def random_walk_baseline(self, df):
        preds = df[MATURITIES].shift(1)
        return preds.iloc[1:]
    
    def linear_baseline(self, df, features_df, train_idx):
        from sklearn.linear_model import Ridge
        
        feature_cols = [c for c in features_df.columns if c not in ['date'] + MATURITIES]
        X = features_df[feature_cols].values
        y = df[MATURITIES].values
        
        model = Ridge(alpha=1.0)
        model.fit(X[:train_idx], y[:train_idx])
        preds = model.predict(X)
        return pd.DataFrame(preds, columns=MATURITIES, index=df.index)
    
    def evaluate_by_regime(self, df, predictions_dict, train_ratio=0.8):
        df = self.assign_regimes(df)
        results = []
        
        for regime in self.regimes.keys():
            regime_mask = df['regime'] == regime
            if regime_mask.sum() < 3:
                continue
            
            regime_df = df[regime_mask]
            for model_name, preds in predictions_dict.items():
                regime_preds = preds.loc[regime_mask]
                mse = self.compute_mse(regime_df, regime_preds)
                
                for mat, val in mse.items():
                    results.append({'regime': regime, 'model': model_name, 
                                    'maturity': mat, 'mse': val})
        
        self.results = pd.DataFrame(results)
        return self.results
    
    def get_summary_table(self):
        if self.results is None:
            return None
        return self.results.pivot_table(values='mse', index='regime', 
                                        columns='model', aggfunc='mean')


def generate_dummy_predictions(df, train_ratio=0.8):
    analyzer = RegimeAnalyzer()
    n_train = int(len(df) * train_ratio)
    
    rw_preds = df[MATURITIES].shift(1).copy()
    rw_preds.iloc[0] = df[MATURITIES].iloc[0]
    train_mean = df[MATURITIES].iloc[:n_train].mean()
    mean_preds = pd.DataFrame([train_mean.values]*len(df), 
                               columns=MATURITIES, index=df.index)
    
    return {'Random Walk': rw_preds, 'Train Mean': mean_preds}


def run_regime_analysis(data_path=None, output_path=None):
    base = Path(__file__).parent.parent.parent
    data_path = data_path or base / "Data" / "Processed" / "aligned_panel.csv"
    output_path = output_path or base / "Experiments" / "Results" / "regime_performance.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path, parse_dates=['date'])
    analyzer = RegimeAnalyzer()
    
    predictions = generate_dummy_predictions(df)
    
    results = analyzer.evaluate_by_regime(df, predictions)
    results.to_csv(output_path, index=False)
    
    summary = analyzer.get_summary_table()
    print("\nRegime-wise MSE Summary:")
    print(summary.round(4))
    print(f"\n✓ Saved detailed results to {output_path}")
    
    return results, summary

if __name__ == "__main__":
    print("=" * 60)
    print("STEP 7B: Regime-wise Performance Analysis")
    print("=" * 60)
    run_regime_analysis()
