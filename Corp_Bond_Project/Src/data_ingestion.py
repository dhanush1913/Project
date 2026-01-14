import pandas as pd
from fredapi import Fred
from pathlib import Path

API_KEY = "b5abeb860fb7e0ee8f432ed47b59f068"
BASE = Path(__file__).parent.parent / "Data"
RAW, PROCESSED = BASE / "Raw", BASE / "Processed"

# FRED series mapping
SERIES = {
    'bond_yields': {'y1': 'DGS1', 'y2': 'DGS2', 'y5': 'DGS5', 'y10': 'DGS10', 'y20': 'DGS20', 'y30': 'DGS30'},
    'macro_factors': {'cpi': 'CPIAUCSL', 'indpro': 'INDPRO', 'unrate': 'UNRATE', 'fedfunds': 'FEDFUNDS', 'm2': 'M2SL'},
    'credit_spreads': {'baa': 'BAA', 'aaa': 'AAA', 'baa10y': 'BAA10Y'},
    'market_indicators': {'vix': 'VIXCLS', 'sp500': 'SP500'}
}

def fetch_and_save():
    fred = Fred(api_key=API_KEY)
    RAW.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    
    dfs = {}
    for category, series_map in SERIES.items():
        data = {}
        for col, sid in series_map.items():
            try:
                s = fred.get_series(sid)
                data[col] = s.resample('ME').last()  # Monthly end
            except Exception as e:
                print(f"Warning: {sid} failed - {e}")
        df = pd.DataFrame(data)
        df.index.name = 'date'
        df.to_csv(RAW / f"{category}.csv")
        dfs[category] = df
        print(f"✓ {category}.csv ({len(df)} rows)")
    return dfs

def create_aligned_panel(dfs):
    panel = pd.concat(dfs.values(), axis=1).dropna()
    panel.to_csv(PROCESSED / "aligned_panel.csv")
    print(f"✓ aligned_panel.csv ({len(panel)} rows)")
    return panel

def create_targets(panel):
    targets = panel[['y1', 'y2', 'y5', 'y10', 'y20', 'y30']].copy()
    targets.to_csv(PROCESSED / "maturity_targets.csv")
    features = panel.drop(columns=['y1', 'y2', 'y5', 'y10', 'y20', 'y30'])
    features.to_csv(PROCESSED / "normalized_features.csv")
    print(f"✓ maturity_targets.csv, normalized_features.csv")

if __name__ == "__main__":
    print("=" * 50)
    print("STEP 1: Data Ingestion from FRED")
    print("=" * 50)
    dfs = fetch_and_save()
    panel = create_aligned_panel(dfs)
    create_targets(panel)
    print(f"\nDate range: {panel.index.min()} to {panel.index.max()}")
    print("Done! Raw data in Data/Raw/, processed in Data/Processed/")
