import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path


def generate_yield_curve_fits(df, output_path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    maturities = ['y2', 'y10', 'y30']
    colors = {'Actual': '#2E86AB', 'Random Walk': '#A23B72'}
    n_train = int(len(df) * 0.8)
    test_df = df.iloc[n_train:].copy()
    rw_preds = df[maturities].shift(1).iloc[n_train:]
    
    for i, mat in enumerate(maturities):
        ax = axes[i]
        ax.plot(test_df['date'], test_df[mat], color=colors['Actual'], 
                lw=2, label='Actual', marker='o', markersize=4)
        ax.plot(test_df['date'], rw_preds[mat], color=colors['Random Walk'],
                lw=2, label='Forecast', linestyle='--', alpha=0.8)
        
        ax.set_title(f'{mat.upper()} Yield', fontsize=12, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Yield (%)' if i == 0 else '')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.tick_params(axis='x', rotation=45)
        if i == 0:
            ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Out-of-Sample Yield Forecasts', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'✓ Saved {output_path.name}')


def generate_regime_breakdowns(df, output_path):
    fig, ax = plt.subplots(figsize=(14, 5))
    maturities = ['y1', 'y2', 'y5', 'y10', 'y20', 'y30']
    
    errors = []
    for mat in maturities:
        rw = df[mat].shift(1)
        err = (df[mat] - rw) ** 2
        errors.append(err)
    
    avg_error = np.mean(errors, axis=0)
    rolling_error = pd.Series(avg_error).rolling(6).mean()
    ax.fill_between(df['date'], 0, rolling_error, alpha=0.3, color='#e74c3c')
    ax.plot(df['date'], rolling_error, color='#c0392b', lw=2, label='Rolling 6M MSE')
    
    regimes = {
        'COVID Shock': '2020-03-01',
        'QE Era': '2020-07-01',
        'Inflation': '2022-01-01',
        'High Rate': '2023-07-01'
    }

    colors_regime = ['#e74c3c', '#27ae60', '#f39c12', '#9b59b6']
    for i, (name, date) in enumerate(regimes.items()):
        ax.axvline(pd.to_datetime(date), color=colors_regime[i], linestyle='--', 
                   lw=2, alpha=0.8, label=name)
    
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Mean Squared Error', fontsize=11)
    ax.set_title('Prediction Error Spikes at Regime Transitions', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', ncol=3)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'✓ Saved {output_path.name}')


def main():
    base = Path(__file__).parent
    data_path = base / 'Data' / 'Processed' / 'aligned_panel.csv'
    figures_path = base / 'Reports' / 'figures'
    figures_path.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(data_path, parse_dates=['date'])
    print(f'Loaded {len(df)} observations')
    
    generate_yield_curve_fits(df, figures_path / 'yield_curve_fits.png')
    generate_regime_breakdowns(df, figures_path / 'regime_breakdowns.png')
    
    print('\n✓ All figures generated')

if __name__ == '__main__':
    main()
