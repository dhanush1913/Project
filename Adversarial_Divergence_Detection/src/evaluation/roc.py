import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve
from pathlib import Path

def evaluate_roc(scores_clean, scores_adv, attack_name='adversarial'):
    y_true = np.concatenate([np.zeros(len(scores_clean)), np.ones(len(scores_adv))])
    y_score = np.concatenate([scores_clean, scores_adv])
    
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    tpr_at_fpr = {}
    for target_fpr in [0.01, 0.05, 0.10]:
        idx = np.argmin(np.abs(fpr - target_fpr))
        tpr_at_fpr[f'tpr_at_fpr_{int(target_fpr*100)}'] = float(tpr[idx])
    
    return {
        'attack': attack_name,
        'auc': float(auc),
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds,
        **tpr_at_fpr
    }

def plot_roc_curves(roc_results, save_path=None):
    plt.figure(figsize=(10, 8))
    
    for result in roc_results:
        plt.plot(result['fpr'], result['tpr'], 
                label=f"{result['attack']} (AUC = {result['auc']:.3f})",
                linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Adversarial Detection', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved ROC curve to {save_path}")
    
    plt.close()

def plot_score_distributions(scores_clean, scores_adv, attack_name='adversarial', save_path=None):
    plt.figure(figsize=(10, 6))
    
    plt.hist(scores_clean, bins=50, alpha=0.6, label='Clean', color='green', density=True)
    plt.hist(scores_adv, bins=50, alpha=0.6, label=attack_name, color='red', density=True)
    plt.xlabel('Detection Score', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title(f'Detection Score Distribution: Clean vs {attack_name}', 
             fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved score distribution to {save_path}")    
    plt.close()

def print_roc_summary(roc_results):
    print("\n" + "="*70)
    print("ROC Evaluation Summary")
    print("="*70 + "\n")
    
    print(f"{'Attack':<20} {'AUC':<10} {'TPR@1%FPR':<12} {'TPR@5%FPR':<12} {'TPR@10%FPR':<12}")
    print("-"*70)
    
    for result in roc_results:
        print(f"{result['attack']:<20} "
              f"{result['auc']:<10.4f} "
              f"{result.get('tpr_at_fpr_1', 0):<12.4f} "
              f"{result.get('tpr_at_fpr_5', 0):<12.4f} "
              f"{result.get('tpr_at_fpr_10', 0):<12.4f}")
    
    print("="*70 + "\n")
