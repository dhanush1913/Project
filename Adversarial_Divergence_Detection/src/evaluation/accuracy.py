import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
import csv
from pathlib import Path

def validate_accuracy_roc(detector_fn, clean_samples, adv_samples, attack_name, seed=42):
    np.random.seed(seed)
    
    print(f"\n{'='*70}")
    print(f"ACCURACY & ROC VALIDATION: {attack_name}")
    print(f"{'='*70}\n")
    print(f"[1/3] Running detection...")
    print(f"  Clean samples: {len(clean_samples)}")
    print(f"  Attack samples: {len(adv_samples)}")
    clean_preds = [detector_fn(img)['prediction'] for img in clean_samples]
    adv_preds = [detector_fn(img)['prediction'] for img in adv_samples]
    y_true = np.concatenate([np.zeros(len(clean_samples)), np.ones(len(adv_samples))])
    y_pred = np.concatenate([clean_preds, adv_preds])
    
    clean_scores = [detector_fn(img).get('score', 0) for img in clean_samples]
    adv_scores = [detector_fn(img).get('score', 0) for img in adv_samples]
    y_score = np.concatenate([clean_scores, adv_scores])
    
    print("  ✓ Detection complete")
    print(f"\n[2/3] Computing metrics...")
    
    auc = roc_auc_score(y_true, y_score)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    fp = np.sum((y_true == 0) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    tpr = recall
    
    metrics = {
        'attack': attack_name,
        'auc': float(auc),
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'tpr': float(tpr),
        'fpr': float(fpr),
        'num_clean': len(clean_samples),
        'num_adv': len(adv_samples),
        'seed': seed,
    }
    
    print("  ✓ Metrics computed")
    
   
    print(f"\n[3/3] Results:")
    print(f"  ROC AUC:       {metrics['auc']:>7.4f}  (target: ≥0.90)")
    print(f"  Accuracy:      {metrics['accuracy']:>7.4f}  (target: ≥0.95)")
    print(f"  Precision:     {metrics['precision']:>7.4f}")
    print(f"  Recall (TPR):  {metrics['recall']:>7.4f}")
    print(f"  FPR:           {metrics['fpr']:>7.4f}  (target: ≤0.01)")
    
    meets_auc = auc >= 0.90
    meets_acc = accuracy >= 0.95
    meets_fpr = fpr <= 0.01
    
    print(f"\nValidation:")
    print(f"  AUC ≥ 0.90:    {'✓' if meets_auc else '✗'}")
    print(f"  Acc ≥ 0.95:    {'✓' if meets_acc else '✗'}")
    print(f"  FPR ≤ 0.01:    {'✓' if meets_fpr else '✗'}")
    
    print(f"{'='*70}\n")
    
    return metrics

def save_metrics_csv(metrics_list, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fieldnames = ['attack', 'auc', 'accuracy', 'precision', 'recall', 'tpr', 'fpr', 
                  'num_clean', 'num_adv', 'seed']
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics_list)
    
    print(f"✓ Saved metrics to {output_path}")
