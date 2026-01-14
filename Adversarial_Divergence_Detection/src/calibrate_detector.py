import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from detector.threshold_detector import ThresholdDetector, calibrate_weights, calibrate_threshold
from features.feature_vector import extract_and_normalize_features
from attacks.adversarial_generator import generate_fgsm_dataset, generate_pgd_dataset
from evaluation.roc import evaluate_roc, plot_roc_curves, print_roc_summary, plot_score_distributions

def calibrate_detector(num_clean=500, num_adv_per_attack=250, epsilon=0.3, fpr_penalty=2.0):
    print("\n" + "="*70)
    print("DETECTOR CALIBRATION")
    print("="*70 + "\n")
    
    print("[1/6] Loading reference statistics...")
    ref_dir = Path(__file__).parent.parent / 'data' / 'reference'
    ref_stats = {
        'class_histograms': np.load(ref_dir / 'class_histograms.npy'),
        'pixel_distributions': np.load(ref_dir / 'pixel_distributions.npy'),
    }
    
    import json
    with open(ref_dir / 'feature_stats.json', 'r') as f:
        feature_stats = json.load(f)
        ref_stats.update(feature_stats)

    print("  ✓ Loaded reference statistics")
    print("[2/6] Loading clean validation data...")
    test_data = np.load(ref_dir.parent / 'raw' / 'mnist_test.npz')
    clean_images = test_data['images'][:num_clean].astype(np.float32) / 255.0
    print(f"  ✓ Loaded {len(clean_images)} clean images")
    # Generate adversarial examples
    print("[3/6] Generating adversarial examples...")
    fgsm_images = generate_fgsm_dataset(clean_images[:num_adv_per_attack], epsilon=epsilon)
    pgd_images = generate_pgd_dataset(clean_images[:num_adv_per_attack], epsilon=epsilon, num_steps=40)
    print(f"  ✓ Generated FGSM: {len(fgsm_images)}, PGD: {len(pgd_images)}")
    
    # Extract features
    print("[4/6] Extracting features...")
    clean_features = np.array([extract_and_normalize_features(img, ref_stats) for img in clean_images])
    fgsm_features = np.array([extract_and_normalize_features(img, ref_stats) for img in fgsm_images])
    pgd_features = np.array([extract_and_normalize_features(img, ref_stats) for img in pgd_images])
    print(f"  ✓ Extracted features (shape: {clean_features.shape})")
    
    # Calibrate weights (use FGSM for calibration, validate on both)
    print("[5/6] Calibrating weights...")
    weights = calibrate_weights(clean_features, fgsm_features, method='roc_auc')
    
    feature_names = ['KL_global', 'JS_global', 'W1_global', 'KL_pixel_mean', 'KL_pixel_max', 'W1_pixel_mean']
    print("\n  Feature weights (ROC AUC-based):")
    for name, weight in zip(feature_names, weights):
        print(f"    {name:<20} {weight:.4f}")
    
    # Compute scores with calibrated weights
    detector_temp = ThresholdDetector(weights=weights, threshold=0.0)
    scores_clean = detector_temp.compute_score(clean_features)
    scores_fgsm = detector_temp.compute_score(fgsm_features)
    scores_pgd = detector_temp.compute_score(pgd_features)
    
    # Calibrate threshold
    print("\n[6/6] Calibrating threshold...")
    threshold = calibrate_threshold(scores_clean, scores_fgsm, fpr_penalty=fpr_penalty)
    print(f"  ✓ Optimal threshold: {threshold:.4f}")
    # Create final detector
    detector = ThresholdDetector(weights=weights, threshold=threshold)
    
    # Save configuration
    config_path = Path(__file__).parent.parent / 'config' / 'detector_config.yaml'
    detector.save_config(config_path)
    print(f"\n  ✓ Saved configuration to {config_path}")
    # Evaluate ROC
    print("\n" + "="*70)
    print("ROC EVALUATION")
    print("="*70)
    
    roc_fgsm = evaluate_roc(scores_clean, scores_fgsm, attack_name='FGSM (ε=0.3)')
    roc_pgd = evaluate_roc(scores_clean, scores_pgd, attack_name='PGD (ε=0.3, 40 steps)')
    roc_results = [roc_fgsm, roc_pgd]
    print_roc_summary(roc_results)
    # Plot ROC curves
    benchmarks_dir = Path(__file__).parent.parent / 'benchmarks'
    benchmarks_dir.mkdir(exist_ok=True)
    
    plot_roc_curves(roc_results, save_path=benchmarks_dir / 'roc_curves.png')
    plot_score_distributions(scores_clean, scores_fgsm, 'FGSM', 
                            save_path=benchmarks_dir / 'scores_fgsm.png')
    plot_score_distributions(scores_clean, scores_pgd, 'PGD',
                            save_path=benchmarks_dir / 'scores_pgd.png')
    
    print("\n" + "="*70)
    print("✓ CALIBRATION COMPLETE")
    print("="*70 + "\n")
    return detector

if __name__ == '__main__':
    calibrate_detector(num_clean=500, num_adv_per_attack=250, epsilon=0.3, fpr_penalty=2.0)
