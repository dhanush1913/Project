import numpy as np
import json
from pathlib import Path

REF_DIR = Path(__file__).parent.parent / 'data' / 'reference'

def validate_reference():
    print("\n" + "="*60)
    print("Reference Distribution Validation")
    print("="*60 + "\n")
    # Load all reference data
    class_hist = np.load(REF_DIR / 'class_histograms.npy')
    pixel_dist = np.load(REF_DIR / 'pixel_distributions.npy')
    
    with open(REF_DIR / 'feature_stats.json', 'r') as f:
        stats = json.load(f)
    # Validation checks
    checks = []
    # Check shapes
    checks.append(("Class histograms shape", class_hist.shape == (10, 32), class_hist.shape))
    checks.append(("Pixel distributions shape", pixel_dist.shape == (28, 28, 32), pixel_dist.shape))
    # Check epsilon smoothing (no zeros)
    checks.append(("Class histograms no zeros", (class_hist > 0).all(), f"Min: {class_hist.min():.2e}"))
    checks.append(("Pixel distributions no zeros", (pixel_dist > 0).all(), f"Min: {pixel_dist.min():.2e}"))
    # Check normalization (sum to 1)
    class_sums = class_hist.sum(axis=1)
    checks.append(("Class histograms normalized", np.allclose(class_sums, 1.0), f"Sums: {class_sums.min():.4f} - {class_sums.max():.4f}"))
    pixel_sums = pixel_dist.sum(axis=2)
    checks.append(("Pixel distributions normalized", np.allclose(pixel_sums, 1.0), f"Range: {pixel_sums.min():.4f} - {pixel_sums.max():.4f}"))
    # Check stats
    checks.append(("Feature stats epsilon", stats['epsilon'] == 1e-8, stats['epsilon']))
    checks.append(("Feature stats bins", stats['bins'] == 32, stats['bins']))
    checks.append(("Training samples", stats['num_training_samples'] == 60000, stats['num_training_samples']))
    # Print results
    print("Validation Results:")
    print("-" * 60)
    for name, passed, detail in checks:
        status = "✓" if passed else "✗"
        print(f"  {status} {name}: {detail}")
    all_passed = all(check[1] for check in checks)
    print("\n" + "="*60)
    if all_passed:
        print("✓ All validation checks passed")
        print("Reference distribution is ready for use.")
    else:
        print("✗ Some validation checks failed")
        print("Review reference distribution build process.")
    print("="*60 + "\n")
    return all_passed

if __name__ == '__main__':
    validate_reference()
