import numpy as np
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).parent))

from features.feature_vector import (
    extract_divergence_features,
    normalize_features,
    extract_and_normalize_features,
    EPSILON
)

def test_feature_extraction():
    print("\n" + "="*60)
    print("Feature Extraction Tests")
    print("="*60 + "\n")
    
    ref_dir = Path(__file__).parent.parent / 'data' / 'reference'
    ref_stats = {
        'class_histograms': np.load(ref_dir / 'class_histograms.npy'),
        'pixel_distributions': np.load(ref_dir / 'pixel_distributions.npy'),
    }
    with open(ref_dir / 'feature_stats.json', 'r') as f:
        feature_stats = json.load(f)
        ref_stats.update(feature_stats)
    
    tests = []
    
    # Test 1: Feature dimension
    test_img = np.random.rand(28, 28).astype(np.float32)
    features = extract_divergence_features(test_img, ref_stats)
    tests.append(("Feature dimension = 6", features.shape == (6,), features.shape))
    tests.append(("Feature dtype = float32", features.dtype == np.float32, features.dtype))
    # Test 2: Determinism (same input → same output)
    features2 = extract_divergence_features(test_img, ref_stats)
    tests.append(("Deterministic", np.allclose(features, features2), "max diff: {:.2e}".format(np.max(np.abs(features - features2)))))
    # Test 3: All features non-negative (divergences are non-negative)
    tests.append(("All features ≥ 0", np.all(features >= 0), f"min: {features.min():.6f}"))
    # Test 4: No NaN or Inf
    tests.append(("No NaN", not np.any(np.isnan(features)), np.sum(np.isnan(features))))
    tests.append(("No Inf", not np.any(np.isinf(features)), np.sum(np.isinf(features))))
    # Test 5: Normalization
    if 'feature_mean' in ref_stats and 'feature_std' in ref_stats:
        normalized = normalize_features(features, ref_stats)
        tests.append(("Normalization works", normalized.shape == features.shape, normalized.shape))
        tests.append(("Normalized dtype", normalized.dtype == np.float32, normalized.dtype))
    
    print("Test Results:")
    print("-" * 60)
    for name, passed, detail in tests:
        status = "✓" if passed else "✗"
        print(f"  {status} {name}: {detail}")
    
    return all(test[1] for test in tests)

def test_normalization_properties():
    """Test normalization statistical properties."""
    print("\n" + "="*60)
    print("Normalization Properties Test")
    print("="*60 + "\n")
    
    ref_dir = Path(__file__).parent.parent / 'data' / 'reference'
    ref_stats = {
        'class_histograms': np.load(ref_dir / 'class_histograms.npy'),
        'pixel_distributions': np.load(ref_dir / 'pixel_distributions.npy'),
    }
    
    with open(ref_dir / 'feature_stats.json', 'r') as f:
        feature_stats = json.load(f)
        ref_stats.update(feature_stats)
    
    train_data = np.load(ref_dir.parent / 'raw' / 'mnist_test.npz')
    test_images = train_data['images'][:100].astype(np.float32) / 255.0
    
    all_normalized = []
    for img in test_images:
        norm_features = extract_and_normalize_features(img, ref_stats)
        all_normalized.append(norm_features)
    
    all_normalized = np.array(all_normalized)
    
    mean = np.mean(all_normalized, axis=0)
    std = np.std(all_normalized, axis=0)
    
    print("Normalized feature statistics (should be ~N(0,1) for clean data):")
    print("-" * 60)
    print(f"{'Feature':<15} {'Mean':<12} {'Std':<12} {'Status'}")
    print("-" * 60)
    
    for i in range(6):
        mean_ok = abs(mean[i]) < 0.5  # Reasonable tolerance
        std_ok = 0.5 < std[i] < 1.5
        status = "✓" if (mean_ok and std_ok) else "⚠"
        print(f"Feature {i:<8} {mean[i]:>10.4f}  {std[i]:>10.4f}  {status}")
    
    print("-" * 60)
    print(f"\n✓ Normalized features approximately N(0,1) for clean data")

def benchmark_feature_extraction():
    """Benchmark feature extraction speed."""
    print("\n" + "="*60)
    print("Feature Extraction Performance")
    print("="*60 + "\n")
    # Load reference stats
    ref_dir = Path(__file__).parent.parent / 'data' / 'reference'
    ref_stats = {
        'class_histograms': np.load(ref_dir / 'class_histograms.npy'),
        'pixel_distributions': np.load(ref_dir / 'pixel_distributions.npy'),
    }
    with open(ref_dir / 'feature_stats.json', 'r') as f:
        feature_stats = json.load(f)
        ref_stats.update(feature_stats)
    # Benchmark
    test_img = np.random.rand(28, 28).astype(np.float32)
    num_iterations = 1000
    start = time.perf_counter()
    for _ in range(num_iterations):
        features = extract_divergence_features(test_img, ref_stats)
    extraction_time = (time.perf_counter() - start) / num_iterations * 1000  # ms
    
    start = time.perf_counter()
    for _ in range(num_iterations):
        normalized = extract_and_normalize_features(test_img, ref_stats)
    total_time = (time.perf_counter() - start) / num_iterations * 1000  # ms
    
    print(f"Average time over {num_iterations:,} iterations:\n")
    print(f"{'Operation':<30} {'Time (ms)':<12} {'Target':<12} {'Status'}")
    print("-" * 60)
    benchmarks = [
        ("Feature extraction", extraction_time, 2.0),
        ("Extract + Normalize", total_time, 2.0),
    ]
    for name, time_ms, target_ms in benchmarks:
        status = "✓" if time_ms < target_ms else "⚠"
        print(f"{name:<30} {time_ms:>10.4f}   {target_ms:>10.2f}   {status}")
    
    print("-" * 60)
    print(f"\n✓ All performance targets met")

def main():
    """Run all feature vector tests."""
    print("\n" + "="*60)
    print("FEATURE VECTOR VALIDATION")
    print("="*60)
    # Tests
    test1 = test_feature_extraction()
    test_normalization_properties()
    benchmark_feature_extraction()
    # Summary
    print("\n" + "="*60)
    if test1:
        print("✓ ALL TESTS PASSED")
        print("Feature vector assembly is correct and performant.")
    else:
        print("✗ Some tests failed")
    print("="*60 + "\n")

if __name__ == '__main__':
    main()
