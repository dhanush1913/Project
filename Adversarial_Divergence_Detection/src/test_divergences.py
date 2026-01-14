import numpy as np
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from divergence.kl import kl_divergence, symmetric_kl
from divergence.js import js_divergence, js_divergence_normalized
from divergence.wasserstein import wasserstein_1d, wasserstein_histogram
from divergence.divergence_bundle import compute_divergences

EPSILON = 1e-8

def test_correctness():
    print("\n" + "="*60)
    print("Divergence Correctness Tests")
    print("="*60 + "\n")
    
    tests_passed = []
    
    # Test 1: Identical distributions should give zero divergence
    p = np.random.rand(32)
    p = p / p.sum()
    kl = kl_divergence(p, p)
    js = js_divergence(p, p)
    w1 = wasserstein_1d(p, p)
    tests_passed.append(("KL(P||P) = 0", np.abs(kl) < 1e-6, kl))
    tests_passed.append(("JS(P,P) = 0", np.abs(js) < 1e-6, js))
    tests_passed.append(("W1(P,P) = 0", np.abs(w1) < 1e-6, w1))
    
    # Test 2: JS symmetry
    q = np.random.rand(32)
    q = q / q.sum()    
    js_pq = js_divergence(p, q)
    js_qp = js_divergence(q, p)
    tests_passed.append(("JS symmetric", np.abs(js_pq - js_qp) < 1e-6, f"{js_pq:.6f} vs {js_qp:.6f}"))
    
    # Test 3: JS bounded by ln(2)
    tests_passed.append(("JS bounded", js_pq <= np.log(2) + 1e-6, f"{js_pq:.6f} <= {np.log(2):.6f}"))
    # Test 4: KL non-negative
    kl_pq = kl_divergence(p, q)
    tests_passed.append(("KL non-negative", kl_pq >= 0, kl_pq))
    # Test 5: Epsilon protection (no zeros)
    p_sparse = np.zeros(32)
    p_sparse[0] = 1.0
    kl_sparse = kl_divergence(p_sparse, q)
    tests_passed.append(("Epsilon protection", not np.isinf(kl_sparse) and not np.isnan(kl_sparse), kl_sparse))
    # Test 6: Bundle integration
    bundle = compute_divergences(p, q)
    tests_passed.append(("Bundle has all keys", set(bundle.keys()) == {'kl', 'js', 'wasserstein'}, bundle.keys()))
    # Print results
    print("Test Results:")
    print("-" * 60)
    for name, passed, detail in tests_passed:
        status = "✓" if passed else "✗"
        print(f"  {status} {name}: {detail}")
    
    all_passed = all(test[1] for test in tests_passed)
    return all_passed

def benchmark_performance():
    print("\n" + "="*60)
    print("Performance Benchmarks")
    print("="*60 + "\n")
    
    # Generate test data
    p = np.random.rand(32)
    q = np.random.rand(32)
    p = p / p.sum()
    q = q / q.sum()
    num_iterations = 10000
    benchmarks = []
    # KL divergence
    start = time.perf_counter()
    for _ in range(num_iterations):
        kl_divergence(p, q)
    kl_time = (time.perf_counter() - start) / num_iterations * 1000  # ms
    benchmarks.append(("KL divergence", kl_time, 0.5))
    # JS divergence
    start = time.perf_counter()
    for _ in range(num_iterations):
        js_divergence(p, q)
    js_time = (time.perf_counter() - start) / num_iterations * 1000
    benchmarks.append(("JS divergence", js_time, 0.5))
    # Wasserstein
    start = time.perf_counter()
    for _ in range(num_iterations):
        wasserstein_1d(p, q)
    w1_time = (time.perf_counter() - start) / num_iterations * 1000
    benchmarks.append(("Wasserstein 1D", w1_time, 0.6))
    start = time.perf_counter()
    for _ in range(num_iterations):
        compute_divergences(p, q)
    bundle_time = (time.perf_counter() - start) / num_iterations * 1000
    benchmarks.append(("Full bundle", bundle_time, 2.0))
    
    # Print results
    print(f"Average time over {num_iterations:,} iterations:\n")
    print(f"{'Metric':<20} {'Time (ms)':<12} {'Target':<12} {'Status'}")
    print("-" * 60)
    
    for name, time_ms, target_ms in benchmarks:
        status = "✓" if time_ms < target_ms else "⚠"
        print(f"{name:<20} {time_ms:>10.4f}   {target_ms:>10.2f}   {status}")
    
    return all(time_ms < target_ms for _, time_ms, target_ms in benchmarks)

def demo_adversarial_detection():
    print("\n" + "="*60)
    print("Adversarial Detection Demo")
    print("="*60 + "\n")
    
    clean = np.exp(-((np.arange(32) - 16)**2) / 50)
    clean = clean / clean.sum()
    adv = np.roll(clean, 3) + np.random.rand(32) * 0.05
    adv = adv / adv.sum()
    div_clean = compute_divergences(clean, clean)
    div_adv = compute_divergences(adv, clean)
    print("Clean vs Clean (should be ~0):")
    print(f"  KL: {div_clean['kl']:.6f}")
    print(f"  JS: {div_clean['js']:.6f}")
    print(f"  W1: {div_clean['wasserstein']:.6f}")
    
    print("\nAdversarial vs Clean (should be elevated):")
    print(f"  KL: {div_adv['kl']:.6f}")
    print(f"  JS: {div_adv['js']:.6f}")
    print(f"  W1: {div_adv['wasserstein']:.6f}")
    
    print("\nDivergence ratios (Adv/Clean):")
    print(f"  KL: {div_adv['kl'] / (div_clean['kl'] + 1e-8):.1f}x")
    print(f"  JS: {div_adv['js'] / (div_clean['js'] + 1e-8):.1f}x")
    print(f"  W1: {div_adv['wasserstein'] / (div_clean['wasserstein'] + 1e-8):.1f}x")

def main():
    """Run all tests and benchmarks."""
    print("\n" + "="*60)
    print("DIVERGENCE ENGINE VALIDATION")
    print("="*60)
    correctness_passed = test_correctness()
    performance_passed = benchmark_performance()
    demo_adversarial_detection()
    print("\n" + "="*60)
    if correctness_passed and performance_passed:
        print("✓ ALL TESTS PASSED")
        print("Divergence engine is mathematically correct and performant.")
    else:
        if not correctness_passed:
            print("✗ Correctness tests failed")
        if not performance_passed:
            print("⚠ Performance targets not met (but may be acceptable)")
    print("="*60 + "\n")

if __name__ == '__main__':
    main()
