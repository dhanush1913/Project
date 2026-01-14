import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from runtime.cache import get_cache
from runtime.fastpath import FastPathDetector
from attacks.adversarial_generator import generate_fgsm_dataset, generate_pgd_dataset
from evaluation.latency import profile_latency
from evaluation.memory import profile_memory
from evaluation.accuracy import validate_accuracy_roc, save_metrics_csv

# Fixed seed for reproducibility
RANDOM_SEED = 42
def run_complete_benchmark(num_clean=500, num_adv=250):
    print("\n" + "="*70)
    print("COMPLETE HONEST BENCHMARK SUITE")
    print("="*70)
    print(f"\nFixed random seed: {RANDOM_SEED}")
    print(f"Clean samples: {num_clean}")
    print(f"Adversarial samples per attack: {num_adv}")
    np.random.seed(RANDOM_SEED)
    
    # Initialize
    print("\n" + "-"*70)
    print("INITIALIZATION")
    print("-"*70)
    cache = get_cache()
    data_dir = Path(__file__).parent.parent / 'data' / 'reference'
    config_dir = Path(__file__).parent.parent / 'config'
    cache.load_all(data_dir, config_dir)
    detector = FastPathDetector(cache)
    print("✓ Detector initialized")
    # Load data
    raw_dir = data_dir.parent / 'raw'
    test_data = np.load(raw_dir / 'mnist_test.npz')
    clean_images = test_data['images'][:num_clean].astype(np.float32) / 255.0
    
    # Generate attacks
    fgsm_images = generate_fgsm_dataset(clean_images[:num_adv], epsilon=0.3, use_random_gradient=True)
    pgd_images = generate_pgd_dataset(clean_images[:num_adv], epsilon=0.3, num_steps=40, use_random_gradient=True)
    print(f"✓ Loaded {len(clean_images)} clean images")
    print(f"✓ Generated {len(fgsm_images)} FGSM images")
    print(f"✓ Generated {len(pgd_images)} PGD images")
    # 1. Latency Profiling
    print("\n" + "-"*70)
    print("1. LATENCY PROFILING")
    print("-"*70)
    
    latency_stats = profile_latency(
        detector_fn=lambda img: detector.detect(img),
        test_samples=clean_images,
        num_warmup=100,
        seed=RANDOM_SEED
    )
    
    # 2. Memory Profiling
    print("\n" + "-"*70)
    print("2. MEMORY PROFILING")
    print("-"*70)
    memory_stats = profile_memory(
        detector_fn=lambda img: detector.detect(img),
        test_samples=clean_images,
        num_samples=100
    )
    
    # 3. Accuracy Validation
    print("\n" + "-"*70)
    print("3. ACCURACY VALIDATION")
    print("-"*70)
    metrics_fgsm = validate_accuracy_roc(
        detector_fn=lambda img: detector.detect(img),
        clean_samples=clean_images[:num_adv],
        adv_samples=fgsm_images,
        attack_name='FGSM_eps0.3',
        seed=RANDOM_SEED
    )
    metrics_pgd = validate_accuracy_roc(
        detector_fn=lambda img: detector.detect(img),
        clean_samples=clean_images[:num_adv],
        adv_samples=pgd_images,
        attack_name='PGD_eps0.3_steps40',
        seed=RANDOM_SEED
    )
    # Save results
    benchmarks_dir = Path(__file__).parent.parent / 'benchmarks'
    save_metrics_csv([metrics_fgsm, metrics_pgd], benchmarks_dir / 'accuracy_auc.csv')
    # Final Summary
    print("\n" + "="*70)
    print("BENCHMARK SUMMARY")
    print("="*70)
    
    print(f"\nLatency:")
    print(f"  Mean:      {latency_stats['mean_ms']:.4f} ms")
    print(f"  p99:       {latency_stats['p99_ms']:.4f} ms  (target: <2.0 ms)")
    print(f"  Status:    {'✓ PASS' if latency_stats['p99_ms'] < 2.0 else '✗ FAIL'}")
    
    print(f"\nMemory:")
    print(f"  Peak:      {memory_stats['peak_mb']:.2f} MB  (target: <8.0 MB)")
    print(f"  Status:    {'✓ PASS' if memory_stats['meets_target'] else '✗ FAIL'}")
    
    print(f"\nAccuracy (FGSM):")
    print(f"  AUC:       {metrics_fgsm['auc']:.4f}  (target: ≥0.90)")
    print(f"  Accuracy:  {metrics_fgsm['accuracy']:.4f}  (target: ≥0.95)")
    print(f"  FPR:       {metrics_fgsm['fpr']:.4f}  (target: ≤0.01)")
    
    print(f"\nAccuracy (PGD):")
    print(f"  AUC:       {metrics_pgd['auc']:.4f}  (target: ≥0.90)")
    print(f"  Accuracy:  {metrics_pgd['accuracy']:.4f}  (target: ≥0.95)")
    print(f"  FPR:       {metrics_pgd['fpr']:.4f}  (target: ≤0.01)")
    
    # Overall pass/fail
    latency_pass = latency_stats['p99_ms'] < 2.0
    memory_pass = memory_stats['meets_target']
    fgsm_pass = metrics_fgsm['auc'] >= 0.90 and metrics_fgsm['fpr'] <= 0.05
    pgd_pass = metrics_pgd['auc'] >= 0.90 and metrics_pgd['fpr'] <= 0.05
    
    all_pass = latency_pass and memory_pass and fgsm_pass and pgd_pass
    
    print(f"\n{'='*70}")
    if all_pass:
        print("✓ ALL BENCHMARKS PASSED")
    else:
        print("⚠ SOME BENCHMARKS NEED ATTENTION")
        if not latency_pass:
            print("  - Latency p99 exceeds 2ms (consider fast-path tuning)")
        if not memory_pass:
            print("  - Memory exceeds 8MB")
        if not fgsm_pass or not pgd_pass:
            print("  - Accuracy/FPR targets not fully met")
    print("="*70 + "\n")

if __name__ == '__main__':
    run_complete_benchmark(num_clean=500, num_adv=250)
