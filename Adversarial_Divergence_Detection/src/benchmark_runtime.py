import numpy as np
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from runtime.cache import get_cache
from runtime.fastpath import FastPathDetector

def benchmark_fast_path(num_samples=1000, num_warmup=100):
    print("\n" + "="*70)
    print("FAST-PATH DETECTOR PERFORMANCE BENCHMARK")
    print("="*70 + "\n")
    
    # Initialize cache
    print("[1/5] Loading runtime cache...")
    cache = get_cache()
    data_dir = Path(__file__).parent.parent / 'data' / 'reference'
    config_dir = Path(__file__).parent.parent / 'config'
    cache.load_all(data_dir, config_dir)
    print("  ✓ Cache loaded")
    
    # Initialize detector
    detector = FastPathDetector(cache)
    print("  ✓ Fast-path detector initialized")
    
    # Load test images
    print("\n[2/5] Loading test images...")
    raw_dir =data_dir.parent / 'raw'  # data_dir is data/reference, so parent is data
    test_data = np.load(raw_dir / 'mnist_test.npz')
    clean_images = test_data['images'][:num_samples].astype(np.float32) / 255.0
    print(f"  ✓ Loaded {len(clean_images)} clean test images")
    
    # Warmup (important for fair benchmarking)
    print("\n[3/5] Warming up...")
    for i in range(num_warmup):
        _ = detector.detect(clean_images[i % len(clean_images)])
    print(f"  ✓ Completed {num_warmup} warmup iterations")
    
    # Benchmark
    print("\n[4/5] Benchmarking...")
    latencies = []
    stage1_exits = 0
    stage2_executions = 0
    
    for img in clean_images:
        start = time.perf_counter()
        result = detector.detect(img)
        latency = (time.perf_counter() - start) * 1000  # ms
        
        latencies.append(latency)
        
        if result['stage'] == 1:
            stage1_exits += 1
        else:
            stage2_executions += 1
    
    latencies = np.array(latencies)
    
    # Statistics
    print("  ✓ Benchmark complete")
    
    # Analysis
    print("\n[5/5] Performance Analysis:")
    print("-" * 70)
    
    stats = {
        'mean': np.mean(latencies),
        'median': np.median(latencies),
        'std': np.std(latencies),
        'min': np.min(latencies),
        'max': np.max(latencies),
        'p50': np.percentile(latencies, 50),
        'p90': np.percentile(latencies, 90),
        'p95': np.percentile(latencies, 95),
        'p99': np.percentile(latencies, 99),
        'p999': np.percentile(latencies, 99.9),
        'stage1_exit_rate': stage1_exits / num_samples,
        'stage2_rate': stage2_executions / num_samples,
    }
    
    print(f"\nLatency Distribution (ms):")
    print(f"  Mean:     {stats['mean']:.4f}")
    print(f"  Median:   {stats['median']:.4f}")
    print(f"  Std Dev:  {stats['std']:.4f}")
    print(f"  Min:      {stats['min']:.4f}")
    print(f"  Max:      {stats['max']:.4f}")
    
    print(f"\nPercentiles (ms):")
    print(f"  p50:      {stats['p50']:.4f}")
    print(f"  p90:      {stats['p90']:.4f}")
    print(f"  p95:      {stats['p95']:.4f}")
    print(f"  p99:      {stats['p99']:.4f}  ← CRITICAL")
    print(f"  p99.9:    {stats['p999']:.4f}")
    
    print(f"\nFast-Path Efficiency:")
    print(f"  Stage 1 exits:     {stage1_exits:4d} ({100*stats['stage1_exit_rate']:.1f}%)")
    print(f"  Stage 2 full runs: {stage2_executions:4d} ({100*stats['stage2_rate']:.1f}%)")
    
    # Validation
    print(f"\nValidation:")
    meets_p99 = stats['p99'] < 2.0
    good_exit_rate = stats['stage1_exit_rate'] > 0.85  # >85% early exit
    
    print(f"  ✓ p99 < 2ms:        {meets_p99} ({stats['p99']:.4f} ms)")
    print(f"  ✓ Stage 1 >85%:     {good_exit_rate} ({100*stats['stage1_exit_rate']:.1f}%)")
    
    print("\n" + "="*70)
    if meets_p99 and good_exit_rate:
        print("✓ PERFORMANCE TARGETS MET")
    else:
        print("⚠ PERFORMANCE TARGETS NOT MET")
    print("="*70 + "\n")
    
    return stats

if __name__ == '__main__':
    benchmark_fast_path(num_samples=1000, num_warmup=100)
