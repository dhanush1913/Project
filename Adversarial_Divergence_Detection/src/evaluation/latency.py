import numpy as np
import time
from pathlib import Path

def profile_latency(detector_fn, test_samples, num_warmup=100, seed=42):
    np.random.seed(seed)
    print("\n" + "="*70)
    print("LATENCY PROFILING")
    print("="*70 + "\n")
    print(f"[1/3] Warmup ({num_warmup} iterations)...")
    for i in range(num_warmup):
        _ = detector_fn(test_samples[i % len(test_samples)])
    print("  ✓ Cache warmed")
    print(f"\n[2/3] Measuring latency ({len(test_samples)} samples)...")
    latencies_ns = []
    for sample in test_samples:
        start = time.perf_counter_ns()
        _ = detector_fn(sample)
        end = time.perf_counter_ns()
        latencies_ns.append(end - start)
    latencies_ms = np.array(latencies_ns) / 1_000_000
    print("  ✓ Measurement complete")
    stats = {
        'mean_ms': float(np.mean(latencies_ms)),
        'median_ms': float(np.median(latencies_ms)),
        'std_ms': float(np.std(latencies_ms)),
        'min_ms': float(np.min(latencies_ms)),
        'max_ms': float(np.max(latencies_ms)),
        'p50_ms': float(np.percentile(latencies_ms, 50)),
        'p90_ms': float(np.percentile(latencies_ms, 90)),
        'p95_ms': float(np.percentile(latencies_ms, 95)),
        'p99_ms': float(np.percentile(latencies_ms, 99)),
        'p999_ms': float(np.percentile(latencies_ms, 99.9)),
        'num_samples': len(test_samples),
        'seed': seed,
    }
    
    # Print report
    print("\n" + "="*70)
    print("LATENCY REPORT")
    print("="*70)
    
    print(f"\nSamples: {stats['num_samples']:,}")
    print(f"Random seed: {stats['seed']}")
    
    print(f"\nLatency Distribution (ms):")
    print(f"  Mean:       {stats['mean_ms']:>8.4f}")
    print(f"  Median:     {stats['median_ms']:>8.4f}")
    print(f"  Std Dev:    {stats['std_ms']:>8.4f}")
    print(f"  Min:        {stats['min_ms']:>8.4f}")
    print(f"  Max:        {stats['max_ms']:>8.4f}")
    
    print(f"\nPercentiles (ms):")
    print(f"  p50  (median): {stats['p50_ms']:>8.4f}")
    print(f"  p90:           {stats['p90_ms']:>8.4f}")
    print(f"  p95:           {stats['p95_ms']:>8.4f}")
    print(f"  p99:           {stats['p99_ms']:>8.4f}  ← CRITICAL")
    print(f"  p99.9:         {stats['p999_ms']:>8.4f}")
    
    # Validation
    target_p99 = 2.0
    meets_target = stats['p99_ms'] < target_p99
    
    print(f"\nValidation:")
    print(f"  Target p99:    {target_p99:.2f} ms")
    print(f"  Actual p99:    {stats['p99_ms']:.4f} ms")
    print(f"  Status:        {'✓ PASS' if meets_target else '✗ FAIL'}")
    print("="*70 + "\n")
    
    return stats

def analyze_latency_breakdown(latencies_by_stage):

    print("\n" + "="*70)
    print("LATENCY BREAKDOWN BY STAGE")
    print("="*70 + "\n")
    for stage_name, latencies in latencies_by_stage.items():
        latencies_ms = np.array(latencies) / 1_000_000
        
        print(f"{stage_name}:")
        print(f"  Count:      {len(latencies):>6,}")
        print(f"  Mean:       {np.mean(latencies_ms):>8.4f} ms")
        print(f"  p99:        {np.percentile(latencies_ms, 99):>8.4f} ms")
    
    print("="*70 + "\n")
