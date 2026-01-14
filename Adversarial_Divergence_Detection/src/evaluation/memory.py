import tracemalloc
import numpy as np

def profile_memory(detector_fn, test_samples, num_samples=100):
    print("\n" + "="*70)
    print("MEMORY PROFILING")
    print("="*70 + "\n")
    tracemalloc.start()
    
    baseline = tracemalloc.take_snapshot()
    baseline_size = sum(stat.size for stat in baseline.statistics('lineno'))
    print(f"[1/3] Baseline memory: {baseline_size / 1024 / 1024:.2f} MB")
    print(f"\n[2/3] Running detection on {num_samples} samples...")
    for i in range(min(num_samples, len(test_samples))):
        _ = detector_fn(test_samples[i])
    
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    current_mb = current / 1024 / 1024
    peak_mb = peak / 1024 / 1024
    increase_mb = (peak - baseline_size) / 1024 / 1024
    
    print(f"  ✓ Detection complete")
    print("\n[3/3] Memory Statistics:")
    print(f"  Current RSS:     {current_mb:>8.2f} MB")
    print(f"  Peak RSS:        {peak_mb:>8.2f} MB")
    print(f"  Increase:        {increase_mb:>8.2f} MB")
    target_mb = 8.0
    meets_target = peak_mb < target_mb
    
    print(f"\nValidation:")
    print(f"  Target:          {target_mb:.2f} MB")
    print(f"  Actual peak:     {peak_mb:.2f} MB")
    print(f"  Status:          {'✓ PASS' if meets_target else '✗ FAIL'}")
    
    print("="*70 + "\n")
    
    return {
        'baseline_mb': baseline_size / 1024 / 1024,
        'current_mb': current_mb,
        'peak_mb': peak_mb,
        'increase_mb': increase_mb,
        'meets_target': meets_target,
    }
