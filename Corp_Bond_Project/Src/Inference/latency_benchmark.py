import time
import numpy as np
import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from Src.Inference.predictor import YieldCurvePredictor


def benchmark_inference(n_runs=1000, input_dim=15, model_type='attention'):

    predictor = YieldCurvePredictor(model_type=model_type, input_dim=input_dim)
    inputs = [np.random.randn(input_dim).astype(np.float32) for _ in range(n_runs)]
    
    latencies = []
    for x in inputs:
        start = time.perf_counter()
        _ = predictor.predict(x)
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # Convert to ms
    
    return {
        'mean_ms': np.mean(latencies),
        'std_ms': np.std(latencies),
        'min_ms': np.min(latencies),
        'max_ms': np.max(latencies),
        'p99_ms': np.percentile(latencies, 99),
        'n_runs': n_runs
    }


def benchmark_batch(batch_sizes=[1, 8, 32, 128], input_dim=15):
    """Benchmark at different batch sizes."""
    predictor = YieldCurvePredictor(model_type='attention', input_dim=input_dim)
    results = []
    
    for bs in batch_sizes:
        x = np.random.randn(bs, input_dim).astype(np.float32)
        x_tensor = torch.tensor(x)
        
        for _ in range(10):
            with torch.no_grad():
                predictor.model(x_tensor)
        
        times = []
        for _ in range(100):
            start = time.perf_counter()
            with torch.no_grad():
                predictor.model(x_tensor)
            times.append((time.perf_counter() - start) * 1000)
        
        results.append({
            'batch_size': bs,
            'mean_ms': np.mean(times),
            'per_sample_ms': np.mean(times) / bs
        })
    
    return results


def run_full_benchmark():
    """Run complete benchmark suite and print results."""
    print("=" * 60)
    print("NEURAL PEA INFERENCE LATENCY BENCHMARK")
    print("=" * 60)
    print(f"Device: CPU | PyTorch {torch.__version__}")
    print()
    
    print("Single Sample Inference (1000 runs):")
    print("-" * 40)
    
    for model in ['base', 'attention']:
        res = benchmark_inference(n_runs=1000, model_type=model)
        print(f"  {model.upper():10} | Mean: {res['mean_ms']:.3f} ms | "
              f"P99: {res['p99_ms']:.3f} ms | Std: {res['std_ms']:.3f} ms")
    
    print()
    print("Batch Inference (Attention Model):")
    print("-" * 40)
    
    batch_results = benchmark_batch()
    for r in batch_results:
        print(f"  Batch {r['batch_size']:3d} | Total: {r['mean_ms']:.3f} ms | "
              f"Per sample: {r['per_sample_ms']:.4f} ms")
    
    print()
    print("=" * 60)
    
    single = benchmark_inference(n_runs=1000, model_type='attention')
    if single['mean_ms'] < 1.0:
        print(f"✓ SUB-MILLISECOND INFERENCE ACHIEVED: {single['mean_ms']:.3f} ms")
    else:
        print(f"⚠ Inference time: {single['mean_ms']:.3f} ms (target: <1 ms)")
    
    print("=" * 60)
    return single


if __name__ == "__main__":
    run_full_benchmark()
