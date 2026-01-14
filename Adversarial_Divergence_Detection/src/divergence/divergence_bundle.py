import numpy as np

try:
    from .kl import kl_divergence, EPSILON
    from .js import js_divergence
    from .wasserstein import wasserstein_histogram, wasserstein_1d
except ImportError:
    from kl import kl_divergence, EPSILON
    from js import js_divergence
    from wasserstein import wasserstein_histogram, wasserstein_1d

def compute_divergences(test_hist, ref_hist, epsilon=EPSILON):
    test_hist = np.maximum(test_hist, epsilon)
    ref_hist = np.maximum(ref_hist, epsilon)
    test_hist = test_hist / test_hist.sum()
    ref_hist = ref_hist / ref_hist.sum()
    kl = float(kl_divergence(test_hist, ref_hist, epsilon))
    js = float(js_divergence(test_hist, ref_hist, epsilon))
    w1 = float(wasserstein_histogram(test_hist, ref_hist, epsilon=epsilon))
    
    return {
        'kl': kl,
        'js': js,
        'wasserstein': w1,
    }

def compute_divergences_batch(test_hists, ref_hist, epsilon=EPSILON):
    test_hists = np.maximum(test_hists, epsilon)
    ref_hist = np.maximum(ref_hist, epsilon)
    test_hists = test_hists / test_hists.sum(axis=1, keepdims=True)
    ref_hist = ref_hist / ref_hist.sum()
    batch_size = test_hists.shape[0]
    kl_batch = kl_divergence(test_hists, ref_hist[np.newaxis, :], epsilon)
    js_batch = js_divergence(test_hists, ref_hist[np.newaxis, :], epsilon)
    w1_batch = np.array([
        wasserstein_histogram(test_hists[i], ref_hist, epsilon=epsilon)
        for i in range(batch_size)
    ])
    
    return {
        'kl': kl_batch,
        'js': js_batch,
        'wasserstein': w1_batch,
    }

def compute_all_divergences(test_image, ref_stats, epsilon=EPSILON):
    from ..features.histogram import compute_histogram
    test_hist = compute_histogram(test_image, bins=32)
    ref_hist = ref_stats['class_histograms'].mean(axis=0)
    return compute_divergences(test_hist, ref_hist, epsilon)
