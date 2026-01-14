import numpy as np
import sys
from pathlib import Path

# Add parent to path for divergence imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from features.histogram import compute_histogram, BINS
from divergence.kl import kl_divergence
from divergence.js import js_divergence
from divergence.wasserstein import wasserstein_1d
from divergence.divergence_bundle import compute_divergences

EPSILON = 1e-6  # For normalization stability

def extract_divergence_features(image, ref_stats, epsilon=1e-8):
    test_hist = compute_histogram(image, bins=BINS)
    ref_hist_global = ref_stats['class_histograms'].mean(axis=0)
    
    global_divs = compute_divergences(test_hist, ref_hist_global, epsilon)
    H, W = image.shape
    kl_rows = []
    w1_rows = []
    
    for i in range(H):
        row_hist = compute_histogram(image[i, :], bins=BINS)
        ref_row = ref_stats['pixel_distributions'][i, :, :].mean(axis=0)
        
        kl = kl_divergence(row_hist, ref_row, epsilon)
        w1 = wasserstein_1d(row_hist, ref_row, epsilon)
        
        kl_rows.append(kl)
        w1_rows.append(w1)
    
    features = np.array([
        global_divs['kl'],
        global_divs['js'],
        global_divs['wasserstein'],
        np.mean(kl_rows),
        np.max(kl_rows),
        np.mean(w1_rows),
    ], dtype=np.float32)
    
    return features

def normalize_features(features, norm_stats, epsilon=EPSILON):
    mean = np.array(norm_stats['feature_mean'], dtype=np.float32)
    std = np.array(norm_stats['feature_std'], dtype=np.float32)
    
    return (features - mean) / (std + epsilon)

def extract_and_normalize_features(image, ref_stats, epsilon=EPSILON):
    features = extract_divergence_features(image, ref_stats, epsilon)
    
    if 'feature_mean' in ref_stats and 'feature_std' in ref_stats:
        features = normalize_features(features, ref_stats, epsilon)
    
    return features
