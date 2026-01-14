import numpy as np
from pathlib import Path

EPSILON = 1e-8

def _compute_histogram_fast(image, bins=32):
    hist, _ = np.histogram(image.ravel(), bins=bins, range=(0.0, 1.0), density=False)
    prob = hist.astype(np.float32) / hist.sum()
    return np.maximum(prob, EPSILON)

def _js_divergence_fast(p, q):
    p = np.maximum(p, EPSILON)
    q = np.maximum(q, EPSILON)
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * (np.log(p) - np.log(m)))
    kl_qm = np.sum(q * (np.log(q) - np.log(m)))
    return np.clip(0.5 * (kl_pm + kl_qm), 0.0, np.log(2))

def _wasserstein_1d_fast(p, q):
    p = np.maximum(p, EPSILON)
    q = np.maximum(q, EPSILON)
    p = p / p.sum()
    q = q / q.sum()
    cdf_p = np.cumsum(p)
    cdf_q = np.cumsum(q)
    return np.sum(np.abs(cdf_p - cdf_q))

class FastPathDetector:

    
    def __init__(self, cache):
        self.cache = cache
        
        self.SAFE_JS_THRESHOLD = 0.05  # Very low JS
        self.SAFE_W1_THRESHOLD = 0.08  # Very low Wasserstein
    
    def detect_stage1(self, image):
        test_hist = _compute_histogram_fast(image, bins=32)
        ref_hist = self.cache.get_ref_histogram_global()
        
        js = _js_divergence_fast(test_hist, ref_hist)
        w1 = _wasserstein_1d_fast(test_hist, ref_hist)
        exit_early = (js < self.SAFE_JS_THRESHOLD) and (w1 < self.SAFE_W1_THRESHOLD)
        return exit_early, float(js), float(w1)
    
    def detect_stage2_full(self, image):
        from features.feature_vector import extract_divergence_features
        
        ref_stats = {
            'class_histograms': self.cache.ref_histograms,
            'pixel_distributions': self.cache.pixel_distributions,
        }
        
        features_raw = extract_divergence_features(image, ref_stats)
        features_norm = (features_raw - self.cache.feature_mean) / (self.cache.feature_std + 1e-6)
        score = np.dot(features_norm, self.cache.detector_weights)
        prediction = score > self.cache.detector_threshold
        
        return {
            'prediction': bool(prediction),
            'score': float(score),
            'features_raw': features_raw,
            'features_norm': features_norm,
        }
    
    def detect(self, image):
        exit_early, js, w1 = self.detect_stage1(image)
        
        if exit_early:
            return {
                'prediction': False,  # Clean
                'score': 0.0,  # Below threshold
                'stage': 1,  # Exited at Stage 1
                'js': js,
                'w1': w1,
            }
        
        result = self.detect_stage2_full(image)
        result['stage'] = 2
        
        return result
    
    def batch_detect(self, images):
        results = []
        for img in images:
            result = self.detect(img)
            results.append(result)
        return results
