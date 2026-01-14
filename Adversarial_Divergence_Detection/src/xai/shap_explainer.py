import numpy as np

def group_pixels(image, group_size=4):
    H, W = image.shape
    groups = []
    
    for i in range(0, H, group_size):
        for j in range(0, W, group_size):
            block = []
            for di in range(group_size):
                for dj in range(group_size):
                    if i+di < H and j+dj < W:
                        block.append((i+di, j+dj))
            if block:
                groups.append(block)
    
    return groups

def compute_feature_importance(image, ref_stats, detector, num_samples=100):
    from features.feature_vector import extract_and_normalize_features
    baseline_features = extract_and_normalize_features(image, ref_stats)
    baseline_score = detector.compute_score(baseline_features)
    groups = group_pixels(image, group_size=4)
    group_scores = []
    for group in groups:
        perturbed = image.copy()
        group_mean = np.mean([perturbed[i, j] for i, j in group])
        for i, j in group:
            perturbed[i, j] = group_mean * 0.5
        perturbed_features = extract_and_normalize_features(perturbed, ref_stats)
        perturbed_score = detector.compute_score(perturbed_features)
        importance = abs(baseline_score - perturbed_score)
        group_scores.append((group, importance))
    
    importance_map = np.zeros_like(image)
    for group, importance in group_scores:
        for i, j in group:
            importance_map[i, j] = importance
    
    if importance_map.max() > 0:
        importance_map = importance_map / importance_map.max()
    
    return {
        'importance_map': importance_map,
        'group_importances': group_scores,
        'baseline_score': float(baseline_score)
    }

def identify_high_importance_regions(importance_map, threshold=0.5):
    H, W = importance_map.shape
    regions = []
    
    quadrants = {
        'top-left': importance_map[:H//2, :W//2].mean(),
        'top-right': importance_map[:H//2, W//2:].mean(),
        'bottom-left': importance_map[H//2:, :W//2].mean(),
        'bottom-right': importance_map[H//2:, W//2:].mean(),
    }
    
    for name, score in quadrants.items():
        if score > threshold:
            regions.append((name, score))
    
    regions.sort(key=lambda x: x[1], reverse=True)
    return regions

def explain_detection(image, ref_stats, detector, num_samples=100):

    result = compute_feature_importance(image, ref_stats, detector, num_samples)
    regions = identify_high_importance_regions(result['importance_map'], threshold=0.3)
    from features.feature_vector import extract_divergence_features
    features_raw = extract_divergence_features(image, ref_stats)
    return {
        'importance_map': result['importance_map'],
        'high_importance_regions': regions,
        'detection_score': result['baseline_score'],
        'feature_breakdown': {
            'kl_global': float(features_raw[0]),
            'js_global': float(features_raw[1]),
            'wasserstein_global': float(features_raw[2]),
            'kl_pixel_mean': float(features_raw[3]),
            'kl_pixel_max': float(features_raw[4]),
            'wasserstein_pixel_mean': float(features_raw[5]),
        }
    }
