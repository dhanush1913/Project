import numpy as np

EPSILON = 1e-8
BINS = 32

def compute_histogram(image, bins=BINS, range_min=0.0, range_max=1.0):
    hist, _ = np.histogram(image.flatten(), bins=bins, range=(range_min, range_max), density=False)
    prob = hist.astype(np.float64) / hist.sum()
    return np.maximum(prob, EPSILON)

def compute_pixel_marginals(images, bins=BINS):
    N, H, W = images.shape
    marginals = np.zeros((H, W, bins), dtype=np.float64)
    
    for i in range(H):
        for j in range(W):
            pixel_values = images[:, i, j]
            hist, _ = np.histogram(pixel_values, bins=bins, range=(0.0, 1.0), density=False)
            marginals[i, j] = np.maximum(hist / N, EPSILON)
    
    return marginals

def compute_class_histograms(images, labels, num_classes=10, bins=BINS):

    class_hists = np.zeros((num_classes, bins), dtype=np.float64)
    
    for c in range(num_classes):
        class_mask = (labels == c)
        if class_mask.sum() > 0:
            class_images = images[class_mask]
            combined = class_images.flatten()
            hist, _ = np.histogram(combined, bins=bins, range=(0.0, 1.0), density=False)
            class_hists[c] = np.maximum(hist / hist.sum(), EPSILON)
    
    return class_hists
