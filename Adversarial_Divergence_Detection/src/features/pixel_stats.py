import numpy as np

def compute_pixel_statistics(images):
    return {
        'mean': np.mean(images, axis=0),
        'std': np.std(images, axis=0),
    }

def compute_global_statistics(images):
    return {
        'global_mean': float(np.mean(images)),
        'global_std': float(np.std(images)),
        'global_min': float(np.min(images)),
        'global_max': float(np.max(images)),
    }
