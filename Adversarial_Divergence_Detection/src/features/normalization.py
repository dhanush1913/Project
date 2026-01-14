import numpy as np

def validate_images(images, expected_shape_suffix=(28, 28), value_range=(0.0, 1.0)):

    if images.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {images.shape}")
    if images.shape[1:] != expected_shape_suffix:
        raise ValueError(f"Expected shape (N, {expected_shape_suffix}), got {images.shape}")
    if np.isnan(images).any():
        raise ValueError("Found NaN values in images")
    if np.isinf(images).any():
        raise ValueError("Found Inf values in images")
    actual_min, actual_max = images.min(), images.max()
    if actual_min < value_range[0] or actual_max > value_range[1]:
        raise ValueError(f"Values out of range [{value_range[0]}, {value_range[1]}]: [{actual_min}, {actual_max}]")
    
    std_per_image = images.reshape(images.shape[0], -1).std(axis=1)
    constant_count = (std_per_image < 1e-6).sum()
    if constant_count / len(images) > 0.01:  # More than 1% constant
        raise ValueError(f"Found {constant_count} constant images ({100*constant_count/len(images):.2f}%)")
    
    return True

def normalize_to_unit_range(images):
    min_val, max_val = images.min(), images.max()
    if min_val < 0 or max_val > 1:
        return (images - min_val) / (max_val - min_val + 1e-10)
    return images
