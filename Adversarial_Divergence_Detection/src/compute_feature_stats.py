import numpy as np
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from features.feature_vector import extract_divergence_features

def compute_feature_statistics(num_samples=1000):
    print("\n" + "="*60)
    print("Computing Feature Normalization Statistics")
    print("="*60 + "\n")
    
    # Paths
    data_dir = Path(__file__).parent.parent / 'data'
    raw_dir = data_dir / 'raw'
    ref_dir = data_dir / 'reference'
    # Load reference stats
    print("[1/3] Loading reference statistics...")
    ref_stats = {
        'class_histograms': np.load(ref_dir / 'class_histograms.npy'),
        'pixel_distributions': np.load(ref_dir / 'pixel_distributions.npy'),
    }
    print(f"  ✓ Loaded reference distributions")
    
    # Load clean training data
    print("[2/3] Extracting features from clean images...")
    train_data = np.load(raw_dir / 'mnist_train.npz')
    images = train_data['images'].astype(np.float32) / 255.0
    # Sample subset for efficiency
    num_samples = min(num_samples, len(images))
    indices = np.random.choice(len(images), num_samples, replace=False)
    sample_images = images[indices]
    # Extract features from all samples
    all_features = []
    for i, img in enumerate(sample_images):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{num_samples} images...", end='\r')
        
        features = extract_divergence_features(img, ref_stats)
        all_features.append(features)
    
    print(f"  ✓ Extracted features from {num_samples} clean images")
    
    # Compute statistics
    print("[3/3] Computing normalization statistics...")
    all_features = np.array(all_features)  # (num_samples, feature_dim)
    feature_mean = np.mean(all_features, axis=0)
    feature_std = np.std(all_features, axis=0)
    # Ensure no zero std (add small epsilon if needed)
    feature_std = np.maximum(feature_std, 1e-8)
    print(f"  ✓ Computed mean and std for {all_features.shape[1]} features")
    
    # Load existing stats and update
    with open(ref_dir / 'feature_stats.json', 'r') as f:
        stats = json.load(f)
    
    stats['feature_mean'] = feature_mean.tolist()
    stats['feature_std'] = feature_std.tolist()
    stats['feature_dim'] = int(all_features.shape[1])
    stats['norm_samples'] = num_samples
    # Save updated stats
    with open(ref_dir / 'feature_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\n✓ Saved normalization statistics to feature_stats.json")
    
    # Print summary
    print("\n" + "="*60)
    print("Feature Normalization Statistics")
    print("="*60)
    print(f"\nFeature dimension: {all_features.shape[1]}")
    print(f"Samples used: {num_samples}")
    print(f"\nMean values:")
    for i, val in enumerate(feature_mean):
        print(f"  Feature {i}: {val:.6f}")
    print(f"\nStd values:")
    for i, val in enumerate(feature_std):
        print(f"  Feature {i}: {val:.6f}")
    print("\n" + "="*60 + "\n")
    
    return stats

if __name__ == '__main__':
    compute_feature_statistics(num_samples=1000)
