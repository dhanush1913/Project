import numpy as np
import json
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from features.normalization import validate_images, normalize_to_unit_range
from features.histogram import compute_class_histograms, compute_pixel_marginals, BINS, EPSILON
from features.pixel_stats import compute_pixel_statistics, compute_global_statistics

# Paths
DATA_DIR = Path(__file__).parent.parent / 'data'
RAW_DIR = DATA_DIR / 'raw'
REF_DIR = DATA_DIR / 'reference'

def load_mnist_train():
    """Load MNIST training data."""
    train_path = RAW_DIR / 'mnist_train.npz'
    
    if not train_path.exists():
        print(f"⚠️  MNIST data not found at {train_path}")
        print("Downloading MNIST dataset...")
        download_mnist()
    
    data = np.load(train_path)
    images = data['images'].astype(np.float32) / 255.0  # Normalize to [0, 1]
    labels = data['labels']
    
    print(f"✓ Loaded {len(images)} training images")
    return images, labels

def download_mnist():
    """Download MNIST dataset if not present."""
    try:
        from tensorflow.keras.datasets import mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        
        # Save in our format
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(RAW_DIR / 'mnist_train.npz', images=x_train, labels=y_train)
        np.savez_compressed(RAW_DIR / 'mnist_test.npz', images=x_test, labels=y_test)
        
        print(f"✓ Downloaded MNIST to {RAW_DIR}")
    except ImportError:
        print("❌ TensorFlow not installed. Install with: pip install tensorflow")
        print("   Or manually place mnist_train.npz in data/raw/")
        sys.exit(1)

def build_reference_distribution():
    """Main pipeline: validate data, compute features, save."""
    print("\n" + "="*60)
    print("Building Reference Distribution")
    print("="*60 + "\n")
    
    # Step 1: Load and validate
    print("[1/4] Loading MNIST training data...")
    images, labels = load_mnist_train()
    
    print("[2/4] Validating data integrity...")
    try:
        validate_images(images, expected_shape_suffix=(28, 28), value_range=(0.0, 1.0))
        print(f"  ✓ Shape: {images.shape}")
        print(f"  ✓ Range: [{images.min():.3f}, {images.max():.3f}]")
        print(f"  ✓ No NaNs or constant images")
    except ValueError as e:
        print(f"  ❌ Validation failed: {e}")
        sys.exit(1)
    
    # Step 2: Compute reference statistics
    print("[3/4] Computing reference statistics...")
    # Class-conditional histograms
    print(f"  - Class histograms ({BINS} bins with ε={EPSILON})...")
    class_histograms = compute_class_histograms(images, labels, num_classes=10, bins=BINS)
    # Per-pixel marginal distributions
    print(f"  - Pixel marginals (28×28×{BINS})...")
    pixel_distributions = compute_pixel_marginals(images, bins=BINS)
    # Global statistics
    print(f"  - Global statistics...")
    pixel_stats = compute_pixel_statistics(images)
    global_stats = compute_global_statistics(images)
    # Step 3: Save to disk
    print("[4/4] Saving reference data...")
    REF_DIR.mkdir(parents=True, exist_ok=True)
    np.save(REF_DIR / 'class_histograms.npy', class_histograms)
    print(f"  ✓ Saved class_histograms.npy: {class_histograms.shape}")
    np.save(REF_DIR / 'pixel_distributions.npy', pixel_distributions)
    print(f"  ✓ Saved pixel_distributions.npy: {pixel_distributions.shape}")
    feature_stats = {
        'epsilon': EPSILON,
        'bins': BINS,
        'pixel_mean': pixel_stats['mean'].tolist(),
        'pixel_std': pixel_stats['std'].tolist(),
        **global_stats,
        'num_training_samples': int(len(images)),
    }
    with open(REF_DIR / 'feature_stats.json', 'w') as f:
        json.dump(feature_stats, f, indent=2)
    print(f"  ✓ Saved feature_stats.json")
    # Validation summary
    print("\n" + "="*60)
    print("Reference Distribution Built Successfully ✓")
    print("="*60)
    print(f"\nStatistics:")
    print(f"  Training samples: {len(images):,}")
    print(f"  Histogram bins: {BINS}")
    print(f"  Epsilon smoothing: {EPSILON}")
    print(f"  Global mean: {global_stats['global_mean']:.4f}")
    print(f"  Global std: {global_stats['global_std']:.4f}")
    print(f"\nReference data saved to: {REF_DIR.absolute()}")
    print("\nReady for divergence computation. ✓\n")

if __name__ == '__main__':
    build_reference_distribution()
