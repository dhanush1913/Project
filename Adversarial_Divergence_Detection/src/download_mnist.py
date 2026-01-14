import numpy as np
from pathlib import Path

try:
    from tensorflow.keras.datasets import mnist
    print("Downloading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    data_dir = Path(__file__).parent.parent / 'data' / 'raw'
    data_dir.mkdir(parents=True, exist_ok=True)
    
    np.savez_compressed(data_dir / 'mnist_train.npz', images=x_train, labels=y_train)
    np.savez_compressed(data_dir / 'mnist_test.npz', images=x_test, labels=y_test)
    
    print(f"✓ Saved {len(x_train)} train + {len(x_test)} test images to {data_dir}")
    
except ImportError:
    print("❌ Install tensorflow: pip install tensorflow")
