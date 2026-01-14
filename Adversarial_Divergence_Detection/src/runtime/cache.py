import numpy as np
import json
from pathlib import Path

class RuntimeCache:
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not RuntimeCache._initialized:
            self.ref_histograms = None
            self.pixel_distributions = None
            self.feature_mean = None
            self.feature_std = None
            self.detector_weights = None
            self.detector_threshold = None
            RuntimeCache._initialized = True
    
    def load_all(self, data_dir, config_dir):
        data_dir = Path(data_dir)
        config_dir = Path(config_dir)
        
        self.ref_histograms = np.load(data_dir / 'class_histograms.npy').astype(np.float32)
        self.pixel_distributions = np.load(data_dir / 'pixel_distributions.npy').astype(np.float32)
        
        with open(data_dir / 'feature_stats.json', 'r') as f:
            stats = json.load(f)
            self.feature_mean = np.array(stats['feature_mean'], dtype=np.float32)
            self.feature_std = np.array(stats['feature_std'], dtype=np.float32)
        
        import yaml
        with open(config_dir / 'detector_config.yaml', 'r') as f:
            config = yaml.safe_load(f)
            self.detector_weights = np.array(config['weights'], dtype=np.float32)
            self.detector_threshold = float(config['threshold'])
    
    def get_ref_histogram_global(self):
        if not hasattr(self, '_ref_hist_global'):
            self._ref_hist_global = self.ref_histograms.mean(axis=0).astype(np.float32)
        return self._ref_hist_global
    
    def clear(self):
        self.ref_histograms = None
        self.pixel_distributions = None
        self.feature_mean = None
        self.feature_std = None
        self.detector_weights = None
        self.detector_threshold = None

_cache = RuntimeCache()

def get_cache():
    return _cache
