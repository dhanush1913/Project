import numpy as np
import json
from pathlib import Path
from datetime import datetime
from enum import Enum

class DetectionState(Enum):
    CLEAN = "CLEAN"
    ADVERSARIAL = "ADVERSARIAL"
    UNCERTAIN = "UNCERTAIN"  # Human review required

class ProductionDetector:

    def __init__(self, cache, config, enable_logging=True, random_seed=42):
        self.cache = cache
        self.config = config
        self.enable_logging = enable_logging
        self.random_seed = random_seed
        
        np.random.seed(random_seed)
        self.uncertainty_config = config.get('uncertainty_thresholds', {
            'max_divergence_disagreement': 0.5,  # Max diff between divergences
            'min_xai_consistency': 0.3,  # Min XAI-divergence consistency
            'ood_score_threshold': 10.0,  # Extreme scores → OOD
        })
    
    def detect(self, image, generate_explanation=True):

        from runtime.fastpath import FastPathDetector
        from features.feature_vector import extract_divergence_features
        
        fast_detector = FastPathDetector(self.cache)
        result = fast_detector.detect(image)
        
        ref_stats = {
            'class_histograms': self.cache.ref_histograms,
            'pixel_distributions': self.cache.pixel_distributions,
        }
        features_raw = extract_divergence_features(image, ref_stats)
        
        uncertainty_reason = self._check_uncertainty(features_raw, result)
        
        if uncertainty_reason:
            state = DetectionState.UNCERTAIN
            confidence = 0.5  # Uncertain
        elif result['prediction']:
            state = DetectionState.ADVERSARIAL
            confidence = min(1.0, result['score'] / self.cache.detector_threshold)
        else:
            state = DetectionState.CLEAN
            confidence = 1.0 - (result['score'] / self.cache.detector_threshold)
        
        confidence = float(np.clip(confidence, 0.0, 1.0))
        
        audit_log = {
            'timestamp': datetime.utcnow().isoformat(),
            'state': state.value,
            'confidence': confidence,
            'score': result.get('score', 0.0),
            'threshold': self.cache.detector_threshold,
            'stage': result.get('stage', 2),
            'uncertainty_reason': uncertainty_reason,
            'features': {
                'kl_global': float(features_raw[0]),
                'js_global': float(features_raw[1]),
                'wasserstein_global': float(features_raw[2]),
                'kl_pixel_mean': float(features_raw[3]),
                'kl_pixel_max': float(features_raw[4]),
                'wasserstein_pixel_mean': float(features_raw[5]),
            },
            'random_seed': self.random_seed,
        }
        
        explanation = None
        if generate_explanation and state == DetectionState.ADVERSARIAL:
            from xai.shap_explainer import explain_detection
            from xai.audit_report import generate_audit_report
            from detector.threshold_detector import ThresholdDetector
            
            temp_detector = ThresholdDetector(
                weights=self.cache.detector_weights,
                threshold=self.cache.detector_threshold
            )
            
            explanation = explain_detection(image, ref_stats, temp_detector, num_samples=50)
            audit_log['xai_summary'] = {
                'high_importance_regions': [(r[0], float(r[1])) for r in explanation.get('high_importance_regions', [])[:3]]
            }
        
        if self.enable_logging:
            self._log_to_file(audit_log)
        
        return {
            'state': state,
            'confidence': confidence,
            'audit_log': audit_log,
            'explanation': explanation,
            'requires_human_review': (state == DetectionState.UNCERTAIN),
        }
    
    def _check_uncertainty(self, features_raw, detection_result):

        kl_global = features_raw[0]
        js_global = features_raw[1]
        w1_global = features_raw[2]
        
        divergence_spread = max(kl_global, js_global, w1_global) - min(kl_global, js_global, w1_global)
        if divergence_spread > self.uncertainty_config['max_divergence_disagreement']:
            return f"Divergence disagreement ({divergence_spread:.3f})"
        
        score = detection_result.get('score', 0.0)
        if abs(score) > self.uncertainty_config['ood_score_threshold']:
            return f"Out-of-distribution (score={score:.3f})"
        
        return None
    
    def _log_to_file(self, audit_log):
        log_dir = Path(__file__).parent.parent.parent / 'logs'
        log_dir.mkdir(exist_ok=True)
        
        date_str = datetime.utcnow().strftime('%Y%m%d')
        log_file = log_dir / f'audit_{date_str}.jsonl'
        
        with open(log_file, 'a') as f:
            f.write(json.dumps(audit_log) + '\n')
