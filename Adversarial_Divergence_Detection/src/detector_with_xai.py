import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from detector.threshold_detector import ThresholdDetector
from features.feature_vector import extract_and_normalize_features
from xai.shap_explainer import explain_detection
from xai.audit_report import generate_audit_report, validate_explanation_consistency

class AdversarialDetectorWithXAI:
    
    def __init__(self, config_path, ref_stats):
        self.detector = ThresholdDetector()
        self.detector.load_config(config_path)
        self.ref_stats = ref_stats
    
    def detect(self, image, generate_explanation=True):
        # Step 1: Extract features
        features = extract_and_normalize_features(image, self.ref_stats)
        # Step 2: Make decision (BEFORE explanation)
        result = self.detector.predict_with_score(features)
        prediction = result['prediction']
        score = result['score']
        # Step 3: Post-hoc explanation (AFTER decision)
        explanation = None
        audit_report = None
        consistency = None
        
        if generate_explanation:
            explanation = explain_detection(image, self.ref_stats, self.detector, num_samples=50)
            audit_report = generate_audit_report(explanation, prediction, self.detector.threshold)
            consistency = validate_explanation_consistency(
                explanation['importance_map'],
                explanation['feature_breakdown']
            )
        
        return {
            'prediction': prediction,
            'score': float(score),
            'explanation': explanation,
            'audit_report': audit_report,
            'consistency_check': consistency
        }
    
    def batch_detect(self, images, generate_explanations=False):
        results = []
        for img in images:
            result = self.detect(img, generate_explanation=generate_explanations)
            results.append(result)
        return results

def demo_detection_with_xai():
    print("\n" + "="*70)
    print("ADVERSARIAL DETECTION WITH XAI - DEMO")
    print("="*70 + "\n")
    
    # Load reference stats
    print("[1/4] Loading reference statistics and configuration...")
    ref_dir = Path(__file__).parent.parent / 'data' / 'reference'
    ref_stats = {
        'class_histograms': np.load(ref_dir / 'class_histograms.npy'),
        'pixel_distributions': np.load(ref_dir / 'pixel_distributions.npy'),
    }
    
    import json
    with open(ref_dir / 'feature_stats.json', 'r') as f:
        feature_stats = json.load(f)
        ref_stats.update(feature_stats)
    
    config_path = Path(__file__).parent.parent / 'config' / 'detector_config.yaml'
    
    # Initialize detector
    detector = AdversarialDetectorWithXAI(config_path, ref_stats)
    print("  ✓ Detector initialized")
    # Load test images
    print("[2/4] Loading test images...")
    test_data = np.load(ref_dir.parent / 'raw' / 'mnist_test.npz')
    clean_image = test_data['images'][0].astype(np.float32) / 255.0
    # Generate adversarial example
    from attacks.adversarial_generator import generate_fgsm_dataset
    adv_image = generate_fgsm_dataset(clean_image[np.newaxis, :, :], epsilon=0.3)[0]
    print("  ✓ Loaded clean and adversarial images")
    # Detect clean image
    print("\n[3/4] Detecting CLEAN image...")
    result_clean = detector.detect(clean_image, generate_explanation=True)
    print(f"  Prediction: {'ADVERSARIAL' if result_clean['prediction'] else 'BENIGN'}")
    print(f"  Score: {result_clean['score']:.4f}")
    print(f"  Consistency: {result_clean['consistency_check']['consistent']}")
    # Detect adversarial image
    print("\n[4/4] Detecting ADVERSARIAL image...")
    result_adv = detector.detect(adv_image, generate_explanation=True)
    
    print(f"  Prediction: {'ADVERSARIAL' if result_adv['prediction'] else 'BENIGN'}")
    print(f"  Score: {result_adv['score']:.4f}")
    print(f"  Consistency: {result_adv['consistency_check']['consistent']}")
    print("\n" + "="*70)
    print("AUDIT REPORT (Adversarial Image)")
    print("="*70)
    print(result_adv['audit_report'])
    print("\n" + "="*70)
    print("✓ DEMO COMPLETE")
    print("="*70 + "\n")

if __name__ == '__main__':
    demo_detection_with_xai()
