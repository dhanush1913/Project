import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from detector_with_xai import AdversarialDetectorWithXAI
from attacks.adversarial_generator import generate_fgsm_dataset, generate_pgd_dataset

def test_post_hoc_guarantee():
    print("\n" + "="*60)
    print("Test: Post-Hoc Explanation Guarantee")
    print("="*60 + "\n")
    
    ref_dir = Path(__file__).parent.parent / 'data' / 'reference'
    ref_stats = {
        'class_histograms': np.load(ref_dir / 'class_histograms.npy'),
        'pixel_distributions': np.load(ref_dir / 'pixel_distributions.npy'),
    }
    import json
    with open(ref_dir / 'feature_stats.json', 'r') as f:
        ref_stats.update(json.load(f))
    
    config_path = Path(__file__).parent.parent / 'config' / 'detector_config.yaml'
    detector = AdversarialDetectorWithXAI(config_path, ref_stats)
    test_data = np.load(ref_dir.parent / 'raw' / 'mnist_test.npz')
    image = test_data['images'][0].astype(np.float32) / 255.0
    result_no_xai = detector.detect(image, generate_explanation=False)
    result_with_xai = detector.detect(image, generate_explanation=True)
    same_prediction = (result_no_xai['prediction'] == result_with_xai['prediction'])
    same_score = np.abs(result_no_xai['score'] - result_with_xai['score']) < 1e-6
    print(f"  ✓ Same prediction: {same_prediction}")
    print(f"  ✓ Same score: {same_score}")
    if same_prediction and same_score:
        print("\n✓ POST-HOC GUARANTEE VERIFIED")
        print("  Explanation does NOT influence detection decision")
    else:
        print("\n✗ POST-HOC GUARANTEE VIOLATED")
        print("  Explanation influenced detection - SYSTEM INVALID")
    
    return same_prediction and same_score

def test_xai_consistency():
    print("\n" + "="*60)
    print("Test: XAI-Divergence Consistency")
    print("="*60 + "\n")
    
    ref_dir = Path(__file__).parent.parent / 'data' / 'reference'
    ref_stats = {
        'class_histograms': np.load(ref_dir / 'class_histograms.npy'),
        'pixel_distributions': np.load(ref_dir / 'pixel_distributions.npy'),
    }
    import json
    with open(ref_dir / 'feature_stats.json', 'r') as f:
        ref_stats.update(json.load(f))
    
    config_path = Path(__file__).parent.parent / 'config' / 'detector_config.yaml'
    detector = AdversarialDetectorWithXAI(config_path, ref_stats)
    
    test_data = np.load(ref_dir.parent / 'raw' / 'mnist_test.npz')
    clean_image = test_data['images'][0].astype(np.float32) / 255.0
    adv_image = generate_fgsm_dataset(clean_image[np.newaxis, :, :], epsilon=0.3)[0]
    result = detector.detect(adv_image, generate_explanation=True)
    consistency = result['consistency_check']
    print(f"  Consistent: {consistency['consistent']}")
    print(f"  High importance fraction: {consistency['high_importance_fraction']:.3f}")
    print(f"  High divergence detected: {consistency['high_divergence_detected']}")
    if consistency['warnings']:
        print(f"\n  Warnings:")
        for warning in consistency['warnings']:
            print(f"    - {warning}")
    
    if consistency['consistent']:
        print("\n✓ XAI-DIVERGENCE CONSISTENCY VERIFIED")
    else:
        print("\n⚠ XAI-DIVERGENCE INCONSISTENCY DETECTED")
        print("  Manual review recommended")
    
    return consistency['consistent']

def test_audit_report_determinism():
    print("\n" + "="*60)
    print("Test: Audit Report Determinism")
    print("="*60 + "\n")
    
    ref_dir = Path(__file__).parent.parent / 'data' / 'reference'
    ref_stats = {
        'class_histograms': np.load(ref_dir / 'class_histograms.npy'),
        'pixel_distributions': np.load(ref_dir / 'pixel_distributions.npy'),
    }
    
    import json
    with open(ref_dir / 'feature_stats.json', 'r') as f:
        ref_stats.update(json.load(f))
    
    config_path = Path(__file__).parent.parent / 'config' / 'detector_config.yaml'
    detector = AdversarialDetectorWithXAI(config_path, ref_stats)
    
    # Test image
    test_data = np.load(ref_dir.parent / 'raw' / 'mnist_test.npz')
    image = test_data['images'][0].astype(np.float32) / 255.0
    # Generate report twice
    result1 = detector.detect(image, generate_explanation=True)
    result2 = detector.detect(image, generate_explanation=True)
    # Compare reports
    report_match = (result1['audit_report'] == result2['audit_report'])
    # Check for probabilistic language
    prohibited_words = ['likely', 'possibly', 'appears', 'seems', 'probably', 'maybe']
    report_text = result1['audit_report'].lower()
    has_prohibited = any(word in report_text for word in prohibited_words)
    print(f"  ✓ Reports identical: {report_match}")
    print(f"  ✓ No probabilistic language: {not has_prohibited}")
    if report_match and not has_prohibited:
        print("\n✓ AUDIT REPORT DETERMINISM VERIFIED")
    else:
        print("\n✗ AUDIT REPORT ISSUES DETECTED")
    return report_match and not has_prohibited

def main():
    print("\n" + "="*60)
    print("XAI VALIDATION TEST SUITE")
    print("="*60)
    test1 = test_post_hoc_guarantee()
    test2 = test_xai_consistency()
    test3 = test_audit_report_determinism()
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"  Post-hoc guarantee: {'✓' if test1 else '✗'}")
    print(f"  XAI consistency: {'✓' if test2 else '✗'}")
    print(f"  Report determinism: {'✓' if test3 else '✗'}")
    
    if test1 and test2 and test3:
        print("\n✓ ALL XAI VALIDATION TESTS PASSED")
    else:
        print("\n⚠ SOME TESTS FAILED - REVIEW REQUIRED")
    
    print("="*60 + "\n")
if __name__ == '__main__':
    main()
