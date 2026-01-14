import numpy as np

ATTACK_SIGNATURES = {
    'high_frequency': {
        'pattern': 'high-frequency noise',
        'attacks': ['FGSM'],
        'indicator': lambda f: f['kl_pixel_max'] > 2 * f['kl_pixel_mean']
    },
    'spatial_shift': {
        'pattern': 'spatial mass redistribution',
        'attacks': ['PGD'],
        'indicator': lambda f: f['wasserstein_global'] > f['kl_global']
    },
    'uniform_perturbation': {
        'pattern': 'uniform distribution shift',
        'attacks': ['Random noise'],
        'indicator': lambda f: abs(f['kl_global'] - f['js_global']) < 0.01
    }
}

def generate_audit_report(explanation, prediction, threshold):
    report = []
    # Header
    report.append("="*70)
    report.append("ADVERSARIAL DETECTION AUDIT REPORT")
    report.append("="*70)
    report.append("")
    # Decision
    decision_text = "ADVERSARIAL" if prediction else "BENIGN"
    report.append(f"Detection Decision: {decision_text}")
    report.append(f"Detection Score: {explanation['detection_score']:.4f}")
    report.append(f"Threshold: {threshold:.4f}")
    report.append("")
    if not prediction:
        report.append("Sample classified as benign. No further analysis required.")
        report.append("="*70)
        return "\n".join(report)
    
    # Contributing factors
    report.append("Primary Contributing Factors:")
    report.append("-" * 70)
    features = explanation['feature_breakdown']
    # Feature analysis
    feature_contrib = []
    if features['kl_global'] > 0.3:
        feature_contrib.append(f"  - High KL divergence ({features['kl_global']:.4f}): Distribution tail deviation")
    
    if features['wasserstein_global'] > 0.2:
        feature_contrib.append(f"  - Elevated Wasserstein distance ({features['wasserstein_global']:.4f}): Spatial mass shift")
    
    if features['kl_pixel_max'] > 1.0:
        feature_contrib.append(f"  - Peak pixel KL ({features['kl_pixel_max']:.4f}): Localized anomaly")
    
    if not feature_contrib:
        feature_contrib.append("  - Multiple divergence metrics exceed reference thresholds")
    report.extend(feature_contrib)
    report.append("")
    
    # Spatial analysis
    if explanation['high_importance_regions']:
        report.append("High-Importance Spatial Regions:")
        for region, score in explanation['high_importance_regions'][:3]:
            report.append(f"  - {region.capitalize()}: importance = {score:.3f}")
        report.append("")
    
    # Attack signature matching
    report.append("Attack Signature Analysis:")
    matched_signatures = []
    
    for sig_name, sig_info in ATTACK_SIGNATURES.items():
        if sig_info['indicator'](features):
            matched_signatures.append(f"  - Pattern: {sig_info['pattern']}")
            matched_signatures.append(f"    Consistent with: {', '.join(sig_info['attacks'])}")
    
    if matched_signatures:
        report.extend(matched_signatures)
    else:
        report.append("  - No standard attack signature matched")
        report.append("    Potentially novel or adaptive attack")
    
    report.append("")
    
    # Divergence breakdown
    report.append("Divergence Metric Breakdown:")
    report.append(f"  KL (global):        {features['kl_global']:.6f}")
    report.append(f"  JS (global):        {features['js_global']:.6f}")
    report.append(f"  Wasserstein (glob): {features['wasserstein_global']:.6f}")
    report.append(f"  KL (pixel mean):    {features['kl_pixel_mean']:.6f}")
    report.append(f"  KL (pixel max):     {features['kl_pixel_max']:.6f}")
    report.append(f"  Wasserstein (pix):  {features['wasserstein_pixel_mean']:.6f}")
    report.append("")
    # Footer
    report.append("="*70)
    report.append("Note: This explanation is post-hoc. Detection decision was made")
    report.append("prior to explanation generation (XAI compliance).")
    report.append("="*70)
    
    return "\n".join(report)

def validate_explanation_consistency(importance_map, feature_breakdown):
    high_importance_fraction = (importance_map > 0.5).mean()
    high_divergence = (feature_breakdown['kl_pixel_max'] > 2 * feature_breakdown['kl_pixel_mean'])
    consistent = True
    warnings = []
    
    if high_importance_fraction > 0.3 and not high_divergence:
        consistent = False
        warnings.append("High spatial importance but flat divergence - potential explanation error")
    if high_divergence and high_importance_fraction < 0.1:
        consistent = False
        warnings.append("High divergence peaks but low spatial importance - explanation may be incomplete")
    
    return {
        'consistent': consistent,
        'warnings': warnings,
        'high_importance_fraction': float(high_importance_fraction),
        'high_divergence_detected': bool(high_divergence)
    }
