import numpy as np

try:
    from .kl import kl_divergence, EPSILON
except ImportError:
    from kl import kl_divergence, EPSILON

def js_divergence(p, q, epsilon=EPSILON):

    p = np.maximum(p, epsilon)
    q = np.maximum(q, epsilon)
    m = 0.5 * (p + q)
    js = 0.5 * kl_divergence(p, m, epsilon) + 0.5 * kl_divergence(q, m, epsilon)
    js = np.clip(js, 0.0, np.log(2))
    return js

def js_divergence_normalized(p, q, epsilon=EPSILON):
    return js_divergence(p, q, epsilon) / np.log(2)

def js_distance(p, q, epsilon=EPSILON):
    return np.sqrt(js_divergence(p, q, epsilon))
