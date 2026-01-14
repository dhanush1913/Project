import numpy as np

EPSILON = 1e-8

def kl_divergence(p, q, epsilon=EPSILON):
    p = np.maximum(p, epsilon)
    q = np.maximum(q, epsilon)
    kl = np.sum(p * (np.log(p) - np.log(q)), axis=-1)
    kl = np.maximum(kl, 0.0)  # KL is non-negative by definition
    return kl

def symmetric_kl(p, q, epsilon=EPSILON):
    return 0.5 * (kl_divergence(p, q, epsilon) + kl_divergence(q, p, epsilon))

def kl_divergence_batch(p_batch, q_ref, epsilon=EPSILON):

    p_batch = np.maximum(p_batch, epsilon)
    q_ref = np.maximum(q_ref, epsilon)
    log_ratio = np.log(p_batch) - np.log(q_ref[np.newaxis, :])
    kl = np.sum(p_batch * log_ratio, axis=1)
    
    return np.maximum(kl, 0.0)
