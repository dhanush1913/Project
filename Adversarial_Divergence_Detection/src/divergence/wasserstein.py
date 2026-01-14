import numpy as np
from scipy.stats import wasserstein_distance as scipy_wasserstein

EPSILON = 1e-8

def wasserstein_1d(p, q, epsilon=EPSILON):
    p = np.maximum(p, epsilon)
    q = np.maximum(q, epsilon)
    p = p / p.sum()
    q = q / q.sum()
    cdf_p = np.cumsum(p)
    cdf_q = np.cumsum(q)
    w1 = np.sum(np.abs(cdf_p - cdf_q))
    return w1

def wasserstein_histogram(hist_p, hist_q, bin_edges=None, epsilon=EPSILON):

    if bin_edges is None:
        bin_edges = np.linspace(0, 1, len(hist_p) + 1)
    
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    hist_p = np.maximum(hist_p, epsilon)
    hist_q = np.maximum(hist_q, epsilon)
    return scipy_wasserstein(bin_centers, bin_centers, hist_p, hist_q)

def wasserstein_2d_marginal(image_p, image_q, epsilon=EPSILON):
    marginal_x_p = image_p.sum(axis=0)  # Sum over rows -> column marginal
    marginal_x_q = image_q.sum(axis=0)
    
    marginal_y_p = image_p.sum(axis=1)  # Sum over columns -> row marginal
    marginal_y_q = image_q.sum(axis=1)
    marginal_x_p = np.maximum(marginal_x_p / marginal_x_p.sum(), epsilon)
    marginal_x_q = np.maximum(marginal_x_q / marginal_x_q.sum(), epsilon)
    marginal_y_p = np.maximum(marginal_y_p / marginal_y_p.sum(), epsilon)
    marginal_y_q = np.maximum(marginal_y_q / marginal_y_q.sum(), epsilon)
    w_x = wasserstein_1d(marginal_x_p, marginal_x_q, epsilon)
    w_y = wasserstein_1d(marginal_y_p, marginal_y_q, epsilon)
    return (w_x + w_y) / 2.0
