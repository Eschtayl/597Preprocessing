import numpy as np


def rank_normalize(scores, reference):
    # Map each score to its percentile rank within a reference distribution.
    # Scale-free, so detectors with different score ranges become comparable.
    scores = np.asarray(scores, dtype=float)
    ref = np.sort(np.asarray(reference, dtype=float))
    if len(ref) == 0:
        return np.zeros_like(scores)
    return np.searchsorted(ref, scores, side='right') / len(ref)


def fuse_scores(if_scores, ae_scores, if_reference, ae_reference,
                method='mean', weight_ae=0.6):
    # Combine Isolation Forest and Autoencoder scores after rank normalization.
    # References should be the *train* scores of each detector so test scores
    # are ranked against the (benign-dominated) training distribution.
    r_if = rank_normalize(if_scores, if_reference)
    r_ae = rank_normalize(ae_scores, ae_reference)

    if method == 'mean':
        return 0.5 * r_if + 0.5 * r_ae
    if method == 'max':
        return np.maximum(r_if, r_ae)
    if method == 'weighted':
        return weight_ae * r_ae + (1.0 - weight_ae) * r_if
    raise ValueError(f'Unknown fusion method: {method}')
