"""
Utility functions for importance sampling.
"""

import numpy as np
from scipy.special import logsumexp


def clip_and_normalize_weights(
    raw_log_weights: np.ndarray,
    percentile: float | None = None,
) -> np.ndarray:
    """
    Normalize the raw log-weights (and optionally clip them first).

    Clipping the largest raw log-weights can reduce the variance of the
    weights, at the cost of introducing a bias. Use with caution!

    Args:
        raw_log_weights: Raw log-weights.
        percentile: (Upper) percentile for clipping. If `None`, no
            clipping is applied (this is the default).

    Returns:
        normalized_weights: Normalized importance sampling weights.
    """

    # Clip the raw log-weights, if desired
    if percentile is not None:
        threshold = np.percentile(raw_log_weights, percentile)
        clipped_weights = np.clip(raw_log_weights, None, threshold)
    else:
        clipped_weights = raw_log_weights

    # Normalize the clipped log-weights
    # In "normal" space, we normalize the raw weights such that they sum to
    # the number of samples, that is:
    #   n_i = w_i * N / sum(w_i) ,
    # where N is the number of samples. We now only have access to the log-
    # weights log(w_i), so we use the following equivalent expression:
    #   n_i = exp{ log(N) + log(w_i) - LSE(log(w_i)) } ,
    # where the LSE is the log-sum-exp function:
    #   LSE(x)) = log{ sum(exp(x)) } .
    # Using the log-sum-exp trick, we can compute the LSE in a numerically
    # stable way. This allows to compute the normalized weights without
    # ever needing access to the (raw) likelihoods, priors, or proposals.
    # For more details about the log-sum-exp trick, see, e.g.,:
    # https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/
    N = len(raw_log_weights)
    normalized_weights = np.exp(
        np.log(N) + clipped_weights - logsumexp(clipped_weights)
    )

    return np.array(normalized_weights)


def compute_is_weights(
    log_likelihoods: np.ndarray,
    log_prior_values: np.ndarray,
    log_probs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the importance sampling weights: both the raw weights in
    log-space and the normalized weights in "normal" space.

    Args:
        log_likelihoods: Log-likelihood values.
        log_prior_values: Log-prior values.
        log_probs: Log-probabilities under the proposal distribution,
            i.e., the probabilities returned by the ML model. Note
            that, technically, these are probability *densities*, so
            they do not have to be in [0, 1].

    Returns:
        raw_log_weights: Raw log-weights (without normalization).
        normalized_weights: Normalized importance sampling weights.
    """

    # Compute the raw log-weights
    # In "normal" space, the raw importance sampling weights are given by:
    #   w_i = L_i * p_i / q_i ,
    # where L_i is the likelihood, p_i is the prior, and q_i is the proposal.
    # However, L_i is usually very small, so we use the log-weights instead.
    raw_log_weights = log_likelihoods + log_prior_values - log_probs

    # Normalize the raw log-weights (by default without weight clipping)
    normalized_weights = clip_and_normalize_weights(
        raw_log_weights=raw_log_weights,
        percentile=None,
    )

    return raw_log_weights, normalized_weights


def compute_effective_sample_size(
    weights: np.ndarray,
    log_prior_values: np.ndarray | None = None,
) -> tuple[float, float, float]:
    """
    Compute the effective sample size, the sampling efficiency, and
    optionally the simulation efficiency.

    Args:
        weights: (Normalized) importance sampling weights.
        log_prior_values: Log-prior values. If `None`, simulator
            efficiency is not computed but set to NaN.

    Returns:
        n_eff: Effective sample size.
        sampling_efficiency: Sampling efficiency, i.e., number of total
            proposal samples divided by the effective sample size.
        simulation_effiency: Simulation efficiency, i.e., number of
            simulator calls divided by the effective sample size. This
            excludes the proposal samples where the prior is zero.
            Trivially, `simulation_efficiency` >= `sampling_efficiency`.
    """

    # Compute the effective sample size and the "raw" sampling efficiency,
    # as it is required, e.g., to compute the log-evidence estimate
    n_eff = float(np.sum(weights) ** 2 / np.sum(weights**2))
    sampling_efficiency = n_eff / len(weights)

    # Optionally: Compute the simulator efficiency
    # For this we need the number of actual simulator calls, which we can get
    # from the number of proposal samples where the prior is non-zero: Those
    # are the samples for which we ran the simulator, even if the likelihood
    # came out as zero (e.g., because the simulation failed or timed out).
    if log_prior_values is not None:
        n_simulator_calls = float(np.sum(log_prior_values > -np.inf))
        simulator_efficiency = n_eff / n_simulator_calls
    else:
        simulator_efficiency = np.nan

    return n_eff, sampling_efficiency, simulator_efficiency


def compute_log_evidence(
    raw_log_weights: np.ndarray,
) -> tuple[float, float]:
    """
    Compute the estimate of the log-evidence and its standard deviation.

    Args:
        raw_log_weights: Raw log-weights.

    Returns:
        log_evidence: Log-evidence estimate.
        log_evidence_std: Standard deviation of the log-evidence.
    """

    # Normalize the raw log-weights
    weights = clip_and_normalize_weights(raw_log_weights)

    # Compute the number of samples and the effective sample size
    n = len(raw_log_weights)
    n_eff, _, _ = compute_effective_sample_size(weights)

    # Copmute the log-evidence estimate and its standard deviation
    # noinspection PyUnresolvedReferences
    log_evidence = float(logsumexp(raw_log_weights) - np.log(n))
    log_evidence_std = float(np.sqrt((n - n_eff) / (n * n_eff)))

    return log_evidence, log_evidence_std
