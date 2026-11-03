"""Normal-Normal Thompson sampling with known observation noise.

No Streamlit dependency: experiments and tests use the same simulation as the UI.
"""

from dataclasses import dataclass
from numbers import Integral

import numpy as np
import pandas as pd

PRIOR_MEAN = 0.0
PRIOR_VARIANCE = 25.0


@dataclass
class SimulationResult:
    summary: pd.DataFrame
    history: pd.DataFrame
    total_reward: float
    recommendation: int  # one-based alternative number


def update_posterior(mean: float, variance: float, reward: float, sigma: float):
    """Return posterior mean and variance after one observation.

    Precision adds: 1/v_new = 1/v_old + 1/sigma**2.
    The mean is the precision-weighted average of prior mean and reward.
    """
    if not all(np.isfinite(x) for x in (mean, variance, reward, sigma)):
        raise ValueError("Posterior inputs must be finite.")
    if variance <= 0 or sigma <= 0:
        raise ValueError("Variance and noise standard deviation must be positive.")
    noise_variance = sigma * sigma
    if not np.isfinite(noise_variance) or noise_variance == 0:
        raise ValueError("Noise standard deviation is outside the supported numeric range.")
    weight = variance / (variance + noise_variance)
    return mean + weight * (reward - mean), noise_variance * weight


def select_alternative(means, variances, rng):
    """Selection depends exclusively on learned posteriors, never true means."""
    return int(np.argmax(rng.normal(means, np.sqrt(variances))))


def simulate(true_means, sigma: float, rounds: int, seed: int) -> SimulationResult:
    true_means = np.asarray(true_means, dtype=float)
    if true_means.shape != (3,) or not np.all(np.isfinite(true_means)):
        raise ValueError("Provide exactly three finite true means.")
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("Noise standard deviation must be positive and finite.")
    if isinstance(rounds, bool) or not isinstance(rounds, Integral) or rounds <= 0:
        raise ValueError("Rounds must be a positive integer.")
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("Seed must be a nonnegative integer.")
    # Also check noise arithmetic before starting a potentially long run.
    update_posterior(PRIOR_MEAN, PRIOR_VARIANCE, 0.0, sigma)
    rng = np.random.default_rng(seed)
    means = np.full(3, PRIOR_MEAN)
    variances = np.full(3, PRIOR_VARIANCE)
    counts = np.zeros(3, dtype=int)
    records = []
    total_reward = 0.0
    for step in range(1, rounds + 1):
        selected = select_alternative(means, variances, rng)
        reward = float(rng.normal(true_means[selected], sigma))
        means[selected], variances[selected] = update_posterior(
            means[selected], variances[selected], reward, sigma
        )
        counts[selected] += 1
        total_reward += reward
        records.append({
            "round": step, "alternative": selected + 1, "reward": reward,
            "cumulative_reward": total_reward,
            **{f"estimated_mean_{i+1}": float(means[i]) for i in range(3)},
        })
    summary = pd.DataFrame({
        "alternative": [1, 2, 3], "true_mean": true_means,
        "posterior_mean": means, "posterior_std": np.sqrt(variances),
        "selection_count": counts,
    })
    return SimulationResult(summary, pd.DataFrame(records), total_reward,
                            int(np.argmax(means)) + 1)
