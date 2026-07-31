import numpy as np
import xarray as xr
from scipy import stats


def estimate_block_length(values: np.ndarray, max_lag: int = 40) -> int:
    """First lag where the series' autocorrelation falls inside the 95%
    white-noise band 1.96/sqrt(length); returns 1 (no blocking) if it never
    does within max_lag, and max_lag if the series stays correlated throughout."""
    series_length = len(values)
    max_lag = min(max_lag, series_length - 1)
    if max_lag < 1:
        return 1
    centered_values = values - values.mean()
    variance = (centered_values ** 2).sum()
    if variance == 0:
        return 1
    white_noise_bound = 1.96 / np.sqrt(series_length)
    for lag in range(1, max_lag + 1):
        autocorrelation = (centered_values[:-lag] * centered_values[lag:]).sum() / variance
        if abs(autocorrelation) < white_noise_bound:
            return lag
    return max_lag


def stationary_bootstrap_indices(series_length: int, block_length: int, rng: np.random.Generator) -> np.ndarray:
    """One resample's index array under the stationary bootstrap: blocks with
    geometrically-distributed lengths (mean block_length) starting at a random
    position and wrapping circularly, concatenated until series_length indices
    are drawn."""
    indices = np.empty(series_length, dtype=int)
    position = 0
    while position < series_length:
        start = rng.integers(0, series_length)
        block = min(rng.geometric(1.0 / block_length), series_length - position)
        indices[position:position + block] = (start + np.arange(block)) % series_length
        position += block
    return indices


def bias_corrected_accelerated_bounds(jackknife_estimates: np.ndarray, bootstrap_statistics: np.ndarray,
                                      point_estimate: float, confidence: float) -> tuple:
    """BCa interval bounds: corrects the plain bootstrap percentile interval
    for both the bootstrap distribution's skew (bias correction, from the
    fraction of resamples below the point estimate) and how fast the statistic
    changes as data is added or removed (acceleration, from a leave-one-out
    jackknife of the statistic itself, supplied by the caller since the
    statistic being jackknifed differs per use)."""
    proportion_below = np.clip(np.mean(bootstrap_statistics < point_estimate), 1e-6, 1 - 1e-6)
    bias_correction = stats.norm.ppf(proportion_below)

    jackknife_mean = jackknife_estimates.mean()
    numerator = ((jackknife_mean - jackknife_estimates) ** 3).sum()
    denominator = 6.0 * (((jackknife_mean - jackknife_estimates) ** 2).sum() ** 1.5)
    acceleration = numerator / denominator if denominator != 0 else 0.0

    alpha = 1 - confidence
    lower_z = stats.norm.ppf(alpha / 2)
    upper_z = stats.norm.ppf(1 - alpha / 2)
    lower_percentile = stats.norm.cdf(
        bias_correction + (bias_correction + lower_z) / (1 - acceleration * (bias_correction + lower_z)))
    upper_percentile = stats.norm.cdf(
        bias_correction + (bias_correction + upper_z) / (1 - acceleration * (bias_correction + upper_z)))
    # If BCa degenerates (non-finite adjusted percentile), fall back to the plain
    # percentile interval rather than crashing np.quantile.
    if not (np.isfinite(lower_percentile) and np.isfinite(upper_percentile)):
        lower_percentile, upper_percentile = alpha / 2, 1 - alpha / 2
    lower_bound, upper_bound = np.quantile(bootstrap_statistics, [lower_percentile, upper_percentile])
    return float(lower_bound), float(upper_bound)


def bootstrap_difference_interval(series_a: xr.DataArray, series_b: xr.DataArray,
                                  num_resamples: int = 10000, confidence: float = 0.95,
                                  seed: int = 42) -> dict:
    """BCa interval on mean(series_a - series_b) for two paired per-forecast
    score series (crps_series, dtr_crps_series, peak_time_ccrps_series,
    conditional_tmax_bias_series, ...), aligned one-to-one on the same
    forecast index. A positive point estimate
    means series_a scores higher; "significant" is true when the interval
    excludes 0.
    """
    differences = np.asarray(series_a.values, dtype=float) - np.asarray(series_b.values, dtype=float)
    differences = differences[np.isfinite(differences)]  # drop NaN pairs (else BCa → NaN percentile)
    series_length = len(differences)
    if series_length < 3:
        return {"point_estimate": float("nan"), "lower_bound": float("nan"),
                "upper_bound": float("nan"), "significant": False, "block_length": 1}
    point_estimate = float(differences.mean())
    block_length = estimate_block_length(differences)

    rng = np.random.default_rng(seed)
    bootstrap_means = np.empty(num_resamples)
    for resample_index in range(num_resamples):
        indices = stationary_bootstrap_indices(series_length, block_length, rng)
        bootstrap_means[resample_index] = differences[indices].mean()

    jackknife_estimates = np.array([np.delete(differences, index).mean() for index in range(series_length)])
    lower_bound, upper_bound = bias_corrected_accelerated_bounds(
        jackknife_estimates, bootstrap_means, point_estimate, confidence)
    return {
        "point_estimate": point_estimate, "lower_bound": lower_bound, "upper_bound": upper_bound,
        "significant": not (lower_bound <= 0 <= upper_bound), "block_length": block_length,
    }


def bootstrap_classification_interval(true_positive_a: xr.DataArray, false_positive_a: xr.DataArray,
                                      false_negative_a: xr.DataArray, true_positive_b: xr.DataArray,
                                      false_positive_b: xr.DataArray, false_negative_b: xr.DataArray,
                                      num_resamples: int = 10000, confidence: float = 0.95,
                                      seed: int = 42) -> dict:
    """BCa interval on F1_a - F1_b for two conditions' per-forecast confusion
    counts (the tp/fp/fn series from binary_classification_metrics's
    counts_series, e.g. hw_tp_series/hw_fp_series/hw_fn_series), aligned
    one-to-one on the same forecast index. F1 is nonlinear in the counts, so
    each resample SUMS the resampled per-forecast counts and recomputes F1
    from the sums, rather than averaging per-forecast F1 values; the same
    resample indices are applied to both conditions to preserve the pairing.
    """
    true_positive_a, false_positive_a, false_negative_a = (
        np.asarray(series.values, dtype=float) for series in (true_positive_a, false_positive_a, false_negative_a))
    true_positive_b, false_positive_b, false_negative_b = (
        np.asarray(series.values, dtype=float) for series in (true_positive_b, false_positive_b, false_negative_b))
    series_length = len(true_positive_a)

    def f1_score(true_positive, false_positive, false_negative):
        precision = true_positive / (true_positive + false_positive + 1e-12)
        recall = true_positive / (true_positive + false_negative + 1e-12)
        return 2 * precision * recall / (precision + recall + 1e-12)

    def f1_difference(indices):
        f1_a = f1_score(true_positive_a[indices].sum(), false_positive_a[indices].sum(), false_negative_a[indices].sum())
        f1_b = f1_score(true_positive_b[indices].sum(), false_positive_b[indices].sum(), false_negative_b[indices].sum())
        return f1_a - f1_b

    all_indices = np.arange(series_length)
    point_estimate = float(f1_difference(all_indices))
    block_length = estimate_block_length(true_positive_a + false_positive_a + false_negative_a)

    rng = np.random.default_rng(seed)
    bootstrap_differences = np.empty(num_resamples)
    for resample_index in range(num_resamples):
        indices = stationary_bootstrap_indices(series_length, block_length, rng)
        bootstrap_differences[resample_index] = f1_difference(indices)

    jackknife_estimates = np.array(
        [f1_difference(np.delete(all_indices, index)) for index in range(series_length)])
    lower_bound, upper_bound = bias_corrected_accelerated_bounds(
        jackknife_estimates, bootstrap_differences, point_estimate, confidence)
    return {
        "point_estimate": point_estimate, "lower_bound": lower_bound, "upper_bound": upper_bound,
        "significant": not (lower_bound <= 0 <= upper_bound), "block_length": block_length,
    }


def equivalence_test(lower_bound: float, upper_bound: float, delta: float) -> dict:
    """Turn a difference CI [lower, upper] on (candidate - reference), lower-is-
    better scores, into the three-way preservation verdict. delta > 0 is the
    pre-specified tolerance (e.g. 10% of the model-over-climatology margin):

      upper < delta  → "preserved"     (candidate gives up at most delta)
      lower > 0      → "degraded"      (candidate is genuinely worse)
      otherwise      → "inconclusive"  (one year can't tell — the outcome a
                                        bare "CI contains 0" silently hides)

    "CI contains 0" is NOT evidence of no difference; equivalence needs the whole
    interval inside the tolerance, which is what this checks.
    """
    if upper_bound < delta:
        verdict = "preserved"
    elif lower_bound > 0:
        verdict = "degraded"
    else:
        verdict = "inconclusive"
    return {"verdict": verdict, "lower_bound": lower_bound, "upper_bound": upper_bound, "delta": delta}


def bootstrap_event_interval(event_lengths_a: np.ndarray, event_lengths_b: np.ndarray, statistic,
                             num_resamples: int = 10000, confidence: float = 0.95,
                             seed: int = 42) -> dict:
    """Percentile interval on statistic(event_lengths_a) - statistic(event_lengths_b)
    for two independent pools of whole heatwave-event lengths
    (spell_length_distribution's "lengths": pass numpy.mean for a duration-bias
    test, len for an event-count test). Resamples whole EVENTS with
    replacement rather than time blocks, since a time-block boundary could
    split one event across two blocks and double-count part of it. The two
    pools are independent draws — model fragmentation means its events don't
    correspond one-to-one with the other run's — so this uses a plain
    percentile interval rather than the paired BCa above, which needs a
    jackknife over a shared index that doesn't exist here.
    """
    event_lengths_a = np.asarray(event_lengths_a, dtype=float)
    event_lengths_b = np.asarray(event_lengths_b, dtype=float)
    if len(event_lengths_a) == 0 or len(event_lengths_b) == 0:
        return {"point_estimate": 0.0, "lower_bound": 0.0, "upper_bound": 0.0, "significant": False}

    point_estimate = float(statistic(event_lengths_a) - statistic(event_lengths_b))
    rng = np.random.default_rng(seed)
    bootstrap_differences = np.empty(num_resamples)
    for resample_index in range(num_resamples):
        resample_a = rng.choice(event_lengths_a, size=len(event_lengths_a), replace=True)
        resample_b = rng.choice(event_lengths_b, size=len(event_lengths_b), replace=True)
        bootstrap_differences[resample_index] = statistic(resample_a) - statistic(resample_b)

    alpha = 1 - confidence
    lower_bound, upper_bound = np.quantile(bootstrap_differences, [alpha / 2, 1 - alpha / 2])
    return {
        "point_estimate": point_estimate, "lower_bound": float(lower_bound), "upper_bound": float(upper_bound),
        "significant": not (lower_bound <= 0 <= upper_bound),
    }
