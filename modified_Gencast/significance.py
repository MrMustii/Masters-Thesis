"""Paired significance testing for the comparison layer.

Decides whether a difference between two runs is real. Daily scores are autocorrelated - a
heatwave makes several consecutive days hard together - so an ordinary bootstrap that
resamples days independently understates the uncertainty and finds differences that are not
there. Everything here is built on a stationary bootstrap, which resamples whole blocks of
consecutive cases and so preserves that dependence.

    estimate_block_length         how long those blocks should be, chosen from the data
    family_block_length           one block length for a whole family of pairs
    bootstrap_paired_difference   interval and p-value for one paired difference series
    bootstrap_paired_metric       the same for a metric that is a nonlinear function of
                                  case means, where per-case values cannot just be averaged
    apply_multiplicity_correction Holm-Bonferroni within each (tier, family) group

Two deliberate choices run through the file. Resamples are shifted onto the null before
p-values are counted, and p is (r + 1) / (B + 1) rather than r / B, so a p-value is never
reported as exactly zero.
"""

import numpy as np


HOLM_ALPHA = 0.05

def estimate_block_length(values: np.ndarray, max_lag: int | None = None) -> float:
    """Politis-White/Patton automatic mean block length for a stationary bootstrap."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    series_length = len(values)
    if series_length < 3:
        return 1.0

    centered_values = values - values.mean()
    if np.allclose(centered_values, 0.0):
        return 1.0

    # How far to look for autocorrelation, and the largest block the series can support.
    block_cap = float(np.ceil(min(3 * np.sqrt(series_length), series_length / 3)))
    consecutive_lags = max(5, int(np.log10(series_length)))
    maximum_lag = int(np.ceil(np.sqrt(series_length))) + consecutive_lags
    if max_lag is not None:
        maximum_lag = min(maximum_lag, max_lag)
    maximum_lag = min(maximum_lag, series_length - 2)
    if maximum_lag < 1:
        return 1.0

    # Autocovariance at every lag, and the same lag normalised into a correlation.
    autocovariance = np.zeros(maximum_lag + 1, dtype=float)
    absolute_autocorrelation = np.full(maximum_lag + 1, np.inf, dtype=float)
    for lag in range(maximum_lag + 1):
        cross_product = np.dot(centered_values[lag:], centered_values[:series_length - lag])
        autocovariance[lag] = cross_product / series_length
        left_norm = np.dot(centered_values[lag + 1:], centered_values[lag + 1:])
        right_norm = np.dot(centered_values[:series_length - lag - 1], centered_values[:series_length - lag - 1])
        if left_norm > 0 and right_norm > 0:
            absolute_autocorrelation[lag] = abs(cross_product) / np.sqrt(left_norm * right_norm)

    # First lag after which consecutive_lags correlations in a row sit inside the noise band.
    critical_value = 2 * np.sqrt(np.log10(series_length) / series_length)
    selected_lag = None
    for lag in range(consecutive_lags, maximum_lag + 1):
        if np.all(absolute_autocorrelation[lag - consecutive_lags:lag] < critical_value):
            selected_lag = lag - consecutive_lags
            break
    bandwidth = 2 * max(selected_lag or 1, 1) if selected_lag is not None else maximum_lag
    bandwidth = min(bandwidth, maximum_lag)

    # Flat-top kernel sums, which the cube-root formula below turns into a block length.
    weighted_lag_sum = 0.0
    long_run_variance = autocovariance[0]
    for lag in range(1, bandwidth + 1):
        weight = 1.0 if lag / bandwidth <= 0.5 else 2.0 * (1.0 - lag / bandwidth)
        weighted_lag_sum += 2.0 * weight * lag * autocovariance[lag]
        long_run_variance += 2.0 * weight * autocovariance[lag]
    if not np.isfinite(long_run_variance) or abs(long_run_variance) <= np.finfo(float).eps:
        return 1.0

    stationary_denominator = 2.0 * long_run_variance ** 2
    estimated = ((2.0 * weighted_lag_sum ** 2 / stationary_denominator) * series_length) ** (1.0 / 3.0)
    if not np.isfinite(estimated):
        return 1.0
    return float(np.clip(estimated, 1.0, block_cap))


def family_block_length(difference_series) -> float:
    """Median block length over difference_series, one paired difference array per pair.

    Per-pair selection is unstable near the selector's critical band: two series with
    near-identical autocorrelation can be handed block lengths differing fivefold, and
    that alone decides which pairs come out significant. Median rather than maximum,
    because the largest value in a family is usually the selector hitting its analytic cap.
    """
    lengths = [estimate_block_length(np.asarray(series, dtype=float))
               for series in difference_series]
    return float(np.median(lengths)) if lengths else 1.0


def stationary_bootstrap_indices(series_length: int, block_length: float, rng: np.random.Generator) -> np.ndarray:
    """One stationary-bootstrap resample with geometric, circular blocks."""
    if series_length < 1:
        raise ValueError("series_length must be positive")
    if block_length < 1:
        raise ValueError("block_length must be at least one")
    indices = np.empty(series_length, dtype=int)
    position = 0
    while position < series_length:
        start = rng.integers(0, series_length)
        block = min(rng.geometric(1.0 / block_length), series_length - position)
        indices[position:position + block] = (start + np.arange(block)) % series_length
        position += block
    return indices


def _stationary_bootstrap_index_matrix(series_length: int, block_length: float, num_resamples: int, rng: np.random.Generator) -> np.ndarray:
    """num_resamples resample index rows at once, the same draw as the loop above."""
    starts = rng.integers(0, series_length, size=(num_resamples, series_length), dtype=np.int32)
    restarts = rng.random((num_resamples, series_length)) < (1.0 / block_length)
    restarts[:, 0] = True
    indices = np.empty((num_resamples, series_length), dtype=np.int32)
    indices[:, 0] = starts[:, 0]
    # Every position either opens a new block or continues the last one, wrapping at the end.
    for position in range(1, series_length):
        indices[:, position] = np.where(
            restarts[:, position], starts[:, position], (indices[:, position - 1] + 1) % series_length)
    return indices


def percentile_interval_bounds(bootstrap_statistics: np.ndarray, confidence: float) -> tuple:
    """Central percentile interval from a bootstrap-statistic distribution."""
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie strictly between 0 and 1")
    alpha = 1 - confidence
    lower_bound, upper_bound = np.quantile(bootstrap_statistics, [alpha / 2, 1 - alpha / 2])
    return float(lower_bound), float(upper_bound)


def bootstrap_paired_difference(differences: np.ndarray, num_resamples: int = 10000, confidence: float = 0.95, seed: int = 42, block_length: float | None = None) -> dict:
    """Stationary-block percentile interval and p-value for one paired difference series.

    block_length overrides the per-series selection; pass one value for every pair in a
    family, since the selector is unstable enough near its critical band to decide
    significance on its own. What it chose is still reported, as block_length_selected.
    """
    differences = np.asarray(differences, dtype=float)
    differences = differences[np.isfinite(differences)]
    series_length = len(differences)
    if series_length < 3:
        return {"point_estimate": float("nan"), "lower_bound": float("nan"),
                "upper_bound": float("nan"), "significant": False, "block_length": 1.0,
                "block_length_selected": 1.0,
                "interval_method": "percentile", "block_length_method": "unavailable",
                "p_value": float("nan")}

    # One resampled mean per draw, in batches so the index matrix stays a manageable size.
    selected_block_length = estimate_block_length(differences)
    block_length = selected_block_length if block_length is None else float(block_length)
    rng = np.random.default_rng(seed)
    bootstrap_means = np.empty(num_resamples, dtype=float)
    batch_size = min(1000, num_resamples)
    for start in range(0, num_resamples, batch_size):
        stop = min(start + batch_size, num_resamples)
        indices = _stationary_bootstrap_index_matrix(
            series_length, block_length, stop - start, rng)
        bootstrap_means[start:stop] = differences[indices].mean(axis=1)

    lower_bound, upper_bound = percentile_interval_bounds(bootstrap_means, confidence)
    # Shift the resamples onto the null, or counting them just inverts the interval.
    point_estimate = float(differences.mean())
    null_means = bootstrap_means - point_estimate
    as_extreme = int((np.abs(null_means) >= abs(point_estimate)).sum())
    # (r + 1) / (B + 1), not r / B, which reports exactly 0 when no resample is as extreme.
    p_value = min(1.0, (as_extreme + 1) / (num_resamples + 1))
    return {
        "point_estimate": point_estimate, "lower_bound": lower_bound,
        "upper_bound": upper_bound, "significant": not (lower_bound <= 0 <= upper_bound),
        "block_length": block_length, "block_length_selected": selected_block_length,
        "interval_method": "percentile",
        "block_length_method": "Politis-White/Patton automatic stationary-bootstrap selector",
        "p_value": p_value,
    }







def apply_multiplicity_correction(rows: list[dict], p_value_key: str = "p_value",
                                  tier_key: str = "tier", family_key: str = "family",
                                  alpha: float | None = None) -> None:
    """In-place Holm-Bonferroni correction within each (tier, family) group of rows.

    A family is one question by its candidates by its metrics at that tier, never pooled
    across families, questions or research questions. Writes p_adjusted, alpha_threshold
    (the number p_adjusted must not exceed) and significant_corrected onto every row. The
    tier splits the families but not the method, so a supporting metric never inflates the
    penalty on a pre-registered one and no pair turns significant purely by its tier.
    """
    # Default every row to untested, then bucket the rows into their correction families.
    groups: dict[tuple, list[dict]] = {}
    threshold = HOLM_ALPHA if alpha is None else alpha
    for row in rows:
        row["p_adjusted"] = float("nan")
        row["alpha_threshold"] = threshold
        row["significant_corrected"] = False
        groups.setdefault((row[tier_key], row[family_key]), []).append(row)

    # Holm step-down: scale each p by the tests still standing, kept monotone by running_max.
    for group_rows in groups.values():
        testable_rows = [row for row in group_rows if np.isfinite(row[p_value_key])]
        if not testable_rows:
            continue
        p_values = np.array([row[p_value_key] for row in testable_rows])
        order = np.argsort(p_values)
        num_tests = len(testable_rows)
        adjusted = np.empty(num_tests)
        running_max = 0.0
        for rank, index in enumerate(order):
            running_max = max(running_max, (num_tests - rank) * p_values[index])
            adjusted[index] = min(running_max, 1.0)

        for row, value in zip(testable_rows, adjusted):
            row["p_adjusted"] = float(value)
            row["significant_corrected"] = bool(value <= threshold)


def bootstrap_paired_metric(candidate_components: dict, reference_components: dict,
                            metric_function, proxy: np.ndarray | None = None,
                            num_resamples: int = 10000, confidence: float = 0.95,
                            seed: int = 42, block_length: float | None = None) -> dict:
    """Paired interval for a metric that is a nonlinear function of case means, as in
    GenCast's Table E2.

    Each component is a per-case array shaped (cases,) or (cases, settings). The case axis
    is resampled and the metric recomputed from the resampled means, not averaged over
    per-case metric values, which would be wrong for any nonlinear metric. The same indices
    go to both runs. Block length comes from a univariate proxy of the paired difference.
    """
    names = sorted(candidate_components)
    candidate = {name: np.asarray(candidate_components[name], dtype=float) for name in names}
    reference = {name: np.asarray(reference_components[name], dtype=float) for name in names}
    series_length = len(candidate[names[0]])
    if series_length < 3:
        return {"point_estimate": float("nan"), "lower_bound": float("nan"),
                "upper_bound": float("nan"), "significant": False, "block_length": 1.0,
                "block_length_selected": 1.0,
                "interval_method": "percentile", "block_length_method": "unavailable",
                "p_value": float("nan")}

    def difference(indices):
        return (metric_function({name: candidate[name][indices].mean(axis=0) for name in names})
                - metric_function({name: reference[name][indices].mean(axis=0) for name in names}))

    # Point estimate on the unresampled cases; the proxy only sets the block length.
    all_indices = np.arange(series_length)
    point_estimate = float(difference(all_indices))
    if proxy is None:
        first = names[0]
        proxy = (candidate[first] - reference[first]).reshape(series_length, -1).mean(axis=1)
    selected_block_length = estimate_block_length(np.asarray(proxy, dtype=float))
    block_length = selected_block_length if block_length is None else float(block_length)

    rng = np.random.default_rng(seed)
    bootstrap_differences = np.empty(num_resamples, dtype=float)
    for resample_index in range(num_resamples):
        bootstrap_differences[resample_index] = difference(
            stationary_bootstrap_indices(series_length, block_length, rng))

    lower_bound, upper_bound = percentile_interval_bounds(bootstrap_differences, confidence)
    # Shift onto the null before counting, exactly as bootstrap_paired_difference does.
    null_differences = bootstrap_differences - point_estimate
    as_extreme = int((np.abs(null_differences) >= abs(point_estimate)).sum())
    p_value = min(1.0, (as_extreme + 1) / (num_resamples + 1))
    return {"point_estimate": point_estimate, "lower_bound": lower_bound,
            "upper_bound": upper_bound, "significant": not (lower_bound <= 0 <= upper_bound),
            "block_length": block_length, "block_length_selected": selected_block_length,
            "interval_method": "percentile",
            "block_length_method": "Politis-White/Patton automatic stationary-bootstrap selector",
            "p_value": p_value}
