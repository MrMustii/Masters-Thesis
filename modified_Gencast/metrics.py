import numpy as np
import pandas as pd
import xarray as xr
import scoringrules as sr
from scipy import stats
from sklearn.metrics import roc_auc_score
import scores.probability

TARGET_VAR = "2m_temperature"


def drop_batch(data_array: xr.DataArray) -> xr.DataArray:
    """Squeeze a singleton `batch` dimension if present."""
    if "batch" in data_array.dims and data_array.sizes["batch"] == 1:
        return data_array.squeeze("batch", drop=True)
    return data_array


def cos_weights(data_array: xr.DataArray) -> xr.DataArray:
    """Cosine-of-latitude area weights aligned to the array's lat coordinate."""
    return xr.DataArray(np.cos(np.deg2rad(data_array.lat)), coords={"lat": data_array.lat})


def _full_day_idx(hour_values: np.ndarray) -> np.ndarray:
    """day_idx label per hour, with partial leading/trailing blocks marked -1."""
    day_idx_values = (hour_values - hour_values[0]) // 24
    labels, counts = np.unique(day_idx_values, return_counts=True)
    full = set(labels[counts == 24].tolist())
    return np.array([d if d in full else -1 for d in day_idx_values])


def get_heatwaves(tmax: xr.DataArray, threshold: xr.DataArray, min_days: int = 3) -> xr.DataArray:
    """Boolean mask of heatwave days per pixel.

    All days in a qualifying spell (min_days or more consecutive days above
    threshold) are flagged, including the first min_days-1 days.
    """
    exceeds = (tmax > threshold).astype(int)
    end_of_spell = (exceeds.rolling(day_idx=min_days, center=False).sum() >= min_days)
    return (
        end_of_spell
        .isel(day_idx=slice(None, None, -1))
        .rolling(day_idx=min_days, center=False, min_periods=1).max()
        .isel(day_idx=slice(None, None, -1))
        .fillna(False)
        .astype(bool)
    )


def mean_forecast(data_array: xr.DataArray) -> xr.DataArray:
    """Mean over forecast and ensemble (if present). Returns (hour, lat, lon)."""
    if "ensemble" in data_array.dims:
        data_array = data_array.mean("ensemble")
    return data_array.mean("forecast")


def spatial_errors(pred: xr.DataArray, target: xr.DataArray) -> dict:
    """Spatial error maps (hour, lat, lon), averaged over forecast.

    For ensemble: errors on ensemble mean. rmse/bias curves are cos-weighted.
    """
    pred = drop_batch(pred)
    target = drop_batch(target)
    is_ensemble = "ensemble" in pred.dims
    pred_mean = pred.mean("ensemble") if is_ensemble else pred
    diff = pred_mean - target
    w = cos_weights(pred_mean)
    rmse_map = np.sqrt((diff ** 2).mean("forecast"))
    bias_map = diff.mean("forecast")
    return {
        "rmse_map": rmse_map,
        "bias_map": bias_map,
        "rmse": np.sqrt((diff ** 2).weighted(w).mean(["lat", "lon"]).mean("forecast")),
        "bias": bias_map.weighted(w).mean(["lat", "lon"]),
    }


def binary_classification_metrics(pred_hw: xr.DataArray, tgt_hw: xr.DataArray) -> dict:
    """Precision, Recall, F1 as curves (day_idx,) and maps (lat, lon), plus
    per-initialization confusion-count series (forecast,) for the block
    bootstrap (F1 must be recomputed from resampled summed counts).

    Takes boolean output of get_heatwaves(). pred_hw must be deterministic.
    """
    w = cos_weights(tgt_hw)

    tp = (pred_hw & tgt_hw).sum("forecast")
    fp = (pred_hw & ~tgt_hw).sum("forecast")
    fn = (~pred_hw & tgt_hw).sum("forecast")

    tp_w = tp.weighted(w).sum(["lat", "lon"])
    fp_w = fp.weighted(w).sum(["lat", "lon"])
    fn_w = fn.weighted(w).sum(["lat", "lon"])
    precision_curve = (tp_w / (tp_w + fp_w)).fillna(0)
    recall_curve = (tp_w / (tp_w + fn_w)).fillna(0)
    f1_curve = (2 * precision_curve * recall_curve / (precision_curve + recall_curve + 1e-12))

    tp_s = tp.sum("day_idx")
    fp_s = fp.sum("day_idx")
    fn_s = fn.sum("day_idx")
    defined = (tp_s + fp_s + fn_s) > 0
    precision_map = (tp_s / (tp_s + fp_s)).fillna(0).where(defined)
    recall_map = (tp_s / (tp_s + fn_s)).fillna(0).where(defined)
    f1_map = (2 * precision_map * recall_map / (precision_map + recall_map + 1e-12)).where(defined)

    counts_series = {
        "tp": (pred_hw & tgt_hw).astype(float).weighted(w).sum(["lat", "lon", "day_idx"]),
        "fp": (pred_hw & ~tgt_hw).astype(float).weighted(w).sum(["lat", "lon", "day_idx"]),
        "fn": (~pred_hw & tgt_hw).astype(float).weighted(w).sum(["lat", "lon", "day_idx"]),
        "tn": (~pred_hw & ~tgt_hw).astype(float).weighted(w).sum(["lat", "lon", "day_idx"]),
    }
    return {
        "precision": precision_curve, "recall": recall_curve, "f1": f1_curve,
        "precision_map": precision_map, "recall_map": recall_map, "f1_map": f1_map,
        "counts_series": counts_series,
    }


def rev(pred_tmax, tgt_tmax, threshold, n_alpha: int = 99) -> xr.DataArray:
    """Relative Economic Value over C/L in [0.01, 0.99]. Needs ensemble dim; custom, unverified against scores.plotdata.rev_impl."""
    if "ensemble" not in pred_tmax.dims:
        raise ValueError("REV requires ensemble predictions.")
    M = pred_tmax.sizes["ensemble"]
    obs = (tgt_tmax > threshold).astype(float)
    prob = (pred_tmax > threshold).astype(float).mean("ensemble")
    w = cos_weights(tgt_tmax)
    mu = float(obs.weighted(w).mean(["lat", "lon"]).mean())
    warning_rates = np.zeros(M + 2)
    miss_rates = np.zeros(M + 2)
    for j in range(M + 2):
        q = (j - 0.5) / M
        forecast_yes = (prob > q)
        warning_rates[j] = float(forecast_yes.astype(float).weighted(w).mean(["lat", "lon"]).mean())
        miss_rates[j] = float(((~forecast_yes) & (obs > 0.5)).astype(float).weighted(w).mean(["lat", "lon"]).mean())
    alpha_values = np.linspace(0.01, 0.99, n_alpha)
    e_clim = np.minimum(alpha_values, mu)
    e_forecasts = alpha_values[:, None] * warning_rates[None, :] + miss_rates[None, :]
    e_min = e_forecasts.min(axis=1)
    denom = e_clim - alpha_values * mu
    rev_values = np.where(denom > 1e-12, np.maximum(0.0, (e_clim - e_min) / denom), 0.0)
    return xr.DataArray(rev_values, dims=["cl_ratio"], coords={"cl_ratio": alpha_values})


def power_spectra_1d(data_array: xr.DataArray) -> tuple:
    values = np.asarray(data_array.values, dtype=float).ravel()
    n = len(values)
    hour_vals = data_array.hour.values
    dt_hours = float(hour_vals[1] - hour_vals[0]) if len(hour_vals) > 1 else 1.0
    freqs = np.fft.rfftfreq(n, d=dt_hours)
    psd = (np.abs(np.fft.rfft(values - values.mean())) ** 2) / n
    return 1.0 / freqs[1:], psd[1:]


def temporal_autocorrelation(data_array: xr.DataArray, max_lag: int = 48) -> xr.DataArray:
    spatial_dims = [d for d in ("lat", "lon") if d in data_array.dims]
    if spatial_dims:
        w = cos_weights(data_array)
        data_array = data_array.weighted(w).mean(spatial_dims)
    n = data_array.sizes["hour"]
    max_lag = min(max_lag, n - 1)
    non_hour = [d for d in data_array.dims if d != "hour"]
    stk = data_array.stack(series=non_hour).transpose("series", "hour")
    vals = stk.values.astype(float)
    vals -= vals.mean(axis=-1, keepdims=True)
    norm = (vals * vals).sum(axis=-1)
    acs = np.stack([
        (vals[:, :-lag] * vals[:, lag:]).sum(axis=-1) / (norm + 1e-12) if lag > 0
        else norm / (norm + 1e-12)
        for lag in range(max_lag + 1)
    ], axis=0)
    ac_mean = acs.mean(axis=-1)
    return xr.DataArray(ac_mean, dims=["lag"], coords={"lag": np.arange(max_lag + 1)})


def crps_field(pred: xr.DataArray, target: xr.DataArray) -> xr.DataArray:
    """Per-point CRPS field (fair ensemble form), before any aggregation.

    Standalone (not inside crps) because the transformed scores need it too:
    DTR-CRPS and HAT-CRPS score per-member TRANSFORMED ensembles (daily
    amplitude, exceedance count) and exp1 scores an absolute-time axis with no
    (forecast, hour) layout to aggregate over. Without an ensemble dim the
    forecast is deterministic — the baselines — and CRPS collapses to the
    absolute error, still the same proper score, so climatology/persistence/
    interpolated-truth get a comparable crps_series instead of no CRPS at all.
    """
    pred = drop_batch(pred)
    target = drop_batch(target)
    if "ensemble" not in pred.dims:
        return abs(pred - target)
    # Ferro-2014 fair CRPS; preserve_dims="all" keeps every non-ensemble dim, so this
    # returns the per-gridpoint field.
    return scores.probability.crps_for_ensemble(
        pred, target, ensemble_member_dim="ensemble", method="fair", preserve_dims="all")


def crps(pred: xr.DataArray, target: xr.DataArray, threshold: xr.DataArray = None) -> dict:
    """CRPS aggregates over (forecast, hour, lat, lon) fields.

    With `threshold` this is the threshold-weighted CRPS via the chaining
    function v(x) = max(x, threshold): members and observation are clipped at
    the cell's day-of-year p90 before scoring, which keeps the score strictly
    proper while weighting exactly the upper tail heatwaves live in (Allen et
    al. 2025). Plain CRPS integrates the Brier score over ALL thresholds and
    dilutes that tail.
    """
    if threshold is not None:
        pred = np.maximum(pred, threshold)
        target = np.maximum(target, threshold)
    field = crps_field(pred, target)
    crps_map = field.mean("forecast")
    w = cos_weights(target)
    crps_growth = field.weighted(w).mean(["lat", "lon"]).mean("forecast")
    crps_series = field.weighted(w).mean(["lat", "lon"]).mean("hour")
    day1_hours = field.hour.values < (field.hour.values.min() + 24)
    crps_day1_series = field.isel(hour=day1_hours).weighted(w).mean(["lat", "lon", "hour"])
    return {"crps_map": crps_map, "crps": crps_growth, "crps_series": crps_series,
            "crps_day1_series": crps_day1_series}


def spread_skill_ratio(pred: xr.DataArray, target: xr.DataArray) -> dict:
    if "ensemble" not in pred.dims:
        raise ValueError("Spread-skill ratio requires ensemble predictions.")
    pred = drop_batch(pred)
    target = drop_batch(target)
    M = pred.sizes["ensemble"]
    w = cos_weights(pred)
    spread_growth = np.sqrt(
        pred.var("ensemble").weighted(w).mean(["lat", "lon"]).mean("forecast")
    ) * np.sqrt((M + 1) / (M - 1))
    skill_growth = np.sqrt(
        ((pred.mean("ensemble") - target) ** 2).weighted(w).mean(["lat", "lon"]).mean("forecast")
    )
    ratio = spread_growth / (skill_growth + 1e-12)
    return {"spread": spread_growth, "skill": skill_growth, "ratio": ratio}


def rank_histogram(pred: xr.DataArray, target: xr.DataArray) -> np.ndarray:
    """Pooled, area-weighted Talagrand rank histogram via scores.probability.rank_histogram."""
    if "ensemble" not in pred.dims:
        raise ValueError("Rank histogram requires ensemble predictions.")
    pred = drop_batch(pred)
    target = drop_batch(target)
    w = cos_weights(pred)
    histogram = scores.probability.rank_histogram(pred, target, "ensemble", reduce_dims="all", weights=w)
    return histogram.values


def ensemble_power_spectra_1d(pred: xr.DataArray) -> tuple:
    """Ensemble-only sibling of power_spectra_1d, which targets/baselines still need."""
    if "ensemble" not in pred.dims:
        raise ValueError("Ensemble power spectra requires ensemble dim.")
    spatial_dims = [d for d in ("lat", "lon") if d in pred.dims]
    if spatial_dims:
        w = cos_weights(pred)
        pred = pred.weighted(w).mean(spatial_dims)
    pred = drop_batch(pred)
    n = pred.sizes["hour"]
    hour_vals = pred.hour.values
    dt_hours = float(hour_vals[1] - hour_vals[0]) if len(hour_vals) > 1 else 1.0
    freqs = np.fft.rfftfreq(n, d=dt_hours)
    vals = pred.transpose("ensemble", "forecast", "hour").values.astype(float)
    vals -= vals.mean(axis=-1, keepdims=True)
    psds = (np.abs(np.fft.rfft(vals, axis=-1)) ** 2) / n
    psds = psds.mean(axis=1)
    return 1.0 / freqs[1:], psds[:, 1:].mean(axis=0), psds[:, 1:].std(axis=0)


def spell_length_distribution(hw_mask: xr.DataArray) -> dict:
    """Spell lengths pooled over (forecast x lat x lon). Report KS statistic only;
    pooled pixels are spatially correlated so the p-value is optimistic."""
    non_day_dims = [d for d in hw_mask.dims if d != "day_idx"]
    if non_day_dims:
        arr = hw_mask.stack(pixel=non_day_dims).transpose("pixel", "day_idx").values.astype(np.int8)
    else:
        arr = hw_mask.values[np.newaxis, :].astype(np.int8)
    N, D = arr.shape
    padded = np.concatenate([np.zeros((N, 1), np.int8), arr, np.zeros((N, 1), np.int8)], axis=1)
    changes = np.diff(padded.astype(np.int16), axis=1)
    _, rise_day = np.where(changes == 1)
    _, fall_day = np.where(changes == -1)
    lengths = fall_day - rise_day
    if len(lengths) == 0:
        return {"mean_length": 0.0, "max_length": 0, "lengths": lengths}
    return {"mean_length": float(lengths.mean()), "max_length": int(lengths.max()), "lengths": lengths}


def heatwave_ks_statistic(pred_lengths: np.ndarray, tgt_lengths: np.ndarray) -> float:
    if len(pred_lengths) == 0 or len(tgt_lengths) == 0:
        return 0.0
    return float(stats.ks_2samp(pred_lengths, tgt_lengths).statistic)


def mean_temporal_difference(pred: xr.DataArray, target: xr.DataArray) -> dict:
    pred = drop_batch(pred)
    target = drop_batch(target)
    w = cos_weights(pred)
    if "ensemble" in pred.dims:
        pred_abs_diff = abs(pred.diff("hour")).mean("ensemble")
    else:
        pred_abs_diff = abs(pred.diff("hour"))
    tgt_abs_diff = abs(target.diff("hour"))
    pred_curve = pred_abs_diff.weighted(w).mean(["lat", "lon"]).mean("forecast")
    tgt_curve = tgt_abs_diff.weighted(w).mean(["lat", "lon"]).mean("forecast")
    return {"pred": pred_curve, "target": tgt_curve}


def _day_blocked(da: xr.DataArray) -> xr.DataArray:
    """Assign day_idx and keep only full 24-h day blocks; shared by every day-block metric."""
    assert "forecast" not in da["hour"].dims, (
        "hour must not depend on forecast; day-blocking assumes one shared lead-relative hour axis.")
    hv = da.hour.values
    labels = _full_day_idx(hv)
    da = da.assign_coords(day_idx=("hour", labels))
    return da.where(da.day_idx >= 0, drop=True)


def tmax_grid(dataset: xr.Dataset) -> xr.DataArray:
    """Daily max of 2m_temperature over FULL 24-h day blocks only."""
    return _day_blocked(drop_batch(dataset[TARGET_VAR])).groupby("day_idx").max()


def diurnal_shape_errors(pred: xr.DataArray, target: xr.DataArray, min_dtr: float = 1.0) -> dict:
    """DTR (daily max-min) and asymmetry ((mean-min)/(max-min)) shape, MAE diagnostic
    + proper CRPS score each, plus every raw per-day value, from one day-block pass."""
    pred = drop_batch(pred)
    target = drop_batch(target)
    pred_day_blocked = _day_blocked(pred)
    target_day_blocked = _day_blocked(target)
    pred_day_groups = pred_day_blocked.groupby("day_idx")
    target_day_groups = target_day_blocked.groupby("day_idx")
    pred_max, pred_min, pred_mean = pred_day_groups.max(), pred_day_groups.min(), pred_day_groups.mean()
    tgt_max, tgt_min, tgt_mean = target_day_groups.max(), target_day_groups.min(), target_day_groups.mean()
    pred_dtr = pred_max - pred_min
    tgt_dtr = tgt_max - tgt_min
    pred_asymmetry = ((pred_mean - pred_min) / pred_dtr).where(tgt_dtr >= min_dtr)
    tgt_asymmetry = ((tgt_mean - tgt_min) / tgt_dtr).where(tgt_dtr >= min_dtr)
    w = cos_weights(target)

    abs_dtr_diff = abs(pred_dtr - tgt_dtr)
    if "ensemble" in abs_dtr_diff.dims:
        abs_dtr_diff = abs_dtr_diff.mean("ensemble")
    dtr_curve = abs_dtr_diff.weighted(w).mean(["lat", "lon"]).mean("forecast")

    asymmetry_diff = pred_asymmetry - tgt_asymmetry
    if "ensemble" in asymmetry_diff.dims:
        asymmetry_diff = asymmetry_diff.mean("ensemble")
    asymmetry_curve = asymmetry_diff.weighted(w).mean(["lat", "lon"]).mean("forecast")

    dtr_crps_field = crps_field(pred_dtr, tgt_dtr)
    dtr_crps_map = dtr_crps_field.mean(["forecast", "day_idx"])
    dtr_crps_series_by_day = dtr_crps_field.weighted(w).mean(["lat", "lon"])
    dtr_crps_series = dtr_crps_series_by_day.mean("day_idx")

    asymmetry_crps_field = crps_field(pred_asymmetry, tgt_asymmetry)
    asymmetry_crps_map = asymmetry_crps_field.mean(["forecast", "day_idx"])
    asymmetry_crps_series_by_day = asymmetry_crps_field.weighted(w).mean(["lat", "lon"])
    asymmetry_crps_series = asymmetry_crps_series_by_day.mean("day_idx")

    return {
        "pred_max": pred_max, "pred_min": pred_min, "pred_mean": pred_mean,
        "tgt_max": tgt_max, "tgt_min": tgt_min, "tgt_mean": tgt_mean,
        "pred_dtr": pred_dtr, "tgt_dtr": tgt_dtr,
        "pred_asymmetry": pred_asymmetry, "tgt_asymmetry": tgt_asymmetry,
        "dtr_curve": dtr_curve, "dtr_scalar": float(dtr_curve.mean()),
        "dtr_crps_map": dtr_crps_map, "dtr_crps_series": dtr_crps_series,
        "dtr_crps_series_by_day": dtr_crps_series_by_day,
        "dtr_crps_scalar": float(dtr_crps_map.weighted(w).mean(["lat", "lon"])),
        "asymmetry_curve": asymmetry_curve,
        "asymmetry_map": asymmetry_diff.mean("forecast").mean("day_idx"),
        "asymmetry_scalar": float(abs(asymmetry_diff).weighted(w).mean(["lat", "lon"]).mean()),
        "asymmetry_crps_map": asymmetry_crps_map, "asymmetry_crps_series": asymmetry_crps_series,
        "asymmetry_crps_series_by_day": asymmetry_crps_series_by_day,
        "asymmetry_crps_scalar": float(asymmetry_crps_map.weighted(w).mean(["lat", "lon"])),
    }


def rmse_by_window_position(pred, target, ar_window_hours: int) -> xr.DataArray:
    pred = drop_batch(pred)
    target = drop_batch(target)
    pred_mean = pred.mean("ensemble") if "ensemble" in pred.dims else pred
    diff = pred_mean - target
    w = cos_weights(pred_mean)
    rmse_per_hour = np.sqrt((diff ** 2).weighted(w).mean(["lat", "lon"]).mean("forecast"))
    total_hours = rmse_per_hour.sizes["hour"]
    positions = np.arange(total_hours) % ar_window_hours
    position_rmse = np.zeros(ar_window_hours)
    counts = np.zeros(ar_window_hours)
    for hour_idx in range(total_hours):
        pos = positions[hour_idx]
        position_rmse[pos] += float(rmse_per_hour.isel(hour=hour_idx).values)
        counts[pos] += 1
    counts = np.where(counts == 0, 1, counts)
    return xr.DataArray(position_rmse / counts, dims=["window_pos"],
                        coords={"window_pos": np.arange(ar_window_hours)})


def variogram_score(pred: xr.DataArray, target: xr.DataArray, order: float = 0.5) -> dict:
    """Per-day temporal variogram score via scoringrules.vs_ensemble on the within-day 24h vector, w_ij = 1/|hour separation| normalised to sum 1."""
    pred = drop_batch(pred)
    target = drop_batch(target)
    pred_blocked = _day_blocked(pred)
    tgt_blocked = _day_blocked(target)
    has_ensemble = "ensemble" in pred_blocked.dims

    # reshape the hour axis into (day_idx, hour_of_day) so day_idx is a real dim and VS runs per day
    hour_of_day = ((tgt_blocked.hour.values - tgt_blocked.hour.values[0]) % 24).astype(int)
    pred_blocked = pred_blocked.assign_coords(hour_of_day=("hour", hour_of_day)).set_index(hour=["day_idx", "hour_of_day"]).unstack("hour")
    tgt_blocked = tgt_blocked.assign_coords(hour_of_day=("hour", hour_of_day)).set_index(hour=["day_idx", "hour_of_day"]).unstack("hour")

    # within-day pairwise weight matrix (24 x 24)
    hours = tgt_blocked.hour_of_day.values
    pair_distance = np.abs(hours[:, None] - hours[None, :])
    pair_weight = np.where(pair_distance > 0, 1.0 / np.maximum(pair_distance, 1), 0.0)
    pair_weight = pair_weight / pair_weight.sum()

    batch_dims = [d for d in ("forecast", "day_idx", "lat", "lon") if d in tgt_blocked.dims]
    tgt_values = tgt_blocked.transpose(*batch_dims, "hour_of_day").values
    if has_ensemble:
        pred_values = pred_blocked.transpose(*batch_dims, "ensemble", "hour_of_day").values
    else:
        pred_values = pred_blocked.transpose(*batch_dims, "hour_of_day").values[..., np.newaxis, :]
    score_values = sr.vs_ensemble(tgt_values, pred_values, w=pair_weight, p=order)
    field = xr.DataArray(score_values, dims=batch_dims,
                         coords={d: tgt_blocked[d] for d in batch_dims})
    w = cos_weights(target)
    score_map = field.mean(["forecast", "day_idx"])
    series_by_day_idx = field.weighted(w).mean(["lat", "lon"])
    series = series_by_day_idx.mean("day_idx")
    return {"map": score_map, "series": series, "series_by_day_idx": series_by_day_idx,
            "scalar": float(score_map.weighted(w).mean(["lat", "lon"]))}


def spatial_variogram_score(pred: xr.DataArray, target: xr.DataArray, order: float = 0.5,
                            max_distance_deg: float = 5.0, normalize: bool = True) -> dict:
    """Spatial variogram score of order p over grid-cell pairs, one value per
    (forecast, hour). Scores the SPATIAL dependence structure — a field that is
    too smooth (over-regularised, the failure a reduced variable set risks)
    fails this even when it passes CRPS. That is why the variable-selection test
    requires passing on both twCRPS and this.

    Computed the efficient variogram way — loop over spatial lag vectors
    (delta_lat, delta_lon) within `max_distance_deg`, shifting the field rather
    than enumerating all O(cells^2) pairs. Weight 1 / great-circle-ish degree
    distance, normalised to sum 1 (mandatory for cross-domain comparison).
    Returns a per-forecast series (area over hour+lag) for the block bootstrap.
    """
    pred = drop_batch(pred)
    target = drop_batch(target)
    pred_mean = pred.mean("ensemble") if "ensemble" in pred.dims else pred
    pred_mean = pred_mean.transpose("forecast", "hour", "lat", "lon")
    target = target.transpose("forecast", "hour", "lat", "lon")
    latitudes = target.lat.values
    longitudes = target.lon.values
    lat_step = float(np.abs(latitudes[1] - latitudes[0])) if len(latitudes) > 1 else 0.25
    lon_step = float(np.abs(longitudes[1] - longitudes[0])) if len(longitudes) > 1 else 0.25
    max_lat_shift = int(round(max_distance_deg / lat_step))
    max_lon_shift = int(round(max_distance_deg / lon_step))

    pred_values = pred_mean.values
    tgt_values = target.values
    num_lat, num_lon = tgt_values.shape[-2], tgt_values.shape[-1]
    accumulated = np.zeros(pred_values.shape[:2])  # (forecast, hour)
    weight_total = 0.0
    for lat_shift in range(-max_lat_shift, max_lat_shift + 1):
        for lon_shift in range(0, max_lon_shift + 1):
            if lon_shift == 0 and lat_shift <= 0:
                continue  # skip (0,0) and the mirror of the +lat / same-lon pairs
            if abs(lat_shift) >= num_lat or lon_shift >= num_lon:
                continue  # no overlapping cells at this lag (grid smaller than the shift)
            distance = np.hypot(lat_shift * lat_step, lon_shift * lon_step)
            if distance > max_distance_deg:
                continue
            pair_weight = 1.0 / distance
            weight_total += pair_weight
            # Overlapping cell (i, j) vs (i+lat_shift, j+lon_shift), valid indices only.
            here_lat = slice(max(0, -lat_shift), num_lat - max(0, lat_shift))
            shift_lat = slice(max(0, lat_shift), num_lat - max(0, -lat_shift))
            here_lon, shift_lon = slice(0, num_lon - lon_shift), slice(lon_shift, num_lon)
            tgt_term = np.abs(tgt_values[..., here_lat, here_lon]
                              - tgt_values[..., shift_lat, shift_lon]) ** order
            pred_term = np.abs(pred_values[..., here_lat, here_lon]
                               - pred_values[..., shift_lat, shift_lon]) ** order
            accumulated += pair_weight * ((tgt_term - pred_term) ** 2).mean(axis=(-2, -1))
    if normalize and weight_total > 0:
        accumulated /= weight_total
    field = xr.DataArray(accumulated, dims=["forecast", "hour"],
                         coords={"forecast": target.forecast, "hour": target.hour})
    series = field.mean("hour")
    return {"series": series, "scalar": float(series.mean())}


def peak_time_ccrps(pred: xr.DataArray, target: xr.DataArray) -> dict:
    """Peak-time error as a PROPER, rankable score: the argmax-hour transform
    per member, then a CIRCULAR fair CRPS with the angular distance
    alpha(a, b) = min(|a-b|, 24-|a-b|) as the (negative-definite) kernel.

    cCRPS = mean_k alpha(theta_k, obs) - 1/(2M(M-1)) sum_{k!=l} alpha(theta_k, theta_l),
    bounded by 12 h, proper because angular distance is a negative-definite
    kernel (Gneiting 1998). Replaces the old argmax-of-the-ensemble-mean MAE,
    which could not rank a probabilistic peak. Deterministic input scores the
    single trajectory's angular distance to the observed peak."""
    pred = drop_batch(pred)
    target = drop_batch(target)
    pred_peak = _day_blocked(pred).groupby("day_idx").map(
        lambda block: block.hour.isel(hour=block.argmax("hour")).drop_vars(["hour", "day_idx"], errors="ignore"))
    tgt_peak = _day_blocked(target).groupby("day_idx").map(
        lambda block: block.hour.isel(hour=block.argmax("hour")).drop_vars(["hour", "day_idx"], errors="ignore"))
    tgt_peak_hour = (tgt_peak % 24)
    if "ensemble" in pred_peak.dims:
        pred_hours = (pred_peak % 24).transpose("ensemble", ...)
        member_axis = 0
        angular_to_truth = np.minimum(np.abs(pred_hours - tgt_peak_hour),
                                      24 - np.abs(pred_hours - tgt_peak_hour))
        term_truth = angular_to_truth.mean("ensemble")
        member_values = pred_hours.values
        num_members = member_values.shape[member_axis]
        pairwise = np.zeros(term_truth.shape)
        for first_member in range(num_members):
            for second_member in range(num_members):
                if first_member == second_member:
                    continue
                difference = np.abs(member_values[first_member] - member_values[second_member])
                pairwise += np.minimum(difference, 24 - difference)
        spread = xr.DataArray(pairwise / (num_members * (num_members - 1)),
                              dims=term_truth.dims, coords=term_truth.coords)
        field = term_truth - 0.5 * spread
    else:
        pred_hour = pred_peak % 24
        field = np.minimum(np.abs(pred_hour - tgt_peak_hour), 24 - np.abs(pred_hour - tgt_peak_hour))
    w = cos_weights(target)
    score_map = field.mean(["forecast", "day_idx"])
    series = field.weighted(w).mean(["lat", "lon", "day_idx"])
    return {"map": score_map, "series": series, "scalar": float(score_map.weighted(w).mean(["lat", "lon"]))}


def event_bias(pred_hw: xr.DataArray, tgt_hw: xr.DataArray) -> dict:
    """Metrics 16-17: event frequency bias N_pred/N_obs over cells with
    N_obs >= 1 (above 1 too many events, catches fragmentation) and event
    duration bias, mean event length difference in days (negative truncates
    or fragments). Events are maximal flagged runs; the >= 3-day sieve is
    already applied by get_heatwaves."""
    w = cos_weights(tgt_hw)
    padded_pred = pred_hw.astype(np.int8).pad(day_idx=(1, 0), constant_values=0)
    padded_tgt = tgt_hw.astype(np.int8).pad(day_idx=(1, 0), constant_values=0)
    pred_events = (padded_pred.diff("day_idx") == 1).sum(["day_idx", "forecast"])
    tgt_events = (padded_tgt.diff("day_idx") == 1).sum(["day_idx", "forecast"])
    frequency_bias_map = (pred_events / tgt_events).where(tgt_events >= 1)
    pred_mean_length = (pred_hw.sum(["day_idx", "forecast"]) / pred_events).where(pred_events >= 1)
    tgt_mean_length = (tgt_hw.sum(["day_idx", "forecast"]) / tgt_events).where(tgt_events >= 1)
    duration_bias_map = pred_mean_length - tgt_mean_length
    return {
        "frequency_bias_map": frequency_bias_map,
        "frequency_bias": float(frequency_bias_map.weighted(w).mean(["lat", "lon"])),
        "duration_bias_map": duration_bias_map,
        "duration_bias": float(duration_bias_map.weighted(w).mean(["lat", "lon"])),
    }


def event_detection_rate(pred_hw: xr.DataArray, tgt_hw: xr.DataArray) -> dict:
    """Fraction of true sieved-heatwave EVENTS the model flags on at least one day, per cell and pooled."""
    non_day_dims = [d for d in tgt_hw.dims if d != "day_idx"]
    stacked_tgt = tgt_hw.stack(pixel=non_day_dims).transpose("pixel", "day_idx")
    pixel_tgt = stacked_tgt.values.astype(np.int8)
    pixel_pred = pred_hw.stack(pixel=non_day_dims).transpose("pixel", "day_idx").values.astype(np.int8)
    num_pixels = pixel_tgt.shape[0]
    padded = np.concatenate([np.zeros((num_pixels, 1), np.int8), pixel_tgt, np.zeros((num_pixels, 1), np.int8)], axis=1)
    changes = np.diff(padded.astype(np.int16), axis=1)
    rise_pixel, rise_day = np.where(changes == 1)
    _, fall_day = np.where(changes == -1)
    hits = np.array([bool(pixel_pred[pixel, start:end].any())
                     for pixel, start, end in zip(rise_pixel, rise_day, fall_day)])
    events_per_pixel = np.bincount(rise_pixel, minlength=num_pixels).astype(float)
    hits_per_pixel = np.bincount(rise_pixel, weights=hits, minlength=num_pixels)
    rate_per_pixel = np.divide(hits_per_pixel, events_per_pixel,
                               out=np.full(num_pixels, np.nan), where=events_per_pixel > 0)
    rate_map = xr.DataArray(rate_per_pixel, dims=["pixel"], coords={"pixel": stacked_tgt.pixel}).unstack("pixel")
    if "forecast" in rate_map.dims:
        rate_map = rate_map.mean("forecast")  # per-cell rate averaged over forecasts
    return {"map": rate_map, "rate": float(hits.mean()) if len(hits) else float("nan"), "num_events": int(len(hits))}


def conditional_tmax_errors(pred_tmax: xr.DataArray, tgt_tmax: xr.DataArray, tgt_hw: xr.DataArray) -> dict:
    """Metric 18: Tmax bias and RMSE only over day-cell pairs where the
    OBSERVED heatwave flag is true (conditioning on observation, not
    prediction). A conditional cold bias explains recall deficits.
    bias_series (forecast,) feeds the block bootstrap."""
    diff = (pred_tmax - tgt_tmax).where(tgt_hw)
    w = cos_weights(tgt_tmax)
    bias = float(diff.weighted(w).mean(["lat", "lon", "day_idx", "forecast"]))
    rmse = float(np.sqrt((diff ** 2).weighted(w).mean(["lat", "lon", "day_idx", "forecast"])))
    bias_series = diff.weighted(w).mean(["lat", "lon", "day_idx"])
    return {"bias": bias, "rmse": rmse, "bias_map": diff.mean(["forecast", "day_idx"]), "bias_series": bias_series}


def tmax_quantile_bias(pred_tmax: xr.DataArray, tgt_tmax: xr.DataArray, quantile_levels=(0.90, 0.95, 0.99)) -> dict:
    """Metric 19: per-cell quantiles of pooled per-member daily Tmax minus the
    observed quantiles. Negative means the upper tail is too cold — the
    regression-to-the-mean signature. Pools members; quantiles of the ensemble
    mean would smooth the tail."""
    pred_pool_dims = [d for d in ("forecast", "day_idx", "ensemble") if d in pred_tmax.dims]
    tgt_pool_dims = [d for d in ("forecast", "day_idx") if d in tgt_tmax.dims]
    pred_quantiles = pred_tmax.quantile(list(quantile_levels), dim=pred_pool_dims)
    tgt_quantiles = tgt_tmax.quantile(list(quantile_levels), dim=tgt_pool_dims)
    delta_map = pred_quantiles - tgt_quantiles
    w = cos_weights(tgt_tmax)
    scalars = {
        f"q{int(round(level * 100))}": float(delta_map.sel(quantile=level).weighted(w).mean(["lat", "lon"]))
        for level in quantile_levels
    }
    return {"map": delta_map, "scalars": scalars}


def tail_ratio(pred_tmax: xr.DataArray, tgt_tmax: xr.DataArray,
               low: float = 0.90, mid: float = 0.95, high: float = 0.99) -> dict:
    """Tail THICKNESS, independent of bias: R = (Q99 - Q90)_model / (Q99 - Q90)_obs
    per cell, from pooled daily Tmax. R < 1 = upper tail too compressed (squashed
    extremes), R > 1 = too stretched. A uniformly cold model can still have R = 1;
    quantile bias gives the tail's LOCATION, this gives its width. The two sub-gaps
    (Q95-Q90, Q99-Q95) show where the compression sits."""
    pred_pool = [d for d in ("forecast", "day_idx", "ensemble") if d in pred_tmax.dims]
    tgt_pool = [d for d in ("forecast", "day_idx") if d in tgt_tmax.dims]
    pred_quantiles = pred_tmax.quantile([low, mid, high], dim=pred_pool)
    tgt_quantiles = tgt_tmax.quantile([low, mid, high], dim=tgt_pool)
    pred_gap = pred_quantiles.sel(quantile=high) - pred_quantiles.sel(quantile=low)
    tgt_gap = tgt_quantiles.sel(quantile=high) - tgt_quantiles.sel(quantile=low)
    ratio_map = (pred_gap / tgt_gap).where(tgt_gap > 1e-6)
    w = cos_weights(tgt_tmax)
    lower_pred = pred_quantiles.sel(quantile=mid) - pred_quantiles.sel(quantile=low)
    upper_pred = pred_quantiles.sel(quantile=high) - pred_quantiles.sel(quantile=mid)
    lower_tgt = tgt_quantiles.sel(quantile=mid) - tgt_quantiles.sel(quantile=low)
    upper_tgt = tgt_quantiles.sel(quantile=high) - tgt_quantiles.sel(quantile=mid)
    return {
        "map": ratio_map,
        "ratio": float(ratio_map.weighted(w).mean(["lat", "lon"])),
        "lower_gap_ratio": float((lower_pred / lower_tgt).where(lower_tgt > 1e-6).weighted(w).mean(["lat", "lon"])),
        "upper_gap_ratio": float((upper_pred / upper_tgt).where(upper_tgt > 1e-6).weighted(w).mean(["lat", "lon"])),
    }


def conditional_rank_histogram(pred_tmax: xr.DataArray, tgt_tmax: xr.DataArray,
                               threshold: xr.DataArray) -> np.ndarray:
    """Tail CALIBRATION: the rank of the observation among {obs, members},
    restricted to cases where the observation exceeds tau. M+1 bins. Reading:
    flat = extreme observations are plausible extreme members (calibrated);
    rising toward high ranks = observations more extreme than members (tails too
    thin); falling toward low ranks = members too extreme (tails too thick). This
    is the diagnostic that discharges the Brehmer-Strokorb obligation — no proper
    score can assess tail calibration. Coarse at M=8; pool over cells for counts."""
    if "ensemble" not in pred_tmax.dims:
        raise ValueError("Conditional rank histogram requires ensemble predictions.")
    num_members = pred_tmax.sizes["ensemble"]
    below = (pred_tmax < tgt_tmax).sum("ensemble")
    tied = (pred_tmax == tgt_tmax).sum("ensemble")
    exceeds = tgt_tmax > threshold
    below, tied, exceeds = xr.broadcast(below, tied, exceeds)
    rng = np.random.default_rng(42)
    ranks = below.values + rng.integers(0, tied.values + 1)
    tail_ranks = ranks[np.asarray(exceeds.values) & np.isfinite(ranks)]
    if tail_ranks.size == 0:
        return np.zeros(num_members + 1)
    histogram = np.bincount(tail_ranks.astype(int).ravel(), minlength=num_members + 1)
    return histogram / histogram.sum()


def probabilistic_detection(pred_tmax: xr.DataArray, tgt_tmax: xr.DataArray, threshold: xr.DataArray,
                            climatology_probability: xr.DataArray = None) -> dict:
    """Probabilistic heatwave detection. p_hat = fraction of members with Tmax > tau
    per cell-day; o = 1 if observed Tmax > tau. Brier score (p_hat - o)^2, Brier
    skill score vs climatology (BSS = 1 - BS_model / BS_climatology; climatology's
    p_hat is ~0.1 by construction of the p90 threshold, or the observed base rate
    if none is given). Reliability: with M members p_hat takes M+1 discrete values,
    used directly as the bins (no binning choice). AUC via the rank-sum identity,
    invariant to monotone recalibration so reported alongside reliability, not
    instead of it."""
    if "ensemble" not in pred_tmax.dims:
        raise ValueError("Probabilistic detection requires ensemble predictions.")
    num_members = pred_tmax.sizes["ensemble"]
    forecast_probability = (pred_tmax > threshold).mean("ensemble")
    observation = (tgt_tmax > threshold).astype(float)
    forecast_probability, observation = xr.broadcast(forecast_probability, observation)
    w = cos_weights(tgt_tmax)
    brier_map = ((forecast_probability - observation) ** 2).mean([d for d in ("forecast", "day_idx")
                                                                  if d in forecast_probability.dims])
    brier = float(((forecast_probability - observation) ** 2).weighted(w).mean(list(forecast_probability.dims)))
    base_rate = float(observation.weighted(w).mean(list(observation.dims)))
    reference = climatology_probability if climatology_probability is not None else base_rate
    brier_reference = float(((reference - observation) ** 2).weighted(w).mean(list(observation.dims)))
    brier_skill_score = 1.0 - brier / brier_reference if brier_reference > 0 else 0.0

    probability_values = forecast_probability.values.ravel()
    observation_values = observation.values.ravel()
    finite = np.isfinite(probability_values) & np.isfinite(observation_values)
    probability_values, observation_values = probability_values[finite], observation_values[finite]
    bin_index = np.rint(probability_values * num_members).astype(int)
    reliability_forecast = np.arange(num_members + 1) / num_members
    reliability_observed = np.full(num_members + 1, np.nan)
    reliability_count = np.zeros(num_members + 1)
    for bin_value in range(num_members + 1):
        in_bin = bin_index == bin_value
        reliability_count[bin_value] = in_bin.sum()
        if in_bin.any():
            reliability_observed[bin_value] = observation_values[in_bin].mean()

    # AUC via the rank-sum identity is exactly sklearn's roc_auc_score; needs both classes present.
    has_both_classes = 0 < observation_values.sum() < observation_values.size
    auc = float(roc_auc_score(observation_values, probability_values)) if has_both_classes else float("nan")
    return {
        "brier_map": brier_map, "brier": brier, "brier_skill_score": brier_skill_score,
        "base_rate": base_rate, "auc": float(auc),
        "reliability_forecast": reliability_forecast, "reliability_observed": reliability_observed,
        "reliability_count": reliability_count,
    }


def over_smoothing_diagnostic(pred: xr.DataArray, target: xr.DataArray) -> dict:
    """Curvature check: mean |X_{h+1} - 2 X_h + X_{h-1}| (the discrete second
    difference along hour) for model vs ERA5, by lead hour. The model curve below
    ERA5's = the diurnal curve is too smooth — the power-variation signature of an
    over-regularised reconstruction. Read as two raw curves, not a single score."""
    pred = drop_batch(pred)
    target = drop_batch(target)
    pred_mean = pred.mean("ensemble") if "ensemble" in pred.dims else pred
    second_pred = np.abs(pred_mean.diff("hour", n=2))
    second_tgt = np.abs(target.diff("hour", n=2))
    w = cos_weights(target)
    return {
        "pred_curve": second_pred.weighted(w).mean(["lat", "lon"]).mean("forecast"),
        "tgt_curve": second_tgt.weighted(w).mean(["lat", "lon"]).mean("forecast"),
    }


