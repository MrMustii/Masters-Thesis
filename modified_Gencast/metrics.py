import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats

TARGET_VAR = "2m_temperature"


def tmax_grid(dataset: xr.Dataset) -> xr.DataArray:
    """Daily max of 2m_temperature over FULL 24-h day blocks only."""

    data_array = dataset[TARGET_VAR]
    if "batch" in data_array.dims and data_array.sizes["batch"] == 1:
        data_array = data_array.squeeze("batch", drop=True)

    # day-blocking assumes one shared lead-relative hour axis, identical
    # across every forecast; if that ever broke, every day would be
    # mis-blocked for every forecast but the first.
    assert "forecast" not in data_array["hour"].dims, (
        "hour must not depend on forecast; day-blocking assumes one shared lead-relative hour axis.")
    hour_values = data_array.hour.values
    day_idx_values = (hour_values - hour_values[0]) // 24
    data_array = data_array.assign_coords(day_idx=("hour", day_idx_values))
    # count hours per day_idx; keep only blocks with a full 24
    
    counts = xr.DataArray(
        np.ones_like(day_idx_values), dims=["hour"],
        coords={"hour": data_array.hour}
    ).assign_coords(day_idx=("hour", day_idx_values)).groupby("day_idx").sum()
    full_days = counts.where(counts == 24, drop=True).day_idx.values
    tmax = data_array.groupby("day_idx").max()
    return tmax.sel(day_idx=full_days)


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

    For ensemble: errors on ensemble mean. rmse/bias curves are cos-weighted;
    mse_series is the per-initialization spatially averaged MSE used by the
    block bootstrap.
    """
    if "batch" in pred.dims and pred.sizes["batch"] == 1:
        pred = pred.squeeze("batch", drop=True)
    if "batch" in target.dims and target.sizes["batch"] == 1:
        target = target.squeeze("batch", drop=True)
    is_ensemble = "ensemble" in pred.dims
    pred_mean = pred.mean("ensemble") if is_ensemble else pred
    diff = pred_mean - target
    weights = np.cos(np.deg2rad(pred_mean.lat))
    w = xr.DataArray(weights, coords={"lat": pred_mean.lat})
    rmse_map = np.sqrt((diff ** 2).mean("forecast"))
    bias_map = diff.mean("forecast")
    return {
        "rmse_map": rmse_map,
        "bias_map": bias_map,
        "rmse": np.sqrt((diff ** 2).weighted(w).mean(["lat", "lon"]).mean("forecast")),
        "bias": bias_map.weighted(w).mean(["lat", "lon"]),
        "mse_series": (diff ** 2).weighted(w).mean(["lat", "lon"]).mean("hour"),
    }


def binary_classification_metrics(pred_hw: xr.DataArray, tgt_hw: xr.DataArray) -> dict:
    """Precision, Recall, F1 as curves (day_idx,) and maps (lat, lon), plus
    per-initialization confusion-count series (forecast,) for the block
    bootstrap (F1 must be recomputed from resampled summed counts).

    Takes boolean output of get_heatwaves(). pred_hw must be deterministic.
    """
    weights = np.cos(np.deg2rad(tgt_hw.lat))
    w = xr.DataArray(weights, coords={"lat": tgt_hw.lat})

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
    precision_map = (tp_s / (tp_s + fp_s)).fillna(0)
    recall_map = (tp_s / (tp_s + fn_s)).fillna(0)
    f1_map = (2 * precision_map * recall_map / (precision_map + recall_map + 1e-12))

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
    """Relative Economic Value over C/L in [0.01, 0.99]. Needs ensemble dim."""
    if "ensemble" not in pred_tmax.dims:
        raise ValueError("REV requires ensemble predictions.")
    M = pred_tmax.sizes["ensemble"]
    obs = (tgt_tmax > threshold).astype(float)
    prob = (pred_tmax > threshold).astype(float).mean("ensemble")
    weights = np.cos(np.deg2rad(tgt_tmax.lat))
    w = xr.DataArray(weights, coords={"lat": tgt_tmax.lat})
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
        weights = np.cos(np.deg2rad(data_array.lat))
        w = xr.DataArray(weights, coords={"lat": data_array.lat})
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


def crps(pred: xr.DataArray, target: xr.DataArray) -> dict:
    if "ensemble" not in pred.dims:
        raise ValueError("CRPS requires ensemble predictions.")
    if "batch" in pred.dims and pred.sizes["batch"] == 1:
        pred = pred.squeeze("batch", drop=True)
    if "batch" in target.dims and target.sizes["batch"] == 1:
        target = target.squeeze("batch", drop=True)
    mae_term = abs(pred - target).mean("ensemble")
    M = pred.sizes["ensemble"]
    ensemble_axis = list(pred.dims).index("ensemble")
    sorted_vals = np.sort(pred.values, axis=ensemble_axis)
    pairwise_w = (2 * np.arange(M) + 1 - M).astype(np.float32)
    pairwise_w = pairwise_w.reshape([M if i == ensemble_axis else 1 for i in range(pred.values.ndim)])
    energy_np = (2.0 / (M * (M - 1))) * (pairwise_w * sorted_vals).sum(axis=ensemble_axis)
    non_ensemble_dims = [d for d in pred.dims if d != "ensemble"]
    non_ensemble_coords = {d: pred.coords[d] for d in non_ensemble_dims if d in pred.coords}
    energy = xr.DataArray(energy_np, dims=non_ensemble_dims, coords=non_ensemble_coords)
    crps_field = mae_term - 0.5 * energy
    crps_map = crps_field.mean("forecast")
    weights = np.cos(np.deg2rad(pred.lat))
    w = xr.DataArray(weights, coords={"lat": pred.lat})
    crps_growth = crps_field.weighted(w).mean(["lat", "lon"]).mean("forecast")
    crps_series = crps_field.weighted(w).mean(["lat", "lon"]).mean("hour")
    return {"crps_map": crps_map, "crps": crps_growth, "crps_series": crps_series}


def spread_skill_ratio(pred: xr.DataArray, target: xr.DataArray) -> dict:
    if "ensemble" not in pred.dims:
        raise ValueError("Spread-skill ratio requires ensemble predictions.")
    if "batch" in pred.dims and pred.sizes["batch"] == 1:
        pred = pred.squeeze("batch", drop=True)
    if "batch" in target.dims and target.sizes["batch"] == 1:
        target = target.squeeze("batch", drop=True)
    M = pred.sizes["ensemble"]
    weights = np.cos(np.deg2rad(pred.lat))
    w = xr.DataArray(weights, coords={"lat": pred.lat})
    spread_growth = np.sqrt(
        pred.var("ensemble").weighted(w).mean(["lat", "lon"]).mean("forecast")
    ) * np.sqrt((M + 1) / (M - 1))
    skill_growth = np.sqrt(
        ((pred.mean("ensemble") - target) ** 2).weighted(w).mean(["lat", "lon"]).mean("forecast")
    )
    ratio = spread_growth / (skill_growth + 1e-12)
    return {"spread": spread_growth, "skill": skill_growth, "ratio": ratio}


def rank_histogram(pred: xr.DataArray, target: xr.DataArray) -> np.ndarray:
    if "ensemble" not in pred.dims:
        raise ValueError("Rank histogram requires ensemble predictions.")
    if "batch" in pred.dims and pred.sizes["batch"] == 1:
        pred = pred.squeeze("batch", drop=True)
    if "batch" in target.dims and target.sizes["batch"] == 1:
        target = target.squeeze("batch", drop=True)
    N = pred.sizes["ensemble"]
    hist = np.zeros(N + 1, dtype=np.float64)
    rng = np.random.default_rng(42)
    dims_without_lat = [d for d in pred.dims if d != "lat"]
    ens_axis = dims_without_lat.index("ensemble")
    for i in range(pred.sizes["lat"]):
        p = np.moveaxis(pred.isel(lat=i).values, ens_axis, 0)
        t = target.isel(lat=i).values
        strictly_below = (p < t[np.newaxis]).sum(axis=0)
        tied = (p == t[np.newaxis]).sum(axis=0)
        ranks = (strictly_below + rng.integers(0, tied + 1, size=strictly_below.shape)).ravel()
        hist += np.bincount(ranks.astype(int), minlength=N + 1)
    return hist / hist.sum()


def ensemble_power_spectra_1d(pred: xr.DataArray) -> tuple:
    if "ensemble" not in pred.dims:
        raise ValueError("Ensemble power spectra requires ensemble dim.")
    spatial_dims = [d for d in ("lat", "lon") if d in pred.dims]
    if spatial_dims:
        weights = np.cos(np.deg2rad(pred.lat))
        w = xr.DataArray(weights, coords={"lat": pred.lat})
        pred = pred.weighted(w).mean(spatial_dims)
    if "batch" in pred.dims and pred.sizes["batch"] == 1:
        pred = pred.squeeze("batch", drop=True)
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
    if "batch" in pred.dims and pred.sizes["batch"] == 1:
        pred = pred.squeeze("batch", drop=True)
    if "batch" in target.dims and target.sizes["batch"] == 1:
        target = target.squeeze("batch", drop=True)
    weights = np.cos(np.deg2rad(pred.lat))
    w = xr.DataArray(weights, coords={"lat": pred.lat})
    if "ensemble" in pred.dims:
        pred_abs_diff = abs(pred.diff("hour")).mean("ensemble")
    else:
        pred_abs_diff = abs(pred.diff("hour"))
    tgt_abs_diff = abs(target.diff("hour"))
    pred_curve = pred_abs_diff.weighted(w).mean(["lat", "lon"]).mean("forecast")
    tgt_curve = tgt_abs_diff.weighted(w).mean(["lat", "lon"]).mean("forecast")
    return {"pred": pred_curve, "target": tgt_curve}


SEASONS = {"DJF": (12, 1, 2), "MAM": (3, 4, 5), "JJA": (6, 7, 8), "SON": (9, 10, 11)}


def seasonal_errors(pred: xr.DataArray, target: xr.DataArray) -> dict:
    """Per-season RMSE curves (hour,) and bias maps (lat, lon); bias is
    reported as maps and by season rather than one scalar."""
    if "batch" in pred.dims and pred.sizes["batch"] == 1:
        pred = pred.squeeze("batch", drop=True)
    if "batch" in target.dims and target.sizes["batch"] == 1:
        target = target.squeeze("batch", drop=True)
    forecast_months = pd.DatetimeIndex(pred.forecast.values).month
    pred_mean = pred.mean("ensemble") if "ensemble" in pred.dims else pred
    weights = np.cos(np.deg2rad(pred.lat))
    w = xr.DataArray(weights, coords={"lat": pred.lat})
    result = {}
    for season, months in SEASONS.items():
        mask = np.isin(forecast_months, months)
        if not mask.any():
            result[season] = None
            continue
        diff = pred_mean.isel(forecast=mask) - target.isel(forecast=mask)
        result[season] = {
            "rmse": np.sqrt((diff ** 2).weighted(w).mean(["lat", "lon"]).mean("forecast")),
            "bias_map": diff.mean(["forecast", "hour"]),
        }
    return result


def _day_blocked(da: xr.DataArray) -> xr.DataArray:
    """Assign day_idx and keep only full 24-h day blocks (drops half days).

    Shared chokepoint for DTR, asymmetry, peak-time, and hours-above-threshold
    — all four assume one shared lead-relative hour axis, identical across
    every forecast (see the assert below)."""
    assert "forecast" not in da["hour"].dims, (
        "hour must not depend on forecast; day-blocking assumes one shared lead-relative hour axis.")
    hv = da.hour.values
    labels = _full_day_idx(hv)
    da = da.assign_coords(day_idx=("hour", labels))
    return da.where(da.day_idx >= 0, drop=True)


def diurnal_shape_errors(pred: xr.DataArray, target: xr.DataArray, min_dtr: float = 1.0) -> dict:
    """Metrics 6 and 8 from one shared set of per-day aggregates.

    DTR error: extremes taken per member, errors averaged afterwards (extremes
    of the ensemble mean are biased low); pred_dtr/tgt_dtr returned for the
    density scatter. Asymmetry index (mean - min)/(max - min): 0.5 symmetric,
    below 0.5 the sharp-afternoon-peak shape; model minus truth, masked where
    DTR < min_dtr since the index degenerates."""
    if "batch" in pred.dims and pred.sizes["batch"] == 1:
        pred = pred.squeeze("batch", drop=True)
    if "batch" in target.dims and target.sizes["batch"] == 1:
        target = target.squeeze("batch", drop=True)
    pred_b = _day_blocked(pred)
    tgt_b = _day_blocked(target)
    pg = pred_b.groupby("day_idx")
    tg = tgt_b.groupby("day_idx")
    pred_max, pred_min, pred_avg = pg.max(), pg.min(), pg.mean()
    tgt_max, tgt_min, tgt_avg = tg.max(), tg.min(), tg.mean()
    pred_dtr = pred_max - pred_min
    tgt_dtr = tgt_max - tgt_min
    weights = np.cos(np.deg2rad(target.lat))
    w = xr.DataArray(weights, coords={"lat": target.lat})

    abs_dtr_diff = abs(pred_dtr - tgt_dtr)
    if "ensemble" in abs_dtr_diff.dims:
        abs_dtr_diff = abs_dtr_diff.mean("ensemble")
    dtr_curve = abs_dtr_diff.weighted(w).mean(["lat", "lon"]).mean("forecast")

    pred_asymmetry = ((pred_avg - pred_min) / pred_dtr).where(pred_dtr >= min_dtr)
    tgt_asymmetry = ((tgt_avg - tgt_min) / tgt_dtr).where(tgt_dtr >= min_dtr)
    asymmetry_diff = pred_asymmetry - tgt_asymmetry
    if "ensemble" in asymmetry_diff.dims:
        asymmetry_diff = asymmetry_diff.mean("ensemble")
    asymmetry_curve = asymmetry_diff.weighted(w).mean(["lat", "lon"]).mean("forecast")
    return {
        "dtr_curve": dtr_curve,
        "dtr_scalar": float(dtr_curve.mean()),
        "pred_dtr": pred_dtr,
        "tgt_dtr": tgt_dtr,
        "asymmetry_curve": asymmetry_curve,
        "asymmetry_map": asymmetry_diff.mean("forecast").mean("day_idx"),
        "asymmetry_scalar": float(abs(asymmetry_diff).weighted(w).mean(["lat", "lon"]).mean()),
    }


def rmse_by_window_position(pred, target, ar_window_hours: int) -> xr.DataArray:
    if "batch" in pred.dims and pred.sizes["batch"] == 1:
        pred = pred.squeeze("batch", drop=True)
    if "batch" in target.dims and target.sizes["batch"] == 1:
        target = target.squeeze("batch", drop=True)
    pred_mean = pred.mean("ensemble") if "ensemble" in pred.dims else pred
    diff = pred_mean - target
    weights = np.cos(np.deg2rad(pred_mean.lat))
    w = xr.DataArray(weights, coords={"lat": pred_mean.lat})
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


def peak_time_error(pred: xr.DataArray, target: xr.DataArray) -> dict:
    """Metric 7: placement of the daily maximum within the day, from the
    ensemble mean, per full day block, with wraparound at midnight. Units
    hours. Also returns the 24x24 joint histogram of predicted vs observed
    peak hour-of-day (UTC) for the heatmap figure."""
    if "batch" in pred.dims and pred.sizes["batch"] == 1:
        pred = pred.squeeze("batch", drop=True)
    if "batch" in target.dims and target.sizes["batch"] == 1:
        target = target.squeeze("batch", drop=True)
    pred_mean = pred.mean("ensemble") if "ensemble" in pred.dims else pred
    pred_b = _day_blocked(pred_mean)
    tgt_b = _day_blocked(target)
    pred_peak_lead = pred_b.groupby("day_idx").map(
        lambda block: block.hour.isel(hour=block.argmax("hour")).drop_vars(["hour", "day_idx"], errors="ignore"))
    tgt_peak_lead = tgt_b.groupby("day_idx").map(
        lambda block: block.hour.isel(hour=block.argmax("hour")).drop_vars(["hour", "day_idx"], errors="ignore"))
    wrapped_delta = ((pred_peak_lead - tgt_peak_lead + 12) % 24) - 12
    weights = np.cos(np.deg2rad(target.lat))
    w = xr.DataArray(weights, coords={"lat": target.lat})
    error_map = abs(wrapped_delta).mean(["forecast", "day_idx"])
    bias_map = wrapped_delta.mean(["forecast", "day_idx"])
    scalar = float(error_map.weighted(w).mean(["lat", "lon"]))
    # Per-initialization series (area + day mean, keep the forecast dim) for the block bootstrap.
    series = abs(wrapped_delta).weighted(w).mean(["lat", "lon", "day_idx"])
    initialization_hours = xr.DataArray(
        pd.DatetimeIndex(pred.forecast.values).hour, dims=["forecast"], coords={"forecast": pred.forecast})
    pred_peak_hour_of_day = (initialization_hours + pred_peak_lead) % 24
    tgt_peak_hour_of_day = (initialization_hours + tgt_peak_lead) % 24
    heatmap_counts, _, _ = np.histogram2d(
        tgt_peak_hour_of_day.values.ravel(), pred_peak_hour_of_day.values.ravel(),
        bins=[np.arange(25), np.arange(25)])
    heatmap = xr.DataArray(
        heatmap_counts, dims=["observed_hour", "predicted_hour"],
        coords={"observed_hour": np.arange(24), "predicted_hour": np.arange(24)})
    return {"map": error_map, "bias_map": bias_map, "scalar": scalar,
            "series": series, "peak_hour_heatmap": heatmap}


def hours_above_threshold_error(pred: xr.DataArray, target: xr.DataArray, threshold: xr.DataArray) -> dict:
    """Metric 15: within-day exceedance duration error — |hours above threshold
    per day, model minus truth|, counted per member then averaged.

    "Threshold" here is the day's p90-of-DAILY-MAX-Tmax climatology value (the
    same threshold get_heatwaves() uses), not an hourly climatological
    percentile. That's a rarer bar than a typical hour clears, so counts are
    small and concentrated near the daily peak by construction — read this as
    "hours above the daily-Tmax p90", not "hours above a typical-hour p90".
    """
    if "batch" in pred.dims and pred.sizes["batch"] == 1:
        pred = pred.squeeze("batch", drop=True)
    if "batch" in target.dims and target.sizes["batch"] == 1:
        target = target.squeeze("batch", drop=True)
    pred_b = _day_blocked(pred)
    tgt_b = _day_blocked(target)
    pred_counts = pred_b.groupby("day_idx").map(
        lambda block: (block > threshold.sel(day_idx=int(block.day_idx.values[0]))).sum("hour"))
    tgt_counts = tgt_b.groupby("day_idx").map(
        lambda block: (block > threshold.sel(day_idx=int(block.day_idx.values[0]))).sum("hour"))
    count_error = abs(pred_counts - tgt_counts)
    if "ensemble" in count_error.dims:
        count_error = count_error.mean("ensemble")
    weights = np.cos(np.deg2rad(target.lat))
    w = xr.DataArray(weights, coords={"lat": target.lat})
    error_map = count_error.mean(["forecast", "day_idx"])
    scalar = float(error_map.weighted(w).mean(["lat", "lon"]))
    # Per-initialization series (area + day mean, keep the forecast dim) for the block bootstrap.
    series = count_error.weighted(w).mean(["lat", "lon", "day_idx"])
    return {"map": error_map, "scalar": scalar, "series": series}


def event_bias(pred_hw: xr.DataArray, tgt_hw: xr.DataArray) -> dict:
    """Metrics 16-17: event frequency bias N_pred/N_obs over cells with
    N_obs >= 1 (above 1 too many events, catches fragmentation) and event
    duration bias, mean event length difference in days (negative truncates
    or fragments). Events are maximal flagged runs; the >= 3-day sieve is
    already applied by get_heatwaves."""
    weights = np.cos(np.deg2rad(tgt_hw.lat))
    w = xr.DataArray(weights, coords={"lat": tgt_hw.lat})
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


def conditional_tmax_errors(pred_tmax: xr.DataArray, tgt_tmax: xr.DataArray, tgt_hw: xr.DataArray) -> dict:
    """Metric 18: Tmax bias and RMSE only over day-cell pairs where the
    OBSERVED heatwave flag is true (conditioning on observation, not
    prediction). A conditional cold bias explains recall deficits.
    bias_series (forecast,) feeds the block bootstrap."""
    diff = (pred_tmax - tgt_tmax).where(tgt_hw)
    weights = np.cos(np.deg2rad(tgt_tmax.lat))
    w = xr.DataArray(weights, coords={"lat": tgt_tmax.lat})
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
    pred_quantiles = pred_tmax.quantile(list(quantile_levels), dim=pred_pool_dims)
    tgt_quantiles = tgt_tmax.quantile(list(quantile_levels), dim=["forecast", "day_idx"])
    delta_map = pred_quantiles - tgt_quantiles
    weights = np.cos(np.deg2rad(tgt_tmax.lat))
    w = xr.DataArray(weights, coords={"lat": tgt_tmax.lat})
    scalars = {
        f"q{int(round(level * 100))}": float(delta_map.sel(quantile=level).weighted(w).mean(["lat", "lon"]))
        for level in quantile_levels
    }
    return {"map": delta_map, "scalars": scalars}


