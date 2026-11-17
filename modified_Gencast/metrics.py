"""Every metric the experiments score, computed per case so it can be resampled later.

Takes a cached prediction and its targets and returns per-case values rather than finished
numbers. A report calls these and writes the result to metric_cases.nc; significance.py
later bootstraps over the case axis. Averaging here would make that impossible, which is why
nothing is pre-aggregated and why spread and skill are returned as two components instead of
a ratio.

    to_scoring_grid               canonical layout: no batch dim, ensemble axis, -180..180
                                  longitudes, ascending latitude, cropped to the region
    area_weighted_crps            cos-weighted fair CRPS; given a threshold it is twCRPS
    variogram_score               temporal variogram per 24-hour vector, short-lag weighted
    peak_time_ccrps               circular CRPS on the hour of the daily maximum
    spread_and_skill_cases        ensemble variance and squared error, kept separate
    reconstruction_cases          daily range, asymmetry, hours above threshold, re-aggregation
    heatwave_conditioned_metrics  the orchestrator all five reports call
    fixed_lead_tmax_fields        builds exp2's (forecast, lead_day) Tmax case grid
    reconstruction_tmax_fields    the same for exp3 and exp4's (forecast, day_idx) grid

Heatwaves are defined in exactly one place, ``get_heatwaves``: three or more consecutive days
above the daily p90. Both case-grid builders derive spells on continuous calendar time and
then read them onto the case grid, so a spell is never truncated at a forecast window's edge,
and both return the two observed days before each forecast - the only history the model's own
heatwave flag is allowed to use.
"""

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.metrics import roc_auc_score
import scoringrules as sr
import scores.probability

TARGET_VAR = "2m_temperature"
# Bins for tmax_histogram, 220-325 K in 1 K steps: wide enough for any European daily Tmax.
TMAX_HISTOGRAM_BIN_EDGES = np.arange(220.0, 325.01, 1.0)
# Denmark, the region the heatwave calendar plot is drawn over; the extra cell is Bornholm
DENMARK_CALENDAR_REGION = (54.5, 58.0, 8.0, 13.0)
DENMARK_CALENDAR_EXTRA_CELLS = ((55.1, 14.9),)


def drop_batch(data_array: xr.DataArray) -> xr.DataArray:
    """Squeeze a singleton `batch` dimension if present."""
    if "batch" in data_array.dims and data_array.sizes["batch"] == 1:
        return data_array.squeeze("batch", drop=True)
    return data_array


def cos_weights(data_array: xr.DataArray) -> xr.DataArray:
    """Cosine-of-latitude area weights aligned to the array's lat coordinate."""
    return xr.DataArray(np.cos(np.deg2rad(data_array.lat)), coords={"lat": data_array.lat})




def get_heatwaves(tmax: xr.DataArray, threshold: xr.DataArray, min_days: int = 3, day_dim: str = "day_idx", flag_full_spell: bool = True) -> xr.DataArray:
    """Boolean mask of heatwave days per pixel.

    flag_full_spell=True (default): every day of a qualifying spell (min_days or
    more consecutive days above threshold) is flagged, including the first
    min_days-1 days. flag_full_spell=False: only day min_days onward of a
    qualifying spell counts; isolated hot days and the ramp-up never qualify.
    """
    exceeds = (tmax > threshold).astype(int)
    if exceeds.chunks is not None:
        exceeds = exceeds.chunk({day_dim: -1})
    end_of_spell = (exceeds.rolling({day_dim: min_days}, center=False).sum() >= min_days).fillna(False)
    if not flag_full_spell:
        return end_of_spell.astype(bool)
    return (end_of_spell.isel({day_dim: slice(None, None, -1)}).rolling({day_dim: min_days}, center=False, min_periods=1).max()
        .isel({day_dim: slice(None, None, -1)}).fillna(False).astype(bool))



def relative_economic_value(event_rate, warning_rates, miss_rates, cl_ratios=None) -> np.ndarray:
    """Value of acting on the forecast, per cost-loss ratio, against climatology.

    For each cost-loss ratio, compares the expected cost of the best available warning
    threshold against two reference costs: always acting at the climatological rate, and a
    perfect forecast. 1 means as good as perfect, 0 means no better than climatology.

    Returns one value per ratio, by default the 99 ratios from 0.01 to 0.99. Picking the best
    threshold per ratio is what makes this a potential value rather than a realised one.
    Negative values are clipped to 0, since a user would fall back to climatology instead.
    """
    alpha_values = (np.linspace(0.01, 0.99, 99) if cl_ratios is None
                    else np.asarray(cl_ratios, dtype=float))
    event_rate = float(event_rate)
    warning_rates = np.asarray(warning_rates, dtype=float)
    miss_rates = np.asarray(miss_rates, dtype=float)
    if warning_rates.shape != miss_rates.shape:
        raise ValueError("warning_rates and miss_rates must have the same shape")
    if warning_rates.ndim != 1:
        raise ValueError("warning_rates and miss_rates must be one-dimensional")

    climatology_cost = np.minimum(alpha_values, event_rate)
    forecast_cost = alpha_values[:, None] * warning_rates[None, :] + miss_rates[None, :]
    best_forecast_cost = forecast_cost.min(axis=1)
    perfect_cost = alpha_values * event_rate
    denominator = climatology_cost - perfect_cost
    with np.errstate(divide="ignore", invalid="ignore"):
        values = np.where(denominator > 1e-12,(climatology_cost - best_forecast_cost) / denominator, 0.0)
    return np.maximum(values, 0.0)




def crps_field(pred: xr.DataArray, target: xr.DataArray) -> xr.DataArray:
    """Per-point CRPS field in fair ensemble form, before any aggregation.

    Falls back to absolute error when there is no ensemble dimension, which is what makes
    the deterministic baselines scoreable on the same path as the models.
    """
    pred = drop_batch(pred)
    target = drop_batch(target)
    if "ensemble" not in pred.dims:
        return abs(pred - target)
    return scores.probability.crps_for_ensemble(pred, target, ensemble_member_dim="ensemble", method="fair", preserve_dims="all")


def day_blocks_reshaped(da: xr.DataArray) -> xr.DataArray:
    """Keep only whole 24-h day blocks and split the hour axis into real (day_idx, hour_of_day) dims."""
    assert "forecast" not in da["hour"].dims, ("hour must not depend on forecast; day-blocking assumes one shared lead-relative hour axis.")
    hour_values = da.hour.values
    # "full day" means 24 samples, so anything other than 1h spacing would be silently wrong
    assert len(hour_values) < 2 or np.all(np.diff(hour_values) == 1), (f"day-blocking needs 1h-spaced hours, got spacing {np.unique(np.diff(hour_values))}")
    day_idx_values = (hour_values - hour_values[0]) // 24
    labels, counts = np.unique(day_idx_values, return_counts=True)
    full = set(labels[counts == 24].tolist())
    da = da.assign_coords(day_idx=("hour", np.array([d if d in full else -1 for d in day_idx_values])))
    da = da.where(da.day_idx >= 0, drop=True)
    hour_of_day = ((da.hour.values - da.hour.values[0]) % 24).astype(int)
    return da.assign_coords(hour_of_day=("hour", hour_of_day)).set_index(hour=["day_idx", "hour_of_day"]).unstack("hour")






def variogram_score(pred: xr.DataArray, target: xr.DataArray, order: float = 0.5, forecast_batch_size: int = 4) -> dict:
    """ Per-day temporal variogram score on each within-day 24-hour vector.

    The 1/|i-j| weights concentrate on short lags, so this measures local temporal coherence:
    whether consecutive reconstructed hours follow each other plausibly. Diurnal amplitude and
    peak timing are scored separately, by reconstruction_cases and peak_time_ccrps. """
    pred = drop_batch(pred)
    target = drop_batch(target)
    pred_blocked = day_blocks_reshaped(pred)
    tgt_blocked = day_blocks_reshaped(target)
    has_ensemble = "ensemble" in pred_blocked.dims

    # within-day pairwise weight matrix (24 x 24)
    hours = tgt_blocked.hour_of_day.values
    pair_distance = np.abs(hours[:, None] - hours[None, :])
    pair_weight = np.where(pair_distance > 0, 1.0 / np.maximum(pair_distance, 1), 0.0)
    pair_weight = pair_weight / pair_weight.sum()

    batch_dims = [d for d in ("forecast", "day_idx", "lat", "lon") if d in tgt_blocked.dims]
    target_order = [*batch_dims, "hour_of_day"]
    prediction_order = ([*batch_dims, "ensemble", "hour_of_day"] if has_ensemble else [*batch_dims, "hour_of_day"])
    if forecast_batch_size < 1:
        raise ValueError("forecast_batch_size must be at least 1")
    score_batches = []
    for start in range(0, tgt_blocked.sizes["forecast"], forecast_batch_size):
        stop = min(start + forecast_batch_size, tgt_blocked.sizes["forecast"])
        target_values = np.asarray(tgt_blocked.isel(forecast=slice(start, stop)).transpose(*target_order).values, dtype=np.float32)
        prediction_values = np.asarray(pred_blocked.isel(forecast=slice(start, stop)).transpose(*prediction_order).values, dtype=np.float32)
        if not has_ensemble:
            prediction_values = prediction_values[..., np.newaxis, :]
        score_batches.append(sr.vs_ensemble(target_values, prediction_values, w=pair_weight, p=order))
    score_values = np.concatenate(score_batches, axis=0)
    field = xr.DataArray(score_values, dims=batch_dims, coords={d: tgt_blocked[d] for d in batch_dims})
    w = cos_weights(target)
    score_map = field.mean(["forecast", "day_idx"])
    series_by_day_idx = field.weighted(w).mean(["lat", "lon"])
    series = series_by_day_idx.mean("day_idx")
    return {"map": score_map, "series": series, "series_by_day_idx": series_by_day_idx, "scalar": float(score_map.weighted(w).mean(["lat", "lon"]))}



def peak_time_ccrps(pred: xr.DataArray, target: xr.DataArray) -> dict:
    """Error in the hour of the daily maximum, as a proper and therefore rankable score.

    Takes each member's argmax hour, then scores those against the observed hour with a fair
    CRPS built on the angular distance min(|a-b|, 24-|a-b|). That kernel is negative
    definite, which is what keeps the score proper, and circular, so hour 23 and hour 0 are
    one apart rather than 23.
    """

    pred = drop_batch(pred)
    target = drop_batch(target)
    pred_day_blocked = day_blocks_reshaped(pred)
    tgt_day_blocked = day_blocks_reshaped(target)

    pred_peak = pred_day_blocked.hour_of_day.isel(hour_of_day=pred_day_blocked.argmax("hour_of_day").compute())
    tgt_peak = tgt_day_blocked.hour_of_day.isel(hour_of_day=tgt_day_blocked.argmax("hour_of_day").compute())
    tgt_peak_hour = (tgt_peak % 24)
    
    if "ensemble" in pred_peak.dims:
        pred_hours = (pred_peak % 24).transpose("ensemble", ...)
        member_axis = 0
        angular_to_truth = np.minimum(np.abs(pred_hours - tgt_peak_hour), 24 - np.abs(pred_hours - tgt_peak_hour))
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
        spread = xr.DataArray(pairwise / (num_members * (num_members - 1)), dims=term_truth.dims, coords=term_truth.coords)
        field = term_truth - 0.5 * spread
    else:
        pred_hour = pred_peak % 24
        field = np.minimum(np.abs(pred_hour - tgt_peak_hour), 24 - np.abs(pred_hour - tgt_peak_hour))
    w = cos_weights(target)
    score_map = field.mean(["forecast", "day_idx"])
    series_by_day_idx = field.weighted(w).mean(["lat", "lon"])
    series = series_by_day_idx.mean("day_idx")
    return { "map": score_map, "series_by_day_idx": series_by_day_idx, "series": series, "scalar": float(score_map.weighted(w).mean(["lat", "lon"]))}




def conditional_tmax_errors(pred_tmax: xr.DataArray, tgt_tmax: xr.DataArray, tgt_hw: xr.DataArray, day_dim: str | None = "day_idx") -> dict:
    """Tmax bias and RMSE counting only the days and cells that were observed to be in a heatwave."""

    diff = (pred_tmax - tgt_tmax).where(tgt_hw)
    w = cos_weights(tgt_tmax)
    pool_dims = ["lat", "lon"] + ([day_dim] if day_dim else [])
    non_spatial_dims = [dimension for dimension in diff.dims if dimension not in ("lat", "lon")]
    bias_series = diff.weighted(w).mean(pool_dims)
    squared_error_series = (diff ** 2).weighted(w).mean(pool_dims)
    rmse_series = np.sqrt(squared_error_series)
    bias = float(bias_series.mean())
    rmse = float(np.sqrt(squared_error_series.mean()))
    return {"bias": bias, "rmse": rmse, "bias_map": diff.mean(non_spatial_dims),
            "bias_series": bias_series, "rmse_series": rmse_series, "squared_error_series": squared_error_series}


def to_scoring_grid(data: xr.DataArray, region: tuple[float, float, float, float]) -> xr.DataArray:
    """Put one cached field on the canonical scoring layout: no batch dim, ensemble member axis,
    -180..180 longitudes, ascending latitude, cropped to the scoring region."""
    if "batch" in data.dims:
        if data.sizes["batch"] != 1:
            raise ValueError(f"Expected one cached batch, got {data.sizes['batch']}")
        data = data.squeeze("batch", drop=True)
    if "sample" in data.dims:
        data = data.rename({"sample": "ensemble"})
    if float(data.lon.max()) > 180:
        data = data.assign_coords(lon=(data.lon + 180) % 360 - 180).sortby("lon")
    return data.sortby("lat").sel(lat=slice(region[0], region[1]), lon=slice(region[2], region[3]))


def area_weighted_crps(prediction: xr.DataArray, target: xr.DataArray, threshold: xr.DataArray | None = None) -> xr.DataArray:
    """Fair ensemble CRPS between prediction and target, averaged over the region.

    Computes the per-point CRPS, then takes a cos-latitude weighted mean over lat and lon.
    Every non-spatial dimension survives, so the result is one CRPS value per case.

    With a threshold, prediction and target are both clipped from below before scoring, which
    makes this threshold-weighted CRPS: everything under the threshold collapses onto it, so
    only the hot tail can separate two forecasts.
    """
    if threshold is not None:
        prediction = np.maximum(prediction, threshold)
        target = np.maximum(target, threshold)
    weights = xr.DataArray(np.cos(np.deg2rad(target.lat)), dims=["lat"], coords={"lat": target.lat})
    return crps_field(prediction, target).weighted(weights).mean(["lat", "lon"]).reset_coords(drop=True)


def fixed_lead_hourly_threshold(prediction: xr.DataArray, hourly_p90: xr.DataArray) -> xr.DataArray:
    """Select the p90 field at every fixed-lead forecast valid time."""
    forecasts = pd.DatetimeIndex(prediction.forecast.values)
    hours = np.asarray(prediction.hour.values, dtype=int)
    valid = forecasts.values[:, None] + hours[None, :] * np.timedelta64(1, "h")
    
    day_of_year = xr.DataArray(np.minimum(pd.DatetimeIndex(valid.ravel()).dayofyear.values.reshape(valid.shape), 365),
        dims=["forecast", "hour"], coords={"forecast": prediction.forecast, "hour": prediction.hour})

    hour_of_day = xr.DataArray( pd.DatetimeIndex(valid.ravel()).hour.values.reshape(valid.shape),
        dims=["forecast", "hour"], coords={"forecast": prediction.forecast, "hour": prediction.hour})
    
    return hourly_p90.sel(dayofyear=day_of_year, hour_of_day=hour_of_day).interp(lat=prediction.lat, lon=prediction.lon)


def fixed_lead_tmax_fields(prediction: xr.DataArray, true_daily: xr.DataArray, p90: xr.DataArray
                            ) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:
    """Build exp2's (forecast, lead_day) Tmax case grid from a fixed-lead hourly forecast.

    Takes the daily maximum of the forecast, keeping only lead days that carry both the 00
    and 12 UTC frames, then looks up the matching truth and threshold for every cell.

    Returns six arrays:
        members             predicted daily Tmax, ensemble dimension retained
        truth               observed daily Tmax on the same grid
        threshold           p90 for each valid day
        target_spell        observed heatwave mask, derived on continuous calendar time so a
                            spell is not cut off where a forecast window ends
        day_indexer         the valid date of every (forecast, lead_day) cell
        antecedent_exceeds  whether each of the two days before the forecast was above p90,
                            the only history the model's own heatwave flag may use
    """
    # takes the daily maximum of the forecast, keeping only lead days with both 00 and 12 UTC
    lead_hours = np.asarray(prediction.hour.values, dtype=int)
    initialisation_hours = pd.DatetimeIndex(prediction.forecast.values).hour
    groups = []
    for initialisation_hour in np.unique(initialisation_hours):
        group = prediction.isel(forecast=(initialisation_hours == initialisation_hour))
        lead_day = (int(initialisation_hour) + lead_hours) // 24
        counts = pd.Series(lead_day).value_counts()
        full_days = sorted(counts[counts == 2].index.tolist())
        if not full_days:
            raise ValueError(f"No complete Tmax lead days for initialisation hour {initialisation_hour}")
        group = group.assign_coords(lead_day=("hour", lead_day))
        groups.append(group.groupby("lead_day").max().sel(lead_day=full_days))
    common_days = sorted(set.intersection(*[set(group.lead_day.values.tolist()) for group in groups]))
    if not common_days:
        raise ValueError("No common Tmax lead days across initialisation hours")
    members = xr.concat([group.sel(lead_day=common_days) for group in groups], dim="forecast").sortby("forecast")

    # looks up the matching truth and threshold for every forecast/lead-day cell
    forecasts = pd.DatetimeIndex(members.forecast.values).floor("D")
    lead_days = np.asarray(members.lead_day.values, dtype=int)
    valid_days = forecasts.values[:, None] + lead_days[None, :] * np.timedelta64(1, "D")
    day_indexer = xr.DataArray(valid_days, dims=["forecast", "lead_day"],coords={"forecast": members.forecast, "lead_day": members.lead_day})
    truth = true_daily.sel(time=day_indexer).interp(lat=members.lat, lon=members.lon)
    day_of_year = xr.DataArray( np.minimum(pd.DatetimeIndex(valid_days.ravel()).dayofyear.values.reshape(valid_days.shape), 365),
        dims=["forecast", "lead_day"], coords={"forecast": members.forecast, "lead_day": members.lead_day})
    threshold = p90.sel(dayofyear=day_of_year).interp(lat=members.lat, lon=members.lon)

    # finds the observed heatwaves on the full calendar, then reads them onto the case grid
    aligned_true_daily = true_daily.interp(lat=members.lat, lon=members.lon)
    continuous_day_of_year = xr.DataArray(np.minimum(pd.DatetimeIndex(true_daily.time.values).dayofyear, 365),
        dims=["time"], coords={"time": true_daily.time})
    aligned_continuous_threshold = p90.sel(dayofyear=continuous_day_of_year).interp(lat=members.lat, lon=members.lon)
    continuous_spell = get_heatwaves(aligned_true_daily, aligned_continuous_threshold, day_dim="time", flag_full_spell=True)
    target_spell = continuous_spell.sel(time=day_indexer)

    # collects the two observed days before each forecast, the only history its heatwave flag may use
    continuous_exceeds = aligned_true_daily > aligned_continuous_threshold
    first_valid_days = forecasts.values + int(lead_days.min()) * np.timedelta64(1, "D")
    antecedent_days = xr.DataArray(
        first_valid_days[:, None] + np.array([-2, -1])[None, :] * np.timedelta64(1, "D"),
        dims=["forecast", "antecedent"],
        coords={"forecast": members.forecast, "antecedent": [-2, -1]})
    # reindex so forecasts at the start of the record read False instead of raising on a missing date
    antecedent_exceeds = continuous_exceeds.reindex(
        time=np.union1d(continuous_exceeds.time.values, np.unique(antecedent_days.values)),
        fill_value=False).sel(time=antecedent_days).drop_vars("time")
    return members, truth, threshold, target_spell, day_indexer, antecedent_exceeds


def heatwave_event_primitives(prediction: xr.DataArray, target: xr.DataArray,  prediction_event: xr.DataArray, target_event: xr.DataArray) -> xr.Dataset:
    """Per-case reliability, decision and error components for a binary event, from event masks
    passed in so either the threshold or the sieved heatwave definition can be scored."""
    # turns the ensemble into an event probability and the truth into a 0/1 event
    is_ensemble = "ensemble" in prediction.dims
    num_members = int(prediction.sizes["ensemble"]) if is_ensemble else 1
    prediction_probability = (prediction_event.astype(float).mean("ensemble") if is_ensemble else prediction_event.astype(float))
    target_indicator = target_event.astype(float)
    prediction_probability, target_indicator = xr.broadcast(prediction_probability, target_indicator)
    spatial_dims = [dimension for dimension in ("lat", "lon") if dimension in prediction_probability.dims]
    case_dims = [dimension for dimension in prediction_probability.dims if dimension not in spatial_dims]
    is_finite = np.isfinite(prediction_probability) & np.isfinite(target_indicator)
    target_is_event = target_indicator > 0.5
    weights = cos_weights(target)

    # counts cells per forecast-probability bin, and how many were events, for the reliability diagram
    probability_bin = np.rint(prediction_probability * num_members).astype(np.int16)
    probability_bin_values = np.arange(num_members + 1, dtype=np.int16)
    probability_bin_count = xr.concat(
        [((probability_bin == value) & is_finite).sum(spatial_dims).astype(np.int32) for value in probability_bin_values],
        dim=xr.IndexVariable("probability_bin", probability_bin_values),
    ).transpose(*case_dims, "probability_bin")
    probability_bin_event_count = xr.concat(
        [((probability_bin == value) & target_is_event & is_finite).sum(spatial_dims).astype(np.int32) for value in probability_bin_values],
        dim=xr.IndexVariable("probability_bin", probability_bin_values),
    ).transpose(*case_dims, "probability_bin")

    # Brier score, how often the event happened, and the Tmax error over observed-event cells only
    valid_prediction_probability = prediction_probability.where(is_finite)
    valid_target_indicator = target_indicator.where(is_finite)
    brier_score = ((valid_prediction_probability - valid_target_indicator) ** 2).weighted(weights).mean(spatial_dims)
    event_rate = valid_target_indicator.weighted(weights).mean(spatial_dims)
    point_prediction = prediction.mean("ensemble") if is_ensemble else prediction
    conditional_mask = target_is_event & is_finite
    conditional_error_sum = abs(point_prediction - target).where(conditional_mask).weighted(weights).sum(spatial_dims)
    conditional_support = conditional_mask.astype(float).weighted(weights).sum(spatial_dims)

    # sweeps every warning threshold a member vote can produce and records the confusion shares at each;
    # only three are stored because TNR = 1 - TPR - FPR - FNR
    decision_index = np.arange(num_members + 2, dtype=np.int16)
    decision_threshold = (decision_index.astype(float) - 0.5) / num_members
    components = {name: [] for name in (
        "true_positive_share", "false_positive_share", "false_negative_share",
    )}
    for threshold_value in decision_threshold:
        warning = (prediction_probability > threshold_value) & is_finite
        component_masks = {
            "true_positive_share": warning & target_is_event,
            "false_positive_share": warning & ~target_is_event,
            "false_negative_share": ~warning & target_is_event,
        }
        for name, mask in component_masks.items():
            components[name].append(mask.astype(float).where(is_finite).weighted(weights).mean(spatial_dims))

    dataset = xr.Dataset({
        "probability_bin_count": probability_bin_count,
        "probability_bin_event_count": probability_bin_event_count,
        "brier_score": brier_score.astype(np.float32),
        "event_rate": event_rate.astype(np.float32),
        "conditional_abs_error_sum": conditional_error_sum.astype(np.float32),
        "conditional_error_support": conditional_support.astype(np.float32),
        **{name: xr.concat(values, dim=xr.IndexVariable("decision", decision_index)).transpose(*case_dims, "decision").astype(np.float32)
           for name, values in components.items()},
    })

    # AUC per case, left as NaN when a case is all events or all non-events
    if is_ensemble:
        stacked_prediction_probability = prediction_probability.stack(_case=case_dims, _cell=spatial_dims).transpose("_case", "_cell")
        stacked_target_indicator = target_indicator.stack(_case=case_dims, _cell=spatial_dims).transpose("_case", "_cell")
        auc = np.full(stacked_prediction_probability.sizes["_case"], np.nan, dtype=np.float32)
        for index, (prediction_probability_values, target_indicator_values) in enumerate(
                zip(stacked_prediction_probability.values, stacked_target_indicator.values, strict=True)):
            valid_cells = np.isfinite(prediction_probability_values) & np.isfinite(target_indicator_values)
            observed_values = target_indicator_values[valid_cells]
            if observed_values.size and 0 < observed_values.sum() < observed_values.size:
                auc[index] = roc_auc_score(observed_values, prediction_probability_values[valid_cells])
        dataset["auc_case"] = xr.DataArray(auc, dims=["_case"], coords={"_case": stacked_prediction_probability._case}).unstack("_case").transpose(*case_dims)
    else:
        dataset["auc_case"] = xr.full_like(event_rate, np.nan, dtype=np.float32)
    dataset = dataset.assign_coords(
        forecast_probability=("probability_bin", probability_bin_values.astype(float) / num_members),
        decision_probability_threshold=("decision", decision_threshold),
    )
    return dataset


def tmax_histogram(prediction: xr.DataArray, target: xr.DataArray, bin_edges: np.ndarray = TMAX_HISTOGRAM_BIN_EDGES) -> xr.Dataset:
    """Daily-Tmax histogram of model against ERA5 on shared bins, over every day in the run."""
    point_prediction = prediction.mean("ensemble") if "ensemble" in prediction.dims else prediction
    prediction_values = np.asarray(point_prediction.values, dtype=float).ravel()
    target_values = np.asarray(target.values, dtype=float).ravel()
    prediction_counts, _ = np.histogram(prediction_values[np.isfinite(prediction_values)], bins=bin_edges)
    target_counts, _ = np.histogram(target_values[np.isfinite(target_values)], bins=bin_edges)
    return xr.Dataset({
        "tmax_histogram_prediction": xr.DataArray(prediction_counts.astype(np.int64), dims=["tmax_bin"]),
        "tmax_histogram_target": xr.DataArray(target_counts.astype(np.int64), dims=["tmax_bin"]),
        "tmax_bin_edges": xr.DataArray(np.asarray(bin_edges, dtype=np.float32), dims=["tmax_bin_edge"]),
    })

def tmax_scatter_sample(prediction: xr.DataArray, target: xr.DataArray, target_exceeds: xr.DataArray,
                        target_spell: xr.DataArray, num_points: int = 4000, seed: int = 42) -> xr.Dataset:
    """Fixed-size random sample of daily-Tmax prediction/target pairs for a scatter plot,
    each tagged by its observed category: heatwave, above threshold, or neither."""
    point_prediction = prediction.mean("ensemble") if "ensemble" in prediction.dims else prediction
    point_prediction, target, target_exceeds, target_spell = xr.broadcast(
        point_prediction, target, target_exceeds, target_spell)
    flat_prediction = np.asarray(point_prediction.values, dtype=float).ravel()
    flat_target = np.asarray(target.values, dtype=float).ravel()
    flat_exceeds = np.asarray(target_exceeds.values, dtype=bool).ravel()
    flat_spell = np.asarray(target_spell.values, dtype=bool).ravel()
    finite = np.flatnonzero(np.isfinite(flat_prediction) & np.isfinite(flat_target))
    rng = np.random.default_rng(seed)
    sampled = rng.choice(finite, size=min(num_points, len(finite)), replace=False)
    category = np.where(flat_spell[sampled], "heatwave",
                        np.where(flat_exceeds[sampled], "threshold", "none"))
    return xr.Dataset({
        "tmax_scatter_prediction": xr.DataArray(flat_prediction[sampled].astype(np.float32), dims=["sample"]),
        "tmax_scatter_target": xr.DataArray(flat_target[sampled].astype(np.float32), dims=["sample"]),
        "tmax_scatter_category": xr.DataArray(category, dims=["sample"]),
    })


def heatwave_conditioned_metrics(prediction: xr.DataArray, target: xr.DataArray,
                                 threshold: xr.DataArray, day_dim: str = "day_idx",
                                 target_spell: xr.DataArray | None = None,
                                 valid_day: xr.DataArray | None = None,
                                 antecedent_exceeds: xr.DataArray | None = None) -> xr.Dataset:
    """Event primitives and conditional Tmax errors under both heatwave definitions: one day
    above threshold ("_threshold"), and a >= 3 day spell counted from day 3 ("_heatwave")."""
    # scores the definition where any day above p90 counts as an event
    point_prediction = prediction.mean("ensemble") if "ensemble" in prediction.dims else prediction
    conditional_pool_dim = day_dim if "forecast" in prediction.dims else None
    target_exceeds = target > threshold
    threshold_events = heatwave_event_primitives(prediction, target, prediction > threshold, target_exceeds)
    threshold_conditional = conditional_tmax_errors(
        point_prediction, target, target_exceeds, day_dim=conditional_pool_dim)

    if target_spell is None:
        target_spell = get_heatwaves(target, threshold, day_dim=day_dim, flag_full_spell=True)

    # decides which forecast days belong to a 3-day spell, using the two observed days before the forecast
    model_exceeds = prediction > threshold
    if antecedent_exceeds is not None and "forecast" not in prediction.dims:
        prediction_spell = (antecedent_exceeds.isel(antecedent=0) & antecedent_exceeds.isel(antecedent=1) & model_exceeds)
    elif antecedent_exceeds is not None:
        antecedent = antecedent_exceeds.rename({"antecedent": day_dim})
        if "ensemble" in model_exceeds.dims:
            antecedent = antecedent.expand_dims(ensemble=model_exceeds.ensemble)
        spliced = xr.concat([antecedent, model_exceeds], dim=day_dim).astype(int)
        if spliced.chunks is not None:
            spliced = spliced.chunk({day_dim: -1})
        end_of_spell = (spliced.rolling({day_dim: 3}, center=False).sum() >= 3).fillna(False)
        prediction_spell = (
            end_of_spell
            .isel({day_dim: slice(None, None, -1)})
            .rolling({day_dim: 3}, center=False, min_periods=1).max()
            .isel({day_dim: slice(None, None, -1)})
            .fillna(False).astype(bool)
            .isel({day_dim: slice(2, None)})
        )
    elif prediction.sizes.get(day_dim, 1) < 3:
        prediction_spell = xr.zeros_like(model_exceeds, dtype=bool)
    else:
        prediction_spell = get_heatwaves(prediction, threshold, day_dim=day_dim, flag_full_spell=True)

    heatwave_events = heatwave_event_primitives(prediction, target, prediction_spell, target_spell)
    heatwave_conditional = conditional_tmax_errors(point_prediction, target, target_spell, day_dim=conditional_pool_dim)
    distribution = tmax_histogram(prediction, target)
    scatter_sample = tmax_scatter_sample(prediction, target, target_exceeds, target_spell)

    merge_pieces = [
        threshold_events.rename({name: f"{name}_threshold" for name in threshold_events.data_vars}),
        heatwave_events.rename({name: f"{name}_heatwave" for name in heatwave_events.data_vars}),
        xr.Dataset({
            "conditional_tmax_squared_error_case_threshold":threshold_conditional["squared_error_series"].astype(np.float32),
            "conditional_tmax_squared_error_case_heatwave":heatwave_conditional["squared_error_series"].astype(np.float32),
            "tmax_threshold_mean": xr.DataArray(np.float32(threshold.mean())),
        }),
        distribution,
        scatter_sample,
    ]
    # how much of the region was hot each calendar day, observed and as a share of members
    if valid_day is not None:
        spatial_dims = [dimension for dimension in ("lat", "lon") if dimension in target.dims]
        calendar_lead = ({day_dim: prediction[day_dim].values.min()} if "forecast" in prediction.dims else {})
        calendar_valid_day = valid_day.sel(calendar_lead) if calendar_lead else valid_day
        dates = pd.DatetimeIndex(np.asarray(calendar_valid_day.values, dtype="datetime64[ns]").ravel())
        min_lat, max_lat, min_lon, max_lon = DENMARK_CALENDAR_REGION
        half_cell_lat = float(abs(prediction.lat[1] - prediction.lat[0])) / 2
        half_cell_lon = float(abs(prediction.lon[1] - prediction.lon[0])) / 2
        in_region = ((prediction.lat >= min_lat) & (prediction.lat <= max_lat)
                     & (prediction.lon >= min_lon) & (prediction.lon <= max_lon))
        for cell_lat, cell_lon in DENMARK_CALENDAR_EXTRA_CELLS:
            in_region = in_region | ((abs(prediction.lat - cell_lat) <= half_cell_lat)
                                     & (abs(prediction.lon - cell_lon) <= half_cell_lon))
        region_weights = cos_weights(prediction) * in_region

        def to_series(per_case):
            per_case = (per_case.sel(calendar_lead) if calendar_lead else per_case).broadcast_like(calendar_valid_day)
            series = pd.Series(np.asarray(per_case.values, dtype=float).ravel(),index=dates).groupby(level=0).mean().sort_index()
            return xr.DataArray(series.values.astype(np.float32), dims=["date"], coords={"date": series.index.values})

        member_coverage = model_exceeds.astype(float).weighted(region_weights).mean(spatial_dims)
        majority_by_member = (member_coverage > 0.5).astype(float)
        calendar = {"calendar_model_certainty": to_series(
            majority_by_member.mean("ensemble") if "ensemble" in prediction.dims else majority_by_member)}
        calendar["calendar_observed_share"] = to_series(
            target_exceeds.astype(float).weighted(region_weights).mean(spatial_dims))

        # turns the hot days into observed heatwaves, region first so heat moving across it still counts
        hot_day = calendar["calendar_observed_share"] > 0.5
        gapless_days = pd.date_range(hot_day.date.values.min(), hot_day.date.values.max(), freq="D")
        run_ends = hot_day.reindex(date=gapless_days, fill_value=False).astype(int).rolling(date=3).sum() >= 3
        calendar["calendar_observed_heatwave"] = (
            run_ends.isel(date=slice(None, None, -1))
            .rolling(date=3, min_periods=1).max()
            .isel(date=slice(None, None, -1))
            .fillna(False).astype(np.float32)
            .reindex(date=hot_day.date))
        merge_pieces.append(xr.Dataset(calendar))
    return xr.merge(merge_pieces, compat="equals", join="exact")


def spread_and_skill_cases(prediction: xr.DataArray, target: xr.DataArray) -> xr.Dataset:
    """Per-case ensemble variance and squared error of the ensemble mean, kept as the two
    components so any window or bootstrap resample can form the spread-skill ratio itself."""
    weights = xr.DataArray(np.cos(np.deg2rad(target.lat)), dims=["lat"], coords={"lat": target.lat})
    if "ensemble" in prediction.dims:
        members = int(prediction.sizes["ensemble"])
        variance = prediction.var("ensemble").weighted(weights).mean(["lat", "lon"])
        point_prediction = prediction.mean("ensemble")
    else:
        members = 1
        variance = xr.zeros_like(target.isel(lat=0, lon=0, drop=True))
        point_prediction = prediction
    squared_error = ((point_prediction - target) ** 2).weighted(weights).mean(["lat", "lon"])
    return xr.Dataset({
        "ensemble_variance_case": variance.astype(np.float32).reset_coords(drop=True),
        "squared_error_case": squared_error.astype(np.float32).reset_coords(drop=True),
    }, attrs={"ensemble_members": members})


def reconstruction_cases(prediction: xr.DataArray, target: xr.DataArray, threshold: xr.DataArray) -> xr.Dataset:
    """Daily range, mean-position asymmetry and hours above threshold, scored per case on full
    24-hour day blocks. Peak timing is scored separately by peak_time_ccrps."""
    prediction_days = day_blocks_reshaped(drop_batch(prediction))
    target_days = day_blocks_reshaped(drop_batch(target))
    threshold_days = day_blocks_reshaped(drop_batch(threshold))
    weights = cos_weights(target)

    # measures how far the predicted warmest hour sits from the observed one on a 24-hour clock
    prediction_peak = prediction_days.hour_of_day.isel(hour_of_day=prediction_days.argmax("hour_of_day").compute()) % 24
    target_peak = target_days.hour_of_day.isel(hour_of_day=target_days.argmax("hour_of_day").compute()) % 24
    angular = np.minimum(abs(prediction_peak - target_peak), 24 - abs(prediction_peak - target_peak))
    peak_time_error = (angular.mean("ensemble") if "ensemble" in angular.dims else angular)

    # takes the daily statistics back out of the reconstructed hours and scores them against truth
    reaggregated = {}
    for name, reducer in (("tmax", "max"), ("tmin", "min"), ("tmean", "mean")):
        predicted = getattr(prediction_days, reducer)("hour_of_day")
        observed = getattr(target_days, reducer)("hour_of_day")
        point = predicted.mean("ensemble") if "ensemble" in predicted.dims else predicted
        reaggregated[f"{name}_reaggregation_crps_case"] = ( crps_field(predicted, observed).weighted(weights).mean(["lat", "lon"]).astype(np.float32))
        reaggregated[f"{name}_reaggregation_bias_case"] = ((point - observed).weighted(weights).mean(["lat", "lon"]).astype(np.float32))
        reaggregated[f"{name}_daily"] = (predicted, observed)

    prediction_range = reaggregated["tmax_daily"][0] - reaggregated["tmin_daily"][0]
    target_range = reaggregated["tmax_daily"][1] - reaggregated["tmin_daily"][1]
    reaggregated = {name: value for name, value in reaggregated.items() if not name.endswith("_daily")}
    prediction_hours_above = (prediction_days > threshold_days).sum("hour_of_day").astype(float)
    target_hours_above = (target_days > threshold_days).sum("hour_of_day").astype(float)

    # measures where in its own daily range the cycle spends its time, 0.5 being symmetric
    min_valid_range = 1.0
    prediction_asymmetry = ((prediction_days.mean("hour_of_day") - prediction_days.min("hour_of_day")) / prediction_range)
    target_asymmetry = ((target_days.mean("hour_of_day") - target_days.min("hour_of_day")) / target_range)
    valid_asymmetry = target_range >= min_valid_range
    prediction_asymmetry = prediction_asymmetry.where(prediction_range >= min_valid_range, 0.5)
    target_asymmetry = target_asymmetry.where(valid_asymmetry)
    point_range = (prediction_range.mean("ensemble") if "ensemble" in prediction_range.dims else prediction_range)
    point_asymmetry = (prediction_asymmetry.mean("ensemble") if "ensemble" in prediction_asymmetry.dims else prediction_asymmetry)

    return xr.Dataset({
        "daily_range_crps_case": crps_field(prediction_range, target_range).weighted(weights).mean(["lat", "lon"]).astype(np.float32),
        "daily_range_bias_case": (point_range - target_range).weighted(weights).mean(["lat", "lon"]).astype(np.float32),
        "target_daily_range_case": target_range.weighted(weights).mean(["lat", "lon"]).astype(np.float32),
        "asymmetry_crps_case": crps_field(prediction_asymmetry, target_asymmetry).weighted(weights).mean(["lat", "lon"]).astype(np.float32),
        "asymmetry_bias_case": (point_asymmetry - target_asymmetry).weighted(weights).mean(["lat", "lon"]).astype(np.float32),
        "target_asymmetry_case": target_asymmetry.weighted(weights).mean(["lat", "lon"]).astype(np.float32),
        "hours_above_threshold_crps_case": crps_field(prediction_hours_above, target_hours_above).weighted(weights).mean(["lat", "lon"]).astype(np.float32),
        "target_hours_above_threshold_case": target_hours_above.weighted(weights).mean(["lat", "lon"]).astype(np.float32),
        **reaggregated,
    }).reset_coords(drop=True)


def reconstruction_tmax_fields(prediction: xr.DataArray, target: xr.DataArray, p90: xr.DataArray
                               ) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:
    """Build exp3 and exp4's (forecast, day_idx) Tmax case grid from an hourly reconstruction.

    The counterpart of fixed_lead_tmax_fields for runs that produce whole 24-hour days:
    daily Tmax is the maximum over each day block rather than over two fixed frames.

    Returns the same six arrays, with day_idx in place of lead_day. The observed spells are
    found by first collapsing every forecast's days into one dated series, so the three-day
    rule runs over real consecutive calendar days instead of within a single forecast window.
    """
    # takes the daily maximum of each reconstructed day and its matching truth and threshold
    prediction_days = day_blocks_reshaped(drop_batch(prediction))
    target_days = day_blocks_reshaped(drop_batch(target))
    members = prediction_days.max("hour_of_day")
    truth = target_days.max("hour_of_day")
    forecast = pd.DatetimeIndex(members.forecast.values).floor("D")
    day_index = np.asarray(members.day_idx.values, dtype=int)
    valid_days = forecast.values[:, None] + day_index[None, :] * np.timedelta64(1, "D")
    day_of_year = xr.DataArray(
        np.minimum(pd.DatetimeIndex(valid_days.ravel()).dayofyear.values.reshape(valid_days.shape), 365),
        dims=["forecast", "day_idx"],
        coords={"forecast": members.forecast, "day_idx": members.day_idx})
    threshold = p90.sel(dayofyear=day_of_year).interp(lat=members.lat, lon=members.lon)
    day_indexer = xr.DataArray(valid_days, dims=["forecast", "day_idx"], coords={"forecast": members.forecast, "day_idx": members.day_idx})

    # puts the observed days into a single dated series, one entry per day, so the 3-day rule
    # runs over real consecutive days instead of within one forecast's window
    stacked_truth = truth.stack(case=("forecast", "day_idx"))
    continuous_truth = (stacked_truth.drop_vars(["case", "forecast", "day_idx"])
                        .assign_coords(case=pd.DatetimeIndex(valid_days.ravel()))
                        .groupby("case").max().rename({"case": "time"}).sortby("time"))
    continuous_truth = continuous_truth.reindex( time=pd.date_range(continuous_truth.time.values.min(), continuous_truth.time.values.max(), freq="D"))
    continuous_day_of_year = xr.DataArray( np.minimum(pd.DatetimeIndex(continuous_truth.time.values).dayofyear, 365), dims=["time"], coords={"time": continuous_truth.time})
    continuous_threshold = p90.sel(dayofyear=continuous_day_of_year).interp( lat=members.lat, lon=members.lon)
    continuous_spell = get_heatwaves(continuous_truth, continuous_threshold, day_dim="time", flag_full_spell=True)
    target_spell = continuous_spell.sel(time=day_indexer)

    # collects the two observed days before each forecast, the only history the model's flag may use
    continuous_exceeds = continuous_truth > continuous_threshold
    first_valid_days = forecast.values + int(day_index.min()) * np.timedelta64(1, "D")

    antecedent_days = xr.DataArray(
        first_valid_days[:, None] + np.array([-2, -1])[None, :] * np.timedelta64(1, "D"),
        dims=["forecast", "antecedent"],
        coords={"forecast": members.forecast, "antecedent": [-2, -1]})
    antecedent_exceeds = continuous_exceeds.reindex( time=np.union1d(continuous_exceeds.time.values, np.unique(antecedent_days.values)), fill_value=False).sel(time=antecedent_days).drop_vars("time")
    return members, truth, threshold, target_spell, day_indexer, antecedent_exceeds
