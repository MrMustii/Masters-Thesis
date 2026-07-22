import os
import numpy as np
import pandas as pd
import s3fs
import xarray as xr

from modified_Gencast.lumio_buckets import lumio, load, bucket_name, GENCAST_LEVELS
from modified_Gencast.metrics import (
    TARGET_VAR, spatial_errors, mean_forecast, temporal_autocorrelation,
    seasonal_errors, power_spectra_1d, mean_temporal_difference,
    crps, spread_skill_ratio, rank_histogram, ensemble_power_spectra_1d,
    binary_classification_metrics,
)
from modified_Gencast.plots import (
    plot_temp_lead_curve, plot_temporal_errors, plot_hour_of_day_curves,
    plot_spatial_error_maps, plot_seasonal_bias_maps,
    plot_temporal_autocorrelation, plot_power_spectra, plot_mean_temporal_difference,
    plot_crps_growth, plot_spread_skill, plot_rank_histogram, plot_ensemble_power_spectra,
)


def prepare_datasets(predictions_ds, targets_ds, region):
    """Squeeze the batch dim, crop to the region box, and detach from dask."""
    if "batch" in predictions_ds.dims and predictions_ds.sizes["batch"] == 1:
        predictions_ds = predictions_ds.squeeze("batch", drop=True)
    if "batch" in targets_ds.dims and targets_ds.sizes["batch"] == 1:
        targets_ds = targets_ds.squeeze("batch", drop=True)

    if region is not None:
        inner_lat = predictions_ds.lat.values[
            (predictions_ds.lat.values >= region[0]) & (predictions_ds.lat.values <= region[1])]
        inner_lon = predictions_ds.lon.values[
            (predictions_ds.lon.values >= region[2]) & (predictions_ds.lon.values <= region[3])]
        predictions_ds = predictions_ds.sel(lat=inner_lat, lon=inner_lon)
        targets_ds = targets_ds.sel(lat=inner_lat, lon=inner_lon)
        assert predictions_ds.sizes["lat"] > 0 and predictions_ds.sizes["lon"] > 0, \
            "Region crop produced an empty grid; check region/coord order."

    predictions_ds = xr.Dataset(
        {name: xr.DataArray(np.asarray(predictions_ds[name].values), dims=predictions_ds[name].dims,
                            coords=predictions_ds[name].coords) for name in predictions_ds.data_vars})
    targets_ds = xr.Dataset(
        {name: xr.DataArray(np.asarray(targets_ds[name].values), dims=targets_ds[name].dims,
                            coords=targets_ds[name].coords) for name in targets_ds.data_vars})
    return predictions_ds, targets_ds


def compute_common_metrics(predictions_ds, targets_ds, num_ensemble, base_attrs):
    """Metrics every experiment reports for 2m temperature.

    Returns (metrics_vars, metrics_attrs). The caller adds its experiment
    specific entries and writes the combined dataset.
    """
    total_hours = int(predictions_ds[TARGET_VAR].sizes["hour"])
    hour_values = predictions_ds[TARGET_VAR].hour.values
    hour_spacing = int(np.median(np.diff(hour_values))) if total_hours > 1 else 24
    fine_time_resolution = hour_spacing <= 6 and total_hours >= 8


    temp_spatial = spatial_errors(predictions_ds[TARGET_VAR], targets_ds[TARGET_VAR])
    rmse_ts_temp = temp_spatial["rmse"]
    bias_ts_temp = temp_spatial["bias"]
    print(f"  2m-T RMSE: {float(rmse_ts_temp.mean()):.4f} K  bias: {float(bias_ts_temp.mean()):.4f} K")

    metrics_vars = {
        "rmse_map_temp": temp_spatial["rmse_map"], "bias_map_temp": temp_spatial["bias_map"],
        "rmse_ts_temp": rmse_ts_temp, "bias_ts_temp": bias_ts_temp,
        "mse_series": temp_spatial["mse_series"],
    }
    metrics_attrs = dict(base_attrs)

    pred_diurnal = mean_forecast(predictions_ds[TARGET_VAR])
    tgt_diurnal = mean_forecast(targets_ds[TARGET_VAR])
    seasonal_result = seasonal_errors(predictions_ds[TARGET_VAR], targets_ds[TARGET_VAR])
    metrics_vars.update({"mean_forecast_pred": pred_diurnal, "mean_forecast_tgt": tgt_diurnal})

    if fine_time_resolution:
        pred_autocorr = temporal_autocorrelation(predictions_ds[TARGET_VAR])
        tgt_autocorr = temporal_autocorrelation(targets_ds[TARGET_VAR])
        pred_temp_series = predictions_ds[TARGET_VAR]
        if "ensemble" in pred_temp_series.dims:
            pred_temp_series = pred_temp_series.mean("ensemble")
        pred_temp_series = pred_temp_series.mean(["lat", "lon"]).mean("forecast")
        tgt_temp_series = targets_ds[TARGET_VAR].mean(["lat", "lon"]).mean("forecast")
        pred_periods, pred_psd = power_spectra_1d(pred_temp_series)
        tgt_periods, tgt_psd = power_spectra_1d(tgt_temp_series)
        metrics_vars.update({
            "autocorr_pred": pred_autocorr, "autocorr_tgt": tgt_autocorr,
            "psd_pred": xr.DataArray(pred_psd, dims=["freq"], coords={"period_hours": ("freq", pred_periods)}),
            "psd_tgt": xr.DataArray(tgt_psd, dims=["freq"], coords={"period_hours": ("freq", tgt_periods)}),
        })
    else:
        print(f"  Skipping autocorrelation/power spectra: {hour_spacing}h spacing over {total_hours} "
              f"leads is too coarse to resolve the diurnal cycle.")
    for season, season_errors in seasonal_result.items():
        if season_errors is None:
            metrics_vars[f"seasonal_rmse_{season.lower()}"] = xr.full_like(rmse_ts_temp, fill_value=np.nan)
        else:
            metrics_vars[f"seasonal_rmse_{season.lower()}"] = season_errors["rmse"]
            metrics_vars[f"seasonal_bias_map_{season.lower()}"] = season_errors["bias_map"]

    if total_hours >= 2:
        mtd_result = mean_temporal_difference(predictions_ds[TARGET_VAR], targets_ds[TARGET_VAR])
        print(f"  Frame diff — Model: {float(mtd_result['pred'].mean()):.4f} K  "
              f"ERA5: {float(mtd_result['target'].mean()):.4f} K")
        metrics_vars.update({
            "temporal_diff_pred": mtd_result["pred"], "temporal_diff_tgt": mtd_result["target"],
        })

    if num_ensemble > 1:
        crps_result = crps(predictions_ds[TARGET_VAR], targets_ds[TARGET_VAR])
        spread_skill_result = spread_skill_ratio(predictions_ds[TARGET_VAR], targets_ds[TARGET_VAR])
        rank_hist = rank_histogram(predictions_ds[TARGET_VAR], targets_ds[TARGET_VAR])
        print(f"  Mean CRPS: {float(crps_result['crps'].mean()):.4f} K")
        print(f"  Mean SSR:  {float(spread_skill_result['ratio'].mean()):.4f}")
        metrics_vars.update({
            "crps_map": crps_result["crps_map"], "crps_growth": crps_result["crps"],
            "crps_series": crps_result["crps_series"],
            "spread_growth": spread_skill_result["spread"], "skill_growth": spread_skill_result["skill"],
            "spread_skill_ratio": spread_skill_result["ratio"],
            "rank_histogram": xr.DataArray(rank_hist, dims=["rank"], coords={"rank": np.arange(len(rank_hist))}),
        })
        metrics_attrs["crps_mean"] = float(crps_result["crps"].mean())
        metrics_attrs["ssr_mean"] = float(spread_skill_result["ratio"].mean())
        if fine_time_resolution:
            ens_periods, ens_psd_mean, ens_psd_std = ensemble_power_spectra_1d(predictions_ds[TARGET_VAR])
            metrics_vars.update({
                "ens_psd_mean": xr.DataArray(ens_psd_mean, dims=["freq"], coords={"period_hours": ("freq", ens_periods)}),
                "ens_psd_std": xr.DataArray(ens_psd_std, dims=["freq"], coords={"period_hours": ("freq", ens_periods)}),
            })

    return metrics_vars, metrics_attrs


def plot_common(metrics_vars, predictions_ds, targets_ds, plot_dir, tag,
                region, num_ensemble, ar_window_hours, hour_axis, rmse_map_tmax=None):
    """Draw the plots shared by every experiment.

    `hour_axis` is "lead" for AR forecasts (exp2/exp3) and "hour_of_day" for the
    daily reconstruction (exp4); it only changes the three line plots whose
    x-axis meaning depends on it. `rmse_map_tmax` lets a report add its Tmax
    RMSE map as a third panel on the shared spatial-error-maps plot instead of
    drawing a second, overlapping figure. Heatwave / diurnal-shape plots belong
    to the individual experiment reports.
    """
    rmse_ts_temp = metrics_vars["rmse_ts_temp"]
    bias_ts_temp = metrics_vars["bias_ts_temp"]
    rmse_map_temp = metrics_vars["rmse_map_temp"]
    bias_map_temp = metrics_vars["bias_map_temp"]
    lat = rmse_map_temp.lat.values
    lon = rmse_map_temp.lon.values

    if hour_axis == "hour_of_day":
        plot_hour_of_day_curves({"model": rmse_ts_temp}, "RMSE (K)", "2m temperature RMSE by hour of day",
                                os.path.join(plot_dir, f"temporal_errors_{tag}.png"))
        plot_hour_of_day_curves({"model": bias_ts_temp}, "Bias (K)", "2m temperature bias by hour of day",
                                os.path.join(plot_dir, f"bias_by_hour_{tag}.png"))
    else:
        plot_temp_lead_curve(predictions_ds[TARGET_VAR], targets_ds[TARGET_VAR],
                             os.path.join(plot_dir, f"temp_lead_curve_{tag}.png"), ar_window_hours=ar_window_hours)
        plot_temporal_errors(rmse_ts_temp, bias_ts_temp,
                             os.path.join(plot_dir, f"temporal_errors_{tag}.png"), ar_window_hours=ar_window_hours)

    plot_spatial_error_maps(rmse_map_temp.mean("hour"), bias_map_temp.mean("hour"), rmse_map_tmax,
                            lat, lon, plot_dir, tag, region=region)

    seasonal_bias_maps = {name.split("seasonal_bias_map_")[1]: metrics_vars[name]
                          for name in metrics_vars if name.startswith("seasonal_bias_map_")}
    if seasonal_bias_maps:
        plot_seasonal_bias_maps(seasonal_bias_maps, lat, lon, plot_dir, tag, region=region)

    if "autocorr_pred" in metrics_vars:
        plot_temporal_autocorrelation(metrics_vars["autocorr_pred"], metrics_vars["autocorr_tgt"],
                                      os.path.join(plot_dir, f"autocorrelation_{tag}.png"))
    if "psd_pred" in metrics_vars:
        plot_power_spectra(metrics_vars["psd_pred"].period_hours.values, metrics_vars["psd_pred"].values,
                           metrics_vars["psd_tgt"].period_hours.values, metrics_vars["psd_tgt"].values,
                           os.path.join(plot_dir, f"power_spectra_{tag}.png"))

    if "temporal_diff_pred" in metrics_vars:
        plot_mean_temporal_difference(metrics_vars["temporal_diff_pred"], metrics_vars["temporal_diff_tgt"],
                                      os.path.join(plot_dir, f"temporal_diff_{tag}.png"), ar_window_hours=ar_window_hours)

    if num_ensemble > 1:
        if hour_axis == "hour_of_day":
            plot_hour_of_day_curves({"model": metrics_vars["crps_growth"]}, "CRPS (K)", "CRPS by hour of day",
                                    os.path.join(plot_dir, f"crps_growth_{tag}.png"))
        else:
            plot_crps_growth(metrics_vars["crps_growth"], os.path.join(plot_dir, f"crps_growth_{tag}.png"),
                             ar_window_hours=ar_window_hours)
        plot_spread_skill(metrics_vars["spread_growth"], metrics_vars["skill_growth"], metrics_vars["spread_skill_ratio"],
                          os.path.join(plot_dir, f"spread_skill_{tag}.png"), ar_window_hours=ar_window_hours)
        plot_rank_histogram(metrics_vars["rank_histogram"].values, os.path.join(plot_dir, f"rank_histogram_{tag}.png"))
        if "ens_psd_mean" in metrics_vars:
            plot_ensemble_power_spectra(metrics_vars["ens_psd_mean"].period_hours.values, metrics_vars["ens_psd_mean"].values,
                                        metrics_vars["ens_psd_std"].values,
                                        metrics_vars["psd_tgt"].period_hours.values, metrics_vars["psd_tgt"].values,
                                        os.path.join(plot_dir, f"ensemble_power_spectra_{tag}.png"))


# == RQ1 Tmax + heatwave detection (shared by the 12h models: exp1, exp2) =======
#
# A 12h model only outputs 00Z and 12Z, so it cannot resolve the true afternoon
# peak. The model daily Tmax proxy is max(00Z, 12Z) per calendar day; the TRUE
# daily Tmax is the max over the 24 hourly ERA5 steps of the day. Detection is a
# single-day exceedance of the climatological p90 threshold — no spell length or
# duration (those need the multi-day horizons of exp3).

def load_true_daily_tmax(start_date, end_date, loading_region, scoring_region):
    """True daily Tmax from hourly ERA5, on the scoring-region grid, indexed by day."""
    wanted_days = pd.date_range(start_date, end_date, freq="D")
    daily_tmax = None
    daily_bucket = bucket_name(GENCAST_LEVELS, loading_region) + "-daily"
    try:
        store = s3fs.S3Map(root=daily_bucket, s3=lumio(), check=False)
        daily = xr.open_zarr(store, consolidated=True)
        covers = (pd.Timestamp(daily.time.min().values) <= wanted_days[0]
                  and pd.Timestamp(daily.time.max().values) >= wanted_days[-1])
        if "2m_temperature_daily_max" in daily and covers:
            daily_tmax = daily["2m_temperature_daily_max"].sel(time=wanted_days)
            print(f"  True Tmax: daily bucket {daily_bucket}")
    except Exception as error:
        print(f"  Daily bucket unavailable ({error}); loading hourly instead.")
    if daily_tmax is None:
        hourly = load(str(start_date.date()), str(end_date.date()),
                      variables=["2m_temperature"], region=loading_region, freq="1h")["2m_temperature"]
        daily_tmax = hourly.resample(time="1D").max()
        print("  True Tmax: hourly load → daily max")

    if float(daily_tmax.lon.max()) > 180:
        daily_tmax = daily_tmax.assign_coords(lon=(daily_tmax.lon + 180) % 360 - 180).sortby("lon")
    daily_tmax = daily_tmax.sortby("lat")
    lat_min, lat_max, lon_min, lon_max = scoring_region
    return daily_tmax.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))


def load_p90_threshold():
    """Climatological p90 Tmax threshold on its native grid, lon in [-180, 180)."""
    clim_store = s3fs.S3Map(root="era5-lumi/historical_tmax", s3=lumio(), check=False)
    threshold_p90 = xr.open_zarr(clim_store, consolidated=True)["tmax_p90"]
    if "latitude" in threshold_p90.dims:
        threshold_p90 = threshold_p90.rename({"latitude": "lat", "longitude": "lon"})
    if float(threshold_p90.lon.max()) > 180:
        threshold_p90 = threshold_p90.assign_coords(lon=(threshold_p90.lon + 180) % 360 - 180).sortby("lon")
    return threshold_p90


def score_daily_tmax(model_tmax, true_daily_tmax, threshold_p90, weights):
    """Score a model daily Tmax field against the true daily Tmax and the p90 threshold.

    `model_tmax` has dims (ensemble?, day, lat, lon); it is collapsed to the
    ensemble median (deterministic detection). Returns Tmax RMSE/bias (map +
    scalar), single-day heatwave-detection precision/recall/F1 (map + scalar),
    and the aligned median/true fields for QQ plotting. Shared by exp1 (one
    lead-agnostic field) and exp2 (one field per lead).
    """
    if "ensemble" in model_tmax.dims:
        model_tmax = model_tmax.median("ensemble")
    model_days = pd.DatetimeIndex(model_tmax.day.values)
    true_tmax = (true_daily_tmax.sel(time=model_days).rename({"time": "day"})
                 .assign_coords(day=model_days).interp(lat=model_tmax.lat, lon=model_tmax.lon))
    day_of_year = xr.DataArray(model_days.dayofyear, dims="day", coords={"day": model_days})
    threshold = threshold_p90.sel(dayofyear=day_of_year).interp(lat=model_tmax.lat, lon=model_tmax.lon)

    diff = model_tmax - true_tmax
    bias_map = diff.mean("day")
    model_exceed = (model_tmax > threshold).rename({"day": "forecast"}).expand_dims({"day_idx": [0]})
    true_exceed = (true_tmax > threshold).rename({"day": "forecast"}).expand_dims({"day_idx": [0]})
    detection = binary_classification_metrics(model_exceed, true_exceed)
    true_positive = float((model_exceed & true_exceed).astype(float).weighted(weights).sum().values)
    false_positive = float((model_exceed & ~true_exceed).astype(float).weighted(weights).sum().values)
    false_negative = float((~model_exceed & true_exceed).astype(float).weighted(weights).sum().values)
    precision = true_positive / (true_positive + false_positive + 1e-12)
    recall = true_positive / (true_positive + false_negative + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    return {
        "rmse_map": np.sqrt((diff ** 2).mean("day")), "bias_map": bias_map,
        "rmse": float(np.sqrt((diff ** 2).weighted(weights).mean(["lat", "lon", "day"]))),
        "bias": float(bias_map.weighted(weights).mean(["lat", "lon"])),
        "precision_map": detection["precision_map"], "recall_map": detection["recall_map"],
        "f1_map": detection["f1_map"], "precision": precision, "recall": recall, "f1": f1,
        "model_tmax": model_tmax, "true_tmax": true_tmax,
    }


def compute_tmax_detection(temp_by_forecast_hour, true_daily_tmax, threshold_p90):
    """Daily Tmax proxy and single-day heatwave detection as a function of lead DAY.

    A 12h fixed-step rollout has `hour` = a set of 12h leads spanning several days
    (e.g. 12..168 h). This regroups them into (init, lead_day): the model daily
    Tmax proxy for each lead day is max(00Z, 12Z) of that day, keeping only lead
    days that carry both frames. Detection is single-day p90 exceedance;
    binary_classification_metrics then gives precision/recall/F1 as curves over
    lead day (day_idx) and as maps aggregated over all inits and lead days. Tmax
    RMSE/bias are likewise reported vs lead day. Returns metric variables, scalar
    attributes, and the pooled model/true Tmax arrays for the QQ plot.
    """
    temperature = temp_by_forecast_hour
    if "ensemble" in temperature.dims:
        temperature = temperature.median("ensemble")
    weights = xr.DataArray(np.cos(np.deg2rad(temperature.lat)), coords={"lat": temperature.lat})
    lead_hours = temperature.hour.values
    init_hours = pd.DatetimeIndex(temperature.forecast.values).hour

    # hour -> lead_day depends only on the init hour, so split by init hour, group
    # each subset onto lead_day, and keep the lead days that got both 00Z and 12Z.
    model_tmax_groups = []
    for init_hour in np.unique(init_hours):
        group = temperature.isel(forecast=(init_hours == init_hour))
        lead_day = (int(init_hour) + lead_hours) // 24
        frames_per_day = pd.Series(lead_day).value_counts()
        full_days = sorted(frames_per_day[frames_per_day == 2].index.tolist())
        group = group.assign_coords(lead_day=("hour", lead_day))
        model_tmax_groups.append(group.groupby("lead_day").max().sel(lead_day=full_days))
    common_lead_days = sorted(set.intersection(*[set(g.lead_day.values.tolist()) for g in model_tmax_groups]))
    model_tmax = xr.concat([g.sel(lead_day=common_lead_days) for g in model_tmax_groups], dim="forecast")

    # True daily Tmax and p90 threshold at each (init, lead_day)'s absolute date.
    init_dates = pd.DatetimeIndex(model_tmax.forecast.values).floor("D")
    lead_day_values = np.array(common_lead_days)
    absolute_dates = init_dates.values[:, None] + lead_day_values[None, :] * np.timedelta64(1, "D")
    absolute_dates_da = xr.DataArray(absolute_dates, dims=["forecast", "lead_day"],
                                     coords={"forecast": model_tmax.forecast, "lead_day": lead_day_values})
    true_tmax = true_daily_tmax.sel(time=absolute_dates_da).interp(lat=model_tmax.lat, lon=model_tmax.lon)
    day_of_year = xr.DataArray(
        pd.DatetimeIndex(absolute_dates.ravel()).dayofyear.values.reshape(absolute_dates.shape),
        dims=["forecast", "lead_day"], coords={"forecast": model_tmax.forecast, "lead_day": lead_day_values})
    threshold = threshold_p90.sel(dayofyear=day_of_year).interp(lat=model_tmax.lat, lon=model_tmax.lon)

    diff = model_tmax - true_tmax
    rmse_by_lead_day = np.sqrt((diff ** 2).weighted(weights).mean(["lat", "lon", "forecast"]))
    bias_map = diff.mean(["forecast", "lead_day"])
    rmse_map = np.sqrt((diff ** 2).mean(["forecast", "lead_day"]))

    model_exceed = (model_tmax > threshold).rename({"lead_day": "day_idx"})
    true_exceed = (true_tmax > threshold).rename({"lead_day": "day_idx"})
    detection = binary_classification_metrics(model_exceed, true_exceed)

    metrics_vars = {
        "tmax_rmse_by_lead_day": rmse_by_lead_day, "tmax_rmse_map": rmse_map, "tmax_bias_map": bias_map,
        "hw_precision_by_lead_day": detection["precision"], "hw_recall_by_lead_day": detection["recall"],
        "hw_f1_by_lead_day": detection["f1"],
        "hw_precision_map": detection["precision_map"], "hw_recall_map": detection["recall_map"],
        "hw_f1_map": detection["f1_map"],
    }
    metrics_attrs = {
        "tmax_rmse": float(rmse_by_lead_day.mean()),
        "hw_precision": float(detection["precision"].mean()),
        "hw_recall": float(detection["recall"].mean()),
        "hw_f1": float(detection["f1"].mean()),
    }
    print(f"  Lead days {list(map(int, lead_day_values))}: "
          f"Tmax RMSE {metrics_attrs['tmax_rmse']:.3f} K  detection F1 {metrics_attrs['hw_f1']:.3f} "
          f"(P {metrics_attrs['hw_precision']:.3f} R {metrics_attrs['hw_recall']:.3f})")
    return metrics_vars, metrics_attrs, {"model_tmax": model_tmax, "true_tmax": true_tmax}
