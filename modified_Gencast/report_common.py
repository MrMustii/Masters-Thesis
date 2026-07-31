import os
import numpy as np
import pandas as pd
import s3fs
import xarray as xr

from modified_Gencast.lumio_buckets import lumio, load, bucket_name, GENCAST_LEVELS
from modified_Gencast.metrics import (
    TARGET_VAR, spatial_errors, mean_forecast, temporal_autocorrelation,
    power_spectra_1d, mean_temporal_difference,
    crps, crps_field, spread_skill_ratio, rank_histogram, ensemble_power_spectra_1d,
    binary_classification_metrics, probabilistic_detection, rev,
    tmax_quantile_bias, tail_ratio, conditional_rank_histogram,
)
from modified_Gencast.plots import (
    plot_temp_lead_curve, plot_temporal_errors, plot_hour_of_day_curves,
    plot_spatial_error_maps,
    plot_temporal_autocorrelation, plot_power_spectra, plot_mean_temporal_difference,
    plot_crps_growth, plot_spread_skill, plot_rank_histogram, plot_ensemble_power_spectra,
)


def prepare_datasets(predictions_ds, targets_ds, region):
    """Squeeze batch dim, crop to region box, detach from dask."""
    # drop singleton dims
    for dim in list(predictions_ds.dims):
        if predictions_ds.sizes[dim] == 1:
            predictions_ds = predictions_ds.squeeze(dim, drop=True)
    for dim in list(targets_ds.dims):
        if targets_ds.sizes[dim] == 1:
            targets_ds = targets_ds.squeeze(dim, drop=True)

    # crop to the interior scoring box
    if region is not None:
        inner_lat = predictions_ds.lat.values[
            (predictions_ds.lat.values >= region[0]) & (predictions_ds.lat.values <= region[1])]
        inner_lon = predictions_ds.lon.values[
            (predictions_ds.lon.values >= region[2]) & (predictions_ds.lon.values <= region[3])]
        predictions_ds = predictions_ds.sel(lat=inner_lat, lon=inner_lon)
        targets_ds = targets_ds.sel(lat=inner_lat, lon=inner_lon)
        assert predictions_ds.sizes["lat"] > 0 and predictions_ds.sizes["lon"] > 0, \
            "Region crop produced an empty grid; check region/coord order."

    # keep only TARGET_VAR, materialise to numpy (day-block ops can't run on dask)
    predictions_ds = xr.Dataset(
        {TARGET_VAR: xr.DataArray(np.asarray(predictions_ds[TARGET_VAR].values),
                                  dims=predictions_ds[TARGET_VAR].dims,
                                  coords=predictions_ds[TARGET_VAR].coords)})
    targets_ds = xr.Dataset(
        {TARGET_VAR: xr.DataArray(np.asarray(targets_ds[TARGET_VAR].values),
                                  dims=targets_ds[TARGET_VAR].dims,
                                  coords=targets_ds[TARGET_VAR].coords)})
    return predictions_ds, targets_ds


def compute_common_metrics(predictions_ds, targets_ds, num_ensemble, base_attrs):
    """Metrics every experiment reports for 2m temperature; caller always runs prepare_datasets() first."""

    # resolution gate
    total_hours = int(predictions_ds[TARGET_VAR].sizes["hour"])
    hour_values = predictions_ds[TARGET_VAR].hour.values
    hour_spacing = int(np.median(np.diff(hour_values))) if total_hours > 1 else 24
    fine_time_resolution = hour_spacing <= 6 and total_hours >= 8

    # RMSE / bias
    temp_spatial = spatial_errors(predictions_ds[TARGET_VAR], targets_ds[TARGET_VAR])
    rmse_ts_temp = temp_spatial["rmse"]
    bias_ts_temp = temp_spatial["bias"]
    print(f"  2m-T RMSE: {float(rmse_ts_temp.mean()):.4f} K  bias: {float(bias_ts_temp.mean()):.4f} K")

    metrics_vars = {
        "rmse_map_temp": temp_spatial["rmse_map"], "bias_map_temp": temp_spatial["bias_map"],
        "rmse_ts_temp": rmse_ts_temp, "bias_ts_temp": bias_ts_temp,
    }
    metrics_attrs = dict(base_attrs)

    # mean diurnal curve
    pred_diurnal = mean_forecast(predictions_ds[TARGET_VAR])
    tgt_diurnal = mean_forecast(targets_ds[TARGET_VAR])
    metrics_vars.update({"mean_forecast_pred": pred_diurnal, "mean_forecast_tgt": tgt_diurnal})

    # autocorrelation + power spectra
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

    # frame-to-frame difference (dX)
    if total_hours >= 2:
        mean_temporal_diff_result = mean_temporal_difference(predictions_ds[TARGET_VAR], targets_ds[TARGET_VAR])
        print(f"  Frame diff — Model: {float(mean_temporal_diff_result['pred'].mean()):.4f} K  "
              f"ERA5: {float(mean_temporal_diff_result['target'].mean()):.4f} K")
        metrics_vars.update({
            "temporal_diff_pred": mean_temporal_diff_result["pred"],
            "temporal_diff_tgt": mean_temporal_diff_result["target"],
        })

    # p90 tail threshold at each (init, lead) valid date
    forecast_values = np.asarray(predictions_ds.forecast.values)
    hour_values_1d = np.asarray(hour_values)
    while forecast_values.ndim > 1 and forecast_values.shape[0] == 1:
        forecast_values = forecast_values.squeeze(axis=0)
    while hour_values_1d.ndim > 1 and hour_values_1d.shape[0] == 1:
        hour_values_1d = hour_values_1d.squeeze(axis=0)
    if forecast_values.ndim != 1:
        raise ValueError(f"forecast coordinate must be 1D, got shape {forecast_values.shape}")
    if hour_values_1d.ndim != 1:
        raise ValueError(f"hour coordinate must be 1D, got shape {hour_values_1d.shape}")

    forecast_index = pd.DatetimeIndex(forecast_values)
    valid_times = forecast_index.values[:, None] + hour_values_1d.astype(int)[None, :] * np.timedelta64(1, "h")
    valid_day_of_year = np.minimum(
        pd.DatetimeIndex(valid_times.ravel()).dayofyear.values.reshape(valid_times.shape), 365)
    day_of_year_da = xr.DataArray(valid_day_of_year, dims=["forecast", "hour"],
                                  coords={"forecast": predictions_ds.forecast, "hour": predictions_ds[TARGET_VAR].hour})
    tail_threshold = (load_p90_threshold().sel(dayofyear=day_of_year_da)
                      .interp(lat=predictions_ds.lat, lon=predictions_ds.lon))

    # CRPS + twCRPS
    crps_result = crps(predictions_ds[TARGET_VAR], targets_ds[TARGET_VAR])
    twcrps_result = crps(predictions_ds[TARGET_VAR], targets_ds[TARGET_VAR], threshold=tail_threshold)
    print(f"  Mean CRPS: {float(crps_result['crps'].mean()):.4f} K"
          f"  twCRPS: {float(twcrps_result['crps'].mean()):.4f} K")
    metrics_vars.update({
        "crps_map": crps_result["crps_map"], "crps_growth": crps_result["crps"],
        "crps_series": crps_result["crps_series"], "crps_day1_series": crps_result["crps_day1_series"],
        "twcrps_map": twcrps_result["crps_map"], "twcrps_growth": twcrps_result["crps"],
        "twcrps_series": twcrps_result["crps_series"], "twcrps_day1_series": twcrps_result["crps_day1_series"],
    })
    metrics_attrs["crps_mean"] = float(crps_result["crps"].mean())
    metrics_attrs["twcrps_mean"] = float(twcrps_result["crps"].mean())
    metrics_attrs["crps_mean_day1"] = float(crps_result["crps_day1_series"].mean())
    metrics_attrs["twcrps_mean_day1"] = float(twcrps_result["crps_day1_series"].mean())

    # ensemble calibration
    if num_ensemble > 1:
        spread_skill_result = spread_skill_ratio(predictions_ds[TARGET_VAR], targets_ds[TARGET_VAR])
        rank_hist = rank_histogram(predictions_ds[TARGET_VAR], targets_ds[TARGET_VAR])
        print(f"  Mean SSR:  {float(spread_skill_result['ratio'].mean()):.4f}")
        metrics_vars.update({
            "spread_growth": spread_skill_result["spread"], "skill_growth": spread_skill_result["skill"],
            "spread_skill_ratio": spread_skill_result["ratio"],
            "rank_histogram": xr.DataArray(rank_hist, dims=["rank"], coords={"rank": np.arange(len(rank_hist))}),
        })
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
    """Plots shared by every experiment. hour_axis "lead" (AR) or "hour_of_day" (daily); rmse_map_tmax adds a Tmax panel."""
    rmse_ts_temp = metrics_vars["rmse_ts_temp"]
    bias_ts_temp = metrics_vars["bias_ts_temp"]
    rmse_map_temp = metrics_vars["rmse_map_temp"]
    bias_map_temp = metrics_vars["bias_map_temp"]
    lat = rmse_map_temp.lat.values
    lon = rmse_map_temp.lon.values

    # lead / hour-of-day line plots
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

    # spatial maps
    plot_spatial_error_maps(rmse_map_temp.mean("hour"), bias_map_temp.mean("hour"), rmse_map_tmax,
                            lat, lon, plot_dir, tag, region=region)

    # autocorr / spectra / dX
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

    # ensemble diagnostics
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
    """Windowed p90 daily-Tmax threshold (±15-day, 1981-2010), built by build_tmax_threshold.py."""
    climatology_store = s3fs.S3Map(root="era5-lumi/tmax_p90_windowed", s3=lumio(), check=False)
    threshold_p90 = xr.open_zarr(climatology_store, consolidated=True)["tmax_p90"]
    if "latitude" in threshold_p90.dims:
        threshold_p90 = threshold_p90.rename({"latitude": "lat", "longitude": "lon"})
    if float(threshold_p90.lon.max()) > 180:
        threshold_p90 = threshold_p90.assign_coords(lon=(threshold_p90.lon + 180) % 360 - 180).sortby("lon")
    return threshold_p90


def score_daily_tmax(model_tmax, true_daily_tmax, threshold_p90, weights):
    """Score a single daily-Tmax field vs true Tmax + p90: RMSE/bias + single-day detection P/R/F1. Used by exp1."""
    # ensemble median + aligned truth/threshold
    if "ensemble" in model_tmax.dims:
        model_tmax = model_tmax.median("ensemble")
    model_days = pd.DatetimeIndex(model_tmax.day.values)
    true_tmax = (true_daily_tmax.sel(time=model_days).rename({"time": "day"})
                 .assign_coords(day=model_days).interp(lat=model_tmax.lat, lon=model_tmax.lon))
    day_of_year = xr.DataArray(model_days.dayofyear, dims="day", coords={"day": model_days})
    threshold = threshold_p90.sel(dayofyear=day_of_year).interp(lat=model_tmax.lat, lon=model_tmax.lon)

    # RMSE/bias + weighted detection counts
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
        "counts_series": detection["counts_series"],
    }


def heatwave_tail_metrics(tmax_members, true_tmax, threshold):
    """Tail + probabilistic heatwave diagnostics on a per-member daily-Tmax proxy. Shared by exp1/exp2."""
    # quantile bias + tail ratio (deterministic-safe)
    quantile_bias = tmax_quantile_bias(tmax_members, true_tmax)
    tail = tail_ratio(tmax_members, true_tmax)
    tail_vars = {"tmax_quantile_bias_map": quantile_bias["map"], "tail_ratio_map": tail["map"]}
    tail_attrs = {f"tmax_qbias_{name}": value for name, value in quantile_bias["scalars"].items()}
    tail_attrs.update({"tail_ratio": tail["ratio"], "tail_ratio_lower": tail["lower_gap_ratio"],
                       "tail_ratio_upper": tail["upper_gap_ratio"]})
    # Brier/BSS/AUC/reliability/REV/cond-rank-hist (ensemble only)
    if "ensemble" in tmax_members.dims:
        probabilistic = probabilistic_detection(tmax_members, true_tmax, threshold)
        reliability_bins = np.arange(len(probabilistic["reliability_forecast"]))
        tail_vars.update({
            "hw_brier_map": probabilistic["brier_map"],
            "reliability_forecast": xr.DataArray(probabilistic["reliability_forecast"],
                                                 dims=["probability_bin"], coords={"probability_bin": reliability_bins}),
            "reliability_observed": xr.DataArray(probabilistic["reliability_observed"],
                                                 dims=["probability_bin"], coords={"probability_bin": reliability_bins}),
            "reliability_count": xr.DataArray(probabilistic["reliability_count"],
                                              dims=["probability_bin"], coords={"probability_bin": reliability_bins}),
            "rev_curve": rev(tmax_members, true_tmax, threshold),
        })
        conditional_ranks = conditional_rank_histogram(tmax_members, true_tmax, threshold)
        tail_vars["conditional_rank_histogram"] = xr.DataArray(
            conditional_ranks, dims=["cond_rank"], coords={"cond_rank": np.arange(len(conditional_ranks))})
        tail_attrs.update({"hw_brier": probabilistic["brier"], "hw_brier_skill_score": probabilistic["brier_skill_score"],
                           "hw_auc": probabilistic["auc"], "hw_base_rate": probabilistic["base_rate"]})
    return tail_vars, tail_attrs


def compute_tmax_detection(temp_by_forecast_hour, true_daily_tmax, threshold_p90):
    """exp2 daily-Tmax proxy max(00Z,12Z) + single-day detection as a function of lead DAY. Returns vars, attrs, QQ arrays."""
    # regroup 12h leads into lead-days (keep days with both frames)
    temperature = temp_by_forecast_hour
    weights = xr.DataArray(np.cos(np.deg2rad(temperature.lat)), coords={"lat": temperature.lat})
    lead_hours = temperature.hour.values
    init_hours = pd.DatetimeIndex(temperature.forecast.values).hour

    model_tmax_groups = []
    for init_hour in np.unique(init_hours):
        group = temperature.isel(forecast=(init_hours == init_hour))
        lead_day = (int(init_hour) + lead_hours) // 24
        frames_per_day = pd.Series(lead_day).value_counts()
        full_days = sorted(frames_per_day[frames_per_day == 2].index.tolist())
        group = group.assign_coords(lead_day=("hour", lead_day))
        model_tmax_groups.append(group.groupby("lead_day").max().sel(lead_day=full_days))
    common_lead_days = sorted(set.intersection(*[set(g.lead_day.values.tolist()) for g in model_tmax_groups]))
    model_tmax_members = xr.concat([g.sel(lead_day=common_lead_days) for g in model_tmax_groups], dim="forecast")
    model_tmax = model_tmax_members.median("ensemble") if "ensemble" in model_tmax_members.dims else model_tmax_members

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

    # Tmax-twCRPS (RQ1 primary ranking score)
    clipped_members = np.maximum(model_tmax_members, threshold)
    clipped_true = np.maximum(true_tmax, threshold)
    tmax_twcrps_field = crps_field(clipped_members, clipped_true)  # (forecast, lead_day, lat, lon)
    tmax_twcrps_series = tmax_twcrps_field.weighted(weights).mean(["lat", "lon", "lead_day"]).reset_coords(drop=True)
    tmax_twcrps_mean = float(tmax_twcrps_field.weighted(weights).mean(["lat", "lon", "lead_day", "forecast"]))

    # day-1 slice keyed by valid date (pairs against exp1)
    day1_lead = int(lead_day_values.min())
    day1_field = tmax_twcrps_field.sel(lead_day=day1_lead)  # (forecast, lat, lon)
    day1_valid = pd.DatetimeIndex(model_tmax.forecast.values).floor("D") + pd.Timedelta(days=day1_lead)
    tmax_twcrps_day1_series = (day1_field.weighted(weights).mean(["lat", "lon"])
                               .reset_coords(drop=True).assign_coords(forecast=day1_valid.values))
    tmax_twcrps_day1_mean = float(day1_field.weighted(weights).mean(["lat", "lon", "forecast"]))

    # Tmax RMSE/bias vs lead day
    diff = model_tmax - true_tmax
    rmse_by_lead_day = np.sqrt((diff ** 2).weighted(weights).mean(["lat", "lon", "forecast"]))
    bias_map = diff.mean(["forecast", "lead_day"])
    rmse_map = np.sqrt((diff ** 2).mean(["forecast", "lead_day"]))

    # detection P/R/F1 + tail metrics
    model_exceed = (model_tmax > threshold).rename({"lead_day": "day_idx"})
    true_exceed = (true_tmax > threshold).rename({"lead_day": "day_idx"})
    detection = binary_classification_metrics(model_exceed, true_exceed)
    tail_vars, tail_attrs = heatwave_tail_metrics(
        model_tmax_members.rename({"lead_day": "day_idx"}),
        true_tmax.rename({"lead_day": "day_idx"}),
        threshold.rename({"lead_day": "day_idx"}))

    metrics_vars = {
        "tmax_rmse_by_lead_day": rmse_by_lead_day, "tmax_rmse_map": rmse_map, "tmax_bias_map": bias_map,
        "hw_precision_by_lead_day": detection["precision"], "hw_recall_by_lead_day": detection["recall"],
        "hw_f1_by_lead_day": detection["f1"],
        "hw_precision_map": detection["precision_map"], "hw_recall_map": detection["recall_map"],
        "hw_f1_map": detection["f1_map"],
        "hw_tp_series": detection["counts_series"]["tp"], "hw_fp_series": detection["counts_series"]["fp"],
        "hw_fn_series": detection["counts_series"]["fn"], "hw_tn_series": detection["counts_series"]["tn"],
        "tmax_twcrps_series": tmax_twcrps_series,
        "tmax_twcrps_day1_series": tmax_twcrps_day1_series,
    }
    metrics_attrs = {
        "tmax_rmse": float(rmse_by_lead_day.mean()),
        "tmax_twcrps": tmax_twcrps_mean,
        "tmax_twcrps_day1": tmax_twcrps_day1_mean,
        "hw_precision": float(detection["precision"].mean()),
        "hw_recall": float(detection["recall"].mean()),
        "hw_f1": float(detection["f1"].mean()),
    }
    metrics_vars.update(tail_vars)
    metrics_attrs.update(tail_attrs)
    print(f"  Lead days {list(map(int, lead_day_values))}: "
          f"Tmax RMSE {metrics_attrs['tmax_rmse']:.3f} K  Tmax-twCRPS {tmax_twcrps_mean:.3f} K  "
          f"detection F1 {metrics_attrs['hw_f1']:.3f} "
          f"(P {metrics_attrs['hw_precision']:.3f} R {metrics_attrs['hw_recall']:.3f})"
          + (f"  Brier {metrics_attrs['hw_brier']:.4f}  AUC {metrics_attrs['hw_auc']:.3f}"
             if "hw_brier" in metrics_attrs else "")
          + f"  tail-ratio {metrics_attrs['tail_ratio']:.3f}  q99-bias {metrics_attrs['tmax_qbias_q99']:+.3f} K")
    return metrics_vars, metrics_attrs, {"model_tmax": model_tmax, "true_tmax": true_tmax}
