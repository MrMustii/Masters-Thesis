import os

import numpy as np
import pandas as pd
import xarray as xr

from modified_Gencast.metrics import (
    TARGET_VAR, tmax_grid, get_heatwaves, binary_classification_metrics, rev,
    spell_length_distribution, heatwave_ks_statistic, event_bias,
    hours_above_threshold_error, conditional_tmax_errors, tmax_quantile_bias,
    diurnal_shape_errors, peak_time_error, rmse_by_window_position,
)
from modified_Gencast.plots import (
    plot_heatwave_metrics, plot_metric_map, plot_qq_tmax, plot_spell_length_histogram,
    plot_tmax_bias_map, plot_peak_hour_heatmap, plot_rev, plot_rmse_by_window_position,
    plot_single_day_profile, plot_weekly_temp, plot_dtr_error, plot_dtr_scatter,
    plot_diurnal_asymmetry,
)
from modified_Gencast.helper_fun import save_to_s3
from modified_Gencast.report_common import (
    prepare_datasets, compute_common_metrics, plot_common, load_p90_threshold,
)

MIN_HEATWAVE_DAYS = 3


def run_exp3_report(predictions_ds, targets_ds, region, num_ensemble, start_date, run_id,
                    out_dir, tag, no_s3, model_path="", autoregressive_steps=0, ar_window_hours=None):
    print("\nComputing metrics...")
    predictions_ds, targets_ds = prepare_datasets(predictions_ds, targets_ds, region)
    total_hours = int(predictions_ds[TARGET_VAR].sizes["hour"])
    is_single_shot = (ar_window_hours is None) or (ar_window_hours >= total_hours)

    base_attrs = {
        "checkpoint": model_path, "run_id": run_id,
        "autoregressive_steps": autoregressive_steps,
        "ar_window_hours": ar_window_hours if ar_window_hours is not None else 0,
    }
    metrics_vars, metrics_attrs = compute_common_metrics(predictions_ds, targets_ds, num_ensemble, base_attrs)

    pred_tmax = tmax_grid(predictions_ds)
    tgt_tmax = tmax_grid(targets_ds)
    num_days = int(tgt_tmax.sizes["day_idx"])
    has_full_days = num_days >= 1
    if not has_full_days:
        print(f"  Skipping all day-block metrics: no full 24-h day blocks.")

    pred_tmax_median = None
    rmse_map_tmax = None
    if has_full_days:
        pred_tmax_median = pred_tmax.median("ensemble") if "ensemble" in pred_tmax.dims else pred_tmax
        diff_tmax = pred_tmax_median - tgt_tmax
        w_tmax = xr.DataArray(np.cos(np.deg2rad(tgt_tmax.lat)), coords={"lat": tgt_tmax.lat})
        rmse_map_tmax = np.sqrt((diff_tmax ** 2).mean("forecast"))
        bias_map_tmax = diff_tmax.mean("forecast")
        rmse_tmax = float(np.sqrt((diff_tmax ** 2).weighted(w_tmax).mean(["lat", "lon", "day_idx", "forecast"])))
        print(f"  Tmax RMSE (member median): {rmse_tmax:.4f} K")
        metrics_vars.update({
            "rmse_map_tmax": rmse_map_tmax, "bias_map_tmax": bias_map_tmax,
            "pred_tmax": pred_tmax, "tgt_tmax": tgt_tmax,
        })
        metrics_attrs["rmse_tmax"] = rmse_tmax

    heatwave_days = num_days
    run_heatwave = has_full_days and heatwave_days >= 2 * MIN_HEATWAVE_DAYS
    if not run_heatwave:
        print(f"  Skipping heatwave/REV: {heatwave_days} consecutive days, need >= {2 * MIN_HEATWAVE_DAYS}.")
    else:
        tmax_p90 = load_p90_threshold()
        forecast_dt = pd.DatetimeIndex(tgt_tmax.forecast.values)
        day_idx_vals = tgt_tmax.day_idx.values.astype(int)
        actual_doy = np.array([
            pd.DatetimeIndex([t + pd.Timedelta(days=int(d)) for d in day_idx_vals]).dayofyear
            for t in forecast_dt])
        doy_da = xr.DataArray(actual_doy, dims=["forecast", "day_idx"],
                              coords={"forecast": tgt_tmax.forecast, "day_idx": tgt_tmax.day_idx})
        threshold = tmax_p90.sel(dayofyear=doy_da)
        threshold_hourly = threshold
        assert not bool(threshold.isnull().all()), \
            "p90 threshold is all-NaN after coordinate alignment; check tmax_p90 grid vs predictions grid."

        pred_hw = get_heatwaves(pred_tmax_median, threshold)
        tgt_hw = get_heatwaves(tgt_tmax, threshold)
        hw_metrics = binary_classification_metrics(pred_hw, tgt_hw)
        spell_pred = spell_length_distribution(pred_hw)
        spell_tgt = spell_length_distribution(tgt_hw)
        spell_ks = heatwave_ks_statistic(spell_pred["lengths"], spell_tgt["lengths"])
        event_result = event_bias(pred_hw, tgt_hw)
        hours_above_result = hours_above_threshold_error(
            predictions_ds[TARGET_VAR], targets_ds[TARGET_VAR], threshold_hourly)
        conditional_result = conditional_tmax_errors(pred_tmax_median, tgt_tmax, tgt_hw)
        quantile_result = tmax_quantile_bias(pred_tmax, tgt_tmax)
        print(f"  HW F1 (mean): {float(hw_metrics['f1'].mean()):.4f}  spell KS: {spell_ks:.4f}")
        print(f"  Event frequency bias: {event_result['frequency_bias']:.3f}  "
              f"duration bias: {event_result['duration_bias']:.3f} days")
        print(f"  Hours-above-threshold error: {hours_above_result['scalar']:.3f} h/day")
        print(f"  Conditional Tmax bias: {conditional_result['bias']:+.3f} K  RMSE: {conditional_result['rmse']:.3f} K")
        print("  Tmax quantile bias: " + "  ".join(
            f"{name}: {value:+.3f} K" for name, value in quantile_result["scalars"].items()))
        metrics_vars.update({
            "hw_precision": hw_metrics["precision"], "hw_recall": hw_metrics["recall"], "hw_f1": hw_metrics["f1"],
            "hw_precision_map": hw_metrics["precision_map"], "hw_recall_map": hw_metrics["recall_map"],
            "hw_f1_map": hw_metrics["f1_map"],
            "hw_tp_series": hw_metrics["counts_series"]["tp"], "hw_fp_series": hw_metrics["counts_series"]["fp"],
            "hw_fn_series": hw_metrics["counts_series"]["fn"], "hw_tn_series": hw_metrics["counts_series"]["tn"],
            "spell_ks_statistic": xr.DataArray([spell_ks]),
            "spell_lengths_pred": xr.DataArray(spell_pred["lengths"], dims=["spell_pred"]),
            "spell_lengths_tgt": xr.DataArray(spell_tgt["lengths"], dims=["spell_tgt"]),
            "event_frequency_bias_map": event_result["frequency_bias_map"],
            "event_duration_bias_map": event_result["duration_bias_map"],
            "hours_above_threshold_error_map": hours_above_result["map"],
            "hours_above_threshold_error_series": hours_above_result["series"],
            "conditional_tmax_bias_map": conditional_result["bias_map"],
            "conditional_tmax_bias_series": conditional_result["bias_series"],
            "tmax_quantile_bias_map": quantile_result["map"],
        })
        metrics_attrs.update({
            "hw_f1_mean": float(hw_metrics["f1"].mean()), "spell_ks": spell_ks,
            "event_frequency_bias": event_result["frequency_bias"],
            "event_duration_bias": event_result["duration_bias"],
            "hours_above_threshold_error": hours_above_result["scalar"],
            "conditional_tmax_bias": conditional_result["bias"], "conditional_tmax_rmse": conditional_result["rmse"],
            **{f"tmax_quantile_bias_{name}": value for name, value in quantile_result["scalars"].items()},
        })

    diurnal_shape = None
    peak_timing = None
    if has_full_days:
        diurnal_shape = diurnal_shape_errors(predictions_ds[TARGET_VAR], targets_ds[TARGET_VAR])
        print(f"  DTR MAE (per member): {diurnal_shape['dtr_scalar']:.4f} K")
        print(f"  Diurnal asymmetry MAE: {diurnal_shape['asymmetry_scalar']:.4f}")
        metrics_vars.update({
            "dtr_error_curve": diurnal_shape["dtr_curve"],
            "diurnal_asymmetry_curve": diurnal_shape["asymmetry_curve"],
            "diurnal_asymmetry_map": diurnal_shape["asymmetry_map"],
        })
        metrics_attrs["dtr_mae"] = diurnal_shape["dtr_scalar"]
        metrics_attrs["diurnal_asymmetry_mae"] = diurnal_shape["asymmetry_scalar"]

        hour_spacing = int(np.median(np.diff(predictions_ds[TARGET_VAR].hour.values))) if total_hours > 1 else 24
        if hour_spacing > 3:
            print(f"  Skipping peak-time error: {hour_spacing}h spacing > 3h.")
        else:
            peak_timing = peak_time_error(predictions_ds[TARGET_VAR], targets_ds[TARGET_VAR])
            print(f"  Peak-time error: {peak_timing['scalar']:.2f} h")
            metrics_vars.update({
                "peak_time_error_map": peak_timing["map"], "peak_time_bias_map": peak_timing["bias_map"],
                "peak_time_error_series": peak_timing["series"],
                "peak_hour_heatmap": peak_timing["peak_hour_heatmap"],
            })
            metrics_attrs["peak_time_error"] = peak_timing["scalar"]

    window_rmse = None
    if (ar_window_hours is not None) and (not is_single_shot):
        window_rmse = rmse_by_window_position(predictions_ds[TARGET_VAR], targets_ds[TARGET_VAR], ar_window_hours)
        metrics_vars["rmse_by_window_position"] = window_rmse
        print(f"  RMSE window-position peak: {float(window_rmse.max()):.4f} K at pos {int(window_rmse.argmax().values)}")

    rev_result = None
    if num_ensemble > 1 and run_heatwave:
        rev_result = rev(pred_tmax, tgt_tmax, threshold)
        print(f"  Max REV:   {float(rev_result.max()):.4f}")
        metrics_vars["rev_curve"] = rev_result
        metrics_attrs["rev_max"] = float(rev_result.max())

    metrics_ds = xr.Dataset(metrics_vars, attrs=metrics_attrs)
    metrics_path = os.path.join(out_dir, f"metrics_{tag}.nc")
    metrics_ds.to_netcdf(metrics_path)
    print(f"Metrics saved → {metrics_path}")
    if not no_s3:
        save_to_s3(metrics_path, run_id)

    plot_dir = os.path.join(os.getcwd(), "figs", run_id)
    os.makedirs(plot_dir, exist_ok=True)
    print(f"\nGenerating plots → {plot_dir}")
    plot_common(metrics_vars, predictions_ds, targets_ds, plot_dir, tag, region, num_ensemble,
                ar_window_hours, hour_axis="lead",
                rmse_map_tmax=rmse_map_tmax.mean("day_idx") if has_full_days else None)

    lat = predictions_ds.lat.values
    lon = predictions_ds.lon.values
    if run_heatwave:
        hw_lat = hw_metrics["precision_map"].lat.values
        hw_lon = hw_metrics["precision_map"].lon.values
        plot_heatwave_metrics(hw_metrics["precision"], hw_metrics["recall"], hw_metrics["f1"],
                              hw_metrics["precision_map"], hw_metrics["recall_map"], hw_metrics["f1_map"],
                              hw_lat, hw_lon, plot_dir, tag, region=region)
        plot_metric_map(event_result["frequency_bias_map"], hw_lat, hw_lon,
                        "Heatwave event frequency bias (N_pred / N_obs)", "Frequency ratio",
                        os.path.join(plot_dir, f"event_frequency_bias_{tag}.png"), region=region, center=1.0)
        plot_metric_map(event_result["duration_bias_map"], hw_lat, hw_lon,
                        "Heatwave event duration bias (pred − obs)", "Days",
                        os.path.join(plot_dir, f"event_duration_bias_{tag}.png"), region=region, center=0.0)
        plot_metric_map(hours_above_result["map"], hw_lat, hw_lon,
                        "Hours above threshold per day — absolute error", "Hours",
                        os.path.join(plot_dir, f"hours_above_threshold_{tag}.png"), region=region)
        plot_metric_map(conditional_result["bias_map"], hw_lat, hw_lon,
                        "Tmax bias on observed heatwave days", "Bias (K)",
                        os.path.join(plot_dir, f"conditional_tmax_bias_{tag}.png"), region=region, center=0.0)
        plot_qq_tmax(pred_tmax, tgt_tmax, os.path.join(plot_dir, f"qq_tmax_{tag}.png"))
        plot_spell_length_histogram(spell_pred["lengths"], spell_tgt["lengths"],
                                    os.path.join(plot_dir, f"spell_length_{tag}.png"))
        if rev_result is not None:
            plot_rev(rev_result, os.path.join(plot_dir, f"rev_{tag}.png"))

    if has_full_days:
        plot_tmax_bias_map(bias_map_tmax, lat, lon, plot_dir, tag, region=region)
    if peak_timing is not None:
        plot_metric_map(peak_timing["bias_map"], lat, lon, "Peak-time bias (pred − obs)", "Hours",
                        os.path.join(plot_dir, f"peak_time_bias_{tag}.png"), region=region, center=0.0)
        plot_peak_hour_heatmap(peak_timing["peak_hour_heatmap"], os.path.join(plot_dir, f"peak_hour_heatmap_{tag}.png"))
    if window_rmse is not None:
        plot_rmse_by_window_position(window_rmse, os.path.join(plot_dir, f"rmse_window_position_{tag}.png"))

    plot_day = str(pd.Timestamp(start_date).date() + pd.Timedelta(days=1))
    plot_single_day_profile(predictions_ds, targets_ds, plot_day, os.path.join(plot_dir, f"day_profile_{tag}_{plot_day}.png"))
    week_start = str(pd.DatetimeIndex(predictions_ds.forecast.values)[len(predictions_ds.forecast) // 2].date())
    plot_weekly_temp(predictions_ds, targets_ds, week_start, os.path.join(plot_dir, f"weekly_temp_{tag}_{week_start}.png"))

    if diurnal_shape is not None:
        plot_dtr_error(diurnal_shape["dtr_curve"], os.path.join(plot_dir, f"dtr_error_{tag}.png"))
        plot_dtr_scatter(diurnal_shape["pred_dtr"], diurnal_shape["tgt_dtr"], os.path.join(plot_dir, f"dtr_scatter_{tag}.png"))
        plot_diurnal_asymmetry(diurnal_shape["asymmetry_curve"], os.path.join(plot_dir, f"diurnal_asymmetry_{tag}.png"))

    print(f"\nDone. All outputs in {out_dir}")
