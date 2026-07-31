import os

import numpy as np
import pandas as pd
import xarray as xr

from modified_Gencast.metrics import (
    TARGET_VAR, tmax_grid, get_heatwaves, binary_classification_metrics, rev,
    spell_length_distribution, heatwave_ks_statistic, event_bias, event_detection_rate,
    conditional_tmax_errors, tmax_quantile_bias, crps_field, cos_weights,
    diurnal_shape_errors, rmse_by_window_position,
    variogram_score, peak_time_ccrps, tail_ratio, conditional_rank_histogram,
    probabilistic_detection, over_smoothing_diagnostic,
)
from modified_Gencast.plots import (
    plot_heatwave_metrics, plot_metric_map, plot_qq_tmax, plot_spell_length_histogram,
    plot_tmax_bias_map, plot_rev, plot_rmse_by_window_position,
    plot_single_day_profile, plot_weekly_temp, plot_dtr_error,
    plot_diurnal_asymmetry, plot_rank_histogram, plot_reliability_diagram, plot_over_smoothing,
)
from modified_Gencast.helper_fun import save_to_s3
from modified_Gencast.report_common import (
    prepare_datasets, compute_common_metrics, plot_common, load_p90_threshold,
)

MIN_HEATWAVE_DAYS = 3


def run_diurnal_report(predictions_ds, targets_ds, region, num_ensemble, start_date, run_id,
                       out_dir, tag, no_s3, layout="rollout", model_path="",
                       autoregressive_steps=0, ar_window_hours=None):
    """RQ2 hourly report for exp3 (layout="rollout") and exp4 (layout="daily"); layout only changes the day-block axis, peak-time gating, window RMSE, and which line plots draw."""
    assert layout in ("rollout", "daily"), f"unknown layout {layout!r}"
    is_rollout = layout == "rollout"
    hour_axis = "lead" if is_rollout else "hour_of_day"

    # common metrics
    print("\nComputing metrics...")
    predictions_ds, targets_ds = prepare_datasets(predictions_ds, targets_ds, region)
    total_hours = int(predictions_ds[TARGET_VAR].sizes["hour"])
    is_single_shot = (ar_window_hours is None) or (ar_window_hours >= total_hours)
    base_attrs = {
        "checkpoint": model_path, "run_id": run_id,
        "autoregressive_steps": autoregressive_steps,
        "ar_window_hours": ar_window_hours if (is_rollout and ar_window_hours is not None) else 0,
    }
    metrics_vars, metrics_attrs = compute_common_metrics(predictions_ds, targets_ds, num_ensemble, base_attrs)

    # daily Tmax RMSE / bias
    prediction_tmax = tmax_grid(predictions_ds).compute()
    target_tmax = tmax_grid(targets_ds).compute()
    num_days = int(target_tmax.sizes["day_idx"])
    has_full_days = num_days >= 1
    if not has_full_days:
        print("  Skipping all day-block metrics: no full 24-h day blocks.")

    prediction_tmax_median = None
    rmse_map_tmax = None
    bias_map_tmax = None
    tmax_weights = None
    detection_result = None
    tail_rank_histogram = None
    if has_full_days:
        prediction_tmax_median = prediction_tmax.median("ensemble") if "ensemble" in prediction_tmax.dims else prediction_tmax
        tmax_diff = prediction_tmax_median - target_tmax
        tmax_weights = xr.DataArray(np.cos(np.deg2rad(target_tmax.lat)), coords={"lat": target_tmax.lat})
        rmse_map_tmax = np.sqrt((tmax_diff ** 2).mean("forecast"))
        bias_map_tmax = tmax_diff.mean("forecast")
        rmse_tmax = float(np.sqrt((tmax_diff ** 2).weighted(tmax_weights).mean(["lat", "lon", "day_idx", "forecast"])))
        print(f"  Tmax RMSE (member median): {rmse_tmax:.4f} K")
        metrics_vars.update({
            "rmse_map_tmax": rmse_map_tmax, "bias_map_tmax": bias_map_tmax,
            "pred_tmax": prediction_tmax, "tgt_tmax": target_tmax,
        })
        metrics_attrs["rmse_tmax"] = rmse_tmax

    # heatwave block (needs >= 6 consecutive days)
    heatwave_metrics = None
    event_result = None
    conditional_result = None
    spell_prediction = None
    spell_target = None
    if is_rollout:
        heatwave_days = num_days
    else:
        forecast_gaps = ((np.diff(predictions_ds.forecast.values) / np.timedelta64(1, "D"))
                         if predictions_ds.sizes["forecast"] > 1 else np.array([]))
        heatwave_days = int(predictions_ds.sizes["forecast"]) if bool(np.all(forecast_gaps == 1)) else 0
    run_heatwave = has_full_days and heatwave_days >= 2 * MIN_HEATWAVE_DAYS
    if not run_heatwave:
        print(f"  Skipping heatwave/REV: {heatwave_days} consecutive days, need >= {2 * MIN_HEATWAVE_DAYS}.")
    else:
        # threshold + day axis per layout
        tmax_p90 = load_p90_threshold()
        if is_rollout:
            forecast_datetimes = pd.DatetimeIndex(target_tmax.forecast.values)
            day_index_values = target_tmax.day_idx.values.astype(int)
            actual_day_of_year = np.array([
                pd.DatetimeIndex([t + pd.Timedelta(days=int(d)) for d in day_index_values]).dayofyear
                for t in forecast_datetimes])
            day_of_year = xr.DataArray(actual_day_of_year, dims=["forecast", "day_idx"],
                                       coords={"forecast": target_tmax.forecast, "day_idx": target_tmax.day_idx})
            threshold = tmax_p90.sel(dayofyear=day_of_year)
            detection_threshold = threshold
            prediction_tmax_heatwave = prediction_tmax_median
            target_tmax_heatwave = target_tmax
            prediction_tmax_pool = prediction_tmax
            target_tmax_pool = target_tmax
        else:
            day_dates = pd.DatetimeIndex(predictions_ds.forecast.values)
            first_day = predictions_ds.forecast.values[0]
            day_index_values = np.arange(heatwave_days)
            prediction_tmax_days = (prediction_tmax.squeeze("day_idx", drop=True).rename({"forecast": "day_idx"})
                                    .assign_coords(day_idx=day_index_values).expand_dims({"forecast": [first_day]}))
            target_tmax_days = (target_tmax.squeeze("day_idx", drop=True).rename({"forecast": "day_idx"})
                                .assign_coords(day_idx=day_index_values).expand_dims({"forecast": [first_day]}))
            day_of_year = xr.DataArray(day_dates.dayofyear.values[None, :], dims=["forecast", "day_idx"],
                                       coords={"forecast": [first_day], "day_idx": day_index_values})
            hourly_day_of_year = xr.DataArray(day_dates.dayofyear.values[:, None], dims=["forecast", "day_idx"],
                                              coords={"forecast": predictions_ds.forecast, "day_idx": [0]})
            threshold = tmax_p90.sel(dayofyear=day_of_year)
            detection_threshold = tmax_p90.sel(dayofyear=hourly_day_of_year)
            prediction_tmax_days_median = prediction_tmax_days.median("ensemble") if "ensemble" in prediction_tmax_days.dims else prediction_tmax_days
            prediction_tmax_heatwave = prediction_tmax_days_median
            target_tmax_heatwave = target_tmax_days
            prediction_tmax_pool = prediction_tmax_days
            target_tmax_pool = target_tmax_days
        assert not bool(threshold.isnull().all()), \
            "p90 threshold is all-NaN after coordinate alignment; check tmax_p90 grid vs predictions grid."

        # detection / spells / events / tail
        prediction_heatwave = get_heatwaves(prediction_tmax_heatwave, threshold)
        target_heatwave = get_heatwaves(target_tmax_heatwave, threshold)
        heatwave_metrics = binary_classification_metrics(prediction_heatwave, target_heatwave)
        spell_prediction = spell_length_distribution(prediction_heatwave)
        spell_target = spell_length_distribution(target_heatwave)
        spell_ks = heatwave_ks_statistic(spell_prediction["lengths"], spell_target["lengths"])
        event_result = event_bias(prediction_heatwave, target_heatwave)
        detection_rate_result = event_detection_rate(prediction_heatwave, target_heatwave)
        conditional_result = conditional_tmax_errors(prediction_tmax_heatwave, target_tmax_heatwave, target_heatwave)
        quantile_result = tmax_quantile_bias(prediction_tmax_pool, target_tmax_pool)
        tmax_twcrps_field = crps_field(np.maximum(prediction_tmax_pool, threshold), np.maximum(target_tmax_pool, threshold))
        pool_weights = cos_weights(target_tmax_pool)
        tmax_twcrps_mean = float(tmax_twcrps_field.weighted(pool_weights).mean(["lat", "lon", "day_idx", "forecast"]))
        # rollout: one value per forecast; daily: the pseudo-forecast is size 1, so re-key day_idx onto the real per-day dates
        if is_rollout:
            tmax_twcrps_series = tmax_twcrps_field.weighted(pool_weights).mean(["lat", "lon", "day_idx"]).reset_coords(drop=True)
        else:
            tmax_twcrps_series = (tmax_twcrps_field.weighted(pool_weights).mean(["lat", "lon"]).squeeze("forecast", drop=True)
                                  .rename({"day_idx": "forecast"}).assign_coords(forecast=day_dates))
        print(f"  HW F1 (mean): {float(heatwave_metrics['f1'].mean()):.4f}  spell KS: {spell_ks:.4f}  "
              f"event detection rate: {detection_rate_result['rate']:.3f} ({detection_rate_result['num_events']} events)")
        print(f"  Event frequency bias: {event_result['frequency_bias']:.3f}  "
              f"duration bias: {event_result['duration_bias']:.3f} days")
        print(f"  Conditional Tmax bias: {conditional_result['bias']:+.3f} K  RMSE: {conditional_result['rmse']:.3f} K")
        print(f"  Tmax-twCRPS: {tmax_twcrps_mean:.3f} K")
        print("  Tmax quantile bias: " + "  ".join(
            f"{name}: {value:+.3f} K" for name, value in quantile_result["scalars"].items()))
        metrics_vars.update({
            "hw_precision": heatwave_metrics["precision"], "hw_recall": heatwave_metrics["recall"], "hw_f1": heatwave_metrics["f1"],
            "hw_precision_map": heatwave_metrics["precision_map"], "hw_recall_map": heatwave_metrics["recall_map"],
            "hw_f1_map": heatwave_metrics["f1_map"],
            "spell_ks_statistic": xr.DataArray([spell_ks]),
            "spell_lengths_pred": xr.DataArray(spell_prediction["lengths"], dims=["spell_pred"]),
            "spell_lengths_tgt": xr.DataArray(spell_target["lengths"], dims=["spell_tgt"]),
            "event_frequency_bias_map": event_result["frequency_bias_map"],
            "event_duration_bias_map": event_result["duration_bias_map"],
            "event_detection_rate_map": detection_rate_result["map"],
            "conditional_tmax_bias_map": conditional_result["bias_map"],
            "tmax_quantile_bias_map": quantile_result["map"],
            "tmax_twcrps_series": tmax_twcrps_series,
        })
        metrics_attrs.update({
            "hw_f1_mean": float(heatwave_metrics["f1"].mean()), "spell_ks": spell_ks,
            "event_frequency_bias": event_result["frequency_bias"],
            "event_duration_bias": event_result["duration_bias"],
            "event_detection_rate": detection_rate_result["rate"],
            "event_detection_rate_num_events": detection_rate_result["num_events"],
            "conditional_tmax_bias": conditional_result["bias"], "conditional_tmax_rmse": conditional_result["rmse"],
            "tmax_twcrps": tmax_twcrps_mean,
            **{f"tmax_quantile_bias_{name}": value for name, value in quantile_result["scalars"].items()},
        })

        # Tmax-twCRPS day-1 slice (ranking; full week is plots-only)
        if is_rollout:
            first_day_idx = int(tmax_twcrps_field.day_idx.min())
            tmax_twcrps_day1 = tmax_twcrps_field.sel(day_idx=first_day_idx).weighted(pool_weights).mean(["lat", "lon"]).reset_coords(drop=True)
            metrics_vars["tmax_twcrps_day1_series"] = tmax_twcrps_day1
            metrics_attrs["tmax_twcrps_day1"] = float(tmax_twcrps_day1.mean())

        # per-forecast confusion counts (block-bootstrap inputs)
        if is_rollout:
            metrics_vars.update({
                "hw_tp_series": heatwave_metrics["counts_series"]["tp"], "hw_fp_series": heatwave_metrics["counts_series"]["fp"],
                "hw_fn_series": heatwave_metrics["counts_series"]["fn"], "hw_tn_series": heatwave_metrics["counts_series"]["tn"],
                "conditional_tmax_bias_series": conditional_result["bias_series"],
            })
        else:
            for count_name, count_mask in (("tp", prediction_heatwave & target_heatwave), ("fp", prediction_heatwave & ~target_heatwave),
                                           ("fn", ~prediction_heatwave & target_heatwave), ("tn", ~prediction_heatwave & ~target_heatwave)):
                day_counts = count_mask.astype(float).weighted(tmax_weights).sum(["lat", "lon"]).squeeze("forecast", drop=True)
                metrics_vars[f"hw_{count_name}_series"] = (
                    day_counts.rename({"day_idx": "forecast"}).assign_coords(forecast=day_dates))
            metrics_vars["conditional_tmax_bias_series"] = (
                (prediction_tmax_heatwave - target_tmax_heatwave).where(target_heatwave)
                .weighted(tmax_weights).mean(["lat", "lon"]).squeeze("forecast", drop=True)
                .rename({"day_idx": "forecast"}).assign_coords(forecast=day_dates))

        # probabilistic detection (ensemble only)
        if num_ensemble > 1:
            tail_rank_histogram = conditional_rank_histogram(prediction_tmax, target_tmax, detection_threshold)
            detection_result = probabilistic_detection(prediction_tmax, target_tmax, detection_threshold)
            print(f"  Brier: {detection_result['brier']:.4f}  BSS: {detection_result['brier_skill_score']:+.3f}  "
                  f"AUC: {detection_result['auc']:.3f}")
            probability_bins = detection_result["reliability_forecast"]
            metrics_vars.update({
                "conditional_rank_histogram": xr.DataArray(
                    tail_rank_histogram, dims=["rank"], coords={"rank": np.arange(len(tail_rank_histogram))}),
                "brier_map": detection_result["brier_map"],
                "reliability_observed": xr.DataArray(detection_result["reliability_observed"],
                                                     dims=["probability_bin"], coords={"probability_bin": probability_bins}),
                "reliability_count": xr.DataArray(detection_result["reliability_count"],
                                                  dims=["probability_bin"], coords={"probability_bin": probability_bins}),
            })
            metrics_attrs.update({
                "brier": detection_result["brier"], "brier_skill_score": detection_result["brier_skill_score"],
                "detection_auc": detection_result["auc"],
            })

    # diurnal shape (DTR / asymmetry / variogram / tail)
    diurnal_shape = None
    peak_time_ccrps_result = None
    over_smoothing_result = None
    tail_ratio_result = None
    if has_full_days:
        diurnal_shape = diurnal_shape_errors(predictions_ds[TARGET_VAR], targets_ds[TARGET_VAR])
        variogram_result = variogram_score(predictions_ds[TARGET_VAR], targets_ds[TARGET_VAR])
        print(f"  DTR MAE (per member): {diurnal_shape['dtr_scalar']:.4f} K  "
              f"DTR-CRPS: {diurnal_shape['dtr_crps_scalar']:.4f} K")
        print(f"  Variogram score: {variogram_result['scalar']:.4f}")
        print(f"  Diurnal asymmetry MAE: {diurnal_shape['asymmetry_scalar']:.4f}  "
              f"ASY-CRPS: {diurnal_shape['asymmetry_crps_scalar']:.4f}")
        metrics_vars.update({
            "dtr_error_curve": diurnal_shape["dtr_curve"],
            "diurnal_asymmetry_curve": diurnal_shape["asymmetry_curve"],
            "diurnal_asymmetry_map": diurnal_shape["asymmetry_map"],
            "variogram_map": variogram_result["map"], "variogram_series": variogram_result["series"],
            "dtr_crps_map": diurnal_shape["dtr_crps_map"], "dtr_crps_series": diurnal_shape["dtr_crps_series"],
            "asymmetry_crps_map": diurnal_shape["asymmetry_crps_map"],
            "asymmetry_crps_series": diurnal_shape["asymmetry_crps_series"],
        })
        metrics_attrs["dtr_mae"] = diurnal_shape["dtr_scalar"]
        metrics_attrs["diurnal_asymmetry_mae"] = diurnal_shape["asymmetry_scalar"]
        metrics_attrs["variogram_score"] = variogram_result["scalar"]
        metrics_attrs["dtr_crps"] = diurnal_shape["dtr_crps_scalar"]
        metrics_attrs["asymmetry_crps"] = diurnal_shape["asymmetry_crps_scalar"]

        if is_rollout:
            # Day-1 slice for ranking (full week is for plots/the lead-time comparison only).
            first_day_idx = int(variogram_result["series_by_day_idx"].day_idx.min())
            variogram_day1 = variogram_result["series_by_day_idx"].sel(day_idx=first_day_idx).reset_coords(drop=True)
            dtr_crps_day1 = diurnal_shape["dtr_crps_series_by_day"].sel(day_idx=first_day_idx).reset_coords(drop=True)
            asymmetry_crps_day1 = diurnal_shape["asymmetry_crps_series_by_day"].sel(day_idx=first_day_idx).reset_coords(drop=True)
            metrics_vars.update({
                "variogram_day1_series": variogram_day1,
                "dtr_crps_day1_series": dtr_crps_day1,
                "asymmetry_crps_day1_series": asymmetry_crps_day1,
            })
            metrics_attrs["variogram_score_day1"] = float(variogram_day1.mean())
            metrics_attrs["dtr_crps_day1"] = float(dtr_crps_day1.mean())
            metrics_attrs["asymmetry_crps_day1"] = float(asymmetry_crps_day1.mean())

        over_smoothing_result = over_smoothing_diagnostic(predictions_ds[TARGET_VAR], targets_ds[TARGET_VAR])
        tail_ratio_result = tail_ratio(prediction_tmax, target_tmax)
        print(f"  Tail ratio (Q99−Q90 model/obs): {tail_ratio_result['ratio']:.3f} "
              f"(lower {tail_ratio_result['lower_gap_ratio']:.3f}, upper {tail_ratio_result['upper_gap_ratio']:.3f})")
        metrics_vars.update({
            "over_smoothing_pred_curve": over_smoothing_result["pred_curve"],
            "over_smoothing_tgt_curve": over_smoothing_result["tgt_curve"],
            "tail_ratio_map": tail_ratio_result["map"],
        })
        metrics_attrs.update({
            "tail_ratio": tail_ratio_result["ratio"],
            "tail_ratio_lower_gap": tail_ratio_result["lower_gap_ratio"],
            "tail_ratio_upper_gap": tail_ratio_result["upper_gap_ratio"],
        })

        # peak-time cCRPS (needs <=3h spacing)
        hour_spacing = int(np.median(np.diff(predictions_ds[TARGET_VAR].hour.values))) if total_hours > 1 else 24
        if is_rollout and hour_spacing > 3:
            print(f"  Skipping peak-time error: {hour_spacing}h spacing > 3h.")
        else:
            peak_time_ccrps_result = peak_time_ccrps(predictions_ds[TARGET_VAR], targets_ds[TARGET_VAR])
            print(f"  PTE-cCRPS (circular, proper): {peak_time_ccrps_result['scalar']:.3f} h")
            metrics_vars.update({
                "peak_time_ccrps_map": peak_time_ccrps_result["map"],
                "peak_time_ccrps_series": peak_time_ccrps_result["series"],
            })
            metrics_attrs["peak_time_ccrps"] = peak_time_ccrps_result["scalar"]

    # window-position RMSE (AR seams, rollout only)
    window_rmse = None
    if is_rollout and (ar_window_hours is not None) and (not is_single_shot):
        window_rmse = rmse_by_window_position(predictions_ds[TARGET_VAR], targets_ds[TARGET_VAR], ar_window_hours)
        metrics_vars["rmse_by_window_position"] = window_rmse
        print(f"  RMSE window-position peak: {float(window_rmse.max()):.4f} K at pos {int(window_rmse.argmax().values)}")

    # REV (ensemble + heatwave)
    rev_result = None
    if num_ensemble > 1 and run_heatwave:
        rev_result = rev(prediction_tmax_pool, target_tmax_pool, threshold)
        print(f"  Max REV:   {float(rev_result.max()):.4f}")
        metrics_vars["rev_curve"] = rev_result
        metrics_attrs["rev_max"] = float(rev_result.max())

    # save
    metrics_ds = xr.Dataset(metrics_vars, attrs=metrics_attrs)
    metrics_path = os.path.join(out_dir, f"metrics_{tag}.nc")
    metrics_ds.to_netcdf(metrics_path)
    print(f"Metrics saved → {metrics_path}")
    if not no_s3:
        save_to_s3(metrics_path, run_id)

    # plots
    plot_dir = os.path.join(os.getcwd(), "figs", run_id)
    os.makedirs(plot_dir, exist_ok=True)
    print(f"\nGenerating plots → {plot_dir}")
    plot_common(metrics_vars, predictions_ds, targets_ds, plot_dir, tag, region, num_ensemble,
                ar_window_hours=(ar_window_hours if is_rollout else None), hour_axis=hour_axis,
                rmse_map_tmax=rmse_map_tmax.mean("day_idx") if has_full_days else None)

    lat = predictions_ds.lat.values
    lon = predictions_ds.lon.values
    # heatwave plots
    if run_heatwave:
        heatwave_lat = heatwave_metrics["precision_map"].lat.values
        heatwave_lon = heatwave_metrics["precision_map"].lon.values
        if is_rollout:
            plot_heatwave_metrics(heatwave_metrics["precision"], heatwave_metrics["recall"], heatwave_metrics["f1"],
                                  heatwave_metrics["f1_map"], heatwave_lat, heatwave_lon, plot_dir, tag, region=region)
        else:
            plot_metric_map(heatwave_metrics["f1_map"], heatwave_lat, heatwave_lon, "Heatwave F1", "F1",
                            os.path.join(plot_dir, f"heatwave_f1_map_{tag}.png"), region=region, cmap="RdYlGn")
        plot_metric_map(event_result["frequency_bias_map"], heatwave_lat, heatwave_lon,
                        "Heatwave event frequency bias (N_pred / N_obs)", "Frequency ratio",
                        os.path.join(plot_dir, f"event_frequency_bias_{tag}.png"), region=region, center=1.0)
        plot_metric_map(event_result["duration_bias_map"], heatwave_lat, heatwave_lon,
                        "Heatwave event duration bias (pred − obs)", "Days",
                        os.path.join(plot_dir, f"event_duration_bias_{tag}.png"), region=region, center=0.0)
        plot_metric_map(conditional_result["bias_map"], heatwave_lat, heatwave_lon,
                        "Tmax bias on observed heatwave days", "Bias (K)",
                        os.path.join(plot_dir, f"conditional_tmax_bias_{tag}.png"), region=region, center=0.0)
        plot_qq_tmax(prediction_tmax, target_tmax, os.path.join(plot_dir, f"qq_tmax_{tag}.png"))
        plot_spell_length_histogram(spell_prediction["lengths"], spell_target["lengths"],
                                    os.path.join(plot_dir, f"spell_length_{tag}.png"))
        if rev_result is not None:
            plot_rev(rev_result, os.path.join(plot_dir, f"rev_{tag}.png"))

    # Tmax bias + probabilistic-detection plots
    if has_full_days:
        plot_tmax_bias_map(bias_map_tmax, lat, lon, plot_dir, tag, region=region)
    if detection_result is not None:
        plot_rank_histogram(tail_rank_histogram, os.path.join(plot_dir, f"conditional_rank_histogram_{tag}.png"))
        plot_reliability_diagram(detection_result["reliability_forecast"], detection_result["reliability_observed"],
                                 detection_result["reliability_count"], os.path.join(plot_dir, f"reliability_{tag}.png"))
        plot_metric_map(detection_result["brier_map"], lat, lon, "Brier score — heatwave detection", "Brier",
                        os.path.join(plot_dir, f"brier_map_{tag}.png"), region=region)
    if window_rmse is not None:
        plot_rmse_by_window_position(window_rmse, os.path.join(plot_dir, f"rmse_window_position_{tag}.png"))

    # single-day + weekly profiles
    plot_day = str(pd.Timestamp(start_date).date() + pd.Timedelta(days=1))
    plot_single_day_profile(predictions_ds, targets_ds, plot_day, os.path.join(plot_dir, f"day_profile_{tag}_{plot_day}.png"),
                            ar_window_hours=ar_window_hours if is_rollout else None)
    if is_rollout:
        week_start = str(pd.DatetimeIndex(predictions_ds.forecast.values)[len(predictions_ds.forecast) // 2].date())
        plot_weekly_temp(predictions_ds, targets_ds, week_start, os.path.join(plot_dir, f"weekly_temp_{tag}_{week_start}.png"))

    # diurnal-shape plots
    if diurnal_shape is not None:
        plot_dtr_error(diurnal_shape["dtr_curve"], os.path.join(plot_dir, f"dtr_error_{tag}.png"))
        plot_diurnal_asymmetry(diurnal_shape["asymmetry_curve"], os.path.join(plot_dir, f"diurnal_asymmetry_{tag}.png"))
        plot_over_smoothing(over_smoothing_result["pred_curve"], over_smoothing_result["tgt_curve"],
                            os.path.join(plot_dir, f"over_smoothing_{tag}.png"),
                            xlabel="Lead time (hours)" if is_rollout else "Hour of day (UTC)",
                            ar_window_hours=ar_window_hours if is_rollout else None)
        plot_metric_map(tail_ratio_result["map"], lat, lon, "Tail ratio (Q99−Q90 model / obs)", "Ratio",
                        os.path.join(plot_dir, f"tail_ratio_{tag}.png"), region=region, center=1.0)
        plot_metric_map(diurnal_shape["asymmetry_crps_map"], lat, lon, "Diurnal asymmetry CRPS", "ASY-CRPS",
                        os.path.join(plot_dir, f"asymmetry_crps_{tag}.png"), region=region)

    print(f"\nDone. All outputs in {out_dir}")
