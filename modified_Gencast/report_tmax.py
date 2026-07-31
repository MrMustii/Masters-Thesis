import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import s3fs
import xarray as xr

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..")))

from modified_Gencast.lumio_buckets import get_loading_region, lumio, load
from modified_Gencast.metrics import TARGET_VAR, spatial_variogram_score, crps_field
from modified_Gencast.baselines import load_climatology, build_climatology_prediction
from modified_Gencast.plots import (
    plot_metric_map, plot_qq_tmax, plot_lead_day_curves,
    plot_reliability_diagram, plot_rev, plot_rank_histogram,
)
from modified_Gencast.helper_fun import save_to_s3
from modified_Gencast.report_common import (
    prepare_datasets, compute_common_metrics, plot_common,
    load_true_daily_tmax, load_p90_threshold, compute_tmax_detection, score_daily_tmax,
    heatwave_tail_metrics,
)
from gencast_infernce.config import DEFAULT_OUTPUT


def plot_tmax_detection(metrics_vars, lat, lon, plot_dir, tag, region):
    """Draw the lead-day curves and aggregate maps from stored Tmax/detection metrics."""
    plot_lead_day_curves(
        {"F1": metrics_vars["hw_f1_by_lead_day"], "precision": metrics_vars["hw_precision_by_lead_day"],
         "recall": metrics_vars["hw_recall_by_lead_day"]},
        "Score", "Heatwave detection vs forecast lead day",
        os.path.join(plot_dir, f"hw_detection_by_lead_day_{tag}.png"))
    plot_lead_day_curves(
        {"model": metrics_vars["tmax_rmse_by_lead_day"]},
        "Tmax RMSE (K)", "Daily Tmax RMSE vs forecast lead day",
        os.path.join(plot_dir, f"tmax_rmse_by_lead_day_{tag}.png"))
    plot_metric_map(metrics_vars["tmax_bias_map"], lat, lon, "Tmax bias (model − ERA5), all lead days", "Bias (K)",
                    os.path.join(plot_dir, f"tmax_bias_map_{tag}.png"), region=region, center=0.0)
    plot_metric_map(metrics_vars["hw_f1_map"], lat, lon, "Heatwave detection F1, all lead days", "F1",
                    os.path.join(plot_dir, f"hw_f1_map_{tag}.png"), region=region, cmap="RdYlGn")

    if "reliability_forecast" in metrics_vars:
        plot_reliability_diagram(metrics_vars["reliability_forecast"], metrics_vars["reliability_observed"],
                                 metrics_vars["reliability_count"],
                                 os.path.join(plot_dir, f"reliability_hw_{tag}.png"))
        plot_rev(metrics_vars["rev_curve"], os.path.join(plot_dir, f"rev_hw_{tag}.png"))
        plot_rank_histogram(metrics_vars["conditional_rank_histogram"].values,
                            os.path.join(plot_dir, f"conditional_rank_histogram_{tag}.png"),
                            title="Conditional rank histogram (obs > τ)")


def run_tmax_report(predictions_ds, targets_ds, region, num_ensemble, run_id,
                    out_dir, tag, no_s3, setup, model_path="", autoregressive_steps=0, ar_window_hours=None):
    """exp2 report: common metrics + lead-day Tmax detection + spatial VS."""
    # common metrics
    print("\nComputing metrics...")
    predictions_ds, targets_ds = prepare_datasets(predictions_ds, targets_ds, region)
    base_attrs = {
        "checkpoint": model_path, "run_id": run_id,
        "autoregressive_steps": autoregressive_steps,
        "ar_window_hours": ar_window_hours if ar_window_hours is not None else 0,
    }
    metrics_vars, metrics_attrs = compute_common_metrics(predictions_ds, targets_ds, num_ensemble, base_attrs)

    # Tmax detection vs lead day
    loading_region = get_loading_region(setup.config["task_config"])
    valid_all = np.concatenate([pd.DatetimeIndex(predictions_ds.forecast.values) + pd.Timedelta(hours=int(h))
                                for h in predictions_ds[TARGET_VAR].hour.values])
    start_day = pd.Timestamp(valid_all.min()).floor("D")
    end_day = pd.Timestamp(valid_all.max()).floor("D")
    true_daily_tmax = load_true_daily_tmax(start_day, end_day, loading_region, tuple(region))
    threshold_p90 = load_p90_threshold()
    tmax_vars, tmax_attrs, tmax_arrays = compute_tmax_detection(
        predictions_ds[TARGET_VAR], true_daily_tmax, threshold_p90)
    metrics_vars.update(tmax_vars)
    metrics_attrs.update(tmax_attrs)

    # spatial VS (2nd variable-selection criterion)
    spatial_vs = spatial_variogram_score(predictions_ds[TARGET_VAR], targets_ds[TARGET_VAR])
    metrics_vars["spatial_vs_series"] = spatial_vs["series"]
    metrics_attrs["spatial_vs"] = spatial_vs["scalar"]
    print(f"  Spatial VS(0.5): {spatial_vs['scalar']:.4f}")

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
    plot_common(metrics_vars, predictions_ds, targets_ds, plot_dir, tag,
                region, num_ensemble, ar_window_hours, hour_axis="lead",
                rmse_map_tmax=tmax_vars["tmax_rmse_map"])
    lat = predictions_ds.lat.values
    lon = predictions_ds.lon.values
    plot_tmax_detection(metrics_vars, lat, lon, plot_dir, tag, region)
    plot_qq_tmax(tmax_arrays["model_tmax"], tmax_arrays["true_tmax"], os.path.join(plot_dir, f"qq_tmax_{tag}.png"))
    print(f"\nDone. All outputs in {out_dir}")


def run_tmax_plots_only(predictions_ds, targets_ds, region, num_ensemble, run_id, out_dir, tag):
    """Redraw from the cached metrics_{tag}.nc without recomputing any metric."""
    metrics_path = os.path.join(out_dir, f"metrics_{tag}.nc")
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"plots-only needs cached metrics at {metrics_path}")
    print(f"\nLoading cached metrics → {metrics_path}")
    metrics_ds = xr.open_dataset(metrics_path)
    metrics_vars = {name: metrics_ds[name] for name in metrics_ds.data_vars}
    ar_window_hours = int(metrics_ds.attrs.get("ar_window_hours", 0)) or None

    predictions_ds, targets_ds = prepare_datasets(predictions_ds, targets_ds, region)
    plot_dir = os.path.join(os.getcwd(), "figs", run_id)
    os.makedirs(plot_dir, exist_ok=True)
    print(f"\nGenerating plots → {plot_dir}")
    plot_common(metrics_vars, predictions_ds, targets_ds, plot_dir, tag,
                region, num_ensemble, ar_window_hours, hour_axis="lead",
                rmse_map_tmax=metrics_vars.get("tmax_rmse_map"))
    lat = predictions_ds.lat.values
    lon = predictions_ds.lon.values
    plot_tmax_detection(metrics_vars, lat, lon, plot_dir, tag, region)
    print(f"\nDone. Plots in {plot_dir}")


def main():
    """Standalone exp1 report: scores the exp1 zarr + climatology + persistence on the same 00/12 grid."""
    parser = argparse.ArgumentParser(description="RQ1 Tmax / heatwave-detection metrics for the exp1 zarr.")
    parser.add_argument("--zarr", type=str, required=True, help="Path (local or s3) to the exp1 predictions zarr.")
    parser.add_argument("--config", type=str, required=True, help="exp2 config json to borrow the scoring region from.")
    parser.add_argument("--run-id", type=str, default="exp1")
    parser.add_argument("--no-s3", action="store_true")
    args = parser.parse_args()

    # region + load exp1 zarr, crop to scoring box
    task_config = json.load(open(args.config))["task_config"]
    region = task_config["region"]
    scoring_region = (region["min_lat"], region["max_lat"], region["min_lon"], region["max_lon"])
    loading_region = get_loading_region(task_config)
    bucket_path = args.zarr.split(":", 1)[-1] if ":" in args.zarr else args.zarr
    store = s3fs.S3Map(root=bucket_path, s3=lumio(), check=False)
    predictions = xr.open_zarr(store, consolidated=True)
    temperature = predictions[TARGET_VAR]
    if "sample" in temperature.dims:
        temperature = temperature.rename({"sample": "ensemble"})
    if float(temperature.lon.max()) > 180:
        temperature = temperature.assign_coords(lon=(temperature.lon + 180) % 360 - 180).sortby("lon")
    lat_min, lat_max, lon_min, lon_max = scoring_region
    temperature = temperature.sortby("lat").sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))

    # true daily Tmax + threshold + hourly truth
    days = pd.DatetimeIndex(temperature.time.values).floor("D")
    start_day = pd.Timestamp(days.min())
    end_day = pd.Timestamp(days.max())
    true_daily_tmax = load_true_daily_tmax(start_day, end_day, loading_region, scoring_region)
    threshold_p90 = load_p90_threshold()
    weights = xr.DataArray(np.cos(np.deg2rad(temperature.lat)), coords={"lat": temperature.lat})
    prediction_times = pd.DatetimeIndex(temperature.time.values)
    truth_hourly = load(str(start_day.date()), str(prediction_times.max()),
                        variables=[TARGET_VAR], region=loading_region, freq="1h")[TARGET_VAR]
    if float(truth_hourly.lon.max()) > 180:
        truth_hourly = truth_hourly.assign_coords(lon=(truth_hourly.lon + 180) % 360 - 180).sortby("lon")
    truth_hourly = truth_hourly.sortby("lat")
    truth_states = truth_hourly.sel(time=temperature.time).interp(lat=temperature.lat, lon=temperature.lon)
    day_of_year_da = xr.DataArray(np.minimum(prediction_times.dayofyear.values, 365),
                                  dims=["time"], coords={"time": temperature.time})
    hour_of_day_da = xr.DataArray(prediction_times.hour.values,
                                  dims=["time"], coords={"time": temperature.time})
    tail_threshold = threshold_p90.sel(dayofyear=day_of_year_da).interp(lat=temperature.lat, lon=temperature.lon)

    # baselines on the same 00/12 grid
    climatology = load_climatology(loading_region)
    # Reuses build_climatology_prediction via a dummy hour=[0] lead, since climatology only needs valid time.
    climatology_input = xr.Dataset({TARGET_VAR: temperature.rename({"time": "forecast"}).expand_dims(hour=[0])})
    climatology_prediction = (build_climatology_prediction(climatology_input, climatology)[TARGET_VAR]
                              .isel(hour=0, drop=True).rename({"forecast": "time"}))

    # Can't reuse build_persistence_prediction: it needs a real init/lead axis exp1's zarr doesn't have.
    has_previous_day = prediction_times >= (start_day + pd.Timedelta(hours=24))
    persistence_times = prediction_times[has_previous_day]
    persistence_prediction = (truth_hourly.sel(time=persistence_times - pd.Timedelta(hours=24))
                              .assign_coords(time=persistence_times)
                              .interp(lat=temperature.lat, lon=temperature.lon))

    # score exp1 + both baselines identically
    scored_exp1 = None
    for run_label, run_temperature in ((args.run_id, temperature),
                                       ("climatology", climatology_prediction),
                                       ("persistence", persistence_prediction)):
        # daily Tmax detection
        run_days = pd.DatetimeIndex(run_temperature.time.values).floor("D")
        model_tmax = run_temperature.assign_coords(day=("time", run_days)).groupby("day").max()
        scored = score_daily_tmax(model_tmax, true_daily_tmax, threshold_p90, weights)

        # CRPS / twCRPS on the raw 00/12 states
        run_truth = truth_states.sel(time=run_temperature.time)
        run_threshold = tail_threshold.sel(time=run_temperature.time)
        crps_series = crps_field(run_temperature, run_truth).weighted(weights).mean(["lat", "lon"])
        twcrps_series = crps_field(np.maximum(run_temperature, run_threshold),
                                   np.maximum(run_truth, run_threshold)).weighted(weights).mean(["lat", "lon"])

        # daily Tmax-twCRPS (pairs against exp2 day-1) + tail metrics
        model_days = pd.DatetimeIndex(model_tmax.day.values)
        daily_doy = xr.DataArray(np.minimum(model_days.dayofyear.values, 365),
                                 dims=["day"], coords={"day": model_tmax.day})
        daily_threshold = threshold_p90.sel(dayofyear=daily_doy).interp(lat=temperature.lat, lon=temperature.lon)
        tmax_twcrps_day1_field = crps_field(np.maximum(model_tmax, daily_threshold),
                                            np.maximum(scored["true_tmax"], daily_threshold))
        tmax_twcrps_day1_series = tmax_twcrps_day1_field.weighted(weights).mean(["lat", "lon"]).reset_coords(drop=True)
        tmax_twcrps_day1_mean = float(tmax_twcrps_day1_field.weighted(weights).mean(["lat", "lon", "day"]))
        tail_vars, tail_attrs = heatwave_tail_metrics(
            model_tmax.rename({"day": "day_idx"}),
            scored["true_tmax"].rename({"day": "day_idx"}),
            daily_threshold.rename({"day": "day_idx"}))
        print(f"  {run_label} — Tmax RMSE {scored['rmse']:.3f} K  bias {scored['bias']:+.3f} K  "
              f"detection F1 {scored['f1']:.3f} (P {scored['precision']:.3f} R {scored['recall']:.3f})  "
              f"CRPS {float(crps_series.mean()):.3f} K  twCRPS {float(twcrps_series.mean()):.3f} K  "
              f"Tmax-twCRPS {tmax_twcrps_day1_mean:.3f} K")

        metrics_vars = {
            "tmax_rmse_map": scored["rmse_map"], "tmax_bias_map": scored["bias_map"],
            "hw_precision_map": scored["precision_map"], "hw_recall_map": scored["recall_map"],
            "hw_f1_map": scored["f1_map"],
            "hw_tp_series": scored["counts_series"]["tp"].rename({"forecast": "day"}).reset_coords(drop=True),
            "hw_fp_series": scored["counts_series"]["fp"].rename({"forecast": "day"}).reset_coords(drop=True),
            "hw_fn_series": scored["counts_series"]["fn"].rename({"forecast": "day"}).reset_coords(drop=True),
            "hw_tn_series": scored["counts_series"]["tn"].rename({"forecast": "day"}).reset_coords(drop=True),
            "crps_series": crps_series.rename({"time": "forecast"}).reset_coords(drop=True),
            "twcrps_series": twcrps_series.rename({"time": "forecast"}).reset_coords(drop=True),
            "tmax_twcrps_day1_series": tmax_twcrps_day1_series,
        }
        metrics_attrs = {
            "run_id": run_label,
            "tmax_rmse": scored["rmse"], "tmax_bias": scored["bias"],
            "hw_precision": scored["precision"], "hw_recall": scored["recall"], "hw_f1": scored["f1"],
            "crps_mean": float(crps_series.mean()), "twcrps_mean": float(twcrps_series.mean()),
            "tmax_twcrps_day1": tmax_twcrps_day1_mean,
        }
        metrics_vars.update(tail_vars)
        metrics_attrs.update(tail_attrs)
        # save one metrics_*.nc per run label
        out_dir = os.path.join(DEFAULT_OUTPUT, run_label)
        os.makedirs(out_dir, exist_ok=True)
        tag = (f"{args.run_id}_{start_day.date()}_{end_day.date()}" if run_label == args.run_id
               else f"{run_label}_{args.run_id}_{start_day.date()}_{end_day.date()}")
        metrics_path = os.path.join(out_dir, f"metrics_{tag}.nc")
        xr.Dataset(metrics_vars, attrs=metrics_attrs).to_netcdf(metrics_path)
        print(f"Metrics saved → {metrics_path}")
        if not args.no_s3:
            save_to_s3(metrics_path, run_label)
        if run_label == args.run_id:
            scored_exp1 = scored
            exp1_tag = tag
            tail_vars_exp1 = tail_vars

    # exp1-only plots
    plot_dir = os.path.join(os.getcwd(), "figs", args.run_id)
    os.makedirs(plot_dir, exist_ok=True)
    lat = temperature.lat.values
    lon = temperature.lon.values
    plot_metric_map(scored_exp1["bias_map"], lat, lon, "Tmax bias (exp1 − ERA5)", "Bias (K)",
                    os.path.join(plot_dir, f"tmax_bias_{exp1_tag}.png"), region=list(scoring_region), center=0.0)
    plot_metric_map(scored_exp1["f1_map"], lat, lon, "Heatwave detection F1 (exp1)", "F1",
                    os.path.join(plot_dir, f"hw_f1_{exp1_tag}.png"), region=list(scoring_region), cmap="RdYlGn")
    plot_qq_tmax(scored_exp1["model_tmax"], scored_exp1["true_tmax"], os.path.join(plot_dir, f"qq_tmax_{exp1_tag}.png"))
    if "reliability_forecast" in tail_vars_exp1:
        plot_reliability_diagram(tail_vars_exp1["reliability_forecast"], tail_vars_exp1["reliability_observed"],
                                 tail_vars_exp1["reliability_count"],
                                 os.path.join(plot_dir, f"reliability_hw_{exp1_tag}.png"))
        plot_rev(tail_vars_exp1["rev_curve"], os.path.join(plot_dir, f"rev_hw_{exp1_tag}.png"))
        plot_rank_histogram(tail_vars_exp1["conditional_rank_histogram"].values,
                            os.path.join(plot_dir, f"conditional_rank_histogram_{exp1_tag}.png"),
                            title="Conditional rank histogram (obs > τ)")
    print(f"\nDone. Outputs in {os.path.join(DEFAULT_OUTPUT, args.run_id)}")


if __name__ == "__main__":
    main()
