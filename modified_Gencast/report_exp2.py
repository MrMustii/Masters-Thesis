import os

import numpy as np
import pandas as pd
import xarray as xr

from modified_Gencast.lumio_buckets import get_loading_region
from modified_Gencast.metrics import TARGET_VAR
from modified_Gencast.plots import plot_metric_map, plot_qq_tmax, plot_lead_day_curves
from modified_Gencast.helper_fun import save_to_s3
from modified_Gencast.report_common import (
    prepare_datasets, compute_common_metrics, plot_common,
    load_true_daily_tmax, load_p90_threshold, compute_tmax_detection,
)


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


def run_exp2_report(predictions_ds, targets_ds, region, num_ensemble, run_id,
                    out_dir, tag, no_s3, setup, model_path="", autoregressive_steps=0, ar_window_hours=None):
    print("\nComputing metrics...")
    predictions_ds, targets_ds = prepare_datasets(predictions_ds, targets_ds, region)

    base_attrs = {
        "checkpoint": model_path, "run_id": run_id,
        "autoregressive_steps": autoregressive_steps,
        "ar_window_hours": ar_window_hours if ar_window_hours is not None else 0,
    }
    metrics_vars, metrics_attrs = compute_common_metrics(predictions_ds, targets_ds, num_ensemble, base_attrs)

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

    metrics_ds = xr.Dataset(metrics_vars, attrs=metrics_attrs)
    metrics_path = os.path.join(out_dir, f"metrics_{tag}.nc")
    metrics_ds.to_netcdf(metrics_path)
    print(f"Metrics saved → {metrics_path}")
    if not no_s3:
        save_to_s3(metrics_path, run_id)

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


def run_exp2_plots_only(predictions_ds, targets_ds, region, num_ensemble, run_id, out_dir, tag):
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
