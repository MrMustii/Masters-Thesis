import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import s3fs
import xarray as xr

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..")))

from modified_Gencast.lumio_buckets import get_loading_region, lumio
from modified_Gencast.metrics import TARGET_VAR
from modified_Gencast.plots import plot_metric_map, plot_qq_tmax
from modified_Gencast.helper_fun import save_to_s3
from modified_Gencast.report_common import load_true_daily_tmax, load_p90_threshold, score_daily_tmax
from gencast_infernce.config import DEFAULT_OUTPUT


def main():
    parser = argparse.ArgumentParser(description="RQ1 Tmax / heatwave-detection metrics for the exp1 zarr.")
    parser.add_argument("--zarr", type=str, required=True, help="Path (local or s3) to the exp1 predictions zarr.")
    parser.add_argument("--config", type=str, required=True, help="exp2 config json to borrow the scoring region from.")
    parser.add_argument("--run-id", type=str, default="exp1")
    parser.add_argument("--no-s3", action="store_true")
    args = parser.parse_args()

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

    days = pd.DatetimeIndex(temperature.time.values).floor("D")
    model_tmax = temperature.assign_coords(day=("time", days)).groupby("day").max()
    start_day = pd.Timestamp(days.min())
    end_day = pd.Timestamp(days.max())

    true_daily_tmax = load_true_daily_tmax(start_day, end_day, loading_region, scoring_region)
    threshold_p90 = load_p90_threshold()
    weights = xr.DataArray(np.cos(np.deg2rad(model_tmax.lat)), coords={"lat": model_tmax.lat})
    scored = score_daily_tmax(model_tmax, true_daily_tmax, threshold_p90, weights)
    print(f"  exp1 — Tmax RMSE {scored['rmse']:.3f} K  bias {scored['bias']:+.3f} K  "
          f"detection F1 {scored['f1']:.3f} (P {scored['precision']:.3f} R {scored['recall']:.3f})")

    metrics_vars = {
        "tmax_rmse_map": scored["rmse_map"], "tmax_bias_map": scored["bias_map"],
        "hw_precision_map": scored["precision_map"], "hw_recall_map": scored["recall_map"],
        "hw_f1_map": scored["f1_map"],
    }
    metrics_attrs = {
        "run_id": args.run_id,
        "tmax_rmse": scored["rmse"], "tmax_bias": scored["bias"],
        "hw_precision": scored["precision"], "hw_recall": scored["recall"], "hw_f1": scored["f1"],
    }
    out_dir = os.path.join(DEFAULT_OUTPUT, args.run_id)
    os.makedirs(out_dir, exist_ok=True)
    tag = f"{args.run_id}_{start_day.date()}_{end_day.date()}"
    metrics_path = os.path.join(out_dir, f"metrics_{tag}.nc")
    xr.Dataset(metrics_vars, attrs=metrics_attrs).to_netcdf(metrics_path)
    print(f"Metrics saved → {metrics_path}")
    if not args.no_s3:
        save_to_s3(metrics_path, args.run_id)

    plot_dir = os.path.join(os.getcwd(), "figs", args.run_id)
    os.makedirs(plot_dir, exist_ok=True)
    lat = model_tmax.lat.values
    lon = model_tmax.lon.values
    plot_metric_map(scored["bias_map"], lat, lon, "Tmax bias (exp1 − ERA5)", "Bias (K)",
                    os.path.join(plot_dir, f"tmax_bias_{tag}.png"), region=list(scoring_region), center=0.0)
    plot_metric_map(scored["f1_map"], lat, lon, "Heatwave detection F1 (exp1)", "F1",
                    os.path.join(plot_dir, f"hw_f1_{tag}.png"), region=list(scoring_region), cmap="RdYlGn")
    plot_qq_tmax(scored["model_tmax"], scored["true_tmax"], os.path.join(plot_dir, f"qq_tmax_{tag}.png"))
    print(f"\nDone. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
