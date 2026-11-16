"""Build or extend the daily mean/max/min bucket. Run once per region, then to extend it.

Coarsens every time-dependent variable in the hourly store to one value per day, three ways,
and writes ``<era5 bucket>-daily``. Two consumers: exp4 reads it as the model's input, and
``load_true_daily_tmax`` reads it as observed Tmax when scoring every experiment.

Appends rather than rewrites, so re-running only fetches the days that are missing. Partial
days at the end of the span are dropped so every stored day has all 24 hours behind it.

    python modified_Gencast/build_daily_bucket.py
    python modified_Gencast/build_daily_bucket.py --start-date 2021-01-01 --end-date 2021-12-31
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import s3fs
import xarray as xr
from dask.distributed import LocalCluster, Client

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from gencast_infernce.config import CONFIGS_DIR
except ImportError:
    CONFIGS_DIR = os.path.join(os.path.dirname(__file__), "traninig configs")
    print(f"Warning: Could not import CONFIGS_DIR. Falling back to: {CONFIGS_DIR}")

from modified_Gencast.lumio_buckets import DOWNLOAD_TIME_CHUNK, GENCAST_LEVELS, get_loading_region, bucket_name, load, lumio, missing_data

HOURS_PER_DAY = 24
SEGMENT_DAYS = 4 * DOWNLOAD_TIME_CHUNK  # multiples of the write chunk keep appends chunk-aligned


def main():
    parser = argparse.ArgumentParser(description="Build (or extend) the daily-statistics bucket.")
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    args = parser.parse_args()

    # Region and time span are shared across all experiment configs; read them from all_gencast.
    reference = json.load(open(os.path.join(CONFIGS_DIR, "experiment_2_all_gencast.json")))
    training_config = reference["training_config"]
    region = get_loading_region(reference["task_config"])
    time_start = args.start_date or min( training_config.get("validation_time_start", training_config["training_time_start"]))
    time_end = args.end_date or max( training_config.get("validation_time_end", training_config["training_time_end"]))

    print(f"[daily] Opening hourly store {time_start} → {time_end} ...")
    hourly = load(time_start, time_end, region=region, freq="1h")
    time_variables = [name for name in hourly.data_vars if "time" in hourly[name].dims]
    print(f"[daily] {len(time_variables)} time-dependent variables: {sorted(time_variables)}")

    times = pd.DatetimeIndex(hourly.time.values)
    if times[0].hour != 0:
        raise ValueError(f"Hourly store starts at {times[0]}, not midnight.")
    if not (np.diff(times.values) == np.timedelta64(1, "h")).all():
        raise ValueError("Hourly time axis is not continuous at 1h.")
    whole_days = len(times) // HOURS_PER_DAY
    hourly = hourly[time_variables].isel(time=slice(0, whole_days * HOURS_PER_DAY))
    day_starts = pd.DatetimeIndex(hourly.time.values[::HOURS_PER_DAY])

    store_bucket = bucket_name(GENCAST_LEVELS, region) + "-daily"
    if not lumio().exists(store_bucket):
        lumio().mkdir(store_bucket)
    store = s3fs.S3Map(root=store_bucket, s3=lumio(), check=False)
    if lumio().exists(f"{store_bucket}/.zmetadata"):
        existing_times = pd.DatetimeIndex(xr.open_zarr(store, chunks=None, consolidated=True).time.values)
        store_is_empty = False
    else:
        existing_times = None
        store_is_empty = True

    day_ranges = missing_data(day_starts[0], day_starts[-1], existing_times, freq="1D")
    if not day_ranges:
        print(f"[{store_bucket}] All {len(day_starts)} days already in store.")
        return

    segments = []
    for range_start, range_end in day_ranges:
        range_days = pd.date_range(range_start, range_end, freq="1D")
        for segment_start in range(0, len(range_days), SEGMENT_DAYS):
            segments.append(range_days[segment_start:segment_start + SEGMENT_DAYS])

    with LocalCluster(n_workers=24, threads_per_worker=2, processes=True, dashboard_address=None, memory_limit="11GB") as cluster, Client(cluster):
        print(f"[{store_bucket}] Dask cluster up; {len(segments)} segments of ≤{SEGMENT_DAYS} days")
        started_at = time.time()
        for segment_index, segment_days in enumerate(segments):
            window = hourly.sel(time=slice(segment_days[0], segment_days[-1] + pd.Timedelta(hours=HOURS_PER_DAY - 1)))
            coarsened = window.coarsen(time=HOURS_PER_DAY)
            daily_segment = xr.merge([
                coarsened.mean().rename({name: f"{name}_daily_mean" for name in time_variables}),
                coarsened.max().rename({name: f"{name}_daily_max" for name in time_variables}),
                coarsened.min().rename({name: f"{name}_daily_min" for name in time_variables}),
            ])
            daily_segment = daily_segment.assign_coords(time=segment_days.values)
            daily_segment = daily_segment.chunk({"time": DOWNLOAD_TIME_CHUNK, "lat": -1, "lon": -1})
            for name in daily_segment.variables:
                daily_segment[name].encoding.pop("chunks", None)
            if store_is_empty:
                daily_segment.to_zarr(store, mode="w-", consolidated=True)
                store_is_empty = False
            else:
                daily_segment.to_zarr(store, mode="a", append_dim="time", consolidated=True, align_chunks=True)
            average_seconds = (time.time() - started_at) / (segment_index + 1)
            remaining_minutes = average_seconds * (len(segments) - segment_index - 1) / 60
            print(
                f"[{store_bucket}] {segment_days[0].date()} → {segment_days[-1].date()}  "
                f"segment {segment_index + 1}/{len(segments)}  eta {remaining_minutes:.0f} min",
                flush=True,
            )

    print(f"[{store_bucket}] Daily store up to date.")


if __name__ == "__main__":
    main()
