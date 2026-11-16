#!/usr/bin/env python3
"""Build the hourly climatology for one region. Run once per region, then never again.

Averages 2m temperature over the config's training window into an average year of
365 x 24 slots, so each (dayofyear, hour) cell holds the mean across years. This is the
field the climatology baseline forecasts, and therefore the CRPSS denominator. Feb 29 is
dropped so every year contributes exactly 365 days.

Reference period is the training window (2005-2019), which is *not* the 1981-2010 period
build_tmax_threshold uses. That difference is known and unreconciled, not a design choice.

    python modified_Gencast/build_climatology.py
    python modified_Gencast/build_climatology.py --min-lat .. --max-lat .. --min-lon .. --max-lon ..
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

from modified_Gencast.lumio_buckets import GENCAST_LEVELS, get_loading_region, bucket_name, load, lumio

TARGET_VAR = "2m_temperature"
HOURS_PER_YEAR = 365 * 24


def main():
    parser = argparse.ArgumentParser(description="Build the hourly climatology for one scoring region.")
    parser.add_argument("--min-lat", type=float, default=None)
    parser.add_argument("--max-lat", type=float, default=None)
    parser.add_argument("--min-lon", type=float, default=None)
    parser.add_argument("--max-lon", type=float, default=None)
    parser.add_argument("--region-name", type=str, default=None)
    args = parser.parse_args()

    reference = json.load(open(os.path.join(CONFIGS_DIR, "experiment_2_all_gencast.json")))
    override_values = (args.min_lat, args.max_lat, args.min_lon, args.max_lon)
    if any(value is not None for value in override_values):
        if not all(value is not None for value in override_values):
            parser.error("regional climatology requires all four latitude/longitude bounds")
        reference["task_config"].setdefault("region", {})
        reference["task_config"]["region"].update({
            "min_lat": args.min_lat, "max_lat": args.max_lat,
            "min_lon": args.min_lon, "max_lon": args.max_lon,
        })
        print("[climatology] Region override", args.region_name, reference["task_config"]["region"])

    region = get_loading_region(reference["task_config"])

    store_bucket = bucket_name(GENCAST_LEVELS, region) + "-climatology"
    if lumio().exists(f"{store_bucket}/.zmetadata"):
        print(f"[{store_bucket}] Already built. Delete the bucket to rebuild.")
        return

    time_start = reference["training_config"]["training_time_start"]
    time_end = reference["training_config"]["training_time_end"]
    start_year = pd.Timestamp(time_start).year
    number_of_years = pd.Timestamp(time_end).year - start_year + 1
    print(f"[climatology] Opening hourly {TARGET_VAR} {time_start} → {time_end} ...")
    hourly = load(time_start, time_end, variables=[TARGET_VAR], region=region, freq="1h")[TARGET_VAR]

    # Make sure all days and all hours are there from can 01 00 
    times = pd.DatetimeIndex(hourly.time.values)
    if times[0].month != 1 or times[0].day != 1 or times[0].hour != 0:
        raise ValueError(f"Hourly store starts at {times[0]}, not {time_start} 00:00.")
    if not (np.diff(times.values) == np.timedelta64(1, "h")).all():
        raise ValueError("Hourly time axis is not continuous at 1h.")

    # Feb 29 and partial years are dropped 
    not_leap_day = ~((times.month == 2) & (times.day == 29))
    hourly = hourly.isel(time=not_leap_day)
    hourly = hourly.isel(time=slice(0, number_of_years * HOURS_PER_YEAR))
    if hourly.sizes["time"] != number_of_years * HOURS_PER_YEAR:
        raise ValueError(f"Expected {number_of_years * HOURS_PER_YEAR} hours after dropping Feb 29, "
                         f"got {hourly.sizes['time']}; the reference period is not {number_of_years} whole years.")

    
    hour_of_year = np.arange(hourly.sizes["time"]) % HOURS_PER_YEAR
    hourly = hourly.assign_coords(hour_of_year=("time", hour_of_year))

    with LocalCluster(n_workers=24, threads_per_worker=2, processes=True,
                      dashboard_address=None, memory_limit="11GB") as cluster, Client(cluster):
        print(f"[{store_bucket}] Dask cluster up; averaging {number_of_years} years over {HOURS_PER_YEAR} slots")
        started_at = time.time()
        climatology = hourly.groupby("hour_of_year").mean("time")
        climatology = climatology.assign_coords(dayofyear=("hour_of_year", (np.arange(HOURS_PER_YEAR) // 24) + 1),
            hour=("hour_of_year", np.arange(HOURS_PER_YEAR) % 24),
        ).set_index(hour_of_year=["dayofyear", "hour"]).unstack("hour_of_year")
        climatology = climatology.transpose("dayofyear", "hour", "lat", "lon").rename("climatology")
        climatology = climatology.chunk({"dayofyear": 31, "hour": 24, "lat": 64, "lon": 64})
        climatology.encoding.pop("chunks", None)

        if not lumio().exists(store_bucket):
            lumio().mkdir(store_bucket)
        store = s3fs.S3Map(root=store_bucket, s3=lumio(), check=False)
        climatology.to_dataset().to_zarr(store, mode="w-", consolidated=True)
        print(f"[{store_bucket}] Built in {(time.time() - started_at) / 60:.1f} min  "
              f"dims={dict(climatology.sizes)}")


if __name__ == "__main__":
    main()
