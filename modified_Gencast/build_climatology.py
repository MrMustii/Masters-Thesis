#!/usr/bin/env python3
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

from modified_Gencast.data_loaders import GENCAST_LEVELS, get_loading_region, bucket_name, load, lumio

TARGET_VAR = "2m_temperature"
REFERENCE_START_YEAR = 2005
REFERENCE_END_YEAR = 2019
HOURS_PER_YEAR = 365 * 24


def main():
    reference = json.load(open(os.path.join(CONFIGS_DIR, "experiment_2_all_gencast.json")))
    region = get_loading_region(reference["task_config"])

    store_bucket = bucket_name(GENCAST_LEVELS, region) + "-climatology"
    if lumio().exists(f"{store_bucket}/.zmetadata"):
        print(f"[{store_bucket}] Already built. Delete the bucket to rebuild.")
        return

    time_start = f"{REFERENCE_START_YEAR}-01-01"
    time_end = f"{REFERENCE_END_YEAR}-12-31"
    print(f"[climatology] Opening hourly {TARGET_VAR} {time_start} → {time_end} ...")
    hourly = load(time_start, time_end, variables=[TARGET_VAR], region=region, freq="1h")[TARGET_VAR]

    times = pd.DatetimeIndex(hourly.time.values)
    if times[0].month != 1 or times[0].day != 1 or times[0].hour != 0:
        raise ValueError(f"Hourly store starts at {times[0]}, not {REFERENCE_START_YEAR}-01-01 00:00.")
    if not (np.diff(times.values) == np.timedelta64(1, "h")).all():
        raise ValueError("Hourly time axis is not continuous at 1h.")

    not_leap_day = ~((times.month == 2) & (times.day == 29))
    hourly = hourly.isel(time=not_leap_day)
    number_of_years = REFERENCE_END_YEAR - REFERENCE_START_YEAR + 1
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
        # Mean over the years that share each hour-of-year slot, then split the
        # 8760-long slot axis into (dayofyear 1..365, hour 0..23).
        climatology = hourly.groupby("hour_of_year").mean("time")
        climatology = climatology.assign_coords(
            dayofyear=("hour_of_year", (np.arange(HOURS_PER_YEAR) // 24) + 1),
            hour=("hour_of_year", np.arange(HOURS_PER_YEAR) % 24),
        ).set_index(hour_of_year=["dayofyear", "hour"]).unstack("hour_of_year")
        climatology = climatology.transpose("dayofyear", "hour", "lat", "lon").rename("climatology")
        climatology = climatology.chunk({"dayofyear": -1, "hour": -1, "lat": -1, "lon": -1})
        climatology.encoding.pop("chunks", None)

        if not lumio().exists(store_bucket):
            lumio().mkdir(store_bucket)
        store = s3fs.S3Map(root=store_bucket, s3=lumio(), check=False)
        climatology.to_dataset().to_zarr(store, mode="w-", consolidated=True)
        print(f"[{store_bucket}] Built in {(time.time() - started_at) / 60:.1f} min  "
              f"dims={dict(climatology.sizes)}")


if __name__ == "__main__":
    main()
