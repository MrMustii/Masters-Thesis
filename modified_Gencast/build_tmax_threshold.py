"""Build the p90 temperature thresholds that define a hot day. Run once per region.

For each day of year, pools a +/-15 day window across 1981-2010 - 31 years x 31 days = 961
samples per grid cell - and takes the 90th percentile. One pass over the hourly data
produces both stores, which is why the full hourly array is held rather than reduced to
daily maxima as it is read:

    tmax_p90_windowed          daily p90 of Tmax; decides which days count as hot, and so
                               which spells count as heatwaves
    temp_p90_hourly_windowed   p90 per (dayofyear, hour); the tail weight for hourly twCRPS

Reference period is 1981-2010, which excludes the 2021 evaluation year. It differs from the
climatology's 2005-2019 window - known and unreconciled, not a design choice.

    python modified_Gencast/build_tmax_threshold.py
"""

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

from gencast_infernce.config import CONFIGS_DIR
from modified_Gencast.lumio_buckets import lumio, download, get_loading_region, GENCAST_LEVELS


TARGET_VAR = "2m_temperature"
REFERENCE_START_YEAR = 1981
REFERENCE_END_YEAR = 2010
WINDOW_DAYS = 15
DAYS_PER_YEAR = 365
HOURS_PER_YEAR = DAYS_PER_YEAR * 24
STORE_BUCKET = "era5-lumi/tmax_p90_windowed"
HOURLY_STORE_BUCKET = "era5-lumi/temp_p90_hourly_windowed"
PERCENTILE = 90


def main():
    reference = json.load(open(os.path.join(CONFIGS_DIR, "experiment_2_all_gencast.json")))
    region = get_loading_region(reference["task_config"])

    if lumio().exists(f"{STORE_BUCKET}/.zmetadata") and lumio().exists(f"{HOURLY_STORE_BUCKET}/.zmetadata"):
        print(f"[{STORE_BUCKET}] and [{HOURLY_STORE_BUCKET}] already built. Delete a store to rebuild it.")
        return
    reference_years = list(range(REFERENCE_START_YEAR, REFERENCE_END_YEAR + 1))
    print(f"[tmax_p90] {TARGET_VAR} only, from WB2, region={region}, "
          f"{reference_years[0]}–{reference_years[-1]} ({len(reference_years)} years) ...")

    # Full hourly array kept (not reduced to daily max) so the same pass also builds the
    # hourly-resolved threshold; preallocated to avoid a list-then-stack memory doubling.
    hourly_values = None
    latitudes = longitudes = None
    with LocalCluster(n_workers=24, threads_per_worker=2, processes=True,
                      dashboard_address=None, memory_limit="8GB") as cluster, Client(cluster):
        started_at = time.time()
        for year_index, year in enumerate(reference_years):
            year_started = time.time()
            hourly = download(f"{year}-01-01", f"{year}-12-31 23:00", [TARGET_VAR], GENCAST_LEVELS, region, "1h")[TARGET_VAR].transpose("time", "lat", "lon")
            times = pd.DatetimeIndex(hourly.time.values)
            hourly = hourly.isel(time=~((times.month == 2) & (times.day == 29)))
            if hourly.sizes["time"] != HOURS_PER_YEAR:
                raise ValueError(f"{year}: expected {HOURS_PER_YEAR} hours after dropping Feb 29," f"got {hourly.sizes['time']}.")
            latitudes, longitudes = hourly.lat.values, hourly.lon.values
            if hourly_values is None:
                hourly_values = np.empty(
                    (len(reference_years), DAYS_PER_YEAR, 24, len(latitudes), len(longitudes)), dtype=np.float32)
            hourly_values[year_index] = hourly.values.reshape(DAYS_PER_YEAR, 24, len(latitudes), len(longitudes))
            elapsed = time.time() - started_at
            per_year = elapsed / (year_index + 1)
            remaining_minutes = per_year * (len(reference_years) - year_index - 1) / 60
            print(f"[tmax_p90] {year}: {time.time() - year_started:.0f}s  "
                  f"elapsed {elapsed / 60:.1f} min  ETA {remaining_minutes:.1f} min  "
                  f"({year_index + 1}/{len(reference_years)})", flush=True)

    daily_max_values = hourly_values.max(axis=2)  # (year, dayofyear, lat, lon)
    print(f"[tmax_p90] hourly {hourly_values.shape} / daily-max {daily_max_values.shape} in memory; "
          f"pooling ±{WINDOW_DAYS}-day windows ...")

    daily_threshold_values = np.empty((DAYS_PER_YEAR, len(latitudes), len(longitudes)), dtype=np.float32)
    hourly_threshold_values = np.empty((DAYS_PER_YEAR, 24, len(latitudes), len(longitudes)), dtype=np.float32)
    for day_of_year in range(1, DAYS_PER_YEAR + 1):
        window_indices = [(day_of_year - 1 + offset) % DAYS_PER_YEAR
                          for offset in range(-WINDOW_DAYS, WINDOW_DAYS + 1)]
        daily_pool = daily_max_values[:, window_indices, :, :].reshape(-1, len(latitudes), len(longitudes))
        daily_threshold_values[day_of_year - 1] = np.percentile(daily_pool, PERCENTILE, axis=0)
        hourly_pool = hourly_values[:, window_indices, :, :, :].reshape(-1, 24, len(latitudes), len(longitudes))
        hourly_threshold_values[day_of_year - 1] = np.percentile(hourly_pool, PERCENTILE, axis=0)

    daily_threshold = xr.DataArray(
        daily_threshold_values, dims=["dayofyear", "lat", "lon"],
        coords={"dayofyear": np.arange(1, DAYS_PER_YEAR + 1), "lat": latitudes, "lon": longitudes},
        name="tmax_p90").chunk({"dayofyear": -1, "lat": -1, "lon": -1})
    daily_threshold.encoding.pop("chunks", None)
    hourly_threshold = xr.DataArray(
        hourly_threshold_values, dims=["dayofyear", "hour_of_day", "lat", "lon"],
        coords={"dayofyear": np.arange(1, DAYS_PER_YEAR + 1), "hour_of_day": np.arange(24),
               "lat": latitudes, "lon": longitudes},
        name="temp_p90_hourly").chunk({"dayofyear": -1, "hour_of_day": -1, "lat": -1, "lon": -1})
    hourly_threshold.encoding.pop("chunks", None)

    for bucket, threshold in ((STORE_BUCKET, daily_threshold), (HOURLY_STORE_BUCKET, hourly_threshold)):
        if lumio().exists(f"{bucket}/.zmetadata"):
            print(f"[{bucket}] already built, skipping.")
            continue
        if not lumio().exists(bucket):
            lumio().mkdir(bucket)
        store = s3fs.S3Map(root=bucket, s3=lumio(), check=False)
        threshold.to_dataset().to_zarr(store, mode="w-", consolidated=True)
        print(f"[{bucket}] built  dims={dict(threshold.sizes)}")
    print(f"Done in {(time.time() - started_at) / 60:.1f} min  "
          f"pool={len(reference_years) * (2 * WINDOW_DAYS + 1)} values/day")


if __name__ == "__main__":
    main()
