"""LUMI-O object storage: the cached ERA5 store and the derived stores read off it.

Every other module gets its data through here. ERA5 is pulled from WeatherBench2 on Google
Cloud and cached as zarr on LUMI-O, one bucket per (levels, region, tp12) combination, so a
run reads object storage instead of refetching from the cloud every time.

    load                       the entry point: fetches whatever the request needs and the
                               store lacks, appends it, then returns the slice asked for
    get_loading_region         widens a config's region of interest by its boundary margin,
                               which is what makes training see the belt
    load_p90_threshold         daily p90 of Tmax, written by build_tmax_threshold
    load_hourly_p90_threshold  p90 per hour of day, the tail weight for hourly twCRPS
    load_true_daily_tmax       observed daily Tmax, cropped to a given scoring grid

Bucket roots are written with ``mode='w-'``: zarr's ``'w'`` wipe goes through s3fs and
removes the bucket itself, which destroyed a store once. Stats and threshold stores live at
a prefix inside ``era5-lumi`` rather than at a bucket root, so those can use ``'w'`` safely.
"""

import s3fs
import os
import pandas as pd
import numpy as np
import xarray as xr
from dask.distributed import LocalCluster, Client
import time
import zarr
zarr.config.set({'default_zarr_format': 2})



DOWNLOAD_TIME_CHUNK = 48
DOWNLOAD_READ_CHUNK = 4
APPEND_SEGMENT_STEPS = 4800
WB2_SOURCE = 'gs://weatherbench2/datasets/era5/1959-2023_01_10-full_37-1h-0p25deg-chunk-1.zarr'
GENCAST_SOURCE = 'gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr'
GENCAST_LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]


def lumio():
    """S3 filesystem for LUMI-O, taking credentials from the environment or ~/.lumio_creds."""
    creds_file = os.path.expanduser('~/.lumio_creds')
    if not ('LUMIO_ACCESS_KEY' in os.environ and 'LUMIO_SECRET_KEY' in os.environ):
        if os.path.exists(creds_file):
            import subprocess
            out = subprocess.check_output(['bash', '-c', f'source {creds_file} && echo $LUMIO_ACCESS_KEY && echo $LUMIO_SECRET_KEY'], text=True)
            key, secret = out.strip().split('\n')
            os.environ['LUMIO_ACCESS_KEY'] = key
            os.environ['LUMIO_SECRET_KEY'] = secret
    return s3fs.S3FileSystem(
        key=os.environ['LUMIO_ACCESS_KEY'],
        secret=os.environ['LUMIO_SECRET_KEY'],
        client_kwargs={'endpoint_url': 'https://lumidata.eu'},
        skip_instance_cache=True,
    )


def bucket_name(levels, region=None, tp12=False):
    """Bucket name encoding the level set, region and tp12 flag; dots become 'p' to stay S3-safe."""
    if levels is None:
        levels = 'lall'
    elif not levels:
        levels = 'lsurf'
    elif len(levels) > 10:
        levels = 'full-gencast'
    else:
        levels = 'l' + '-'.join(str(l) for l in sorted(levels))
    if region is None:
        name = f'era5-{levels}'
    else:
        lat_min, lat_max, lon_min, lon_max = region
        name = f'era5-{levels}-rlat{lat_min}-{lat_max}lon{lon_min}-{lon_max}'
    if tp12:
        name += '-tp12'
    return name.replace('.', 'p')


def missing_data(time_start, time_end, existing_times, freq='1h'):
    """Contiguous (start, end) ranges that the request needs but the store does not yet hold."""
    needed = pd.date_range(time_start, time_end, freq=freq)
    if existing_times is None or len(existing_times) == 0:
        return [(needed[0], needed[-1])]
    missing = needed.difference(existing_times)
    if len(missing) == 0:
        return []
    gaps = np.where(np.diff(missing.values) > pd.Timedelta(freq).to_timedelta64())[0]
    starts = np.concatenate([[0], gaps + 1])
    ends = np.concatenate([gaps, [len(missing) - 1]])
    return sorted((missing[s], missing[e]) for s, e in zip(starts, ends))

def download(time_start, time_end, variables, levels, region=None, freq='1h'):
    """Fetch one span straight from the cloud source, cropped to region and levels.

    Precipitation only exists in the GenCast-derived dataset, so asking for it switches
    source. Longitude is normalised to [-180, 180] and a region crossing the meridian is
    fetched as two slices and concatenated.
    """
    source = GENCAST_SOURCE if "total_precipitation_12hr" in variables else WB2_SOURCE
    src = xr.open_zarr(source, chunks={'time': DOWNLOAD_READ_CHUNK}, storage_options={'token': 'anon'})
    wanted_times = pd.date_range(time_start, time_end, freq=freq)
    subset = src.sel(time=wanted_times)
    if variables is not None:
        subset = subset[variables]
    if levels and 'level' in subset.dims:
        subset = subset.sel(level=levels)
    if 'latitude' in subset.dims:
        subset = subset.rename({'latitude': 'lat', 'longitude': 'lon'})
    if region is not None:
        lat_min, lat_max, lon_min, lon_max = region
        if subset.lat.values[0] > subset.lat.values[-1]:
            subset = subset.sel(lat=slice(lat_max, lat_min)).isel(lat=slice(None, None, -1))
        else:
            subset = subset.sel(lat=slice(lat_min, lat_max))
        if float(subset.lon.max()) > 180:
            lon_start = lon_min % 360
            lon_end = lon_max % 360
            if lon_start <= lon_end:
                subset = subset.sel(lon=slice(lon_start, lon_end))
            else:
                western = subset.sel(lon=slice(lon_start, 360))
                eastern = subset.sel(lon=slice(0, lon_end))
                subset = xr.concat([western, eastern], dim='lon')
            subset = subset.assign_coords(
                lon=(((subset.lon + 180) % 360) - 180)).sortby('lon')
        else:
            subset = subset.sel(lon=slice(lon_min, lon_max))
    subset = subset.chunk({'time': DOWNLOAD_TIME_CHUNK, 'lat': -1, 'lon': -1})
    for name in subset.variables:
        subset[name].encoding.pop('chunks', None)
    return subset


def checkAndDownload(time_start, time_end, variables=None, levels=None, region=None, freq='1h'):
    """
    Ensure the store for this (levels, region) combo contains all requested data.
    Each unique (levels, region) is its own zarr bucket on LUMI-O.
    - Missing times   → fetched and appended.
    - New variables   → fetched for the full stored time span and merged in.
    freq sets the storage cadence (e.g. '1h', '6h') and must exist in the source.
    """
    uses_tp12 = variables is not None and 'total_precipitation_12hr' in variables
    store_bucket = bucket_name(GENCAST_LEVELS, region, tp12=uses_tp12)
    if not lumio().exists(store_bucket):
        lumio().mkdir(store_bucket)
    store = s3fs.S3Map(root=store_bucket, s3=lumio(), check=False)
    store_has_metadata = lumio().exists(f'{store_bucket}/.zmetadata')

    if not store_has_metadata:
        existing_times, existing_vars = None, None
    else:
        meta_ds = xr.open_zarr(store, chunks=None, consolidated=True)
        existing_times, existing_vars = pd.DatetimeIndex(meta_ds.time.values), list(meta_ds.data_vars)
    store_is_empty = existing_times is None

    new_vars = [v for v in variables if v not in existing_vars] if (not store_is_empty and variables is not None) else None
    time_ranges = missing_data(time_start, time_end, existing_times, freq)
    if not new_vars and not time_ranges:
        print(f"[{store_bucket}] All requested data already in store.")
        return

    num_workers = 24
    with LocalCluster(n_workers=num_workers, threads_per_worker=2, processes=True, dashboard_address=None, memory_limit="11GB") as cluster, Client(cluster):
        print(f"[{store_bucket}] Dask cluster: {num_workers} workers × 2 threads")

        if new_vars:
            t_start = min(existing_times.min(), pd.Timestamp(time_start))
            t_end = max(existing_times.max(), pd.Timestamp(time_end))
            print(f"[{store_bucket}] New variables {new_vars}: fetching {t_start} → {t_end} ...")
            chunk = download(t_start, t_end, new_vars, GENCAST_LEVELS, region, freq)
            chunk.to_zarr(store, mode='a', consolidated=True)
            meta_ds = xr.open_zarr(store, chunks=None, consolidated=True)
            existing_times, existing_vars = pd.DatetimeIndex(meta_ds.time.values), list(meta_ds.data_vars)
            time_ranges = missing_data(time_start, time_end, existing_times, freq)

        vars_to_fetch = existing_vars if (not store_is_empty and existing_vars) else variables
        segments = []
        for t_start, t_end in time_ranges:
            range_times = pd.date_range(t_start, t_end, freq=freq)
            for segment_start in range(0, len(range_times), APPEND_SEGMENT_STEPS):
                segments.append(range_times[segment_start:segment_start + APPEND_SEGMENT_STEPS])
        started_at = time.time()
        for segment_index, segment_times in enumerate(segments):
            chunk = download(segment_times[0], segment_times[-1], vars_to_fetch, GENCAST_LEVELS, region, freq)
            if store_is_empty:
                chunk.to_zarr(store, mode='w-', consolidated=True)
                store_is_empty = False
            else:
                if existing_times is not None:
                    chunk = chunk.sel(time=~pd.DatetimeIndex(chunk.time.values).isin(existing_times))
                    if chunk.time.size == 0:
                        continue
                chunk.to_zarr(store, mode='a', append_dim='time', consolidated=True, align_chunks=True)
            average_seconds = (time.time() - started_at) / (segment_index + 1)
            remaining_minutes = average_seconds * (len(segments) - segment_index - 1) / 60
            print(
                f"[{store_bucket}] {segment_times[0]} → {segment_times[-1]}  "
                f"segment {segment_index + 1}/{len(segments)}  eta {remaining_minutes:.0f} min",
                flush=True)
    print(f"[{store_bucket}] Store up to date.")


def load(time_start, time_end, variables=None, levels=None, region=None, chunks='auto', freq='1h'):
    """Return dataset for the requested range/region, downloading anything missing first."""
    checkAndDownload(time_start, time_end, variables, levels, region, freq)
    uses_tp12 = variables is not None and 'total_precipitation_12hr' in variables
    store_bucket = bucket_name(GENCAST_LEVELS, region, tp12=uses_tp12)
    store = s3fs.S3Map(root=store_bucket, s3=lumio(), check=False)
    store = xr.open_zarr(store, chunks=chunks, consolidated=True)
    store = store.sortby("time")
    result = store.sel(time=slice(time_start, time_end))
    if result.sizes["time"] > 1:
        native_step = pd.Timedelta(np.diff(result.time.values[:2])[0])
        stride_steps = round(pd.Timedelta(freq) / native_step)
        if stride_steps > 1:
            result = result.isel(time=slice(None, None, stride_steps))
    if variables is not None:
        result = result[variables]
    if levels and 'level' in result.dims:
        result = result.sel(level=levels)
    return format_and_slice(result, region)
GENERATED_VARS = {
    "year_progress_sin", "year_progress_cos",
    "day_progress_sin",  "day_progress_cos",
    "toa_incident_solar_radiation"}

def get_loading_region(task_config: dict):
    """Computes the loading region including any boundary margin."""
    region_dict = task_config.get("region") or {}
    if not region_dict:
        return None

    boundary_margin = task_config.get("boundary_margin_degrees", 0.0)
    
    lat_min, lat_max = region_dict["min_lat"], region_dict["max_lat"]
    lon_min, lon_max = region_dict["min_lon"], region_dict["max_lon"]

    if boundary_margin > 0:
        return (
            lat_min - boundary_margin, lat_max + boundary_margin,
            lon_min - boundary_margin, lon_max + boundary_margin)
    return (lat_min, lat_max, lon_min, lon_max)

def format_and_slice(data_store: xr.Dataset, region) -> xr.Dataset:
    """Normalise longitude to [-180, 180], sort lat and lon ascending, then crop to region."""
    if "datetime" not in data_store.coords:
        data_store = data_store.assign_coords(datetime=data_store.time)
    if 'lon' in data_store.coords:
        if float(data_store.lon.max()) > 180:
            data_store = data_store.assign_coords(lon=(((data_store.lon + 180) % 360) - 180))
        data_store = data_store.sortby('lon')
    if 'lat' in data_store.coords:
        data_store = data_store.sortby('lat')
    if region is not None:
        lat_min, lat_max, lon_min, lon_max = region
        data_store = data_store.sel(
            lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
    return data_store


## Diffrente region code 
EXP5_THRESHOLD_SUFFIXES = {"36.5-73.5--143.5--96.5", "6.5-43.5-1.5-48.5"}
def region_suffix(region):
    """Region tuple to bucket-name string: (36.5, 73.5, -143.5, -96.5) -> "36.5-73.5--143.5--96.5"."""
    if region is None:
        return None
    return "-".join(str(value) for value in region)


def threshold_bucket(prefix, region):
    """Path of one threshold store: no suffix for the default region, region suffix otherwise."""
    suffix = region_suffix(region)
    if not suffix:
        return f"era5-lumi/{prefix}"
    version = "_v2" if suffix in EXP5_THRESHOLD_SUFFIXES else ""
    return f"era5-lumi/{prefix}_{suffix}{version}"


def load_p90_threshold(region=None):
    """p90 of daily Tmax per (dayofyear, lat, lon). Decides which days count as hot for heatwaves."""
    store = s3fs.S3Map(root=threshold_bucket("tmax_p90_windowed", region), s3=lumio(), check=False)
    threshold_p90 = xr.open_zarr(store, consolidated=True)["tmax_p90"]
    if "latitude" in threshold_p90.dims:
        threshold_p90 = threshold_p90.rename({"latitude": "lat", "longitude": "lon"})
    if float(threshold_p90.lon.max()) > 180:
        threshold_p90 = threshold_p90.assign_coords(lon=(threshold_p90.lon + 180) % 360 - 180).sortby("lon")
    return threshold_p90


def load_hourly_p90_threshold(region=None):
    """p90 of temperature per (dayofyear, hour_of_day, lat, lon). Tail weight for hourly twCRPS."""
    store = s3fs.S3Map(root=threshold_bucket("temp_p90_hourly_windowed", region), s3=lumio(), check=False)
    threshold_p90 = xr.open_zarr(store, consolidated=True)["temp_p90_hourly"]
    if "latitude" in threshold_p90.dims:
        threshold_p90 = threshold_p90.rename({"latitude": "lat", "longitude": "lon"})
    if float(threshold_p90.lon.max()) > 180:
        threshold_p90 = threshold_p90.assign_coords(lon=(threshold_p90.lon + 180) % 360 - 180).sortby("lon")
    return threshold_p90


def load_true_daily_tmax(start_date, end_date, loading_region, scoring_region):
    """Observed daily Tmax from ERA5, one value per day on the scoring-region grid."""
    wanted_days = pd.date_range(start_date, end_date, freq="D")
    daily_tmax = None
    daily_bucket = bucket_name(GENCAST_LEVELS, loading_region) + "-daily"
    try:
        store = s3fs.S3Map(root=daily_bucket, s3=lumio(), check=False)
        daily = xr.open_zarr(store, consolidated=True)
        available_days = wanted_days.intersection(pd.DatetimeIndex(daily.time.values))
        if "2m_temperature_daily_max" in daily and len(available_days):
            daily_tmax = daily["2m_temperature_daily_max"].sel(time=available_days)
            print(f"  True Tmax: daily bucket {daily_bucket}, {len(available_days)} days "
                  f"{available_days[0].date()}..{available_days[-1].date()} "
                  f"({len(wanted_days) - len(available_days)} requested days not stored)")
        else:
            print(f"  Daily bucket {daily_bucket} unusable: "
                  f"has_daily_max={'2m_temperature_daily_max' in daily}, "
                  f"overlap={len(available_days)} days, variables={list(daily.data_vars)}; "
                  f"loading hourly instead.")
    except Exception as error:
        print(f"  Daily bucket unavailable ({error}); loading hourly instead.")
    if daily_tmax is None:
        hourly = load(str(wanted_days[0].date()), str(wanted_days[-1].date()),
                      variables=["2m_temperature"], region=loading_region, freq="1h")["2m_temperature"]
        daily_tmax = hourly.resample(time="1D").max()
        print("  True Tmax: hourly load → daily max")

    if float(daily_tmax.lon.max()) > 180:
        daily_tmax = daily_tmax.assign_coords(lon=(daily_tmax.lon + 180) % 360 - 180).sortby("lon")
    daily_tmax = daily_tmax.sortby("lat")
    lat_min, lat_max, lon_min, lon_max = scoring_region
    return daily_tmax.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))


# == Bucket management =========================================================
#
# Existing rclone remotes: lumi-465002687-private  /  lumi-465002687-public
#   (check which points to era5-lumi: rclone lsd lumi-465002687-private:)
#  
# module load lumio
#  
# List all stores in bucket:
#   rclone lsd   lumi-465002687-private:era5-lumi
#   rclone ls    lumi-465002687-private:era5-lumi
#
# Inspect a specific store:
#   rclone ls    lumi-465002687-private:era5-lumi/era5_L50-100-150-200-250-300-400-500-600-700-850-925-1000.zarr
#   rclone size  lumi-465002687-private:era5-lumi/era5_L50-100-150-200-250-300-400-500-600-700-850-925-1000.zarr
#
# Delete a store:
#   rclone purge lumi-465002687-private:era5-lumi/era5_L50-100-150-200-250-300-400-500-600-700-850-925-1000.zarr
#
# Delete everything in bucket:
#   rclone purge lumi-465002687-private:era5-lumi
#
# Download a store locally:
#   rclone copy  lumi-465002687-private:era5-lumi/era5_L50-...-1000.zarr ./local/ --progress
#
# Upload a local store:
#   rclone copy  ./local/era5_L50-...-1000.zarr lumi-465002687-private:era5-lumi/ --progress
#
# Dry run (preview without transferring):
#   rclone copy  --dry-run lumi-465002687-private:era5-lumi/... ./local/
#
# =============================================================================
