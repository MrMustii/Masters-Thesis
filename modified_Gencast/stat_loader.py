"""Normalisation statistics, computed once per configuration and cached on LUMI-O.

GenCast normalises its inputs and predicts a normalised residual, so a run needs four fields
before it can start. Three are normalisers: mean and stddev for the inputs, diffs_stddev for
the residual. The fourth, min, normalises nothing - it is the fill value NaNCleaner uses for
sea surface temperature's missing cells.

``training.py`` and the four ``eval*.py`` entry points call ``load_stats`` with a run config
and pass the result down to graphcast's ``InputsAndResiduals``.

diffs_stddev is the field that changes shape between experiments, because each one predicts
a different residual:

    load_stats_simple    one Dataset, std of increments one output step apart     (exp2)
    load_stats_variable  dict keyed by lead time, one entry per lead              (exp3)
    load_stats_daily     dict keyed by hour of day, each hour minus its own day's
                         mean, which is the quantity exp4 reconstructs            (exp4)

``load_stats`` chooses between those on sampler type, except that a variable set containing
total_precipitation_12hr takes Google's published statistics instead, since that means exp1
is running their pretrained weights and must match how they were trained.

Caches are keyed by cadence, region and training window, so changing any of those builds a
new cache rather than silently reusing a wrong one.
"""

import dask
import numpy as np
import pandas as pd
import s3fs
import xarray as xr
from modified_Gencast.lumio_buckets import (
    lumio, bucket_name, load, format_and_slice, get_loading_region, GENCAST_LEVELS)
import zarr
import os
import sys
zarr.config.set({'default_zarr_format': 2})
gc_path = os.path.normpath( os.path.join(os.path.dirname(__file__), "..", "Googles_gencast") )
if gc_path not in sys.path:
    sys.path.insert(0, gc_path)

from graphcast import data_utils
zarr.config.set({'default_zarr_format': 2})

def load_stats(config: dict) -> dict:
    """Return the stat flavour this config needs: Google's for tp12, else by sampler type."""
    task_config = config["task_config"]
    all_variables = (
        list(task_config["input_variables"])
        + list(task_config.get("target_surface_variables", []))
        + list(task_config.get("target_atmospheric_variables", []))
    )
    if  "total_precipitation_12hr" in all_variables:
        from google.cloud import storage
        gcs_bucket = storage.Client.create_anonymous_client().get_bucket("dm_graphcast")
        stats = {}
        for name in ("mean_by_level", "stddev_by_level", "min_by_level", "diffs_stddev_by_level"):
            with gcs_bucket.blob(f"gencast/stats/{name}.nc").open("rb") as f:
                stats[name] = xr.load_dataset(f).astype(np.float32).compute()
        return stats
    sampler_type = config["training_config"].get("sampler_type", "fixed")
    if sampler_type == "daily_stats":
        return load_stats_daily(config)
    if sampler_type == "fixed":
        return load_stats_simple(config)
    return load_stats_variable(config)


def load_stats_variable(config: dict) -> dict:
    """Variable-lead-time stats: one diffs_stddev per lead time, for exp3's conditioning.

    One cache per (data_step, region, training window). Lead times are computed in groups of
    24 and the whole set is rewritten after each group, so a timeout costs at most one group
    and a rerun resumes from the last saved lead time.
    """
    task_config = config["task_config"]
    training_config = config["training_config"]

    region = get_loading_region(task_config)
    time_start = training_config["training_time_start"]
    time_end = training_config["training_time_end"]
    data_step = training_config.get("data_step", "1h")
    output_step_hours = int(training_config.get("output_step", training_config["time_step"]).rstrip("h"))
    max_horizon_hours = int(config["lead_time_encoder_config"]["max_horizon_hours"])
    lead_times = list(range(output_step_hours, max_horizon_hours + 1, output_step_hours))

    region_str = "global" if region is None else "-".join(str(x) for x in region)
    base_key = f"era5-lumi/stats/variable_{data_step}_{region_str}_{time_start}_{time_end}"
    s3 = lumio()
    print(f"[load_stats] Stats key: {base_key}")

    reuse_mean_stddev_min = False
    mean_result = stddev_result = min_result = None

    try:
        mean_result = xr.open_zarr(s3fs.S3Map(root=f"{base_key}/mean_by_level", s3=s3, check=False), consolidated=True).load()
        stddev_result = xr.open_zarr(s3fs.S3Map(root=f"{base_key}/stddev_by_level", s3=s3, check=False), consolidated=True).load()
        min_result = xr.open_zarr(s3fs.S3Map(root=f"{base_key}/min_by_level", s3=s3, check=False), consolidated=True).load()
        reuse_mean_stddev_min = True
    except Exception as error:
        print(f"[load_stats] No cached mean/std/min ({error}); computing from scratch ...")
        reuse_mean_stddev_min = False
        mean_result = stddev_result = min_result = None

    cached_diffs_by_lead_time = {}
    try:
        cached_diffs_ds = xr.open_zarr(s3fs.S3Map(root=f"{base_key}/diffs_stddev_by_level", s3=s3, check=False), consolidated=True)
        cached_diffs_by_lead_time = {int(lead_time): cached_diffs_ds.sel(lead_time=lead_time).drop_vars("lead_time").load() for lead_time in cached_diffs_ds.lead_time.values}
        print(f"[load_stats] Cached diffs cover {min(cached_diffs_by_lead_time)}h..{max(cached_diffs_by_lead_time)}h ({len(cached_diffs_by_lead_time)} leads).")
    except Exception:
        print(f"[load_stats] No cached diffs.")

    missing_lead_times = [lead_time for lead_time in lead_times if lead_time not in cached_diffs_by_lead_time]
    if reuse_mean_stddev_min and not missing_lead_times:
        print(f"[load_stats] Everything cached; nothing to compute.")
        return {
            "diffs_stddev_by_level": {lead_time: cached_diffs_by_lead_time[lead_time] for lead_time in lead_times},
            "mean_by_level": mean_result,
            "stddev_by_level": stddev_result,
            "min_by_level": min_result,
        }
 
    print(f"[load_stats] Loading {time_start} → {time_end} at {data_step} ...")
    raw = load(time_start, time_end, region=region, freq=data_step)
    datastore = format_and_slice(raw, region)
    data_utils.add_derived_vars(datastore)

    deltas = np.diff(datastore.time.values) / np.timedelta64(1, "h")
    step_hours = float(deltas[0])
    if not np.allclose(deltas, step_hours):
        raise ValueError("Time axis is not uniformly spaced; cannot map hour lags to indices.")

    non_static_vars = [v for v in datastore.data_vars if "time" in datastore[v].dims]
    reduction_dims = [d for d in datastore.dims if d != "level"]
    diffs_reduction_dims = [d for d in datastore[non_static_vars].dims if d != "level"]

    if not reuse_mean_stddev_min:
        print(f"[load_stats] Computing mean/std/min over {datastore.sizes['time']} time steps ...")
        mean_result, stddev_result, min_result = dask.compute(
            datastore.mean(reduction_dims),
            datastore.std(reduction_dims),
            datastore.min(reduction_dims),
        )
        mean_result = mean_result.load()
        stddev_result = stddev_result.load()
        min_result = min_result.load()
        for name, dataset in [("mean_by_level", mean_result), ("stddev_by_level", stddev_result), ("min_by_level", min_result)]:
            dataset.to_zarr(s3fs.S3Map(root=f"{base_key}/{name}", s3=s3, check=False), mode="w", consolidated=True)
        print(f"[load_stats] mean/std/min saved.")

    print(f"[load_stats] Computing diffs_stddev for {len(missing_lead_times)} missing lead times (of {len(lead_times)} needed) ...")
    n = datastore.sizes["time"]
    diffs_by_lead_time = dict(cached_diffs_by_lead_time)
    for group_start in range(0, len(missing_lead_times), 24):
        lead_time_group = missing_lead_times[group_start:group_start + 24]
        lazy_diffs = {}
        for lead_time in lead_time_group:
            lag_idx = round(lead_time / step_hours)
            shifted = datastore[non_static_vars].isel(time=slice(lag_idx, None))
            base = datastore[non_static_vars].isel(time=slice(None, n - lag_idx))
            base = base.assign_coords(time=shifted.time.values)  # subtract position-wise, not by coord
            lazy_diffs[lead_time] = (shifted - base).std(diffs_reduction_dims)
        computed_diffs = dask.compute(*lazy_diffs.values())
        diffs_by_lead_time.update({lt: ds.load() for lt, ds in zip(lazy_diffs.keys(), computed_diffs)})
        # Save after every group: a timeout costs at most one group, and reruns resume from here.
        saved_lead_times = sorted(diffs_by_lead_time)
        diffs_to_save = xr.concat([diffs_by_lead_time[lt].expand_dims(lead_time=[lt]) for lt in saved_lead_times], dim="lead_time")
        diffs_to_save.to_zarr(s3fs.S3Map(root=f"{base_key}/diffs_stddev_by_level", s3=s3, check=False), mode="w", consolidated=True)
        print(f"[load_stats] diffs {lead_time_group[0]}h..{lead_time_group[-1]}h done and saved ({len(saved_lead_times)}/{len(lead_times)})", flush=True)

    print(f"[load_stats] Stats complete: {base_key}")
 
    return {
        "diffs_stddev_by_level": {lead_time: diffs_by_lead_time[lead_time] for lead_time in lead_times},
        "mean_by_level": mean_result,
        "stddev_by_level": stddev_result,
        "min_by_level": min_result,
    }



def load_stats_simple(config: dict) -> dict:
    """Fixed-lead stats: a single diffs_stddev at the model's own output step, for exp2."""
    task_config = config["task_config"]
    training_config = config["training_config"]

    region = get_loading_region(task_config)
    time_start = training_config["training_time_start"]
    time_end = training_config["training_time_end"]
    data_step = training_config.get("data_step", "1h")
    data_step_hours = int(data_step.rstrip("h"))
    output_step_hours = int(training_config.get("output_step", training_config["time_step"]).rstrip("h"))
    if output_step_hours % data_step_hours != 0:
        raise ValueError(f"Model step {output_step_hours}h is not a multiple of the {data_step_hours}h store cadence.")
    lag_steps = output_step_hours // data_step_hours  # store indices between the states a diff spans

    region_str = "global" if region is None else "-".join(str(x) for x in region)
    base_key = f"era5-lumi/stats/fixed_{data_step}_diffs{output_step_hours}h_{region_str}_{time_start}_{time_end}"
    s3 = lumio()
    print(f"[load_stats_simple] Stats key: {base_key}")

    try:
        loaded = { name: xr.open_zarr(s3fs.S3Map(root=f"{base_key}/{name}", s3=s3, check=False), consolidated=True).load()
            for name in ("mean_by_level", "stddev_by_level", "min_by_level", "diffs_stddev_by_level")}
        print(f"[load_stats_simple] Loaded from cache.")
        return loaded
    except Exception as error:
        print(f"[load_stats_simple] No usable cache ({error}); computing from scratch ...")

    print(f"[load_stats_simple] Loading {time_start} → {time_end} at {data_step} ...")
    raw = load(time_start, time_end, region=region, freq=data_step)
    datastore = format_and_slice(raw, region)
    data_utils.add_derived_vars(datastore)

    non_static_vars = [v for v in datastore.data_vars if "time" in datastore[v].dims]
    reduction_dims = [d for d in datastore.dims if d != "level"]

    print(f"[load_stats_simple] Computing mean/std/min over {datastore.sizes['time']} timesteps at the {data_step} cadence ...")
    mean_result, stddev_result, min_result = dask.compute(
        datastore.mean(reduction_dims),
        datastore.std(reduction_dims),
        datastore.min(reduction_dims),
    )
    mean_result = mean_result.load()
    stddev_result = stddev_result.load()
    min_result = min_result.load()

    # diffs_stddev: std of the increments exactly output_step apart, over the full cadence.
    n = datastore.sizes["time"]
    shifted = datastore[non_static_vars].isel(time=slice(lag_steps, None))
    base = datastore[non_static_vars].isel(time=slice(None, n - lag_steps))
    base = base.assign_coords(time=shifted.time.values)
    print(f"[load_stats_simple] Computing diffs_stddev at the {output_step_hours}h lag ...")
    diffs_stddev_result, = dask.compute((shifted - base).std(reduction_dims))
    diffs_stddev_result = diffs_stddev_result.load()

    print(f"[load_stats_simple] Saving stats to: {base_key} ...")
    for name, dataset in [
        ("mean_by_level", mean_result),
        ("stddev_by_level", stddev_result),
        ("min_by_level", min_result),
        ("diffs_stddev_by_level", diffs_stddev_result),
    ]:
        dataset.to_zarr(s3fs.S3Map(root=f"{base_key}/{name}", s3=s3, check=False), mode="w", consolidated=True)
    print(f"[load_stats_simple] Stats saved.")

    return {
        "diffs_stddev_by_level": diffs_stddev_result,
        "mean_by_level": mean_result,
        "stddev_by_level": stddev_result,
        "min_by_level": min_result,
    }



def load_stats_daily(config: dict) -> dict:
    """Daily-deaggregation stats for exp4: hourly and daily stats merged, diffs per hour.

    exp4 predicts each hour as a residual against that day's mean, so diffs_stddev is keyed
    by hour of day (0..23) rather than by lead time, computed from the hourly field minus its
    own day's forward-filled daily mean. Requires the daily bucket to already cover the
    window, and raises rather than silently scoring a short period if it does not.
    """
    task_config = config["task_config"]
    training_config = config["training_config"]

    region = get_loading_region(task_config)
    time_start = training_config["training_time_start"]
    time_end = training_config["training_time_end"]

    region_str = "global" if region is None else "-".join(str(x) for x in region)
    base_key = f"era5-lumi/stats/daily_{region_str}_{time_start}_{time_end}"
    cache_version = training_config.get("stats_cache_version")
    if cache_version:
        base_key += f"_{cache_version}"
    s3 = lumio()
    print(f"[load_stats_daily] Stats key: {base_key}")

    try:
        loaded = {name: xr.open_zarr(s3fs.S3Map(root=f"{base_key}/{name}", s3=s3, check=False), consolidated=True).load()
            for name in ("mean_by_level", "stddev_by_level", "min_by_level")}
        cached_diffs = xr.open_zarr(s3fs.S3Map(root=f"{base_key}/diffs_stddev_by_level", s3=s3, check=False), consolidated=True).load()
        loaded["diffs_stddev_by_level"] = {int(hour): cached_diffs.sel(lead_time=hour).drop_vars("lead_time") for hour in cached_diffs.lead_time.values}
        print(f"[load_stats_daily] Loaded from cache.")
        return loaded
    except Exception as error:
        print(f"[load_stats_daily] No usable cache ({error}); computing from scratch ...")

    print(f"[load_stats_daily] Loading hourly data {time_start} → {time_end} ...")
    raw = load(time_start, time_end, region=region, freq="1h")
    hourly = format_and_slice(raw, region)
    data_utils.add_derived_vars(hourly)

    if pd.Timestamp(hourly.time.values[0]).hour != 0:
        raise ValueError("Hourly store does not start at midnight; cannot slice by hour of day.")
    deltas = np.diff(hourly.time.values) / np.timedelta64(1, "h")
    if not np.allclose(deltas, 1.0):
        raise ValueError("Hourly time axis is not continuous at 1h; cannot slice by hour of day.")
    whole_days = hourly.sizes["time"] // 24
    hourly = hourly.isel(time=slice(0, whole_days * 24))

    daily_bucket = bucket_name(GENCAST_LEVELS, region) + "-daily"
    print(f"[load_stats_daily] Opening daily bucket {daily_bucket} ...")
    daily = xr.open_zarr(s3fs.S3Map(root=daily_bucket, s3=s3, check=False), consolidated=True).sortby("time")
    daily = daily.sel(time=slice(time_start, time_end))
    needed_days = hourly.time.values[::24]
    missing_days = np.setdiff1d(needed_days, daily.time.values)
    if missing_days.size:
        raise ValueError(
            f"Daily bucket is missing {missing_days.size} days in {time_start} → {time_end} "
            f"(first: {missing_days[0]}); run build_daily_bucket.py first."
        )
    daily = daily.sel(time=needed_days)

    hourly_reduction_dims = [d for d in hourly.dims if d != "level"]
    daily_reduction_dims = [d for d in daily.dims if d != "level"]
    print(f"[load_stats_daily] Computing mean/std/min over {hourly.sizes['time']} hourly and {daily.sizes['time']} daily steps ...")
    hourly_mean, hourly_stddev, hourly_min, daily_mean, daily_stddev, daily_min = dask.compute(
        hourly.mean(hourly_reduction_dims),
        hourly.std(hourly_reduction_dims),
        hourly.min(hourly_reduction_dims),
        daily.mean(daily_reduction_dims),
        daily.std(daily_reduction_dims),
        daily.min(daily_reduction_dims),
    )
    mean_result = xr.merge([hourly_mean, daily_mean]).load()
    stddev_result = xr.merge([hourly_stddev, daily_stddev]).load()
    min_result = xr.merge([hourly_min, daily_min]).load()

    residual_names = [name for name in hourly.data_vars if "time" in hourly[name].dims and f"{name}_daily_mean" in daily.data_vars]
    upsampled_daily_mean = (
        daily[[f"{name}_daily_mean" for name in residual_names]]
        .rename({f"{name}_daily_mean": name for name in residual_names})
        .reindex(time=hourly.time.values, method="ffill")
    )
    residuals = hourly[residual_names] - upsampled_daily_mean
    diffs_reduction_dims = [d for d in residuals.dims if d != "level"]
    print(f"[load_stats_daily] Computing diffs_stddev for hours 0..23 ...")
    lazy_diffs = {hour: residuals.isel(time=slice(hour, None, 24)).std(diffs_reduction_dims) for hour in range(24)}
    computed_diffs = dask.compute(*lazy_diffs.values())
    diffs_by_hour = {hour: ds.load() for hour, ds in zip(lazy_diffs.keys(), computed_diffs)}
    diffs_to_save = xr.concat([diffs_by_hour[hour].expand_dims(lead_time=[hour]) for hour in range(24)], dim="lead_time")

    print(f"[load_stats_daily] Saving stats to lumio: {base_key} ...")
    for name, dataset in [
        ("mean_by_level", mean_result),
        ("stddev_by_level", stddev_result),
        ("min_by_level", min_result),
        ("diffs_stddev_by_level", diffs_to_save),
    ]:
        dataset.to_zarr(s3fs.S3Map(root=f"{base_key}/{name}", s3=s3, check=False), mode="w", consolidated=True)
    print(f"[load_stats_daily] Stats saved.")

    return {
        "diffs_stddev_by_level": diffs_by_hour,
        "mean_by_level": mean_result,
        "stddev_by_level": stddev_result,
        "min_by_level": min_result,
    }
