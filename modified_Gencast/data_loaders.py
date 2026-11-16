"""Data loading for training and evaluation: one dataset held in memory, three samplers.

``Era5DataLoaderV2`` reads the configured region and span from LUMI-O once and keeps it,
then a ``WindowSampler`` cuts (inputs, targets, forcings) windows out of it and the loader
batches them. Which sampler is used is what makes each experiment different:

    FixedLeadTimeSampler     one fixed output step, autoregressive rollout (exp2)
    VariableLeadTimeSampler  a lead time drawn per sample, for the lead-time-conditioned
                             denoiser (exp3)
    DailyStatsSampler        daily mean/max/min as input, the 24 hours of that day as
                             targets (exp4)
"""

import abc
import dataclasses
import os
import sys
import time
import dask
import zarr
import numpy as np
import pandas as pd
import s3fs
import xarray as xr
from dask.distributed import LocalCluster, Client


from modified_Gencast.lumio_buckets import ( load, format_and_slice, get_loading_region, GENERATED_VARS)  



gc_path = os.path.normpath( os.path.join(os.path.dirname(__file__), "..", "Googles_gencast") )
if gc_path not in sys.path:
    sys.path.insert(0, gc_path)

from graphcast import data_utils
zarr.config.set({'default_zarr_format': 2})

class WindowSampler(abc.ABC):
    """Defines what one training sample looks like.

    Subclass and override __call__ to change target selection, lead time
    sampling, or conditioning signals without touching Era5DataLoaderV2.

    window_span must return the total hour span needed so the loader can
    compute valid window start indices.
    """

    @property
    @abc.abstractmethod
    def window_span(self) -> int: 
        ...

    def begin_batch(self) -> None:
        pass

    @abc.abstractmethod
    def __call__(self, ds: xr.Dataset, start_idx: int) -> dict:
        """Return {"inputs", "targets", "forcings", "conditioning": {...}}."""
        ...


class FixedLeadTimeSampler(WindowSampler):
    """Fixed step size, fixed lead time, optional autoregressive steps."""

    def __init__(
        self,
        task_config,
        num_inputs: int,
        step_size: int,
        lead_time: int,
        num_output: int = 1,
        data_step_hours: int = 1,
    ):
        self.task_args = dataclasses.asdict(task_config)
        self.lead_time = lead_time 
        self.step_size = step_size
        self.data_step_hours = data_step_hours
        self.input_hours = [k * step_size for k in range(num_inputs)]
        self.targets_hours = [self.input_hours[-1] + k * lead_time for k in range(1, num_output + 1)]

        self.all_hours = self.input_hours + self.targets_hours
        self.all_indices = [h // data_step_hours for h in self.all_hours]

        self.target_indices = [h // data_step_hours for h in self.targets_hours]
        self.target_lead_time_slice = slice( f"{self.targets_hours[0] - self.input_hours[-1]}h", f"{self.targets_hours[-1] - self.input_hours[-1]}h")
        self.fixed_input_times = np.array([np.timedelta64(h - self.input_hours[-1], 'h') for h in self.input_hours])
        self.fixed_target_times = np.array([h - self.input_hours[-1] for h in self.targets_hours], dtype='timedelta64[h]')

    @property
    def window_span(self) -> int:
        return self.targets_hours[-1] // self.data_step_hours

    def __call__(self, dataset: xr.Dataset, start_idx: int) -> dict:
        next_sample = dataset.isel(time=[start_idx + i for i in self.all_indices])
        next_sample = next_sample.assign_coords(time=(next_sample.time - next_sample.time[len(self.input_hours) - 1]).astype("timedelta64[ns]"))
        raw_inputs, raw_targets = data_utils.extract_input_target_times(
            next_sample.drop_vars("datetime"),
            input_duration=f"{self.step_size * len(self.input_hours)}h",
            target_lead_times=self.target_lead_time_slice,
        )
        raw_inputs = raw_inputs.assign_coords(time=self.fixed_input_times)
        raw_targets = raw_targets.assign_coords(time=self.fixed_target_times)
        return {
            "inputs": raw_inputs[list(self.task_args["input_variables"])].expand_dims(batch=1).compute(),
            "targets": raw_targets[list(self.task_args["target_variables"])].expand_dims(batch=1).compute(),
            "forcings": raw_targets[list(self.task_args["forcing_variables"])].expand_dims(batch=1).compute(),
            "conditioning": {
                "lead_time_hours": self.lead_time,
                "abs_target_times": dataset.time.values[[start_idx + i for i in self.target_indices]], # TODO is this even needed by the model??? what is it used for
            },
        }


class VariableLeadTimeSampler(WindowSampler):
    """Randomly samples lead time per window for lead-time conditioning.

    window_span covers the maximum possible lead time so all sampled
    targets are always in bounds. The sampled value is returned in
    conditioning["lead_time_hours"] for the model to condition on.
    """

    def __init__(
        self,
        task_config,
        num_inputs: int,
        step_size: int,
        lead_times: list,
        rng: np.random.Generator,
        data_step_hours: int = 1,
    ):
        self.task_args = dataclasses.asdict(task_config)
        self.step_size = step_size
        self.data_step_hours = data_step_hours
        self.input_hours = [k * step_size for k in range(num_inputs)]
        self.lead_times = lead_times
        self.rng = rng
        self.batch_lead_time_hours = None
        self.fixed_input_times = np.array([np.timedelta64(h - self.input_hours[-1], 'h') for h in self.input_hours])

    @property
    def window_span(self) -> int:
        return (self.input_hours[-1] + max(self.lead_times)) // self.data_step_hours

    def begin_batch(self) -> None:
        self.batch_lead_time_hours = int(self.rng.choice(self.lead_times))

    def __call__(self, dataset: xr.Dataset, start_idx: int) -> dict:
        all_offsets = self.input_hours + [self.input_hours[-1] + self.batch_lead_time_hours]
        next_sample = dataset.isel(time=[start_idx + h // self.data_step_hours for h in all_offsets])
        next_sample = next_sample.assign_coords(time=(next_sample.time - next_sample.time[len(self.input_hours) - 1]).astype("timedelta64[ns]"))
        raw_inputs, raw_targets = data_utils.extract_input_target_times(
            next_sample.drop_vars("datetime"),
            input_duration=f"{self.step_size * len(self.input_hours)}h",
            target_lead_times=f"{self.batch_lead_time_hours}h",
        )
        raw_inputs = raw_inputs.assign_coords(time=self.fixed_input_times)
        raw_targets = raw_targets.assign_coords(time=[np.timedelta64(self.lead_times[0], 'h')])
        return {
            "inputs": raw_inputs[list(self.task_args["input_variables"])].expand_dims(batch=1).compute(),
            "targets": raw_targets[list(self.task_args["target_variables"])].expand_dims(batch=1).compute(),
            "forcings": raw_targets[list(self.task_args["forcing_variables"])].expand_dims(batch=1).compute(),
            "conditioning": {
                "lead_time_hours": self.batch_lead_time_hours,
                "abs_target_times": dataset.time.values[[start_idx + (self.input_hours[-1] + self.batch_lead_time_hours) // self.data_step_hours]],
            },
        }


class DailyStatsSampler(WindowSampler):
    """Deaggregation: reconstruct the field at one hour of day 0 from daily statistics.

    Each day offset becomes one input timestep (time coordinate = offset in
    days) carrying that day's aggregates, read from the precomputed daily
    bucket: the daily mean under the plain variable name (the residual base
    for day 0), and daily max/min wherever input_variables contains
    <variable>_daily_max / <variable>_daily_min. Generated forcings (progress
    variables) are not in the daily bucket, so their daily mean is computed
    from the hourly window. The target hour within day 0 is sampled once per
    batch and passed via conditioning["lead_time_hours"].

    Requires the loader to be midnight-aligned: store starting at 00:00 and
    stride equal to one day. daily_dataset must be on the same grid as the
    loader's hourly dataset (same region slice and resolution subsampling).
    """

    def __init__(self, task_config, day_offsets, rng, daily_dataset, data_step_hours=1):
        self.task_args = dataclasses.asdict(task_config)
        if 24 % data_step_hours:
            raise ValueError(f"A day is not a whole number of {data_step_hours}h storage steps.")
        self.day_offsets = sorted(int(offset) for offset in day_offsets)
        if 0 not in self.day_offsets:
            raise ValueError(f"day_offsets {day_offsets} must contain day 0 (the target day).")
        self.rng = rng
        self.daily_dataset = daily_dataset
        self.data_step_hours = data_step_hours
        self.steps_per_day = 24 // data_step_hours
        self.target_hours = list(range(0, 24, data_step_hours))
        self.input_times = np.array([np.timedelta64(offset * 24, "h") for offset in self.day_offsets])
        self.batch_target_hour = None

    @property
    def window_span(self) -> int:
        return (self.day_offsets[-1] - self.day_offsets[0] + 1) * self.steps_per_day - 1

    def start_is_available(self, dataset: xr.Dataset, start_idx: int) -> bool:
        """True only when the daily bucket holds every offset day this window needs."""
        day_zero_start = start_idx + (0 - self.day_offsets[0]) * self.steps_per_day
        day_zero_date = dataset.time.values[day_zero_start]
        offset_dates = np.array([day_zero_date + np.timedelta64(offset * 24, "h") for offset in self.day_offsets])
        return bool(np.isin(offset_dates, self.daily_dataset.time.values).all())

    def begin_batch(self) -> None:
        self.batch_target_hour = int(self.rng.choice(self.target_hours))

    def __call__(self, dataset: xr.Dataset, start_idx: int) -> dict:
        static_names = [
            name for name in self.task_args["input_variables"]
            if name in dataset.data_vars and "time" not in dataset[name].dims
        ]
        day_zero_start = start_idx + (0 - self.day_offsets[0]) * self.steps_per_day
        day_zero_date = dataset.time.values[day_zero_start]
        offset_dates = np.array([day_zero_date + np.timedelta64(offset * 24, "h") for offset in self.day_offsets])
        daily_rows = self.daily_dataset.sel(time=offset_dates)  # raises KeyError if the daily bucket misses a day
        aggregated = {}
        for name in self.task_args["input_variables"]:
            if name in static_names:
                continue
            if name.endswith("_daily_max") or name.endswith("_daily_min"):
                aggregated[name] = daily_rows[name]
            elif f"{name}_daily_mean" in daily_rows.data_vars:
                aggregated[name] = daily_rows[f"{name}_daily_mean"]
            else:
                day_means = []
                for offset in self.day_offsets:
                    day_start = start_idx + (offset - self.day_offsets[0]) * self.steps_per_day
                    day_means.append(dataset[name].isel(time=slice(day_start, day_start + self.steps_per_day)).mean("time"))
                aggregated[name] = xr.concat(day_means, dim="time")
        raw_inputs = xr.Dataset(aggregated).assign_coords(time=self.input_times)
        if static_names:
            raw_inputs = xr.merge([raw_inputs, dataset[static_names]])

        target_index = day_zero_start + self.batch_target_hour // self.data_step_hours
        raw_targets = dataset.isel(time=[target_index]).drop_vars("datetime", errors="ignore")

        raw_targets = raw_targets.assign_coords(time=np.array([np.timedelta64(self.data_step_hours, "h")]))
        return {
            "inputs": raw_inputs[list(self.task_args["input_variables"])].expand_dims(batch=1).compute(),
            "targets": raw_targets[list(self.task_args["target_variables"])].expand_dims(batch=1).compute(),
            "forcings": raw_targets[list(self.task_args["forcing_variables"])].expand_dims(batch=1).compute(),
            "conditioning": {
                "lead_time_hours": self.batch_target_hour,
                "abs_target_times": dataset.time.values[[target_index]],
            },
        }


class Era5DataLoaderV2:
    """ERA5 data loader backed by a WindowSampler.

    Handles data fetching, window indexing, shuffling, and batching.
    All sample-level logic lives in the WindowSampler.
    Yields dicts: {"inputs", "targets", "forcings", "conditioning": {...}}
    """

    def __init__(
        self,
        time_start: str,
        time_end: str,
        config: dict,
        sampler: WindowSampler,
        batch_size: int = 1,
        shuffle: bool = True,
        seed: int = 42,
        stride: int = 1,
        resolution: float = 0.25,
        preload: bool = False,
        data_step: str = '1h',
    ):
        self.sampler = sampler
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)

        task_config = config["task_config"]
        all_vars = (
            list(task_config["input_variables"])
            + list(task_config.get("target_surface_variables", []))
            + list(task_config.get("target_atmospheric_variables", []))
        )
        fetch_vars = list(dict.fromkeys(
            v for v in all_vars
            if v not in GENERATED_VARS and not v.endswith(("_daily_mean", "_daily_max", "_daily_min"))
        ))
        levels = list(task_config["pressure_levels"])
        # Widened by boundary_margin_degrees, so the training loss covers the belt; reports crop it back off.
        region = get_loading_region(task_config)
        print(f"[DataLoaderV2] Loading {time_start} → {time_end} ...")
        raw = load(time_start, time_end, variables=fetch_vars, levels=levels, region=region, freq=data_step)
        self.dataset = format_and_slice(raw, region)
        for variable_name, data_array in self.dataset.data_vars.items():
            if data_array.dtype == np.float64:
                self.dataset[variable_name] = data_array.astype(np.float32)
        if resolution == 1.0:
            self.dataset = self.dataset.isel(
                lat=slice(None, None, 4),
                lon=slice(None, None, 4),
            )
        if preload:
            print(f"[DataLoaderV2] Pulling dataset into RAM ...")
            self.dataset = self.dataset.load()
        data_utils.add_derived_vars(self.dataset)

        N = self.dataset.sizes["time"]
        span = sampler.window_span
        self.window_starts = np.arange(0, N - span, stride, dtype=np.int64)
        if hasattr(sampler, "start_is_available"):
            available = np.array([sampler.start_is_available(self.dataset, int(start))
                                  for start in self.window_starts])
            if not available.all():
                margin = max(abs(min(sampler.day_offsets)), abs(max(sampler.day_offsets)))
                daily_min = np.datetime_as_string(sampler.daily_dataset.time.values.min(), unit="D")
                daily_max = np.datetime_as_string(sampler.daily_dataset.time.values.max(), unit="D")
                need_min = np.datetime_as_string(self.dataset.time.values.min(), unit="D")
                need_max = np.datetime_as_string(self.dataset.time.values.max(), unit="D")
                raise ValueError(
                    f"Daily bucket doesn't cover this loader's period: "
                    f"{int((~available).sum())}/{len(available)} windows need daily days it lacks. "
                    f"Daily bucket has {daily_min}..{daily_max}; this loader needs "
                    f"{need_min}..{need_max} (±{margin}d offset margin). Build the daily bucket for "
                    f"the gap:\n  START_DATE={need_min} END_DATE={need_max} "
                    f"sbatch job_scripts/lumi_build_daily_bucket.sh")
        print(
            f"[DataLoaderV2] {N}h  {len(self.window_starts)} windows  "
            f"grid={self.dataset.sizes['lat']}x{self.dataset.sizes['lon']}  "
            f"batch={batch_size}  stride={stride}h  res={resolution}deg"
        )

    def __len__(self) -> int:
        return len(self.window_starts) // self.batch_size

    def first_sample(self) -> dict:
        self.sampler.begin_batch()
        return self.sampler(self.dataset, int(self.window_starts[0]))

    def __iter__(self):
        indices = self.window_starts
        if self.shuffle:
            indices = self.rng.permutation(indices)

        buffer = []
        for idx in indices:
            if not buffer:
                self.sampler.begin_batch()
            buffer.append(self.sampler(self.dataset, int(idx)))
            if len(buffer) == self.batch_size:
                yield {
                    "inputs": xr.concat([s["inputs"] for s in buffer], dim="batch"),
                    "targets": xr.concat([s["targets"] for s in buffer], dim="batch"),
                    "forcings": xr.concat([s["forcings"] for s in buffer], dim="batch"),
                    "conditioning": {
                        k: np.array([s["conditioning"][k] for s in buffer])
                        for k in buffer[0]["conditioning"]
                    },
                }
                buffer = []
