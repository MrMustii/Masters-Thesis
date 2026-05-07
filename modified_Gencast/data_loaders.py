import dataclasses
import os
import sys
import zarr
import numpy as np
import pandas as pd
import s3fs
import xarray as xr

# graphcast lives in Googles_gencast; add to path if needed
_gc_path = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "Googles_gencast")
)
if _gc_path not in sys.path:
    sys.path.insert(0, _gc_path)

from graphcast import data_utils  # noqa: E402

# Force zarr v2 format globally — zarr v3 defaults to format 3 which
# breaks numcodecs Blosc codecs used by the ERA5 source data.
zarr.config.set({'default_zarr_format': 2})

BUCKET = 'era5-lumi'

WB2_SOURCE = (
    'gs://weatherbench2/datasets/era5/'
    '1959-2023_01_10-full_37-1h-0p25deg-chunk-1.zarr'
)


# ========================================================================
def _lumio_fs():
    # To access the LUMI S3 bucket
    _creds_file = os.path.expanduser('~/.lumio_creds')
    if not ('LUMIO_ACCESS_KEY' in os.environ and 'LUMIO_SECRET_KEY' in os.environ):
        if os.path.exists(_creds_file):
            import subprocess
            out = subprocess.check_output(['bash', '-c', f'source {_creds_file} && echo $LUMIO_ACCESS_KEY && echo $LUMIO_SECRET_KEY'], text=True)
            key, secret = out.strip().split('\n')
            os.environ['LUMIO_ACCESS_KEY'] = key
            os.environ['LUMIO_SECRET_KEY'] = secret
    return s3fs.S3FileSystem(
        key=os.environ['LUMIO_ACCESS_KEY'],
        secret=os.environ['LUMIO_SECRET_KEY'],
        client_kwargs={'endpoint_url': 'https://lumidata.eu'},
    )


def _store_key(levels, region=None):
    # creates the name of the object to be stored in the bucket for a given (levels, region) combo
    lvl = 'Lall' if levels is None else 'L' + '-'.join(str(l) for l in sorted(levels))
    if region is None:
        return f'era5_{lvl}.zarr'
    lat_min, lat_max, lon_min, lon_max = region
    return f'era5_{lvl}_Rlat{lat_min}-{lat_max}lon{lon_min}-{lon_max}.zarr'


def _store(levels, region=None):
    return s3fs.S3Map(root=f'{BUCKET}/{_store_key(levels, region)}', s3=_lumio_fs(), check=False)


def _store_exists(levels, region=None):
    return _lumio_fs().exists(f'{BUCKET}/{_store_key(levels, region)}/.zmetadata')


def _open_store(levels, region=None, chunks='auto'):
    return xr.open_zarr(_store(levels, region), chunks=chunks, consolidated=True)


def _store_meta(levels, region=None):
    # Return (times, variables) for the exiting downlaoded data in the store for this (levels, region) combo, 
    if not _store_exists(levels, region):
        return None, None
    ds = _open_store(levels, region, chunks=None)
    return pd.DatetimeIndex(ds.time.values), list(ds.data_vars)


def _missing_time_ranges(time_start, time_end, existing_times):
    # Return list of (start, end) tuples for contiguous missing time ranges between time_start and time_end.
    needed = pd.date_range(time_start, time_end, freq='1h')
    if existing_times is None or len(existing_times) == 0:
        return [(needed[0], needed[-1])]
    missing = needed.difference(existing_times)
    if len(missing) == 0:
        return []
    gaps = np.where(np.diff(missing.asi8) > pd.Timedelta('1h').value)[0]
    starts = np.concatenate([[0], gaps + 1])
    ends = np.concatenate([gaps, [len(missing) - 1]])
    return sorted((missing[s], missing[e]) for s, e in zip(starts, ends))


def _fetch(time_start, time_end, variables, levels, region=None):
    # gets the requested subset of data from the source zarr, applying variable/level/region filters as needed.
    src = xr.open_zarr(WB2_SOURCE, chunks={'time': 24}, storage_options={'token': 'anon'})
    subset = src.sel(time=slice(time_start, time_end))
    if variables is not None:
        subset = subset[variables]
    if levels is not None and 'level' in subset.dims:
        subset = subset.sel(level=levels)
    if region is not None:
        lat_min, lat_max, lon_min, lon_max = region
        subset = subset.sel(latitude=slice(lat_min, lat_max), longitude=slice(lon_min, lon_max))
    # Rename to match Google's GenCast coordinate names
    if 'latitude' in subset.dims:
        subset = subset.rename({'latitude': 'lat', 'longitude': 'lon'})
    return subset


# ========================================================================
def ensure(time_start, time_end, variables=None, levels=None, region=None):
    """
    Ensure the store for this (levels, region) combo contains all requested data.
    Each unique (levels, region) is its own zarr object in the bucket.
    - Missing times   → fetched and appended.
    - New variables   → fetched for the full stored time span and merged in.
    """
    existing_times, existing_vars = _store_meta(levels, region)
    store = _store(levels, region)
    key = _store_key(levels, region)
    first_write = existing_times is None

    # variables
    new_vars = None
    if not first_write and variables is not None:
        new_vars = [v for v in variables if v not in existing_vars]
        if new_vars:
            t_start = min(existing_times.min(), pd.Timestamp(time_start))
            t_end = max(existing_times.max(), pd.Timestamp(time_end))
            print(f"[{key}] New variables {new_vars}: fetching {t_start} → {t_end} ...")
            chunk = _fetch(t_start, t_end, new_vars, levels, region)
            chunk.to_zarr(store, mode='a', consolidated=True)
            existing_times, existing_vars = _store_meta(levels, region)

    # times
    time_ranges = _missing_time_ranges(time_start, time_end, existing_times)
    if not time_ranges:
        if not new_vars:
            print(f"[{key}] All requested data already in store.")
        return

    vars_to_fetch = existing_vars if (not first_write and existing_vars) else variables
    for t_start, t_end in time_ranges:
        print(f"[{key}] Downloading {t_start} → {t_end} ...")
        chunk = _fetch(t_start, t_end, vars_to_fetch, levels, region)
        if first_write:
            chunk.to_zarr(store, mode='w', consolidated=True)
            first_write = False
        else:
            chunk.to_zarr(store, mode='a', append_dim='time', consolidated=True)

    print(f"[{key}] Store up to date.")


def load(time_start, time_end, variables=None, levels=None, region=None, chunks='auto'):
    """Return dataset for the requested range/region, downloading anything missing first."""
    ensure(time_start, time_end, variables, levels, region)
    ds = _open_store(levels, region, chunks=chunks)
    result = ds.sel(time=slice(time_start, time_end))
    if variables is not None:
        result = result[variables]
    if levels is not None and 'level' in result.dims:
        result = result.sel(level=levels)
    return result


# == Bucket management =========================================================
#
# Existing rclone remotes: lumi-465002687-private  /  lumi-465002687-public
#   (check which points to era5-lumi: rclone lsd lumi-465002687-private:)
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


# ── GenCast data loader ───────────────────────────────────────────────────────

# Variables computed on-the-fly by extract_inputs_targets_forcings — not in S3.
_GENERATED_VARS = {
    "year_progress_sin", "year_progress_cos",
    "day_progress_sin",  "day_progress_cos",
    "toa_incident_solar_radiation",
}


class Era5DataLoader:
    """
    Iterable data loader for ERA5 GenCast training.

    Downloads/caches data from the LUMI S3 bucket, then yields overlapping
    windows of shape:

        [t₀, t₀+tbps, …, t₀+(n-1)·tbps,  t₀+(n-1)·tbps+lead]
         └──────── num_prev_states inputs ─────────────┘  └─ target ─┘

    Variables and pressure levels are derived from ``task_config``.
    Train/val splitting is left to the caller — use different time ranges.

    Parameters
    ----------
    time_start, time_end : str  e.g. "2019-01-01" / "2019-12-31"
    task_config           : TaskConfig — source of variables and levels
    region                : (lat_min, lat_max, lon_min, lon_max) or None
    num_prev_states       : number of input time steps (default 2)
    time_between_prev_states : hours between inputs (default 12)
    target_lead_time      : prediction horizon in hours (default 12)
    batch_size            : windows per batch (default 1)
    shuffle               : shuffle window order each iteration (default True)
    seed                  : RNG seed for shuffling (default 42)
    """

    def __init__(
        self,
        time_start: str,
        time_end: str,
        task_config,
        region=None,
        num_prev_states: int = 2,
        time_between_prev_states: int = 12,
        target_lead_time: int = 12,
        batch_size: int = 1,
        shuffle: bool = True,
        seed: int = 42,
    ):
        self.task_config = task_config
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._rng = np.random.default_rng(seed)
        self._lead_str = f"{target_lead_time}h"

        # Hour offsets for isel into the 1-h-resolution dataset, e.g. [0, 12, 24]
        self._hour_offsets = (
            [k * time_between_prev_states for k in range(num_prev_states)]
            + [(num_prev_states - 1) * time_between_prev_states + target_lead_time]
        )

        # Variables to fetch: ERA5 vars only (generated forcings are computed inside
        # extract_inputs_targets_forcings and must not be requested from the store)
        fetch_vars = list(dict.fromkeys(
            v for v in list(task_config.input_variables) + list(task_config.target_variables)
            if v not in _GENERATED_VARS
        ))
        # Empty pressure_levels means no level dimension — pass None so load() doesn't filter
        levels = list(task_config.pressure_levels) or None

        # Do NOT pass region to load(): the source stores longitude in [0, 360) so
        # negative-lon regions would be silently wrong.  Spatial crop happens in
        # _postprocess, after we shift lon to [-180, 180).
        print(f"[DataLoader] Fetching {time_start} → {time_end} ...")
        raw = load(time_start, time_end, variables=fetch_vars, levels=levels)
        self.ds = self._postprocess(raw, region)

        N = self.ds.sizes["time"]
        span = self._hour_offsets[-1]
        if span >= N:
            raise ValueError(f"Window span ({span}h) >= dataset length ({N}h).")

        self._window_starts = np.arange(N - span, dtype=np.int64)
        print(
            f"[DataLoader] {N}h dataset  {len(self._window_starts)} windows  "
            f"grid={self.ds.sizes['lat']}×{self.ds.sizes['lon']}  batch={batch_size}"
        )

    @staticmethod
    def _postprocess(ds: xr.Dataset, region) -> xr.Dataset:
        if "datetime" not in ds.coords:
            ds = ds.assign_coords(datetime=ds.time)
        
        ds = ds.assign_coords(lon=(((ds.lon + 180) % 360) - 180))
        ds = ds.roll(lon=(ds.sizes['lon'] // 2), roll_coords=True).sortby("lat")
        
        if region is not None:
            lat_min, lat_max, lon_min, lon_max = region
            ds = ds.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
        return ds

    def _make_window(self, start_hour: int):
        sl = self.ds.isel(time=[start_hour + h for h in self._hour_offsets])
        sl = sl.assign_coords(time=(sl.time - sl.time[0]).astype("timedelta64[ns]"))

        task_kwargs = dataclasses.asdict(self.task_config)

        if task_kwargs['pressure_levels'] or 'level' in sl.dims:
            # Standard path: dataset has pressure levels.
            inputs, targets, forcings = data_utils.extract_inputs_targets_forcings(
                sl,
                target_lead_times=slice(self._lead_str, self._lead_str),
                **task_kwargs,
            )
        else:
            # Surface-only path: skip the dataset.sel(level=...) that
            # extract_inputs_targets_forcings always performs internally.
            _DERIVED = {data_utils.DAY_PROGRESS, data_utils.YEAR_PROGRESS,
                        f"{data_utils.DAY_PROGRESS}_sin", f"{data_utils.DAY_PROGRESS}_cos",
                        f"{data_utils.YEAR_PROGRESS}_sin", f"{data_utils.YEAR_PROGRESS}_cos"}
            if set(task_kwargs['forcing_variables']) & _DERIVED:
                data_utils.add_derived_vars(sl)
            if 'toa_incident_solar_radiation' in task_kwargs['forcing_variables']:
                data_utils.add_tisr_var(sl)
            sl_nodt = sl.drop_vars('datetime')
            raw_inputs, raw_targets = data_utils.extract_input_target_times(
                sl_nodt,
                input_duration=task_kwargs['input_duration'],
                target_lead_times=slice(self._lead_str, self._lead_str),
            )
            inputs   = raw_inputs[list(task_kwargs['input_variables'])]
            forcings = raw_targets[list(task_kwargs['forcing_variables'])]
            targets  = raw_targets[list(task_kwargs['target_variables'])]

        return (
            inputs.expand_dims(batch=1).compute(),
            targets.expand_dims(batch=1).compute(),
            forcings.expand_dims(batch=1).compute(),
        )

    def __len__(self) -> int:
        """Number of batches per epoch."""
        return len(self._window_starts) // self.batch_size

    def init_batch(self):
        """Return the first window (batch_size=1) for model initialisation."""
        return self._make_window(int(self._window_starts[0]))

    def __iter__(self):
        indices = self._window_starts
        if self.shuffle:
            indices = self._rng.permutation(indices)
        for i in range(0, len(indices), self.batch_size):
            batch = [self._make_window(int(j)) for j in indices[i: i + self.batch_size]]
            yield (
                xr.concat([w[0] for w in batch], dim="batch"),
                xr.concat([w[1] for w in batch], dim="batch"),
                xr.concat([w[2] for w in batch], dim="batch"),
            )


