"""Shared machinery for the eval entry points: checkpoints, predictor stack, loaders.

``eval.py``, ``eval_continuous.py``, ``eval_daily.py`` and ``eval_original_weights.py`` all
go through here, so the differences between the experiments are resolved in one place
instead of four.

    run_settings           find a run's checkpoint and reload the config saved beside it
    load_checkpoint        unpickle parameters into the CheckPoint the predictor expects
    build_predictor_eval   assemble the predictor stack a rollout actually runs
    build_loader           pick the sampler the config asks for and wrap it in a loader
    to_forecast_hour       re-index a rollout from lead time to (forecast, hour)
    save_to_s3             copy a finished prediction file up to LUMI-O
"""

import os
import pickle
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "Googles_gencast"))
sys.path.insert(0, str(REPOSITORY_ROOT))

from graphcast import nan_cleaning
from graphcast import normalization
from modified_Gencast import regional_utils
from modified_Gencast.gencast import GenCast, CheckPoint, load_config, BoundaryForcedPredictor
import numpy as np
import xarray as xr
import s3fs
from modified_Gencast.data_loaders import (
    Era5DataLoaderV2, DailyStatsSampler, FixedLeadTimeSampler, VariableLeadTimeSampler,
)
from modified_Gencast.lumio_buckets import (GENCAST_LEVELS, get_loading_region, bucket_name, format_and_slice, lumio)
from gencast_infernce.config import SCRATCH_PATH, WORKING_DIR



S3_PRED_ROOT = "era5-lumi/predictions"
_LUMI_CHECKPOINTS = "/scratch/project_465002687/checkpoints"
CKPT_DIR = (_LUMI_CHECKPOINTS if os.path.isdir(_LUMI_CHECKPOINTS) else os.path.normpath(os.path.join(WORKING_DIR, SCRATCH_PATH, "checkpoints")))

 
def run_settings(run_id: str, step=None):
    """Resolve a run's checkpoint path and load its config.
 
    Returns (model_path, setup). If step is None, picks the highest saved step.
    """
    ckpt_dir = os.path.join(CKPT_DIR, run_id)
    if step is not None:
        model_path = os.path.join(ckpt_dir, f"step_{step:08d}.pkl")
    else:
        step_files = sorted(
            f for f in os.listdir(ckpt_dir) if f.startswith("step_") and f.endswith(".pkl")
        )
        if not step_files:
            raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")
        model_path = os.path.join(ckpt_dir, step_files[-1])
    setup = load_config(os.path.join(ckpt_dir, "config.json"))
    setup.denoiser_architecture_config.grid2mesh_aggregate_normalization = (16.0 if setup.denoiser_architecture_config.mesh_size == 6 else None)
    return model_path, setup
 
 
def load_checkpoint(path: str, setup) -> CheckPoint:
    """Unpickle parameters and wrap them with the configs the predictor needs.

    Training saves only the parameters, so the architecture comes from the run's own
    config.json rather than from the checkpoint file.
    """
    with open(path, "rb") as f:
        params = pickle.load(f)
    return CheckPoint(
        description="", license="", params=params,
        task_config=setup.task,
        denoiser_architecture_config=setup.denoiser_architecture_config,
        sampler_config=setup.sampler_config,
        noise_config=setup.noise_config,
        noise_encoder_config=setup.noise_encoder_config,
        lead_time_encoder_config=setup.lead_time_encoder_config,
    )
 
 
class InputsAndResidualsDayZero(normalization.InputsAndResiduals):
    """InputsAndResiduals with the residual base taken from the time == 0 input frame.

    The daily-stats sampler can include future days as input frames, so the
    last frame is not necessarily day 0; the residual base must always be the
    day-0 daily mean.
    """

    def _unnormalize_prediction_and_add_input(self, inputs, norm_prediction):
        if norm_prediction.sizes.get("time") != 1:
            raise ValueError("InputsAndResidualsDayZero only supports predicting a single timestep.")
        if norm_prediction.name in inputs:
            prediction = normalization.unnormalize(norm_prediction, self._residual_scales, self._residual_locations)
            return prediction + inputs[norm_prediction.name].sel(time=np.timedelta64(0, "h"))
        return normalization.unnormalize(norm_prediction, self._scales, self._locations)

    def _subtract_input_and_normalize_target(self, inputs, target):
        if target.sizes.get("time") != 1:
            raise ValueError("InputsAndResidualsDayZero only supports a single target timestep.")
        if target.name in inputs:
            residual = target - inputs[target.name].sel(time=np.timedelta64(0, "h"))
            return normalization.normalize(residual, self._residual_scales, self._residual_locations)
        return normalization.normalize(target, self._scales, self._locations)


def build_predictor_eval(stats, ckpt: CheckPoint, boundary_margin_degrees=0.0, sampler_type="fixed"):
    """Assemble the predictor stack for inference, each wrapper outside the previous one.

    GenCast, then residual normalisation, then NaN cleaning when sea surface temperature is
    in the variable set, then boundary forcing when a margin is asked for. The order is not
    arbitrary: BoundaryForcedPredictor overwrites cells with ERA5 in raw units, so it has to
    sit outside normalisation.

    When diffs_stddev is a dict rather than one Dataset - exp4, which has a sigma per hour of
    day - the entries are concatenated along a batch dimension in hour order, so row h of a
    24-row batch is normalised by hour h's sigma. tests/test_eval_daily.py checks that
    alignment holds.
    """
    has_sst = "sea_surface_temperature" in (ckpt.task_config.input_variables + ckpt.task_config.target_variables)
    predictor = GenCast(
        task_config=ckpt.task_config,
        denoiser_architecture_config=ckpt.denoiser_architecture_config,
        sampler_config=ckpt.sampler_config,
        noise_config=ckpt.noise_config,
        noise_encoder_config=ckpt.noise_encoder_config,
        lead_time_encoder_config=ckpt.lead_time_encoder_config,
    )
    diffs_stddev = stats["diffs_stddev_by_level"]
    if isinstance(diffs_stddev, dict):
        diffs_stddev = xr.concat(
            [diffs_stddev[lead].expand_dims({"batch": [h]})
             for h, lead in enumerate(sorted(diffs_stddev.keys()))],
            dim="batch",
        )
    normalization_class = InputsAndResidualsDayZero if sampler_type == "daily_stats" else normalization.InputsAndResiduals
    predictor = normalization_class(
        predictor,
        diffs_stddev_by_level=diffs_stddev,
        mean_by_level=stats["mean_by_level"],
        stddev_by_level=stats["stddev_by_level"],
    )
    if has_sst:
        predictor = nan_cleaning.NaNCleaner(
            predictor=predictor, reintroduce_nans=True,
            fill_value=stats["min_by_level"],
            var_to_clean="sea_surface_temperature",
        )
    if boundary_margin_degrees and boundary_margin_degrees > 0:
        predictor = BoundaryForcedPredictor(predictor, boundary_margin_degrees)
    return predictor
 
 
 
def parse_steps(setup):
    """(input_step, output_step) in hours from the training config."""
    training_config = setup.config["training_config"]
    input_step = int(training_config["time_step"].removesuffix("h"))
    output_step = int(training_config.get("output_step", training_config["time_step"]).removesuffix("h"))
    return input_step, output_step
 
 
def parse_region(setup):
    """Region tuple (min_lat, max_lat, min_lon, max_lon), or None for global."""
    region = setup.config["task_config"].get("region") or {}
    if not region:
        return None
    return (region.get("min_lat"), region.get("max_lat"),region.get("min_lon"), region.get("max_lon"))
 
 
 
def build_loader(setup, time_start, time_end, seed, ar_steps=None, batch_size=1, shuffle=False, resolution=0.25, preload=False, force_sampler_type=None):
    """Build the loader for one run: choose the sampler the config names, then wrap it.

    Deliberately mirrors what training did, so an eval run sees data shaped the same way.
    daily_stats runs also open the ``-daily`` bucket here and widen the slice by the largest
    day offset, so a config asking for day -2 still has day -2 available at the start of the
    window. ``force_sampler_type`` overrides the config's choice, letting a rollout sample
    differently from how the checkpoint was trained.
    """
    regional_utils.patch_samplers_utils()
    training_config = setup.config["training_config"]
    input_step_hours = int(training_config["time_step"].removesuffix("h"))
    output_step_hours = int(training_config.get("output_step", training_config["time_step"]).removesuffix("h"))
    stride = training_config.get("stride", 1)
    data_step = training_config.get("data_step", "1h")
    data_step_hours = int(data_step.removesuffix("h"))
    sampler_type = force_sampler_type if force_sampler_type is not None else training_config.get("sampler_type", "fixed")
    input_duration_hours = int(setup.config["task_config"]["input_duration"].removesuffix("h"))
    num_inputs = input_duration_hours // input_step_hours
    if sampler_type == "daily_stats":
        task_config_dict = setup.config["task_config"]
        loading_region = get_loading_region(task_config_dict)
        daily_bucket = bucket_name(GENCAST_LEVELS, loading_region) + "-daily"
        print(f"[build_loader] Opening daily bucket {daily_bucket} ...")
        daily_dataset = xr.open_zarr(s3fs.S3Map(root=daily_bucket, s3=lumio(), check=False), consolidated=True).sortby("time")
        daily_dataset = format_and_slice(daily_dataset, loading_region)
        levels = list(task_config_dict["pressure_levels"])
        if levels and "level" in daily_dataset.dims:
            daily_dataset = daily_dataset.sel(level=levels)
        daily_names = []
        for name in task_config_dict["input_variables"]:
            if name.endswith(("_daily_max", "_daily_min")):
                daily_names.append(name)
            elif f"{name}_daily_mean" in daily_dataset.data_vars:
                daily_names.append(f"{name}_daily_mean")
        daily_dataset = daily_dataset[list(dict.fromkeys(daily_names))]
        day_offsets = training_config["day_offsets"]
        offset_margin_days = max(abs(min(day_offsets)), abs(max(day_offsets)))
        daily_dataset = daily_dataset.sel(time=slice(
            np.datetime64(time_start) - np.timedelta64(offset_margin_days, "D"),
            np.datetime64(time_end) + np.timedelta64(offset_margin_days, "D"),
        ))
        if resolution == 1.0:
            daily_dataset = daily_dataset.isel(lat=slice(None, None, 4), lon=slice(None, None, 4))
        daily_dataset = daily_dataset.load()
        sampler = DailyStatsSampler(
            task_config=setup.task,
            day_offsets=training_config["day_offsets"],
            rng=np.random.default_rng(seed),
            daily_dataset=daily_dataset,
            data_step_hours=data_step_hours,
        )
    elif sampler_type == "variable":
        max_horizon = int(setup.config["lead_time_encoder_config"]["max_horizon_hours"])
        lead_times = list(range(output_step_hours, max_horizon + 1, output_step_hours))
        sampler = VariableLeadTimeSampler(
            task_config=setup.task,
            num_inputs=num_inputs,
            step_size=input_step_hours,
            lead_times=lead_times,
            rng=np.random.default_rng(seed),
            data_step_hours=data_step_hours,
        )
    else:
        resolved_ar_steps = ar_steps if ar_steps is not None else training_config.get("ar_steps", 1)
        sampler = FixedLeadTimeSampler(
            task_config=setup.task,
            num_inputs=num_inputs,
            step_size=input_step_hours,
            lead_time=output_step_hours,
            num_output=resolved_ar_steps,
            data_step_hours=data_step_hours,
        )
    return Era5DataLoaderV2(
        time_start=time_start,
        time_end=time_end,
        config=setup.config,
        sampler=sampler,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
        stride=stride,
        resolution=resolution,
        preload=preload,
        data_step=data_step,
    )
 
  
def to_forecast_hour(dataset, abs_target_times, hours_arr, forecast_time):
    """Re-index one rollout from lead time to absolute hour, under a forecast coordinate.

    Rollouts come back indexed by lead time, but predictions are stored as
    (forecast, hour) so forecasts from different initialisations can later be aligned on
    the same valid time.
    """
    return (
        dataset
        .assign_coords(time=abs_target_times)
        .assign_coords(hour=("time", hours_arr))
        .swap_dims({"time": "hour"})
        .drop_vars("time", errors="ignore")
        .expand_dims({"forecast": [forecast_time]})
    )
 
 
def print_config(setup, args, target_lead: int, region, out_dir: str):
    """Print the resolved eval configuration, so the Slurm log records exactly what ran."""
    task_config = setup.config["task_config"]
    separator = "=" * 60
    target_variables = list(task_config.get("target_surface_variables", [])) + list(task_config.get("target_atmospheric_variables", []))
    print(separator)
    print("  EVAL CONFIG")
    print(separator)
    print(f"\n[Model]")
    print(f"  run_id: {args.run_id}")
    print(f"  checkpoint: {args.model_path}")
    print(f"\n[What the model sees]")
    print(f"  input_duration: {task_config['input_duration']}")
    print(f"  target_lead_time: {target_lead}h")
    print(f"  input_variables: {list(task_config['input_variables'])}")
    print(f"  target_variables: {target_variables}")
    print(f"  forcing_variables: {list(task_config['forcing_variables'])}")
    print(f"  pressure_levels: {task_config['pressure_levels'] or 'surface-only'}")
    if region:
        print(f"  region: lat=[{region[0]},{region[1]}] lon=[{region[2]},{region[3]}]")
    else:
        print(f"  region: global")
    print(f"\n[Eval run]")
    print(f"  date range: {args.start_date} → {args.end_date}")
    print(f"  ensemble members: {args.num_ensemble}")
    ar_steps = getattr(args, "autoregressive_steps", None) or getattr(args, "forecast_hours", "N/A")
    print(f"  autoregressive steps: {ar_steps}")
    print(f"  rng seed: {args.seed}")
    print(f"  output dir: {out_dir}")
    print(separator + "\n")


def save_to_s3(local_path: str, run_id: str):
    """Copy a finished prediction file to LUMI-O; a failed upload warns and keeps the local copy.

    Deliberately non-fatal: the local file under Predictions/ is the one the report layer
    reads, so losing the upload must not throw away a completed rollout.
    """
    try:
        fs = lumio()
        dest = f"{S3_PRED_ROOT}/{run_id}/{os.path.basename(local_path)}"
        print(f"  uploading {local_path} → s3://{dest} ...")
        fs.put(local_path, dest)
        print(f"  upload done")
    except Exception as error:
        print(f"  WARNING: S3 upload failed ({error}). File kept locally.")
