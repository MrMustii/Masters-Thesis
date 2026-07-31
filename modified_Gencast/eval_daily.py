import argparse
import os
import sys
import time

import jax
jax.config.update("jax_default_matmul_precision", "highest")
import jax.numpy as jnp
import numpy as np
import xarray as xr
import haiku as hk

sys.path.insert(0, '/project/project_465002687/Masters-Thesis/Googles_gencast')
sys.path.insert(0, '/project/project_465002687/Masters-Thesis')

from graphcast import xarray_jax
from modified_Gencast.stat_loader import load_stats
from modified_Gencast.ou_noise import build_noise_stacked
from modified_Gencast.report_diurnal import run_diurnal_report
from modified_Gencast.helper_fun import (
    run_settings, parse_region, build_loader, load_checkpoint,
    build_predictor_eval, save_to_s3,
)
from gencast_infernce.config import DEFAULT_OUTPUT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--start-date", type=str, required=True)
    parser.add_argument("--end-date", type=str, required=True)
    parser.add_argument("--num-ensemble", type=int, default=8)
    parser.add_argument("--noise-method", type=str, default="ou", choices=["ou", "fixed"])
    parser.add_argument("--ou-rho", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preload", action="store_true", help="Pull the hourly eval range into RAM.")
    parser.add_argument("--no-s3", action="store_true")
    parser.add_argument("--plots-only", action="store_true", help="Load cached predictions; skip inference.")
    args = parser.parse_args()

    model_path, setup = run_settings(args.run_id, args.step)
    args.model_path = model_path
    training_config = setup.config["training_config"]
    if training_config.get("sampler_type", "fixed") != "daily_stats":
        raise ValueError("eval_daily.py is for daily-stats (exp4) models; use eval.py / eval_continuous.py otherwise.")
    if setup.lead_time_encoder_config is None:
        raise ValueError("Daily-stats models must have a lead-time encoder (hour conditioning).")

    region = parse_region(setup)
    boundary_margin = setup.config["task_config"].get("boundary_margin_degrees", 0.0)

    run_id = args.run_id
    out_dir = os.path.join(DEFAULT_OUTPUT, run_id)
    os.makedirs(out_dir, exist_ok=True)
    ckpt_step = os.path.splitext(os.path.basename(model_path))[0]
    tag = (f"{run_id}_{ckpt_step}_{args.start_date}_{args.end_date}_daily_{args.noise_method}"
           + (f"_rho{args.ou_rho}" if args.noise_method == "ou" else ""))
    pred_path = os.path.join(out_dir, f"predictions_{tag}.nc")
    target_path = os.path.join(out_dir, f"targets_{tag}.nc")

    separator = "=" * 60
    print(separator)
    print("  DAILY DEAGGREGATION EVAL")
    print(separator)
    print(f"  run_id: {run_id}   checkpoint: {model_path}")
    print(f"  dates: {args.start_date} → {args.end_date}   members: {args.num_ensemble}")
    print(f"  day_offsets: {training_config['day_offsets']}   noise: {args.noise_method}"
          + (f" (rho={args.ou_rho})" if args.noise_method == "ou" else ""))
    print(f"  region: {region}   boundary margin: {boundary_margin} deg (cropped at metrics time)")
    print(f"  output: {out_dir}")
    print(separator + "\n")

    if args.plots_only:
        for path in (pred_path, target_path):
            if not os.path.exists(path):
                raise FileNotFoundError(f"--plots-only requires cached file {path}")
        print(f"Loading cached predictions → {pred_path}")
        run_diurnal_report(
            xr.open_dataset(pred_path), xr.open_dataset(target_path), region, args.num_ensemble,
            args.start_date, run_id, out_dir, tag, args.no_s3, layout="daily",
            model_path=model_path)
        return

    loader = build_loader(setup, args.start_date, args.end_date, args.seed, shuffle=False, preload=args.preload)
    sampler = loader.sampler
    if sampler.data_step_hours != 1:
        raise ValueError("eval_daily.py assumes an hourly store: batch row index must equal the hour.")

    daily_times = set(sampler.daily_dataset.time.values)
    valid_window_starts = []
    for start_idx in loader.window_starts:
        day_zero_index = int(start_idx) - sampler.day_offsets[0] * sampler.steps_per_day
        day_zero_date = loader.dataset.time.values[day_zero_index]
        offset_dates = [day_zero_date + np.timedelta64(offset * 24, "h") for offset in sampler.day_offsets]
        if all(date in daily_times for date in offset_dates):
            valid_window_starts.append(int(start_idx))
    num_days = len(valid_window_starts)
    if num_days < len(loader.window_starts):
        print(f"Dropped {len(loader.window_starts) - num_days} day(s) whose offset days are missing from the daily bucket.")
    if num_days == 0:
        raise ValueError("No evaluable days in the requested range.")

    print("Loading checkpoint...")
    ckpt = load_checkpoint(model_path, setup)
    params = ckpt.params

    print("Loading normalization stats...")
    stats = load_stats(setup.config)
    diffs_stddev = stats["diffs_stddev_by_level"]
    if not (isinstance(diffs_stddev, dict) and sorted(diffs_stddev.keys()) == list(range(24))):
        raise ValueError("Daily stats must be a dict keyed by hours 0..23 so sigma row == batch row == hour.")

    @hk.transform_with_state
    def run_forward(inputs, targets_template, forcings, initial_noise, lead_time_hours):
        return build_predictor_eval(stats, ckpt, 0.0, sampler_type="daily_stats")(
            inputs, targets_template=targets_template, forcings=forcings,
            initial_noise=initial_noise, lead_time_hours=lead_time_hours,
        )

    state = {}
    run_forward_jitted = jax.jit(
        lambda rng, inputs, targets_template, forcings, noise, lead_time_hours:
            run_forward.apply(params, state, rng, inputs, targets_template, forcings, noise, lead_time_hours)[0])
    run_forward_pmap = xarray_jax.pmap(
        lambda rng, inputs, targets_template, forcings, noise, lead_time_hours:
            run_forward_jitted(rng, inputs, targets_template, forcings, noise, lead_time_hours), dim="sample")

    rng = jax.random.PRNGKey(args.seed)
    hours = list(sampler.target_hours)  
    lead_time_schedule = jnp.tile(jnp.array(hours, dtype=jnp.float32)[None, :], (args.num_ensemble, 1))

    print(f"\nRunning inference: {num_days} days x {len(hours)} hours x {args.num_ensemble} members ...")
    all_predictions, all_targets = [], []
    wall_time = time.time()

    for day_index, start_idx in enumerate(valid_window_starts):
        hour_samples = []
        for hour in hours:
            sampler.batch_target_hour = hour
            hour_samples.append(sampler(loader.dataset, start_idx))

        day_inputs = xr.concat(
            [sample["inputs"].squeeze("batch", drop=True).expand_dims({"batch": [hour]})
             for hour, sample in zip(hours, hour_samples)], dim="batch")
        day_targets = xr.concat(
            [sample["targets"].squeeze("batch", drop=True).expand_dims({"batch": [hour]})
             for hour, sample in zip(hours, hour_samples)], dim="batch")
        day_forcings = xr.concat(
            [sample["forcings"].squeeze("batch", drop=True).expand_dims({"batch": [hour]})
             for hour, sample in zip(hours, hour_samples)], dim="batch")
        targets_template = day_targets * np.nan

        member_rng_pairs = [jax.random.split(jax.random.fold_in(rng, day_index * args.num_ensemble + m)) for m in range(args.num_ensemble)]
        noise_rng_keys = [pair[0] for pair in member_rng_pairs]
        model_rng_keys = [pair[1] for pair in member_rng_pairs]

        noise_templates = [
            sample["targets"].squeeze("batch", drop=True).assign_coords(time=[np.timedelta64(hour, "h")])
            for hour, sample in zip(hours, hour_samples)]
        noise_stacked = build_noise_stacked(noise_templates, args.ou_rho, args.noise_method, noise_rng_keys)

        inputs_stacked = xr.concat([day_inputs.expand_dims({"sample": [m]}) for m in range(args.num_ensemble)], dim="sample")
        template_stacked = xr.concat([targets_template.expand_dims({"sample": [m]}) for m in range(args.num_ensemble)], dim="sample")
        forcings_stacked = xr.concat([day_forcings.expand_dims({"sample": [m]}) for m in range(args.num_ensemble)], dim="sample")
        rngs_stacked = jnp.stack(model_rng_keys)

        day_prediction = run_forward_pmap(
            rngs_stacked, inputs_stacked, template_stacked, forcings_stacked, noise_stacked, lead_time_schedule)

        day_prediction = xr.concat(
            [day_prediction.isel(sample=m, drop=True).squeeze("time", drop=True)
             .rename({"batch": "hour"}).expand_dims({"ensemble": [m]})
             for m in range(args.num_ensemble)], dim="ensemble")
        day_prediction = xr.Dataset(
            {name: xr.DataArray(np.asarray(day_prediction[name].values), dims=day_prediction[name].dims,
                                coords=day_prediction[name].coords) for name in day_prediction.data_vars})

        day_date = hour_samples[0]["conditioning"]["abs_target_times"][0]
        day_targets_abs = day_targets.squeeze("time", drop=True).rename({"batch": "hour"})
        all_predictions.append(day_prediction.expand_dims({"forecast": [day_date]}))
        all_targets.append(day_targets_abs.expand_dims({"forecast": [day_date]}))

        if day_index % 5 == 0 or day_index == num_days - 1:
            print(f"  day {day_index + 1}/{num_days}  {np.datetime_as_string(day_date, unit='D')}  "
                  f"{time.time() - wall_time:.1f}s")
            wall_time = time.time()

    predictions_ds = xr.concat(all_predictions, dim="forecast")
    targets_ds = xr.concat(all_targets, dim="forecast")

    print(f"\nSaving predictions → {pred_path}")
    predictions_ds.to_netcdf(pred_path)
    targets_ds.to_netcdf(target_path)
    if not args.no_s3:
        save_to_s3(pred_path, run_id)
        save_to_s3(target_path, run_id)

    run_diurnal_report(
        predictions_ds, targets_ds, region, args.num_ensemble, args.start_date, run_id, out_dir, tag, args.no_s3,
        layout="daily", model_path=model_path)


if __name__ == "__main__":
    main()
