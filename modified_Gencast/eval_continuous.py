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
jax.config.update("jax_default_matmul_precision", "highest")

sys.path.insert(0, '/project/project_465002687/Masters-Thesis/Googles_gencast')
sys.path.insert(0, '/project/project_465002687/Masters-Thesis')

from graphcast import xarray_jax
from modified_Gencast import regional_utils
from modified_Gencast.stat_loader import load_stats
from modified_Gencast.ou_noise import build_noise_stacked
from modified_Gencast.report_diurnal import run_diurnal_report
from modified_Gencast.helper_fun import (
    run_settings, parse_steps, parse_region, build_loader,
    load_checkpoint, build_predictor_eval, to_forecast_hour, save_to_s3, print_config,
)
from gencast_infernce.config import DEFAULT_OUTPUT

TARGET_VAR = "2m_temperature"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--start-date", type=str, required=True)
    parser.add_argument("--end-date", type=str, required=True)
    parser.add_argument("--num-ensemble", type=int, default=8)
    parser.add_argument("--forecast-hours", type=int, default=168, help="Total horizon in hours, AR-rolled in chunks of max_horizon_hours.")
    parser.add_argument("--noise-method", type=str, default="ou", choices=["ou", "fixed"])
    parser.add_argument("--ou-rho", type=float, default=0.1)
    parser.add_argument("--stochastic-churn-rate", type=float, default=None,
                        help="Override the sampler's stochastic_churn_rate at inference. 0 turns churn off (deterministic per-step reverse diffusion)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-s3", action="store_true")
    parser.add_argument("--plots-only", action="store_true", help="Load cached predictions; skip inference.")
    parser.add_argument("--boundary-margin-degrees", type=float, default=0.0,
                        help="Lateral boundary forcing margin in degrees. Default 0 disables truth injection at the domain edge (free-running rollout).")
    args = parser.parse_args()

    model_path, setup = run_settings(args.run_id, args.step)
    args.model_path = model_path
    if args.stochastic_churn_rate is not None:
        setup.sampler_config = setup.sampler_config.replace(stochastic_churn_rate=args.stochastic_churn_rate)
    if setup.lead_time_encoder_config is None:
        raise ValueError("eval_continuous.py requires a time-conditioned model.")
    if setup.config["training_config"].get("sampler_type", "fixed") == "daily_stats":
        raise ValueError("Daily-stats (exp4) models take daily aggregates as inputs; use the daily eval, not eval_continuous.py.")

    input_step, output_step = parse_steps(setup)
    region = parse_region(setup)
    boundary_margin = args.boundary_margin_degrees
    max_horizon_hours = int(setup.config["lead_time_encoder_config"]["max_horizon_hours"])

    chunk_size = max_horizon_hours
    num_ar_steps = int(np.ceil(args.forecast_hours / chunk_size))
    total_hours = num_ar_steps * chunk_size 

    run_id = args.run_id
    out_dir = os.path.join(DEFAULT_OUTPUT, run_id)
    os.makedirs(out_dir, exist_ok=True)

    ckpt_step = os.path.splitext(os.path.basename(model_path))[0]
    tag = (
        f"{run_id}_{ckpt_step}_{args.start_date}_{args.end_date}"
        f"_AR{total_hours}h_{args.noise_method}"
        + (f"_rho{args.ou_rho}" if args.noise_method == "ou" else "")
        + (f"_churn{args.stochastic_churn_rate}" if args.stochastic_churn_rate is not None else "")
        + (f"_margin{boundary_margin}" if boundary_margin > 0 else "")
    )
    print_config(setup, args, output_step, region, out_dir)

    pred_path = os.path.join(out_dir, f"predictions_{tag}.nc")
    target_path  = os.path.join(out_dir, f"targets_{tag}.nc")

    if args.plots_only:
        if not os.path.exists(pred_path):
            raise FileNotFoundError(f"--plots-only requires cached predictions at {pred_path}")
        if not os.path.exists(target_path):
            raise FileNotFoundError(f"--plots-only requires cached targets at {target_path}")
        print(f"\nLoading cached predictions → {pred_path}")
        predictions_ds = xr.open_dataset(pred_path, chunks={"forecast": 24})
        targets_ds = xr.open_dataset(target_path, chunks={"forecast": 24})
        run_diurnal_report(
            predictions_ds, targets_ds, region, args.num_ensemble,
            args.start_date, run_id, out_dir, tag, args.no_s3, layout="rollout",
            autoregressive_steps=total_hours, ar_window_hours=chunk_size)
        return

    loader = build_loader(setup, args.start_date, args.end_date, args.seed, ar_steps=total_hours, force_sampler_type="fixed")
    window_hours = (loader.dataset.time.values[loader.window_starts].astype("datetime64[h]").astype(np.int64)) % 24
    loader.window_starts = loader.window_starts[(window_hours == 0) | (window_hours == 12)]
    num_windows = len(loader)

    print("Loading checkpoint...")
    ckpt = load_checkpoint(model_path, setup)
    params = ckpt.params

    print("Loading normalization stats...")
    stats = load_stats(setup.config)
    if isinstance(stats["diffs_stddev_by_level"], dict):
        stats["diffs_stddev_by_level"] = {
            lead: stats["diffs_stddev_by_level"][lead]
            for lead in sorted(stats["diffs_stddev_by_level"].keys())[:chunk_size]
        }

    @hk.transform_with_state
    def run_forward(inputs, targets_template, forcings, initial_noise, lead_time_hours):
        return build_predictor_eval(stats, ckpt, boundary_margin)(
            inputs, targets_template=targets_template, forcings=forcings,
            initial_noise=initial_noise, lead_time_hours=lead_time_hours,
        )

    state = {}
    run_forward_jitted = jax.jit(lambda rng, inputs, targets_template, forcings, noise, lead_time_hours: run_forward.apply(params, state, rng, inputs, targets_template, forcings, noise, lead_time_hours)[0])
    run_forward_pmap = xarray_jax.pmap(lambda rng, inputs, targets_template, forcings, noise, lead_time_hours: run_forward_jitted(rng, inputs, targets_template, forcings, noise, lead_time_hours), dim="sample")

    rng = jax.random.PRNGKey(args.seed)
    lead_time_schedule = jnp.tile(jnp.arange(1, chunk_size + 1, dtype=jnp.float32)[None, :], (args.num_ensemble, 1))  # shape (num_ensemble, chunk_size) 


    placeholder_time = [np.timedelta64(1, "h")]
    input_times = np.array([np.timedelta64(-input_step, "h"), np.timedelta64(0, "h")])  # [-12h, 0h], matches sampler
    
    input_var_list = list(setup.task.input_variables)
    non_temp_vars = [v for v in input_var_list if v != TARGET_VAR]
    forcing_vars = list(setup.task.forcing_variables)

    print(f"\nPer-member AR: {num_windows} windows x {num_ar_steps} steps x {chunk_size}h "
          f"x {args.num_ensemble} members → {total_hours}h")

    flush_every = 25
    part_dir = os.path.join(out_dir, f"parts_{tag}")
    os.makedirs(part_dir, exist_ok=True)
    existing_parts = sorted(name for name in os.listdir(part_dir)
                            if name.startswith("pred_") and name.endswith(".nc"))
    prediction_parts = [os.path.join(part_dir, name) for name in existing_parts]
    target_parts = [os.path.join(part_dir, f"tgt_{i:04d}.nc") for i in range(len(existing_parts))]
    windows_done = len(existing_parts) * flush_every
    if windows_done:
        print(f"[resume] {len(existing_parts)} parts on disk → skipping the first {windows_done} windows")
        loader.window_starts = loader.window_starts[windows_done:]
    prediction_buffer, target_buffer = [], []
    wall_time_wind = time.time()

    for win_idx, batch in enumerate(loader):
        window_timestamps = np.asarray(batch["conditioning"]["abs_target_times"]).squeeze()
        inputs = batch["inputs"].squeeze("batch", drop=True) if "batch" in batch["inputs"].dims else batch["inputs"]
        targets = batch["targets"].squeeze("batch", drop=True) if "batch" in batch["targets"].dims else batch["targets"]
        window_start = window_timestamps[0] - np.timedelta64(output_step, "h")

        member_inputs = [inputs for _ in range(args.num_ensemble)]
        chunk_start = window_start
        chunck_predictions = [] 
        for step in range(num_ar_steps):
            t_step = time.time()
            rng_offset = (win_idx * num_ar_steps + step) * args.num_ensemble
            member_rng_pairs = [jax.random.split(jax.random.fold_in(rng, rng_offset + m)) for m in range(args.num_ensemble)]
            noise_rng_keys = [p[0] for p in member_rng_pairs]
            model_rng_keys = [p[1] for p in member_rng_pairs]

            if boundary_margin and boundary_margin > 0:
                margin = regional_utils.boundary_margin_mask(targets, boundary_margin)
                targets_template = xr.concat(
                    [targets.isel(time=[step * chunk_size + h]).where(margin)
                     .assign_coords(time=placeholder_time).expand_dims({"batch": [h]})
                     for h in range(chunk_size)], dim="batch")
            else:
                targets_template = xr.concat(
                    [(targets.isel(time=[0]).assign_coords(time=placeholder_time) * np.nan)
                     .expand_dims({"batch": [h]}) for h in range(chunk_size)], dim="batch")

            chunk_timestamps = np.array([chunk_start + np.timedelta64(h + 1, "h") for h in range(chunk_size)])
            chunk_forcings = loader.dataset[forcing_vars].sel(time=chunk_timestamps).compute()
            chunk_forcings = xr.concat(
                [chunk_forcings.isel(time=[h]).assign_coords(time=placeholder_time)
                 .expand_dims({"batch": [h]}) for h in range(chunk_size)], dim="batch")

            inputs_stacked = xr.concat(
                [xr.concat([member_inputs[m].expand_dims({"batch": [b]}) for b in range(chunk_size)], dim="batch")
                 .expand_dims({"sample": [m]}) for m in range(args.num_ensemble)], dim="sample")
            template_stacked = xr.concat([targets_template.expand_dims({"sample": [m]}) for m in range(args.num_ensemble)], dim="sample")
            forcings_stacked = xr.concat([chunk_forcings.expand_dims({"sample": [m]}) for m in range(args.num_ensemble)], dim="sample")

            noise_templates = [ targets.isel(time=[0]).assign_coords(time=[np.timedelta64(h + 1, "h")]) for h in range(chunk_size)]
            noise_stacked = build_noise_stacked( noise_templates, args.ou_rho, args.noise_method, noise_rng_keys)
            rngs_stacked = jnp.stack(model_rng_keys)

            chunk_predictions = run_forward_pmap(rngs_stacked, inputs_stacked, template_stacked, forcings_stacked, noise_stacked, lead_time_schedule)

            chunk_predictions = xr.concat(
                [chunk_predictions.isel(sample=m, drop=True).squeeze("time", drop=True)
                 .rename({"batch": "time"})
                 .assign_coords(time=[np.timedelta64(h + 1, "h") for h in range(chunk_size)])
                 .expand_dims({"ensemble": [m]}) for m in range(args.num_ensemble)], dim="ensemble")
            chunk_predictions = xr.Dataset({v: xr.DataArray(np.asarray(chunk_predictions[v].values), dims=chunk_predictions[v].dims, coords=chunk_predictions[v].coords) for v in chunk_predictions.data_vars})
            chunck_predictions.append(chunk_predictions)

            if step < num_ar_steps - 1:
                next_chunk_start = chunk_start + np.timedelta64(chunk_size, "h")
                next_input_times = np.array([next_chunk_start - np.timedelta64(input_step, "h"), next_chunk_start])
                next_non_temp = (loader.dataset[non_temp_vars]
                                 .sel(time=next_input_times)
                                 .assign_coords(time=input_times)
                                 .compute())
                next_member_inputs = []
                for m in range(args.num_ensemble):
                    t_prev = chunk_predictions[TARGET_VAR].isel(ensemble=m, time=chunk_size - input_step - 1)
                    t_last = chunk_predictions[TARGET_VAR].isel(ensemble=m, time=chunk_size - 1)
                    temp_frame = xr.DataArray(
                        np.stack([t_prev.values, t_last.values], axis=0),
                        dims=["time", "lat", "lon"],
                        coords={"time": input_times, "lat": t_prev.lat, "lon": t_prev.lon},
                        name=TARGET_VAR)
                    next_member_inputs.append(xr.merge([next_non_temp, temp_frame])[input_var_list])
                member_inputs = next_member_inputs
                chunk_start = next_chunk_start

            if win_idx == 0:
                print(f"  [timing] macro step {step}: {time.time() - t_step:.1f}s")

        prediction = xr.concat(chunck_predictions, dim="time").assign_coords(time=[np.timedelta64(h + 1, "h") for h in range(total_hours)])
        hours_arr = np.arange(1, total_hours + 1)
        targets_abs = targets.isel(time=slice(0, total_hours))
        prediction = to_forecast_hour(prediction, window_timestamps[:total_hours], hours_arr, window_start)
        targets_abs = to_forecast_hour(targets_abs, window_timestamps[:total_hours], hours_arr, window_start)
        prediction_buffer.append(prediction)
        target_buffer.append(targets_abs)
        if len(prediction_buffer) >= flush_every:
            pred_part = os.path.join(part_dir, f"pred_{len(prediction_parts):04d}.nc")
            tgt_part = os.path.join(part_dir, f"tgt_{len(target_parts):04d}.nc")
            xr.concat(prediction_buffer, dim="forecast").to_netcdf(pred_part)
            xr.concat(target_buffer, dim="forecast").to_netcdf(tgt_part)
            prediction_parts.append(pred_part)
            target_parts.append(tgt_part)
            prediction_buffer.clear()
            target_buffer.clear()

        if win_idx % 5 == 0 or win_idx == num_windows - 1:
            print(f"  window {win_idx + 1}/{num_windows}  "
                  f"forecast={np.datetime_as_string(window_start, unit='h')}  "
                  f"{time.time() - wall_time_wind:.1f}s")
            wall_time_wind = time.time()

    # Flush the final partial buffer.
    if prediction_buffer:
        pred_part = os.path.join(part_dir, f"pred_{len(prediction_parts):04d}.nc")
        tgt_part = os.path.join(part_dir, f"tgt_{len(target_parts):04d}.nc")
        xr.concat(prediction_buffer, dim="forecast").to_netcdf(pred_part)
        xr.concat(target_buffer, dim="forecast").to_netcdf(tgt_part)
        prediction_parts.append(pred_part)
        target_parts.append(tgt_part)
        prediction_buffer.clear()
        target_buffer.clear()

    print(f"\nSaving predictions → {pred_path}")
    xr.open_mfdataset(prediction_parts, combine="nested", concat_dim="forecast").to_netcdf(pred_path)
    xr.open_mfdataset(target_parts, combine="nested", concat_dim="forecast").to_netcdf(target_path)
    for part in prediction_parts + target_parts:
        os.remove(part)
    
    os.rmdir(part_dir)
    
    if not args.no_s3:
        save_to_s3(pred_path, run_id)
        save_to_s3(target_path, run_id)

    predictions_ds = xr.open_dataset(pred_path)
    targets_ds = xr.open_dataset(target_path)

    run_diurnal_report(
        predictions_ds, targets_ds, region, args.num_ensemble,
        args.start_date, run_id, out_dir, tag, args.no_s3, layout="rollout",
        autoregressive_steps=total_hours, ar_window_hours=chunk_size)


if __name__ == "__main__":
    main()
