"""Fixed-lead rollout. Step 3 of the pipeline, for exp1 and exp2. MODE=exp2 in lumi_eval.sh.

Rolls a checkpoint forward in fixed 12h steps from one initialisation per day, which is the
layout the exp2 variable-set sweep and the boundary-margin ablation are scored on. Inputs are
the 00 and 12 UTC frames and the forecast starts at 12 UTC.

Writes ``predictions_<tag>.nc`` and ``targets_<tag>.nc`` under Predictions/<run-id>/, indexed
by (forecast, hour) so forecasts from different days can be aligned on valid time later. The
grid written is the full loading region, boundary belt included - the report layer is what
crops it off. Computes no metrics.

    --autoregressive-steps     how many 12h steps to roll (14 covers a week)
    --boundary-margin-degrees  >0 overwrites the belt with ERA5 each step; default 0 is off
"""

import argparse
import os
import sys
import time

import jax
jax.config.update("jax_default_matmul_precision", "highest")
import numpy as np
import xarray as xr
import haiku as hk

REPOSITORY_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(REPOSITORY_ROOT, 'Googles_gencast'))
sys.path.insert(0, REPOSITORY_ROOT)

from graphcast import rollout
from graphcast import xarray_jax

from modified_Gencast import regional_utils
from modified_Gencast.stat_loader import load_stats
from modified_Gencast.helper_fun import build_loader, run_settings, parse_steps, parse_region, load_checkpoint, build_predictor_eval, to_forecast_hour, print_config, save_to_s3
from gencast_infernce.config import DEFAULT_OUTPUT

TARGET_VAR = "2m_temperature"


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained GenCast checkpoint.")
    parser.add_argument("--run-id",type=str, required=True)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--start-date", type=str, required=True)
    parser.add_argument("--end-date", type=str, required=True)
    parser.add_argument("--num-ensemble", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--autoregressive-steps", type=int, default=1)
    parser.add_argument("--no-s3", action="store_true")
    parser.add_argument("--boundary-margin-degrees", type=float, default=0.0)
    args = parser.parse_args()

    model_path, setup = run_settings(args.run_id, args.step)
    args.model_path = model_path

    if setup.lead_time_encoder_config is not None:
        raise ValueError("Use eval_continuous.py for time-conditioned models.")

    _, output_step = parse_steps(setup)
    region = parse_region(setup)
    boundary_margin = args.boundary_margin_degrees

    run_id  = args.run_id
    out_dir = os.path.join(DEFAULT_OUTPUT, run_id)
    os.makedirs(out_dir, exist_ok=True)

    ckpt_step = os.path.splitext(os.path.basename(model_path))[0]
    tag = (f"{run_id}_{ckpt_step}_{args.start_date}_{args.end_date}_ar{args.autoregressive_steps}"
           + (f"_margin{boundary_margin}" if boundary_margin > 0 else ""))

    pred_path = os.path.join(out_dir, f"predictions_{tag}.nc")
    target_path = os.path.join(out_dir, f"targets_{tag}.nc")

    print_config(setup, args, output_step, region, out_dir)

    loader = build_loader( setup, args.start_date, args.end_date, args.seed, ar_steps=args.autoregressive_steps)
    window_hours = (loader.dataset.time.values[loader.window_starts].astype("datetime64[h]").astype(np.int64)) % 24
    loader.window_starts = loader.window_starts[window_hours == 0]
    num_windows = len(loader)
    print("Loading checkpoint...")
    ckpt   = load_checkpoint(model_path, setup)
    params = ckpt.params

    print("Loading normalization stats...")
    stats = load_stats(setup.config)

    @hk.transform_with_state
    def run_forward(inputs, targets_template, forcings):
        return build_predictor_eval(stats, ckpt, boundary_margin)(
            inputs, targets_template=targets_template, forcings=forcings)

    state = {}
    run_forward_jitted = jax.jit(lambda rng, i, t, f: run_forward.apply(params, state, rng, i, t, f)[0])
    run_forward_pmap = xarray_jax.pmap(run_forward_jitted, dim="sample")
    rng  = jax.random.PRNGKey(args.seed)
    rngs = np.stack([jax.random.fold_in(rng, i) for i in range(args.num_ensemble)], axis=0)

    print(f"\nRunning inference ({num_windows} windows x {args.autoregressive_steps} step(s))...")
    all_predictions = []
    all_targets     = []

    for win_idx, batch in enumerate(loader):
        t_batch = time.time()
        abs_target_times = np.asarray(batch["conditioning"]["abs_target_times"]).squeeze()
        inputs = batch["inputs"]
        targets = batch["targets"]
        forcings = batch["forcings"]
        if win_idx == 0:
            print(f"  [timing] batch load:        {time.time()-t_batch:.2f}s")
        t1 = time.time()

        forecast_time = abs_target_times[0] - np.timedelta64(output_step, "h")
        hours_arr = np.arange(1, args.autoregressive_steps + 1) * output_step

        t_rollout = time.time()

        if boundary_margin and boundary_margin > 0:
            targets_template = targets.where(regional_utils.boundary_margin_mask(targets, boundary_margin))
        else:
            targets_template = targets * np.nan
        chunks = []
        for chunk in rollout.chunked_prediction_generator_multiple_runs(
            predictor_fn=run_forward_pmap,
            rngs=rngs,
            inputs=inputs,
            targets_template=targets_template,
            forcings=forcings,
            num_steps_per_chunk=1,
            num_samples=args.num_ensemble,
            pmap_devices=jax.local_devices(),
        ):
            chunks.append(chunk)
        if win_idx == 0:
            jax.block_until_ready([np.asarray(da.values) for da in chunks[-1].data_vars.values()])
            print(f"  [timing] rollout:            {time.time()-t_rollout:.2f}s")

        t_combine = time.time()
        prediction  = xr.combine_by_coords(chunks).rename({"sample": "ensemble"})
        targets_abs = targets
        if "batch" in targets_abs.dims:
            targets_abs = targets_abs.squeeze("batch", drop=True)

        prediction  = to_forecast_hour(prediction,  abs_target_times, hours_arr, forecast_time)
        targets_abs = to_forecast_hour(targets_abs, abs_target_times, hours_arr, forecast_time)
        if win_idx == 0:
            print(f"  [timing] combine+post:       {time.time()-t_combine:.2f}s")

        all_predictions.append(prediction[[TARGET_VAR]])
        all_targets.append(targets_abs[[TARGET_VAR]])

        if win_idx % 10 == 0 or win_idx == num_windows - 1:
            print(f"  window {win_idx+1}/{num_windows}  "
                    f"forecast={np.datetime_as_string(forecast_time, unit='h')}  "
                    f"One Window time: {time.time()-t1:.1f}s")

    predictions_ds = xr.concat(all_predictions, dim="forecast") 
    targets_ds = xr.concat(all_targets,     dim="forecast")

    print(f"\nSaving predictions → {pred_path}")
    predictions_ds.to_netcdf(pred_path)
    targets_ds.to_netcdf(target_path)

    if not args.no_s3:
        save_to_s3(pred_path, run_id)
        save_to_s3(target_path, run_id)


if __name__ == "__main__":
    main()
