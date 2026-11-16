"""Google's pretrained GenCast as the exp1 reference. Run once; step 3 of the pipeline.

Loads DeepMind's published 0.25 degree checkpoint straight from their GCS bucket and runs it
through the regional architecture, giving RQ1 the comparison it needs: does a regionally
trained model beat the global model it was derived from, on the same region and the same days.

Two things make this different from eval.py. The task config comes from the checkpoint rather
than from one of our JSONs, since the pretrained weights expect exactly the variables they
were trained on - including total_precipitation_12hr, which WeatherBench2 does not carry and
``build_tp12_store.py`` exists to supply. And the statistics are Google's published ones, for
the same reason.

Writes under Predictions/exp1_original_weights/. Computes no metrics.
"""

import argparse
import dataclasses
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

from google.cloud import storage
from graphcast import rollout
from graphcast import xarray_jax
from graphcast import checkpoint as gcs_checkpoint
from graphcast import gencast as original_gencast

from modified_Gencast import regional_utils
from modified_Gencast import denoiser as modified_denoiser
from modified_Gencast.gencast import CheckPoint, load_config
from modified_Gencast.helper_fun import build_loader, parse_steps, parse_region, build_predictor_eval, to_forecast_hour, print_config, save_to_s3
from gencast_infernce.config import DEFAULT_OUTPUT

TARGET_VAR = "2m_temperature"
RUN_ID = "exp1_original_weights"
LOADER_CONFIG = "experiment_2_all_gencast.json"
GCS_MODEL = "GenCast 0p25deg <2019.npz"

# the original config classes are missing the fields this project added, so the checkpoint is
# copied field by field into our versions instead of imported as-is
def repack(modified_cls, original_instance, **overrides):
    fields = {f.name: getattr(original_instance, f.name)
             for f in dataclasses.fields(original_instance)}
    fields.update(overrides)
    return modified_cls(**fields)

def load_original_checkpoint() -> CheckPoint:
    """Download the real GenCast checkpoint from GCS and repack it into this
    project's CheckPoint (region-crop-aware GenCast)."""
    gcs_bucket = storage.Client.create_anonymous_client().get_bucket("dm_graphcast")
    with gcs_bucket.blob(f"gencast/params/{GCS_MODEL}").open("rb") as f:
        raw = gcs_checkpoint.load(f, original_gencast.CheckPoint)

    denoiser_cfg = repack(modified_denoiser.DenoiserArchitectureConfig, raw.denoiser_architecture_config)
    denoiser_cfg.sparse_transformer_config.attention_type = "triblockdiag_mha"
    denoiser_cfg.sparse_transformer_config.mask_type = "full"
    noise_encoder_cfg = repack(modified_denoiser.NoiseEncoderConfig, raw.noise_encoder_config)
    return CheckPoint(
        description=raw.description, license=raw.license, params=raw.params,
        task_config=raw.task_config, denoiser_architecture_config=denoiser_cfg,
        sampler_config=raw.sampler_config, noise_config=raw.noise_config,
        noise_encoder_config=noise_encoder_cfg, lead_time_encoder_config=None,
    )


def main():
    parser = argparse.ArgumentParser(description="Eval the original GenCast checkpoint, exp2-style output.")
    parser.add_argument("--start-date", type=str, required=True)
    parser.add_argument("--end-date", type=str, required=True)
    parser.add_argument("--num-ensemble", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--autoregressive-steps", type=int, default=14)
    parser.add_argument("--no-s3", action="store_true")
    args = parser.parse_args()

    regional_utils.patch_samplers_utils()

    setup = load_config(LOADER_CONFIG)
    _, output_step = parse_steps(setup)
    region = parse_region(setup)
    boundary_margin = 0.0

    out_dir = os.path.join(DEFAULT_OUTPUT, RUN_ID)
    os.makedirs(out_dir, exist_ok=True)
    # the 0.25 in the name is the checkpoint resolution and is kept so the file name matches the existing cache
    tag = f"{RUN_ID}_0.25_{args.start_date}_{args.end_date}_ar{args.autoregressive_steps}_freerun"
    pred_path = os.path.join(out_dir, f"predictions_{tag}.nc")
    target_path = os.path.join(out_dir, f"targets_{tag}.nc")

    args.run_id = RUN_ID
    args.model_path = f"gcs:{GCS_MODEL}"
    print_config(setup, args, output_step, region, out_dir)

    print(f"Downloading original GenCast checkpoint ({GCS_MODEL}) from GCS...")
    t0 = time.time()
    ckpt = load_original_checkpoint()
    print(f"  loaded in {time.time()-t0:.1f}s")

    print(f"Building data loader (region-cropped, {args.start_date} -> {args.end_date}, from {LOADER_CONFIG}) ...")
    loader = build_loader(setup, args.start_date, args.end_date, args.seed, ar_steps=args.autoregressive_steps)
    
    # keeps only the forecasts whose inputs are 00 and 12 UTC, so every forecast starts at 12 UTC like exp2
    window_hours = (loader.dataset.time.values[loader.window_starts].astype("datetime64[h]").astype(np.int64)) % 24
    loader.window_starts = loader.window_starts[window_hours == 0]
    num_windows = len(loader)

    print("Loading normalization stats (real GenCast GCS stats, via all_gencast's total_precipitation_12hr branch)...")
    from modified_Gencast.stat_loader import load_stats
    stats = load_stats(setup.config)

    # sets up the forecast function and gives every ensemble member its own random key
    @hk.transform_with_state
    def run_forward(inputs, targets_template, forcings):
        return build_predictor_eval(stats, ckpt, boundary_margin)(inputs, targets_template=targets_template, forcings=forcings)

    state = {}
    run_forward_jitted = jax.jit(lambda rng, i, t, f: run_forward.apply(ckpt.params, state, rng, i, t, f)[0])
    run_forward_pmap = xarray_jax.pmap(run_forward_jitted, dim="sample")

    rng = jax.random.PRNGKey(args.seed)
    rngs = np.stack([jax.random.fold_in(rng, i) for i in range(args.num_ensemble)], axis=0)

    print(f"\nRunning inference ({num_windows} windows x {args.autoregressive_steps} step(s), original weights)...")
    all_predictions = []
    all_targets = []

    for win_idx, batch in enumerate(loader):
        t1 = time.time()
        abs_target_times = np.asarray(batch["conditioning"]["abs_target_times"]).squeeze()
        inputs = batch["inputs"]
        targets = batch["targets"]
        forcings = batch["forcings"]

        forecast_time = abs_target_times[0] - np.timedelta64(output_step, "h")
        hours_arr = np.arange(1, args.autoregressive_steps + 1) * output_step
        targets_template = targets * np.nan  # free-running: nothing truth-forced

        # rolls the model forward one 12h step at a time until the full week is covered
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

        # relabels the week of output by lead hour and files it under this forecast's start time
        prediction = xr.combine_by_coords(chunks).rename({"sample": "ensemble"})
        targets_abs = targets
        if "batch" in targets_abs.dims:
            targets_abs = targets_abs.squeeze("batch", drop=True)

        prediction = to_forecast_hour(prediction, abs_target_times, hours_arr, forecast_time)
        targets_abs = to_forecast_hour(targets_abs, abs_target_times, hours_arr, forecast_time)

        all_predictions.append(prediction[[TARGET_VAR]])
        all_targets.append(targets_abs[[TARGET_VAR]])

        if win_idx % 10 == 0 or win_idx == num_windows - 1:
            print(f"  window {win_idx+1}/{num_windows}  "
                 f"forecast={np.datetime_as_string(forecast_time, unit='h')}  "
                 f"time={time.time()-t1:.1f}s")
            
    predictions_ds = xr.concat(all_predictions, dim="forecast")
    targets_ds = xr.concat(all_targets, dim="forecast")
    print(f"\nSaving predictions -> {pred_path}")
    predictions_ds.to_netcdf(pred_path)
    targets_ds.to_netcdf(target_path)
    if not args.no_s3:
        save_to_s3(pred_path, RUN_ID)
        save_to_s3(target_path, RUN_ID)


if __name__ == "__main__":
    main()
