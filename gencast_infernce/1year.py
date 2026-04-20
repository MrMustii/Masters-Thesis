import time
t0 = time.time()
import dataclasses
import numpy as np
import xarray
import haiku as hk
import jax
import sys
import os
import pandas as pd
import argparse
from config import GENCAST_LIB, DEFAULT_OUTPUT, SCRATCH_PATH
from datatest import prepare_data, download_data
import datetime

sys.path.insert(0, GENCAST_LIB)
sys.path.insert(0, os.path.dirname(GENCAST_LIB))

from google.cloud import storage
from graphcast import rollout
from graphcast import xarray_jax
from graphcast import normalization
from graphcast import checkpoint
from graphcast import data_utils
from graphcast import gencast
from graphcast import nan_cleaning
from graphcast import denoiser as _denoiser_module
from ploting import plot_map
jax.config.update("jax_default_matmul_precision", "highest")
# jax.config.update("jax_debug_nans", True)

print("Imports done.")
print(f"Import time: {time.time()-t0:.1f}s")
print(jax.devices())


# SCRATCH_PATH =  "../../../scratch/project_465002687/data/era5/wb2_2022_subset.zarr"
# SCRATCH_PATH = "../../../scratch/project_465002687/data/era5/wb2_2022_week1_subset.zarr"
# SCRATCH_PATH = "../../../scratch/project_465002687/data/era5/wb2_2021-12-30_2023-01-01_12h_1deg.zarr"
MIN_LAT, MAX_LAT = 20., 82.
MIN_LON, MAX_LON = -30., 60.



###########################################################################

print("Running inference ...")

################################################### Everything abouve this is correct


def sanity_check_predictions(model="mini"):
    # load google data and weatherbench2 data for the same time period and compare them before and after runing the model
    print("-------------- SANITY CHECKS ON FINAL PREDICTIONS: --------------")
    gcs_client = storage.Client.create_anonymous_client()
    gcs_bucket = gcs_client.get_bucket("dm_graphcast")
    dir_prefix = "gencast/"
    degree = 0.25 if "0.25" in model else 1
    dataset_file = "source-era5_date-2019-03-29_res-1.0_levels-13_steps-01.nc" if degree== 1 else "source-era5_date-2019-03-29_res-0.25_levels-13_steps-01.nc"
    with gcs_bucket.blob(dir_prefix + f"dataset/{dataset_file}").open("rb") as f:
        google = xarray.load_dataset(f)
    print("=== Loaded Google sample batch ===")


    print("=== LoadedWeatherBench2 ERA5 (processed) ===")
    wb2 = prepare_data("gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr", deg=degree)
    wb2 = wb2.sel(time=slice("2019-03-29T00:00:00", "2019-03-30T00:00:00"))

    print("Check if datasets are the same")
    print("=== SHAPE ===")
    print(f"Google: lat={google.dims['lat']} lon={google.dims['lon']} time={google.dims['time']}")
    print(f"WB2:    lat={wb2.dims['lat']} lon={wb2.dims['lon']} time={wb2.dims['time']}")
    print(f"lat match: {google.dims['lat'] == wb2.dims['lat']}")
    print(f"lon match: {google.dims['lon'] == wb2.dims['lon']}")

    print("\n=== LAT/LON VALUES ===")
    print(f"Google lat: first={google.lat.values[0]} last={google.lat.values[-1]} step={google.lat.values[1]-google.lat.values[0]:.4f}")
    print(f"WB2    lat: first={wb2.lat.values[0]} last={wb2.lat.values[-1]} step={wb2.lat.values[1]-wb2.lat.values[0]:.4f}")
    print(f"Google lon: first={google.lon.values[0]} last={google.lon.values[-1]} step={google.lon.values[1]-google.lon.values[0]:.4f}")
    print(f"WB2    lon: first={wb2.lon.values[0]} last={wb2.lon.values[-1]} step={wb2.lon.values[1]-wb2.lon.values[0]:.4f}")

    print("\n=== TIME ===")
    print(f"Google time: {google.time.values}")
    print(f"WB2    time: {wb2.time.values}")

    print("\n=== VARS ===")
    google_vars = set(google.data_vars)
    wb2_vars = set(wb2.data_vars)
    print(f"In Google but not WB2: {google_vars - wb2_vars}")
    print(f"In WB2 but not Google: {wb2_vars - google_vars}")
    print("=== check for nans ===")
    for var in wb2.data_vars:
        num_nans_wb2 = wb2[var].isnull().sum().compute().item()
        num_nans_google = google[var].isnull().sum().compute().item()
        print(f"Variable '{var}' has {num_nans_wb2} NaNs in WB2 data and {num_nans_google} NaNs in Google data.")


    print("\n=== 10 POINT DATA VALUE COMPARISON ===")
    var = "2m_temperature" 

    lat_indices = np.linspace(0, len(google.lat) - 1, 10, dtype=int)
    lon_indices = np.linspace(0, len(google.lon) - 1, 10, dtype=int)

    num_time_steps = min(len(google.time), len(wb2.time))
    print(f"Time steps in Google: {len(google.time)} | Time steps in WB2: {len(wb2.time)}")

    for i in range(10):
        lat_idx = lat_indices[i]
        lon_idx = lon_indices[i]
        
        g_val = google[var].isel(time=1, lat=lat_idx, lon=lon_idx).values.squeeze().item()
        w_val = wb2[var].isel(time=1, lat=lat_idx, lon=lon_idx).values.squeeze().item()
        
        print(f"Point {i+1} (lat_idx={lat_idx}, lon_idx={lon_idx}): Google={g_val:.4f}, WB2={w_val:.4f}, Diff={abs(g_val - w_val):.4f}")

    
    
    print("\n=== RUNNING GOOGLE DATA THROUGH MODEL ===")
    rng = jax.random.PRNGKey(0)
    rngs = np.stack([jax.random.fold_in(rng, i) for i in range(NUM_ENSEMBLE)], axis=0)

    g_eval_inputs, g_eval_targets, g_eval_forcings = data_utils.extract_inputs_targets_forcings(
    google, target_lead_times=slice("12h", f"{(google.sizes['time']-2)*12}h"), **dataclasses.asdict(task_config)
    )
    g_eval_inputs = g_eval_inputs.compute()
    g_eval_targets = g_eval_targets.compute()
    g_eval_forcings = g_eval_forcings.compute()

    g_chunks = []
    for chunk in rollout.chunked_prediction_generator_multiple_runs(
        predictor_fn=run_forward_pmap,
        rngs=rngs, 
        inputs=g_eval_inputs,
        targets_template=g_eval_targets * np.nan, 
        forcings=g_eval_forcings, 
        num_steps_per_chunk=1,
        num_samples=1, 
        pmap_devices=jax.local_devices()
    ):
        g_chunks.append(chunk)
    g_pred = xarray.combine_by_coords(g_chunks)

    print("=== RUNNING WB2 DATA THROUGH MODEL ===")
    wb2 = wb2.assign_coords(datetime=wb2.time)
    data_utils.add_derived_vars(wb2)
    data_utils.add_tisr_var(wb2)
    wb2 = wb2.assign_coords(time=(wb2.time - wb2.time[0]).astype("timedelta64[ns]"))

    w_eval_inputs, w_eval_targets, w_eval_forcings = data_utils.extract_inputs_targets_forcings(
        wb2, target_lead_times=slice("12h", f"{(wb2.sizes['time']-2)*12}h"), **dataclasses.asdict(task_config)
    )
    w_eval_inputs = w_eval_inputs.expand_dims(batch=1).compute()
    w_eval_targets = w_eval_targets.expand_dims(batch=1).compute()
    w_eval_forcings = w_eval_forcings.expand_dims(batch=1).compute()

    w_chunks = []
    for chunk in rollout.chunked_prediction_generator_multiple_runs(
        predictor_fn=run_forward_pmap,
         rngs=rngs,
         inputs=w_eval_inputs,
         targets_template=w_eval_targets * np.nan,
         forcings=w_eval_forcings,
         num_steps_per_chunk=1,
         num_samples=1,
         pmap_devices=jax.local_devices()
    ):
        w_chunks.append(chunk)
    w_pred = xarray.combine_by_coords(w_chunks)


    print("\n=== 10 POINT PREDICTION VALUE COMPARISON ===")
    for i in range(10):
        lat_idx = lat_indices[i]
        lon_idx = lon_indices[i]
        
        g_val = g_pred[var].isel(batch=0, sample=0, time=0, lat=lat_idx, lon=lon_idx).values.item()
        w_val = w_pred[var].isel(batch=0, sample=0, time=0, lat=lat_idx, lon=lon_idx).values.item()
        
        print(f"Pred Point {i+1} (lat_idx={lat_idx}, lon_idx={lon_idx}): Google={g_val:.4f}, WB2={w_val:.4f}, Diff={abs(g_val - w_val):.4f}")
    print(f"---------------- Sanity checks on predictions completed. ----------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="mini", choices=["mini", "1","0.25","0.25_operational","modified"], help='which model checkpoint to use for inference (Choices: %(choices)s)')
    parser.add_argument('--sanity-check', action='store_true', help='whether to run sanity checks comparing google data and predictions to weatherbench2 data')
    parser.add_argument('--start-date', type=str, default="2022-01-01", help='start date for inference in ISO format (default: %(default)s)')
    parser.add_argument('--end-date', type=str, default="2022-01-08", help='end date for inference in ISO format (default: %(default)s)')
    parser.add_argument('--output-path', type=str, default=DEFAULT_OUTPUT, help='directory to save predictions in zarr format (default: %(default)s)')
    parser.add_argument('--num-ensemble', type=int, default=1, help='number of ensemble members to run in parallel (default: %(default)s)')
    parser.add_argument('--save-type', type=str, default="zarr", choices=["nc", "zarr"], help='output file format (default: %(default)s)')
    args = parser.parse_args()
    print("------- Arguments:-------")
    for key, value in sorted(vars(args).items()):
        print(key, '=', value)
    print("------------------------")

    # check if the arguments are valid
    try:       
        start_date = np.datetime64(args.start_date+"T00:00:00")
        end_date = np.datetime64(args.end_date+"T12:00:00")
        assert start_date < end_date, "start-date must be before end-date"
    except Exception as e:
        print(f"Invalid date format: {e}")
    # SCRATCH_PATH = "../../../scratch/project_465002687/data/era5/wb2_2021-12-30_2023-01-01_12h_1deg.zarr"
    degree = 0.25 if ("0.25" in args.model or args.model == "modified") else 1
    OUT_DIR = args.output_path
    os.makedirs(OUT_DIR, exist_ok=True)
    NUM_ENSEMBLE = args.num_ensemble



    ###################################################################################
    print("---------- Loading model... ----------")
    t0 = time.time()
    gcs_client = storage.Client.create_anonymous_client()
    gcs_bucket = gcs_client.get_bucket("dm_graphcast")
    dir_prefix = "gencast/"

    avalible_models = {
        "mini":"GenCast 1p0deg Mini <2019.npz",
        "1":"GenCast 1p0deg <2019.npz",
        "0.25":"GenCast 0p25deg <2019.npz",
        "0.25_operational":"GenCast 0p25deg Operational <2022.npz"
    }

    model_selected = avalible_models["0.25"] if args.model == "modified" else avalible_models[args.model]
    try:
        with gcs_bucket.blob(dir_prefix + f"params/{model_selected}").open("rb") as f:
            ckpt = checkpoint.load(f, gencast.CheckPoint)
    except Exception as e:
        print(f"Error loading model checkpoint: {e}")

    if args.model == "modified":
        from modified_Gencast.denoiser import ModifiedDenoiserArchitecture
        from modified_Gencast import regional_utils
        regional_utils.patch_samplers_utils()
        _denoiser_module._DenoiserArchitecture = ModifiedDenoiserArchitecture
        print("Modified denoiser architecture and sampler patches applied.")
    
    params = ckpt.params
    task_config = ckpt.task_config
    sampler_config = ckpt.sampler_config
    noise_config = ckpt.noise_config
    noise_encoder_config = ckpt.noise_encoder_config
    denoiser_architecture_config = ckpt.denoiser_architecture_config
    denoiser_architecture_config.sparse_transformer_config.attention_type = "triblockdiag_mha"
    denoiser_architecture_config.sparse_transformer_config.mask_type = "full"
    print(f"Model loaded in {time.time()-t0:.1f}s | {model_selected}")

    print("---------- Loading normalization stats... ----------")
    t0 = time.time()
    with gcs_bucket.blob(dir_prefix + "stats/diffs_stddev_by_level.nc").open("rb") as f:
        diffs_stddev_by_level = xarray.load_dataset(f).compute()
    with gcs_bucket.blob(dir_prefix + "stats/mean_by_level.nc").open("rb") as f:
        mean_by_level = xarray.load_dataset(f).compute()
    with gcs_bucket.blob(dir_prefix + "stats/stddev_by_level.nc").open("rb") as f:
        stddev_by_level = xarray.load_dataset(f).compute()
    with gcs_bucket.blob(dir_prefix + "stats/min_by_level.nc").open("rb") as f:
        min_by_level = xarray.load_dataset(f).compute()
    print(f"Stats loaded in {time.time()-t0:.1f}s")


    # INPUT: None (uses global configurations and loaded statistics).
    # OUTPUT: A configured Haiku model object wrapped with normalization and NaN-handling layers.
    def construct_wrapped_gencast():
        predictor = gencast.GenCast(
            sampler_config=sampler_config,
            task_config=task_config,
            denoiser_architecture_config=denoiser_architecture_config,
            noise_config=noise_config,
            noise_encoder_config=noise_encoder_config,
        )
        predictor = normalization.InputsAndResiduals(
            predictor,
            diffs_stddev_by_level=diffs_stddev_by_level,
            mean_by_level=mean_by_level,
            stddev_by_level=stddev_by_level,
        )
        predictor = nan_cleaning.NaNCleaner(
            predictor=predictor,
            reintroduce_nans=True,
            fill_value=min_by_level,
            var_to_clean="sea_surface_temperature",
        )
        return predictor

    # INPUT: 'inputs' (historical weather data), 'targets_template' (empty arrays defining the prediction shape), and 'forcings' (known future variables like solar radiation).
    # OUTPUT: A tuple containing the raw model prediction arrays and an empty Haiku state dictionary.
    @hk.transform_with_state
    def run_forward(inputs, targets_template, forcings):
        predictor = construct_wrapped_gencast()
        return predictor(inputs, targets_template=targets_template, forcings=forcings)

    # INPUT: 'rng' (random seed for the diffusion process), alongside the 'inputs', 'targets_template', and 'forcings' arrays.
    # OUTPUT: The compiled and parallelized prediction arrays, with the empty state dictionary explicitly discarded.
    state = {}
    run_forward_jitted = jax.jit(
        lambda rng, i, t, f: run_forward.apply(params, state, rng, i, t, f)[0]
    )
    run_forward_pmap = xarray_jax.pmap(run_forward_jitted, dim="sample")

    print("---------- Model built. ----------")

    ###########################################################################
    print("---------- Loading data ... ----------")
    t0 = time.time()
    path = SCRATCH_PATH + f"wb2_{args.start_date}_{args.end_date}_6h_{float(degree)}deg.zarr"
    if os.path.exists(path):
        print(f"Found existing data file at {path}. Loading data from this file for inference.")
        wb2 = prepare_data(path, deg=degree, time_slice=slice(args.start_date, args.end_date))
    else:
        print("No existing data file found. Downloading data.")
        wb2 = download_data(path, deg=degree, time_slice=slice(args.start_date, args.end_date))
        print(f"Data downloaded and saved to {path} for future use.")
    
    print(f"Data loading time: {time.time()-t0:.1f}s ")
    print("---------- Data loaded. ----------")
    ####################################################################################
    if args.sanity_check:
        t0 = time.time()
        sanity_check_predictions(model=args.model)
        print(f"Sanity checks completed in {time.time()-t0:.1f}s")


    #############################################################################
    print("Running inference ...")
    t0 = time.time()
    rng = jax.random.PRNGKey(0)
    rngs = np.stack([jax.random.fold_in(rng, i) for i in range(NUM_ENSEMBLE)], axis=0)
    wb2 = wb2.assign_coords(datetime=wb2.time)
    data_utils.add_derived_vars(wb2)
    data_utils.add_tisr_var(wb2)

    first_time = wb2.time[0]    
    wb2 = wb2.assign_coords(time=(wb2.time - wb2.time[0]).astype("timedelta64[ns]"))

    out_path = os.path.join(OUT_DIR, f"predictions_{args.model}_{args.start_date}_{args.end_date}_6h_{degree}deg.{args.save_type}")
    num_predictions = wb2.sizes["time"] - 4
    all_predictions = []

    for i in range(num_predictions):
        t1 = time.time()

        # Pick 3 frames 12h apart from the 6h grid: input at +0h and +12h, target at +24h.
        # Sliding by 1 in 6h steps shifts the window by 6h each iteration.
        wb2_slice = wb2.isel(time=[i, i + 2, i + 4])

        # Reset time coordinate locally so the model treats the start of this slice as '0h'
        wb2_slice = wb2_slice.assign_coords(time=(wb2_slice.time - wb2_slice.time[0]).astype("timedelta64[ns]"))

        w_eval_inputs, w_eval_targets, w_eval_forcings = data_utils.extract_inputs_targets_forcings(
            wb2_slice, target_lead_times=slice("12h", "12h"), **dataclasses.asdict(task_config)
        )
        w_eval_inputs = w_eval_inputs.expand_dims(batch=1).compute()
        w_eval_targets = w_eval_targets.expand_dims(batch=1).compute()
        w_eval_forcings = w_eval_forcings.expand_dims(batch=1).compute()

        chunks = []
        for chunk in rollout.chunked_prediction_generator_multiple_runs(
            predictor_fn=run_forward_pmap,
            rngs=rngs,
            inputs=w_eval_inputs,
            targets_template=w_eval_targets * np.nan,
            forcings=w_eval_forcings,
            num_steps_per_chunk=1,
            num_samples=NUM_ENSEMBLE,
            pmap_devices=jax.local_devices()
        ):
            chunks.append(chunk)
            
        prediction = xarray.combine_by_coords(chunks)
        prediction = prediction.assign_coords(lon=(((prediction.lon + 180) % 360) - 180)).sortby(["lat", "lon"])
        prediction_region = prediction[["2m_temperature"]].sel(lat=slice(MIN_LAT, MAX_LAT), lon=slice(MIN_LON, MAX_LON))





        for name in prediction_region.variables:
            prediction_region[name].encoding.clear()
        prediction_region.encoding = {}

        # The prediction's local time is 12h. Map it back to the absolute time it corresponds to
        prediction_region = prediction_region.drop_vars("datetime", errors="ignore")
        absolute_target_time = first_time + wb2.time[i + 4].astype("timedelta64[ns]")
        prediction_region = prediction_region.assign_coords(time=[absolute_target_time.values])        

        if i == 0:
            print("Time for first prediction chunk: {:.1f}s".format(time.time() - t1))

        all_predictions.append(prediction_region)

    ds = xarray.concat(all_predictions, dim="time")
    if args.save_type == "zarr":
        print("Saving predictions to Zarr...")
        ds.to_zarr(out_path, mode="w")
    else:
        print("Saving predictions to NetCDF...")
        ds.to_netcdf(out_path)
    print(f"Predictions saved to {out_path}")
    print(f"Inference completed in {time.time()-t0:.1f}s")
