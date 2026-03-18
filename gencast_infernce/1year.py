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
sys.path.insert(0, '/project/project_465002687/Masters-Thesis/Googles_gencast')

from google.cloud import storage
from graphcast import rollout
from graphcast import xarray_jax
from graphcast import normalization
from graphcast import checkpoint
from graphcast import data_utils
from graphcast import gencast
from graphcast import nan_cleaning
from ploting import plot_map
print("Imports done.")
print(f"Import time: {time.time()-t0:.1f}s")
print(jax.devices())


OUT_DIR = "/project/project_465002687/Masters-Thesis/gencast_infernce/predictions/"
NUM_ENSEMBLE = 1
os.makedirs(OUT_DIR, exist_ok=True)
# SCRATCH_PATH =  "../../../scratch/project_465002687/data/era5/wb2_2022_subset.zarr"
# SCRATCH_PATH = "../../../scratch/project_465002687/data/era5/wb2_2022_week1_subset.zarr"
SCRATCH_PATH = "../../../scratch/project_465002687/data/era5/wb2_2021-12-30_2023-01-01_12h_1deg.zarr"
def prepare_data(location, time_slice=slice("2019-03-28T12:00", "2019-03-30T00:00:00")):
    wb2 = xarray.open_zarr(location)
    wb2 = wb2.sel(time=wb2.time.dt.hour.isin([0, 12]))
    wb2 = wb2.sel(time=time_slice)
    wb2 = wb2.sel(level=[50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000])
    wb2 = wb2[["2m_temperature", "mean_sea_level_pressure", "10m_u_component_of_wind",
            "10m_v_component_of_wind", "sea_surface_temperature", "total_precipitation_12hr",
            "geopotential_at_surface", "land_sea_mask", "geopotential", "specific_humidity",
            "temperature", "u_component_of_wind", "v_component_of_wind", "vertical_velocity"]]
    if "latitude" in wb2.dims and "longitude" in wb2.dims:
        wb2 = wb2.rename({"latitude": "lat", "longitude": "lon"})
    wb2 = wb2.isel(lat=slice(None, None, 4), lon=slice(None, None, 4))    
    wb2 = wb2.sortby("lat") 
    return wb2



print("Loading model...")
t0 = time.time()
gcs_client = storage.Client.create_anonymous_client()
gcs_bucket = gcs_client.get_bucket("dm_graphcast")
dir_prefix = "gencast/"

avalible_models = [
    name for blob in gcs_bucket.list_blobs(prefix=(dir_prefix + "params/"))
    if (name := blob.name.removeprefix(dir_prefix + "params/"))
]
model_selected = [f for f in avalible_models if "Mini" in f][0]
with gcs_bucket.blob(dir_prefix + f"params/{model_selected}").open("rb") as f:
    ckpt = checkpoint.load(f, gencast.CheckPoint)
params = ckpt.params
task_config = ckpt.task_config
sampler_config = ckpt.sampler_config
noise_config = ckpt.noise_config
noise_encoder_config = ckpt.noise_encoder_config
denoiser_architecture_config = ckpt.denoiser_architecture_config
denoiser_architecture_config.sparse_transformer_config.attention_type = "triblockdiag_mha"
denoiser_architecture_config.sparse_transformer_config.mask_type = "full"
print(f"Model loaded in {time.time()-t0:.1f}s | {model_selected}")

print("Loading normalization stats...")
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

print("Model built.")
###########################################################################
print("Loading data ...")
wb2 = xarray.open_zarr(SCRATCH_PATH)
# wb2 = wb2.sel(time=wb2.time.dt.hour.isin([0, 12]))

print("Data loaded.")
print("Running inference ...")
rng = jax.random.PRNGKey(0)
rngs = np.stack([jax.random.fold_in(rng, i) for i in range(NUM_ENSEMBLE)], axis=0)
wb2 = wb2.assign_coords(datetime=wb2.time)
data_utils.add_derived_vars(wb2)
data_utils.add_tisr_var(wb2)
first_time = wb2.time[0]    
wb2 = wb2.assign_coords(time=(wb2.time - wb2.time[0]).astype("timedelta64[ns]"))

for i in range(wb2.sizes["time"] // 2 - 1):
    w_eval_inputs, w_eval_targets, w_eval_forcings = data_utils.extract_inputs_targets_forcings(
        wb2, target_lead_times=slice(f"{12*(i+1)}h", f"{12*(i+1)}h"), **dataclasses.asdict(task_config)
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
    prediction = xarray.combine_by_coords(w_chunks)
    prediction = prediction.assign_coords(lon=(((prediction.lon + 180) % 360) - 180)).sortby(["lat", "lon"])
    prediction_region = prediction[["2m_temperature"]].sel(lat=slice(45.0, 65.0), lon=slice(-5.0, 25.0))

    out_path = os.path.join(OUT_DIR, "predictions_1year_region.zarr")
    if i == 0:
        # convert time from timedelta back to absolute time for the first chunk, then append subsequent chunks using the same time coordinate
        prediction_region = prediction_region.assign_coords(time=first_time + prediction_region.time.astype("timedelta64[ns]"))
        prediction_region.to_zarr(out_path, mode="w")
    else:
        prediction_region = prediction_region.assign_coords(time=first_time + prediction_region.time.astype("timedelta64[ns]"))
        prediction_region.to_zarr(out_path, mode="a", append_dim="time")
    print(f"Predictions saved to {out_path}")


################################################### Everything abouve this is correct


def sanity_check_predictions():
    # load google data and weatherbench2 data for the same time period and compare them before and after runing the model
    print("SANITY CHECKS ON FINAL PREDICTIONS:")
    gcs_client = storage.Client.create_anonymous_client()
    gcs_bucket = gcs_client.get_bucket("dm_graphcast")
    dir_prefix = "gencast/"
    dataset_file = "source-era5_date-2019-03-29_res-1.0_levels-13_steps-01.nc"
    with gcs_bucket.blob(dir_prefix + f"dataset/{dataset_file}").open("rb") as f:
        google = xarray.load_dataset(f)
    print("=== Loaded Google sample batch ===")


    print("=== LoadedWeatherBench2 ERA5 (processed) ===")
    time_slice = slice("2022-01-01T00:00:00", "2022-01-08T00:00:00")

    wb2 = prepare_data("gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr")
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


