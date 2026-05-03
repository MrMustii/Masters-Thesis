"""Minimal training loop: 15 epochs on 1 week of ERA5, prints loss and timing."""
import dataclasses
import time
import os
import sys

import jax
jax.config.update("jax_default_matmul_precision", "highest")
import jax.numpy as jnp
import xarray
import haiku as hk
import wandb
sys.path.insert(0, '/project/project_465002687/Masters-Thesis/Googles_gencast')
sys.path.insert(0, '/project/project_465002687/Masters-Thesis')

from gencast_infernce.config import SCRATCH_PATH
from gencast_infernce.datatest import prepare_data, download_data

from graphcast import checkpoint
from graphcast import normalization
from graphcast import nan_cleaning
from graphcast import data_utils
from graphcast import xarray_jax
from graphcast import xarray_tree

from modified_Gencast.gencast import (
    GenCast, TASK, SamplerConfig, NoiseConfig,
    denoiser_architecture_cfg, noise_encoder_cfg, CONFIG, CheckPoint
)
from modified_Gencast import regional_utils

_region = CONFIG["task_config"].get("region") or {}
MIN_LAT = _region.get("min_lat", 25.0)
MAX_LAT = _region.get("max_lat", 82.0)
MIN_LON = _region.get("min_lon", -30.0)
MAX_LON = _region.get("max_lon", 60.0)

START_DATE = CONFIG["training_config"]["training_time_start"]
END_DATE   = CONFIG["training_config"]["training_time_end"]
NUM_EPOCHS = CONFIG["training_config"]["num_epochs"]
LR         = CONFIG["training_config"]["learning_rate"]
CLIP_NORM  = CONFIG["training_config"]["gradient_clip_norm"]

# TODO use  Optax
# ── Pure-JAX Adam with global gradient clipping ───────────────────────────────

def adam_init(params, lr=1e-4, beta1=0.9, beta2=0.999, eps=1e-8):
    m = jax.tree_util.tree_map(jnp.zeros_like, params)
    v = jax.tree_util.tree_map(jnp.zeros_like, params)
    return {"step": 0, "m": m, "v": v, "lr": lr, "beta1": beta1, "beta2": beta2, "eps": eps}


def adam_update(opt_state, grads, clip_norm):
    step  = opt_state["step"] + 1
    beta1 = opt_state["beta1"]
    beta2 = opt_state["beta2"]
    eps   = opt_state["eps"]
    lr    = opt_state["lr"]

    # Global gradient clipping.
    leaves = jax.tree_util.tree_leaves(grads)
    global_norm = jnp.sqrt(sum(jnp.sum(g ** 2) for g in leaves))
    scale = jnp.minimum(1.0, clip_norm / (global_norm + 1e-12))
    grads = jax.tree_util.tree_map(lambda g: g * scale, grads)

    m = jax.tree_util.tree_map(lambda m, g: beta1 * m + (1 - beta1) * g, opt_state["m"], grads)
    v = jax.tree_util.tree_map(lambda v, g: beta2 * v + (1 - beta2) * g ** 2, opt_state["v"], grads)

    m_hat = jax.tree_util.tree_map(lambda m: m / (1 - beta1 ** step), m)
    v_hat = jax.tree_util.tree_map(lambda v: v / (1 - beta2 ** step), v)

    updates = jax.tree_util.tree_map(lambda m, v: -lr * m / (jnp.sqrt(v) + eps), m_hat, v_hat)
    new_state = {**opt_state, "step": step, "m": m, "v": v}
    return updates, new_state


def apply_updates(params, updates):
    return jax.tree_util.tree_map(lambda p, u: p + u, params, updates)


CKPT_DIR = os.path.normpath(os.path.join(SCRATCH_PATH, "..", "..", "checkpoints"))


def save_checkpoint(params, epoch):
    os.makedirs(CKPT_DIR, exist_ok=True)
    ckpt = CheckPoint(
        description=f"Regional GenCast fine-tune epoch {epoch}",
        license="",
        params=params,
        task_config=TASK,
        denoiser_architecture_config=denoiser_architecture_cfg,
        sampler_config=SamplerConfig(),
        noise_config=NoiseConfig(),
        noise_encoder_config=noise_encoder_cfg,
    )
    path = os.path.join(CKPT_DIR, f"epoch_{epoch:03d}.npz")
    with open(path, "wb") as f:
        checkpoint.dump(f, ckpt)
    print(f"  checkpoint saved -> {path}")


# ── Model ─────────────────────────────────────────────────────────────────────

def build_predictor(stats):
    predictor = GenCast(
        task_config=TASK,
        denoiser_architecture_config=denoiser_architecture_cfg,
        sampler_config=SamplerConfig(),
        noise_config=NoiseConfig(),
        noise_encoder_config=noise_encoder_cfg,
    )
    predictor = normalization.InputsAndResiduals(
        predictor,
        diffs_stddev_by_level=stats["diffs_stddev_by_level"],
        mean_by_level=stats["mean_by_level"],
        stddev_by_level=stats["stddev_by_level"],
    )
    predictor = nan_cleaning.NaNCleaner(
        predictor=predictor,
        reintroduce_nans=True,
        fill_value=stats["min_by_level"],
        var_to_clean="sea_surface_temperature",
    )
    return predictor


def make_loss_fn(stats):
    @hk.transform_with_state
    def loss_fn(inputs, targets, forcings):
        loss, diagnostics = build_predictor(stats).loss(inputs, targets, forcings)
        return xarray_tree.map_structure(
            lambda x: xarray_jax.unwrap_data(x.mean(), require_jax=True),
            (loss, diagnostics),
        )
    return loss_fn


def prepare_window(wb2, i):
    sl = wb2.isel(time=[i, i + 2, i + 4])
    sl = sl.assign_coords(time=(sl.time - sl.time[0]).astype("timedelta64[ns]"))
    inputs, targets, forcings = data_utils.extract_inputs_targets_forcings(
        sl, target_lead_times=slice("12h", "12h"), **dataclasses.asdict(TASK), # TODO check this 
    )
    return (
        inputs.expand_dims(batch=1).compute(),
        targets.expand_dims(batch=1).compute(),
        forcings.expand_dims(batch=1).compute(),
    )



def _init_wandb():
    api_key = os.environ.get("WANDB_API_KEY", "").strip()
    wandb_mode = os.environ.get("WANDB_MODE", "online").strip().lower()
    if api_key:
        wandb.login(key=api_key, relogin=True)
        return wandb.init(
            project="Gencast",
            name="example_training_run",
            config={
                "training_time_start": START_DATE,
                "training_time_end": END_DATE,
                "num_epochs": NUM_EPOCHS,
                "learning_rate": LR,
                "gradient_clip_norm": CLIP_NORM,
            },
        )

    print("W&B API key not set; running with wandb disabled.")
    return wandb.init(
        mode="disabled" if wandb_mode != "offline" else "offline",
        project="Gencast",
        name="example_training_run",
        config={
            "training_time_start": START_DATE,
            "training_time_end": END_DATE,
            "num_epochs": NUM_EPOCHS,
            "learning_rate": LR,
            "gradient_clip_norm": CLIP_NORM,
        },
    )



def _print_config():
    tc = CONFIG["task_config"]
    tr = CONFIG["training_config"]
    sa = CONFIG["sampler_config"]
    nc = CONFIG["noise_config"]

    sep = "=" * 60
    print(sep)
    print("  TRAINING CONFIG")
    print(sep)

    print("\n[Task]")
    print(f"  input_duration     : {tc['input_duration']}")
    print(f"  pressure_levels    : {tc['pressure_levels']}")
    region = tc.get("region", {})
    if region:
        print(f"  region             : lat=[{region['min_lat']}, {region['max_lat']}]  lon=[{region['min_lon']}, {region['max_lon']}]")
    print(f"  input_variables    : {list(tc['input_variables'])}")
    print(f"  target_surface     : {list(tc['target_surface_variables'])}")
    print(f"  target_atmospheric : {list(tc['target_atmospheric_variables'])}")
    print(f"  forcing_variables  : {list(tc['forcing_variables'])}")
    print(f"  static_variables   : {list(tc['static_variables'])}")

    print("\n[Training]")
    print(f"  data               : {START_DATE} -> {END_DATE}")
    print(f"  num_epochs         : {NUM_EPOCHS}")
    print(f"  learning_rate      : {LR}")
    print(f"  gradient_clip_norm : {CLIP_NORM}")
    print(f"  batch_size         : {tr['batch_size']}")
    print(f"  optimizer          : {tr['optimizer']}")
    print(f"  loss_function      : {tr['loss_function']}")
    print(f"  per_variable_wts   : {tr['per_variable_weights']}")

    print("\n[Noise]")
    print(f"  training_min       : {nc['training_min_noise_level']}")
    print(f"  training_max       : {nc['training_max_noise_level']}")
    print(f"  rho                : {nc['training_noise_level_rho']}")

    print("\n[Sampler]")
    print(f"  num_noise_levels   : {sa['num_noise_levels']}")
    print(f"  min / max          : {sa['min_noise_level']} / {sa['max_noise_level']}")
    print(f"  rho                : {sa['rho']}")
    print(f"  stochastic_churn   : {sa['stochastic_churn_rate']}")

    print(f"\n  checkpoint dir     : {CKPT_DIR}")
    print(sep + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    regional_utils.patch_samplers_utils()
    _print_config()
    _init_wandb()

    print("Loading normalization stats...")
    from google.cloud import storage
    gcs_client = storage.Client.create_anonymous_client()
    gcs_bucket = gcs_client.get_bucket("dm_graphcast")
    stats = {}
    for name in ["diffs_stddev_by_level", "mean_by_level", "stddev_by_level", "min_by_level"]:
        with gcs_bucket.blob(f"gencast/stats/{name}.nc").open("rb") as f:
            ds = xarray.load_dataset(f).compute()
        # Surface variables are stored with a single-element or NaN level coordinate.
        # Squeezing it out prevents xarray from producing NaN when normalizing
        # data that has no level dimension.
        squeezed = {}
        for var in ds.data_vars:
            da = ds[var]
            if "level" in da.dims and da.sizes["level"] == 1:
                squeezed[var] = da.squeeze("level", drop=True)
            elif "level" in da.coords and da.ndim == 0:
                squeezed[var] = da.drop_vars("level")
            else:
                squeezed[var] = da
        stats[name] = xarray.Dataset(squeezed)

    print(f"Loading data {START_DATE} -> {END_DATE}...")
    t0 = time.time()
    data_path = SCRATCH_PATH + f"wb2_{START_DATE}_{END_DATE}_6h_0.25deg.zarr"
    if os.path.exists(data_path):
        wb2 = prepare_data(data_path, deg=0.25, time_slice=slice(START_DATE, END_DATE))
    else:
        wb2 = download_data(data_path, deg=0.25, time_slice=slice(START_DATE, END_DATE))
    print(f"  loaded in {time.time()-t0:.1f}s  ({wb2.sizes['time']} steps)")

    wb2 = wb2.assign_coords(datetime=wb2.time)
    data_utils.add_derived_vars(wb2)
    wb2 = wb2.assign_coords(time=(wb2.time - wb2.time[0]).astype("timedelta64[ns]"))
    wb2 = wb2.assign_coords(lon=((wb2.lon.values + 180) % 360 - 180))
    wb2 = wb2.sortby("lon")
    wb2 = wb2.sel(lat=slice(MIN_LAT, MAX_LAT), lon=slice(MIN_LON, MAX_LON))
    print(f"  region: lat=[{MIN_LAT},{MAX_LAT}] lon=[{MIN_LON},{MAX_LON}]  grid={wb2.sizes['lat']}x{wb2.sizes['lon']}")

    num_windows = wb2.sizes["time"] - 4
    print(f"  windows per epoch: {num_windows}")

    loss_fn = make_loss_fn(stats)

    def grads_fn(params, state, rng, inputs, targets, forcings):
        def _aux(params, state):
            (loss, diag), next_state = loss_fn.apply(params, state, rng, inputs, targets, forcings)
            return loss, (diag, next_state)
        (loss, (diag, next_state)), grads = jax.value_and_grad(_aux, has_aux=True)(params, state)
        return loss, diag, next_state, grads

    grads_fn_jit = jax.jit(grads_fn)

    print("Initializing model...")
    t0 = time.time()
    rng = jax.random.PRNGKey(0)
    init_inputs, init_targets, init_forcings = prepare_window(wb2, 0)
    params, state = jax.jit(loss_fn.init)(rng, init_inputs, init_targets, init_forcings)
    print(f"  initialized in {time.time()-t0:.1f}s")

    opt_state = adam_init(params, lr=LR)

    print(f"\nTraining: {NUM_EPOCHS} epochs x {num_windows} windows\n")
    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0.0
        t_epoch = time.time()
        for i in range(num_windows):
            t_batch = time.time()
            rng, step_rng = jax.random.split(rng)
            inputs, targets, forcings = prepare_window(wb2, i)
            loss, _, state, grads = grads_fn_jit(params, state, step_rng, inputs, targets, forcings)
            updates, opt_state = adam_update(opt_state, grads, clip_norm=CLIP_NORM)
            params = apply_updates(params, updates)
            loss_val = float(loss)
            epoch_loss += loss_val
            step = epoch * num_windows + i + 1
            wandb.log({"loss": loss_val, "epoch": epoch + 1}, step=step)
            print(f"  epoch {epoch+1}/{NUM_EPOCHS}  batch {i+1}/{num_windows}  loss={loss_val:.4f}  ({time.time()-t_batch:.1f}s/batch)")

        avg_loss = epoch_loss / num_windows
        wandb.log({"epoch_avg_loss": avg_loss, "epoch": epoch + 1}, step=(epoch + 1) * num_windows)
        print(f"epoch {epoch+1} done  avg_loss={avg_loss:.4f}  ({time.time()-t_epoch:.1f}s/epoch)\n")
        save_checkpoint(params, epoch + 1)


if __name__ == "__main__":
    main()
