"""The training loop. Launched by job_scripts/lumi_train.sh.

Trains one regional checkpoint from one JSON config, in two stages that share parameters:

    stage 1   1.0 degree data, coarser mesh                  cheap, gets the weights close
    stage 2   0.25 degree data, finer mesh, grid2mesh
              aggregate normalisation 16.0                   the checkpoint experiments use

Stage 2 starts from stage 1's parameters, so ``--stage both`` runs them back to back.
Everything that differs between experiments - variable set, region, sampler type, noise and
lead-time conditioning - comes out of the config, so this one file trains exp2, exp3 and
exp4 without branching.

Writes ``<scratch>/checkpoints/<run_name>_stage<N>/``: the parameters pickled every
``--save-every`` steps, plus the config they were trained with, so a checkpoint can be
reloaded without the original JSON. Losses and per-lead-time validation go to wandb.
"""

import argparse
import json
import os
import pickle
import sys
import time
import numpy as np
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..")))
from gencast_infernce.config import GENCAST_LIB, WORKING_DIR, SCRATCH_PATH
sys.path.insert(0, GENCAST_LIB)
sys.path.insert(0, WORKING_DIR)
import haiku as hk
import jax
import optax
import wandb

from graphcast import nan_cleaning, normalization, xarray_jax, xarray_tree
from modified_Gencast import regional_utils
from modified_Gencast.stat_loader import load_stats
from modified_Gencast.gencast import GenCast, load_config
from modified_Gencast.helper_fun import build_loader, InputsAndResidualsDayZero
jax.config.update("jax_default_matmul_precision", "highest")

CKPT_DIR = os.path.normpath(os.path.join(SCRATCH_PATH, "checkpoints"))


def make_optimizer(peak_lr, warmup_steps, total_steps, weight_decay, gradient_clip_norm, start_step=0):
    """AdamW on a warmup-then-cosine-decay schedule, with global-norm gradient clipping.

    optax's step counter restarts at zero on resume, so start_step shifts the schedule
    forward to keep the learning rate continuous across a restart.
    """
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=peak_lr,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        end_value=0.0,
    )
    # On resume the optimizer's internal counter restarts at 0; shift the schedule
    if start_step:
        base_schedule = schedule
        schedule = lambda count: base_schedule(count + start_step)
    return optax.chain(
        optax.clip_by_global_norm(gradient_clip_norm),
        optax.adamw(learning_rate=schedule, weight_decay=weight_decay),
    )


def save_checkpoint(params, step, run_dir):
    """Pickle the parameters as step_<step>.pkl and return the path."""
    path = os.path.join(run_dir, f"step_{step:08d}.pkl")
    with open(path, "wb") as f:
        pickle.dump(params, f)
    print(f"  checkpoint → {path}")
    return path


def load_checkpoint(path):
    """Unpickle parameters written by save_checkpoint."""
    with open(path, "rb") as f:
        return pickle.load(f)


def init_wandb(config, stage_num):
    """Start a wandb run, falling back to offline or disabled when no API key is set.

    Queries the project first and suffixes the name if it is already taken, so a re-run
    never overwrites an earlier run's history.
    """
    api_key = os.environ.get("WANDB_API_KEY", "").strip()
    mode = os.environ.get("WANDB_MODE", "online").strip().lower()
    online = bool(api_key) and mode not in ("offline", "disabled")
    base_name = f"{config['training_config']['run_name']}_stage{stage_num}"
    if not online:
        return wandb.init(
            project="Gencast",
            mode="offline" if mode == "offline" else "disabled",
            config=config,
        )
    wandb.login(key=api_key, relogin=True)
    api = wandb.Api()
    candidate, counter = base_name, 1
    while list(api.runs("Gencast", filters={"display_name": candidate})):
        candidate = f"{base_name}_{counter}"
        counter += 1
    return wandb.init(project="Gencast", name=candidate, config=config)


def get_diffs_stddev(stats, lead_time_hours_arr):
    """Pick this batch's residual normaliser out of whatever shape stat_loader returned.

    Variable-lead and daily runs get a dict keyed by lead time or hour of day; fixed-lead
    runs get a single Dataset that applies to every batch.
    """
    diffs = stats["diffs_stddev_by_level"]
    if isinstance(diffs, dict):
        return diffs[int(np.asarray(lead_time_hours_arr).flat[0])]
    return diffs


def build_loss_fn(setup, stats):
    """Build the haiku-transformed loss: GenCast wrapped in normalisation, then NaN cleaning.

    daily_stats runs swap in ``InputsAndResidualsDayZero`` instead of graphcast's
    ``InputsAndResiduals``, because exp4's residual is measured against day zero's mean
    rather than the last input frame. NaN cleaning is added only when sea surface
    temperature is in the variable set.
    """
    use_lead_time = setup.lead_time_encoder_config is not None
    sampler_type = setup.config["training_config"].get("sampler_type", "fixed")
    task_config = setup.config["task_config"]
    has_sst = "sea_surface_temperature" in (
        task_config.get("input_variables", [])
        + task_config.get("target_surface_variables", [])
    )

    @hk.transform_with_state
    def loss_fn(inputs, targets, forcings, diffs_stddev, lead_time_hours=None):
        predictor = GenCast(
            task_config=setup.task,
            denoiser_architecture_config=setup.denoiser_architecture_config,
            sampler_config=setup.sampler_config,
            noise_config=setup.noise_config,
            noise_encoder_config=setup.noise_encoder_config,
            lead_time_encoder_config=setup.lead_time_encoder_config,
            loss_config=setup.loss_config,
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
                predictor=predictor,
                reintroduce_nans=True,
                fill_value=stats["min_by_level"],
                var_to_clean="sea_surface_temperature",
            )
        loss, diagnostics = predictor.loss(
            inputs, targets, forcings,
            lead_time_hours=lead_time_hours if use_lead_time else None,
        )
        return xarray_tree.map_structure(
            lambda x: xarray_jax.unwrap_data(x.mean(), require_jax=True),
            (loss, diagnostics),
        )
    return loss_fn, use_lead_time


def run_stage(stage_num, setup, stats, train_loader, val_loader, params, state, rng,
              total_steps, peak_lr, warmup_steps, weight_decay, gradient_clip_norm,
              ckpt_dir, run, save_every, start_step=0):
    """Run one stage to total_steps: validate every 100 steps, checkpoint every save_every.

    Initialises parameters from the first batch only when none are passed in, which is what
    lets stage 2 continue from stage 1 instead of starting over. Returns the parameters,
    haiku state and rng so the next stage can pick them up.
    """
    loss_fn, use_lead_time = build_loss_fn(setup, stats)

    if params is None:
        print("Initializing model ...")
        first = train_loader.first_sample()
        init_diffs_stddev = get_diffs_stddev(stats, first["conditioning"]["lead_time_hours"])
        init_lead_time = (
            jax.numpy.asarray([first["conditioning"]["lead_time_hours"]], dtype=jax.numpy.float32)
            if use_lead_time else jax.numpy.zeros(1, dtype=jax.numpy.float32)
        )
        params, state = jax.jit(loss_fn.init)(
            rng, first["inputs"], first["targets"], first["forcings"], init_diffs_stddev, init_lead_time
        )

    optimizer = make_optimizer(peak_lr, warmup_steps, total_steps, weight_decay, gradient_clip_norm, start_step=start_step)
    opt_state = optimizer.init(params)

    @jax.jit
    def grad_step(params, state, rng, inputs, targets, forcings, diffs_stddev, lead_time_hours):
        def loss_with_state(params, state):
            (loss, diag), next_state = loss_fn.apply(
                params, state, rng, inputs, targets, forcings, diffs_stddev, lead_time_hours
            )
            return loss, (diag, next_state)
        (loss, (_, next_state)), grads = jax.value_and_grad(
            loss_with_state, has_aux=True
        )(params, state)
        return loss, grads, next_state

    @jax.jit
    def apply_grads(params, opt_state, grads):
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), new_opt_state

    @jax.jit
    def val_step(params, state, rng, inputs, targets, forcings, diffs_stddev, lead_time_hours):
        (loss, _), _ = loss_fn.apply(params, state, rng, inputs, targets, forcings, diffs_stddev, lead_time_hours)
        return loss

    def cycle_loader():
        while True:
            yield from train_loader

    print(f"\n[Stage {stage_num}] {total_steps} steps  peak_lr={peak_lr}  warmup={warmup_steps}"
          + (f"  RESUMING at step {start_step} (LR continues, Adam moments re-warm)\n" if start_step else "\n"))

    def _lead_time_arr(conditioning):
        if use_lead_time:
            return jax.numpy.asarray(conditioning["lead_time_hours"], dtype=jax.numpy.float32)
        return jax.numpy.zeros(1, dtype=jax.numpy.float32)

    batch_iter = iter(cycle_loader())
    step = start_step
    while step < total_steps:
        t0 = time.time()

        batch = next(batch_iter)
        rng, step_rng = jax.random.split(rng)
        diffs_stddev = get_diffs_stddev(stats, batch["conditioning"]["lead_time_hours"])
        loss, grads, next_state = grad_step(
            params, state, step_rng,
            batch["inputs"], batch["targets"], batch["forcings"],
            diffs_stddev, _lead_time_arr(batch["conditioning"]),
        )
        params, opt_state = apply_grads(params, opt_state, grads)
        state = next_state
        step += 1
        batch_loss = float(loss)

        run.log({
            "train_loss": batch_loss,
            "stage": stage_num,
        }, step=step)
        if step <= 100:
            print(f"  [s{stage_num}] step {step}/{total_steps}  loss={batch_loss:.4f}  ({time.time()-t0:.1f}s)")

        if step % 100 == 0:
            val_losses = []
            val_by_lead_time = {}
            for val_batch in val_loader:
                rng, val_rng = jax.random.split(rng)
                val_diffs_stddev = get_diffs_stddev(stats, val_batch["conditioning"]["lead_time_hours"])
                vl = val_step(
                    params, state, val_rng,
                    val_batch["inputs"], val_batch["targets"], val_batch["forcings"],
                    val_diffs_stddev, _lead_time_arr(val_batch["conditioning"]),
                )
                val_losses.append(float(vl))
                if use_lead_time:
                    lead_time = int(np.asarray(val_batch["conditioning"]["lead_time_hours"]).flat[0])
                    val_by_lead_time.setdefault(lead_time, []).append(val_losses[-1])
            val_loss = sum(val_losses) / len(val_losses)
            val_log = {"val_loss": val_loss}
            if val_by_lead_time:
                for lead_time, losses in val_by_lead_time.items():
                    val_log[f"val_loss_lt{lead_time}h"] = sum(losses) / len(losses)
            run.log(val_log, step=step)
            print(f"  [s{stage_num}] step {step}/{total_steps}  val={val_loss:.4f}")

        if step % save_every == 0:
            save_checkpoint(params, step, ckpt_dir)

    save_checkpoint(params, step, ckpt_dir)
    return params, state, rng


def main():
    """Load the config and stats once, then run the requested stage or stages in order."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="test_traning.json")
    parser.add_argument("--stage", default="1", choices=["1", "2", "both"])
    parser.add_argument("--resume", default=None, help="Checkpoint path to load params from.")
    parser.add_argument("--start-step", type=int, default=0,
                        help="Resume the (single) stage's LR schedule at this step instead of 0 — "
                             "use when resuming a crashed same-stage checkpoint, not for a fresh 1→2 transfer.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=5000)
    args = parser.parse_args()

    setup = load_config(args.config)
    training_config = setup.config["training_config"]
    stages_to_run = {"1": [1], "2": [2], "both": [1, 2]}[args.stage]

    regional_utils.patch_samplers_utils()

    print("Loading normalization stats...")
    stats = load_stats(setup.config)

    rng = jax.random.PRNGKey(args.seed)
    params = load_checkpoint(args.resume) if args.resume else None
    state = None

    for stage_num in stages_to_run:
        stage_config = training_config[f"stage_{stage_num}"]

        setup.denoiser_architecture_config.mesh_size = stage_config["mesh_size"]
        setup.denoiser_architecture_config.grid2mesh_aggregate_normalization = (
            16.0 if stage_num == 2 else None
        )

        time_start = training_config.get(
            f"stage_{stage_num}_training_time_start",
            training_config["training_time_start"],
        )
        time_end = training_config.get(
            f"stage_{stage_num}_training_time_end",
            training_config["training_time_end"],
        )

        batch_size = stage_config.get("batch_size", training_config.get("batch_size", 1))
        resolution = 1.0 if stage_num == 1 else 0.25
        print(f"\n[Stage {stage_num}] Building data loader {time_start} → {time_end}  res={resolution}deg  batch={batch_size} ...")
        loader = build_loader(setup, time_start, time_end, args.seed, batch_size=batch_size, shuffle=True, resolution=resolution, preload=True)

        val_time_start = training_config["validation_time_start"]
        val_time_end = training_config["validation_time_end"]
        print(f"\n[Stage {stage_num}] Building val loader {val_time_start} → {val_time_end}  res={resolution}deg ...")
        val_loader = build_loader(setup, val_time_start, val_time_end, args.seed, batch_size=batch_size, shuffle=False, resolution=resolution, preload=True)

        run = init_wandb(setup.config, stage_num)
        run_name = training_config["run_name"]
        run_ckpt_dir = os.path.join(CKPT_DIR, f"{run_name}_stage{stage_num}")
        os.makedirs(run_ckpt_dir, exist_ok=True)
        with open(os.path.join(run_ckpt_dir, "config.json"), "w") as f:
            json.dump(setup.config, f, indent=2)

        params, state, rng = run_stage(
            stage_num=stage_num,
            setup=setup,
            stats=stats,
            train_loader=loader,
            val_loader=val_loader,
            params=params,
            state=state,
            rng=rng,
            total_steps=stage_config["num_steps"],
            peak_lr=stage_config["learning_rate"],
            warmup_steps=stage_config["warmup_steps"],
            weight_decay=training_config["weight_decay"],
            gradient_clip_norm=training_config["gradient_clip_norm"],
            ckpt_dir=run_ckpt_dir,
            run=run,
            save_every=args.save_every,
            start_step=args.start_step,
        )

        run.finish()


if __name__ == "__main__":
    main()
