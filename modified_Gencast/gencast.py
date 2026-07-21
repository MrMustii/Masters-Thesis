from typing import Any, Optional, Tuple
import chex
import dataclasses
import sys
import json
import os
from gencast_infernce.config import CONFIGS_DIR
sys.path.insert(0, '/project/project_465002687/Masters-Thesis/Googles_gencast')
sys.path.insert(0, '/project/project_465002687/Masters-Thesis')

from graphcast import casting
from graphcast import dpm_solver_plus_plus_2s
from graphcast import graphcast
from modified_Gencast import losses
from graphcast import predictor_base
from graphcast import samplers_utils
from graphcast import xarray_jax
from graphcast import xarray_tree
from modified_Gencast import denoiser as modified_denoiser
from modified_Gencast import regional_utils
import haiku as hk
import jax
import jax.numpy as jnp
import xarray


@chex.dataclass(frozen=True, eq=True)
class SamplerConfig:
    max_noise_level: float
    min_noise_level: float
    num_noise_levels: int
    rho: float
    stochastic_churn_rate: float
    churn_min_noise_level: float
    churn_max_noise_level: float
    noise_level_inflation_factor: float


@chex.dataclass(frozen=True, eq=True)
class NoiseConfig:
    training_noise_level_rho: float
    training_max_noise_level: float
    training_min_noise_level: float


@dataclasses.dataclass
class LossConfig:
    loss_function: str
    per_variable_weights: dict
    loss_area_weighting: bool
    loss_noise_level_weighting: bool


@dataclasses.dataclass
class GenCastSetup:
    """All instantiated config objects for a single training/eval run."""
    config: dict
    config_name: str
    config_path: str
    task: graphcast.TaskConfig
    sampler_config: SamplerConfig
    noise_config: NoiseConfig
    denoiser_architecture_config: modified_denoiser.DenoiserArchitectureConfig
    noise_encoder_config: modified_denoiser.NoiseEncoderConfig
    lead_time_encoder_config: modified_denoiser.LeadTimeEncoderConfig
    loss_config: LossConfig


def load_config(config_name: str) -> GenCastSetup:
    """Load a training config JSON and return fully instantiated config objects."""
    config_path = config_name if os.path.isabs(config_name) else os.path.join(CONFIGS_DIR, config_name)
    if not os.path.exists(config_path):
        raise ValueError(f"Config not found: {config_path}")
    with open(config_path) as f:
        config = json.load(f)

    for section in ("task_config", "sampler_config", "noise_config",
                    "noise_encoder_config", "denoiser_architecture_config", "training_config"):
        if section not in config:
            raise ValueError(f"Config '{config_name}' is missing required section: '{section}'")

    # Task
    task_config = config["task_config"]
    task = graphcast.TaskConfig(
        input_variables=tuple(task_config["input_variables"]),
        target_variables=(
            tuple(task_config["target_surface_variables"])
            + tuple(task_config["target_atmospheric_variables"])
        ),
        forcing_variables=tuple(task_config["forcing_variables"]),
        pressure_levels=tuple(task_config["pressure_levels"]),
        input_duration=task_config["input_duration"],
    )

    # Sampler
    sampler_config = config["sampler_config"]
    sampler_config = SamplerConfig(
        max_noise_level=sampler_config["max_noise_level"],
        min_noise_level=sampler_config["min_noise_level"],
        num_noise_levels=sampler_config["num_noise_levels"],
        rho=sampler_config["rho"],
        stochastic_churn_rate=sampler_config["stochastic_churn_rate"],
        churn_min_noise_level=sampler_config["churn_min_noise_level"],
        churn_max_noise_level=float(sampler_config["churn_max_noise_level"]),
        noise_level_inflation_factor=sampler_config["noise_level_inflation_factor"],
    )

    # Noise
    noise_config = config["noise_config"]
    noise_config = NoiseConfig(
        training_noise_level_rho=noise_config["training_noise_level_rho"],
        training_max_noise_level=noise_config["training_max_noise_level"],
        training_min_noise_level=noise_config["training_min_noise_level"],
    )

    # Noise encoder
    noise_enc_config = config["noise_encoder_config"]
    noise_encoder_config = modified_denoiser.NoiseEncoderConfig(
        apply_log_first=noise_enc_config["apply_log_first"],
        base_period=noise_enc_config["base_period"],
        num_frequencies=noise_enc_config["num_frequencies"],
        output_sizes=tuple(noise_enc_config["output_sizes"]),
    )

    # Conditioning — noise is always present; lead time is optional (omit section = no conditioning)
    norm_conditioning = ["noise_level_encodings"]
    try:
        lead_time_enc_config = config.get("lead_time_encoder_config")
        if lead_time_enc_config is not None:
            lead_time_encoder_config = modified_denoiser.LeadTimeEncoderConfig(
                max_horizon_hours=lead_time_enc_config["max_horizon_hours"],
                num_frequencies=lead_time_enc_config["num_frequencies"],
                output_sizes=tuple(lead_time_enc_config["output_sizes"]),
            )
            norm_conditioning.append("lead_time_encodings")
        else:
            lead_time_encoder_config = None
    except KeyError as e:
        raise ValueError(
            f"Config '{config_name}' lead_time_encoder_config is missing field: {e}"
        ) from None

    # Denoiser architecture
    transformer_config = config["denoiser_architecture_config"]["sparse_transformer_config"]
    sparse_transformer_cfg = modified_denoiser.SparseTransformerConfig(
        attention_k_hop=transformer_config["attention_k_hop"],
        d_model=transformer_config["d_model"],
        num_layers=transformer_config["num_layers"],
        num_heads=transformer_config["num_heads"],
        attention_type=transformer_config["attention_type"],
        mask_type=transformer_config["mask_type"],
        block_q=transformer_config["block_q"],
        block_kv=transformer_config["block_kv"],
        block_kv_compute=transformer_config["block_kv_compute"],
        block_q_dkv=transformer_config["block_q_dkv"],
        block_kv_dkv=transformer_config["block_kv_dkv"],
        block_kv_dkv_compute=transformer_config["block_kv_dkv_compute"],
        ffw_winit_final_mult=transformer_config["ffw_winit_final_mult"],
        attn_winit_final_mult=transformer_config["attn_winit_final_mult"],
        ffw_hidden=transformer_config["ffw_hidden"],
        name=transformer_config["name"],
    )
    denoise_arch_config = config["denoiser_architecture_config"]
    denoiser_architecture_config = modified_denoiser.DenoiserArchitectureConfig(
        sparse_transformer_config=sparse_transformer_cfg,
        mesh_size=denoise_arch_config["mesh_size"],
        latent_size=denoise_arch_config["latent_size"],
        hidden_layers=denoise_arch_config["hidden_layers"],
        radius_query_fraction_edge_length=denoise_arch_config["radius_query_fraction_edge_length"],
        norm_conditioning_features=tuple(norm_conditioning),
    )

    # Loss config
    train_config = config["training_config"]
    loss_config = LossConfig(
        loss_function=train_config["loss_function"],
        per_variable_weights=train_config["per_variable_weights"],
        loss_area_weighting=train_config["loss_area_weighting"],
        loss_noise_level_weighting=train_config["loss_noise_level_weighting"],
    )

    return GenCastSetup(
        config=config,
        config_name=config_name,
        config_path=config_path,
        task=task,
        sampler_config=sampler_config,
        noise_config=noise_config,
        denoiser_architecture_config=denoiser_architecture_config,
        noise_encoder_config=noise_encoder_config,
        lead_time_encoder_config=lead_time_encoder_config,
        loss_config=loss_config,
    )


##### CHANGES BELOW
@chex.dataclass(frozen=True, eq=True)
class CheckPoint:
    description: str
    license: str
    params: dict[str, Any]
    task_config: graphcast.TaskConfig
    denoiser_architecture_config: modified_denoiser.DenoiserArchitectureConfig
    sampler_config: SamplerConfig
    noise_config: NoiseConfig
    noise_encoder_config: modified_denoiser.NoiseEncoderConfig
    lead_time_encoder_config: Optional[modified_denoiser.LeadTimeEncoderConfig] = None
####### CHANGES ABOVE
class ModifiedSampler(dpm_solver_plus_plus_2s.Sampler):
    """Extends the DPM-Solver++ 2S sampler with optional pre-generated initial noise.

    When initial_noise is None the behaviour is identical to the parent.
    When initial_noise is a Dataset (spherical noise at unit variance), the ODE
    starts from noise_levels[0] * initial_noise and the step-0 internal draw is
    skipped, so the trajectory is fully determined by the supplied noise vector.
    """

    def __call__(
        self,
        inputs: xarray.Dataset,
        targets_template: xarray.Dataset,
        forcings: Optional[xarray.Dataset] = None,
        initial_noise: Optional[xarray.Dataset] = None,
        **kwargs,
    ) -> xarray.Dataset:
        if initial_noise is None:
            return super().__call__(inputs, targets_template, forcings, **kwargs)

        dtype = casting.infer_floating_dtype(targets_template)
        noise_levels = jnp.array(self._noise_levels).astype(dtype)
        per_step_churn_rates = jnp.array(self._per_step_churn_rates).astype(dtype)

        def denoiser(noise_level, x):
            bcast_noise_level = xarray_jax.DataArray(
                jnp.tile(noise_level, x.sizes['batch']), dims=('batch',))
            return self._denoiser(
                inputs=inputs,
                noisy_targets=x,
                noise_levels=bcast_noise_level,
                forcings=forcings,
            )

        def body_fn(i, x):
            noise_level = noise_levels[i]
            if self._stochastic_churn:
                x, noise_level = samplers_utils.apply_stochastic_churn(
                    x, noise_level,
                    stochastic_churn_rate=per_step_churn_rates[i],
                    noise_level_inflation_factor=self._noise_level_inflation_factor,
                )
            next_noise_level = noise_levels[i + 1]
            mid_noise_level = jnp.sqrt(noise_level * next_noise_level)
            mid_over_current = mid_noise_level / noise_level
            x_denoised = denoiser(noise_level, x)
            x_mid = mid_over_current * x + (1 - mid_over_current) * x_denoised
            next_over_current = next_noise_level / noise_level
            x_mid_denoised = denoiser(mid_noise_level, x_mid)
            x_next = next_over_current * x + (1 - next_over_current) * x_mid_denoised
            return samplers_utils.tree_where(next_noise_level == 0, x_denoised, x_next)

        noise_init = initial_noise * noise_levels[0]
        return hk.fori_loop(
            0, len(noise_levels) - 1, body_fun=body_fn, init_val=noise_init)


class GenCast(predictor_base.Predictor):
    """Predictor for a denoising diffusion model following the framework of [1].

    [1] Elucidating the Design Space of Diffusion-Based Generative Models
    Karras, Aittala, Aila and Laine, 2022
    https://arxiv.org/abs/2206.00364

    Unlike the paper, we have a conditional model and our denoising function
    conditions on previous timesteps.

    As the paper demonstrates, the sampling algorithm can be varied independently
    of the denoising model and its training procedure, and it is separately
    configurable here.
    """

    def __init__(
        self,
        task_config: graphcast.TaskConfig,
        denoiser_architecture_config: modified_denoiser.DenoiserArchitectureConfig,
        sampler_config: Optional[SamplerConfig] = None,
        noise_config: Optional[NoiseConfig] = None,
        noise_encoder_config: Optional[modified_denoiser.NoiseEncoderConfig] = None,
        ##### CHANGES BELOW
        lead_time_encoder_config: Optional[modified_denoiser.LeadTimeEncoderConfig] = None,
        loss_config: Optional[LossConfig] = None,
        ####### CHANGES ABOVE
    ):
        """Constructs GenCast."""
        # Output size depends on number of variables being predicted.
        num_surface_vars = len(
            set(task_config.target_variables)
            - set(graphcast.ALL_ATMOSPHERIC_VARS)
        )
        num_atmospheric_vars = len(
            set(task_config.target_variables)
            & set(graphcast.ALL_ATMOSPHERIC_VARS)
        )
        num_outputs = (
            num_surface_vars
            + len(task_config.pressure_levels) * num_atmospheric_vars
        )
        denoiser_architecture_config.node_output_size = num_outputs

        ##### CHANGES BELOW
        self._denoiser = modified_denoiser.ModifiedDenoiser(
            noise_encoder_config,
            denoiser_architecture_config,
            lead_time_encoder_config,
        )
        ####### CHANGES ABOVE
        self._sampler_config = sampler_config
        self._noise_config = noise_config
        ##### CHANGES BELOW
        self._loss_config = loss_config
        ####### CHANGES ABOVE

    def _c_in(self, noise_scale: xarray.DataArray) -> xarray.DataArray:
        """Scaling applied to the noisy targets input to the underlying network."""
        return (noise_scale**2 + 1)**-0.5

    def _c_out(self, noise_scale: xarray.DataArray) -> xarray.DataArray:
        """Scaling applied to the underlying network's raw outputs."""
        return noise_scale * (noise_scale**2 + 1)**-0.5

    def _c_skip(self, noise_scale: xarray.DataArray) -> xarray.DataArray:
        """Scaling applied to the skip connection."""
        return 1 / (noise_scale**2 + 1)

    def _loss_weighting(self, noise_scale: xarray.DataArray) -> xarray.DataArray:
        r"""The loss weighting \lambda(\sigma) from the paper."""
        return self._c_out(noise_scale) ** -2

    def _preconditioned_denoiser(
        self,
        inputs: xarray.Dataset,
        noisy_targets: xarray.Dataset,
        noise_levels: xarray.DataArray,
        forcings: Optional[xarray.Dataset] = None,
        **kwargs,
    ) -> xarray.Dataset:
        """The preconditioned denoising function D from the paper (Eqn 7)."""
        raw_predictions = self._denoiser(
            inputs=inputs,
            noisy_targets=noisy_targets * self._c_in(noise_levels),
            noise_levels=noise_levels,
            forcings=forcings,
            **kwargs,
        )
        return (
            raw_predictions * self._c_out(noise_levels)
            + noisy_targets * self._c_skip(noise_levels)
        )

    def loss_and_predictions(
        self,
        inputs: xarray.Dataset,
        targets: xarray.Dataset,
        forcings: Optional[xarray.Dataset] = None,
    ) -> Tuple[predictor_base.LossAndDiagnostics, xarray.Dataset]:
        return self.loss(inputs, targets, forcings), self(inputs, targets, forcings)

    def loss(
        self,
        inputs: xarray.Dataset,
        targets: xarray.Dataset,
        forcings: Optional[xarray.Dataset] = None,
        ##### CHANGES BELOW
        lead_time_hours=None,
        ####### CHANGES ABOVE
    ) -> predictor_base.LossAndDiagnostics:
        if self._noise_config is None:
            raise ValueError("Noise config must be specified to train GenCast.")

        # Sample noise levels.
        dtype = casting.infer_floating_dtype(targets)  # pytype: disable=wrong-arg-types
        key = hk.next_rng_key()
        batch_size = inputs.sizes["batch"]
        noise_levels = xarray_jax.DataArray(
            data=samplers_utils.rho_inverse_cdf(
                min_value=self._noise_config.training_min_noise_level,
                max_value=self._noise_config.training_max_noise_level,
                rho=self._noise_config.training_noise_level_rho,
                cdf=jax.random.uniform(key, shape=(batch_size,), dtype=dtype),
            ),
            dims=("batch",),
        )

        ##### CHANGES BELOW
        # Sample noise and apply it to targets.
        noise = regional_utils.spherical_white_noise_like(targets) * noise_levels
        noisy_targets = targets + noise

        denoised_predictions = self._preconditioned_denoiser(
            inputs, noisy_targets, noise_levels, forcings,
            lead_time_hours=lead_time_hours,
        )
        per_variable_weights = self._loss_config.per_variable_weights if self._loss_config else {}
        loss, diagnostics = losses.weighted_mse_per_level_regional(
            denoised_predictions,
            targets,
            per_variable_weights=per_variable_weights,
        )
        ####### CHANGES ABOVE
        loss *= self._loss_weighting(noise_levels)
        return loss, diagnostics

    def __call__(
        self,
        inputs: xarray.Dataset,
        targets_template: xarray.Dataset,
        forcings: Optional[xarray.Dataset] = None,
        lead_time_hours=None,
        initial_noise: Optional[xarray.Dataset] = None,
        **kwargs,
    ) -> xarray.Dataset:
        if self._sampler_config is None:
            raise ValueError(
                "Sampler config must be specified to run inference on GenCast."
            )
        def bound_denoiser(inputs, noisy_targets, noise_levels, forcings=None):
            return self._preconditioned_denoiser(
                inputs, noisy_targets, noise_levels, forcings,
                lead_time_hours=lead_time_hours,
            )
        sampler = ModifiedSampler(bound_denoiser, **self._sampler_config)
        return sampler(inputs, targets_template, forcings, initial_noise=initial_noise, **kwargs)


class BoundaryForcedPredictor(predictor_base.Predictor):
    """Lateral boundary forcing for a regional predictor (one-way nesting).

    Wraps any predictor and, after each autoregressive step, overwrites the
    outer `boundary_margin_degrees` band of the prediction with host truth. The
    truth is read from the non-NaN cells of `targets_template`, so the host
    boundary is supplied by passing a template whose interior is NaN and whose
    margin holds the true values. The corrected margin feeds back as the next
    step's input, so the true inflow steers the interior across the rollout.

    Operates in raw (unnormalized) space, so it must wrap the outermost
    predictor (outside InputsAndResiduals / NaNCleaner).
    """

    def __init__(self, predictor, boundary_margin_degrees):
        self._predictor = predictor
        self._boundary_margin_degrees = boundary_margin_degrees

    def __call__(self, inputs, targets_template, forcings=None, **kwargs):
        predictions = self._predictor(inputs, targets_template, forcings, **kwargs)
        if not self._boundary_margin_degrees:
            return predictions
        margin = regional_utils.boundary_margin_mask(
            targets_template, self._boundary_margin_degrees
        )
        return xarray_tree.map_structure(
            lambda prediction, truth: prediction.where(~margin, truth),
            predictions, targets_template,
        )

    def loss(self, inputs, targets, forcings=None, **kwargs):
        return self._predictor.loss(inputs, targets, forcings, **kwargs)