from typing import Any, Optional, Tuple
import chex
import sys
import json
from gencast_infernce.config import SCRATCH_PATH, GENCAST_LIB, DEFAULT_OUTPUT, WORKING_DIR
sys.path.insert(0, '/project/project_465002687/Masters-Thesis/Googles_gencast')
sys.path.insert(0, '/project/project_465002687/Masters-Thesis')

from graphcast import casting
from graphcast import denoiser
from graphcast import dpm_solver_plus_plus_2s
from graphcast import graphcast
from modified_Gencast import losses
from graphcast import predictor_base
from graphcast import samplers_utils
from graphcast import xarray_jax
from graphcast import graphcast 
from modified_Gencast import denoiser as modified_denoiser
from modified_Gencast import regional_utils
import haiku as hk
import jax
import jax.numpy as jnp
import xarray
import os


CONFIG = "test_traning.json"

if CONFIG not in os.listdir(WORKING_DIR + "/modified_Gencast/traninig configs/"):
  raise ValueError(f"Config {CONFIG} not found in configs directory.")
CONFIG = json.load(open(WORKING_DIR + "/modified_Gencast/traninig configs/" + CONFIG, "r"))

INPUT_VARIABLES = tuple(CONFIG["task_config"]["input_variables"])
TARGET_SURFACE_VARS = tuple(CONFIG["task_config"]["target_surface_variables"])
TARGET_ATMOSPHERIC_VARS = tuple(CONFIG["task_config"]["target_atmospheric_variables"])
TARGET_VARIABLES = TARGET_SURFACE_VARS + TARGET_ATMOSPHERIC_VARS
FORCING_VARIABLES = tuple(CONFIG["task_config"]["forcing_variables"])
STATIC_VARIABLES = tuple(CONFIG["task_config"]["static_variables"])

TASK = graphcast.TaskConfig(
    input_variables=INPUT_VARIABLES,
    target_variables=TARGET_VARIABLES,
    forcing_variables=FORCING_VARIABLES,
    pressure_levels=tuple(CONFIG["task_config"]["pressure_levels"]),
    input_duration=CONFIG["task_config"]["input_duration"],
)

@chex.dataclass(frozen=True, eq=True)
class SamplerConfig:
  max_noise_level: float = CONFIG["sampler_config"]["max_noise_level"]
  min_noise_level: float = CONFIG["sampler_config"]["min_noise_level"]
  num_noise_levels: int = CONFIG["sampler_config"]["num_noise_levels"]
  rho: float = CONFIG["sampler_config"]["rho"]
  # Stochastic sampler settings.
  stochastic_churn_rate: float = CONFIG["sampler_config"]["stochastic_churn_rate"]
  churn_min_noise_level: float = CONFIG["sampler_config"]["churn_min_noise_level"]
  churn_max_noise_level: float = float(CONFIG["sampler_config"]["churn_max_noise_level"])
  noise_level_inflation_factor: float = CONFIG["sampler_config"]["noise_level_inflation_factor"]


@chex.dataclass(frozen=True, eq=True)
class NoiseConfig:
  training_noise_level_rho: float = CONFIG["noise_config"]["training_noise_level_rho"]
  training_max_noise_level: float = CONFIG["noise_config"]["training_max_noise_level"]
  training_min_noise_level: float = CONFIG["noise_config"]["training_min_noise_level"]


_stc = CONFIG["denoiser_architecture_config"]["sparse_transformer_config"]
sparse_transformer_cfg = modified_denoiser.SparseTransformerConfig(
    attention_k_hop=_stc["attention_k_hop"],
    d_model=_stc["d_model"],
    num_layers=_stc["num_layers"],
    num_heads=_stc["num_heads"],
    attention_type=_stc["attention_type"],
    mask_type=_stc["mask_type"],
    block_q=_stc["block_q"],
    block_kv=_stc["block_kv"],
    block_kv_compute=_stc["block_kv_compute"],
    block_q_dkv=_stc["block_q_dkv"],
    block_kv_dkv=_stc["block_kv_dkv"],
    block_kv_dkv_compute=_stc["block_kv_dkv_compute"],
    ffw_winit_final_mult=_stc["ffw_winit_final_mult"],
    attn_winit_final_mult=_stc["attn_winit_final_mult"],
    ffw_hidden=_stc["ffw_hidden"],
    name=_stc["name"],
)

_dac = CONFIG["denoiser_architecture_config"]
denoiser_architecture_cfg = modified_denoiser.DenoiserArchitectureConfig(
    sparse_transformer_config=sparse_transformer_cfg,
    mesh_size=_dac["mesh_size"],
    latent_size=_dac["latent_size"],
    hidden_layers=_dac["hidden_layers"],
    radius_query_fraction_edge_length=_dac["radius_query_fraction_edge_length"],
    norm_conditioning_features=tuple(_dac["norm_conditioning_features"]),
    grid2mesh_aggregate_normalization=_dac["grid2mesh_aggregate_normalization"],
)
denoiser._DenoiserArchitecture = modified_denoiser.ModifiedDenoiserArchitecture
_nec = CONFIG["noise_encoder_config"]
noise_encoder_cfg = modified_denoiser.NoiseEncoderConfig(
    apply_log_first=_nec["apply_log_first"],
    base_period=_nec["base_period"],
    num_frequencies=_nec["num_frequencies"],
    output_sizes=tuple(_nec["output_sizes"]),
)


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

    self._denoiser = denoiser.Denoiser(
        noise_encoder_config,
        denoiser_architecture_config,
    )
    self._sampler_config = sampler_config
    # Singleton to avoid re-initializing the sampler for each inference call.
    self._sampler = None
    self._noise_config = noise_config

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
      **kwargs) -> xarray.Dataset:
    """The preconditioned denoising function D from the paper (Eqn 7)."""
    raw_predictions = self._denoiser(
        inputs=inputs,
        noisy_targets=noisy_targets * self._c_in(noise_levels),
        noise_levels=noise_levels,
        forcings=forcings,
        **kwargs)
    return (raw_predictions * self._c_out(noise_levels) +
            noisy_targets * self._c_skip(noise_levels))

  def loss_and_predictions(
      self,
      inputs: xarray.Dataset,
      targets: xarray.Dataset,
      forcings: Optional[xarray.Dataset] = None,
  ) -> Tuple[predictor_base.LossAndDiagnostics, xarray.Dataset]:
    return self.loss(inputs, targets, forcings), self(inputs, targets, forcings)

  def loss(self,
           inputs: xarray.Dataset,
           targets: xarray.Dataset,
           forcings: Optional[xarray.Dataset] = None,
           ) -> predictor_base.LossAndDiagnostics:
    if self._noise_config is None:
      raise ValueError('Noise config must be specified to train GenCast.')

    # Sample noise levels:
    dtype = casting.infer_floating_dtype(targets)  # pytype: disable=wrong-arg-types
    key = hk.next_rng_key()
    batch_size = inputs.sizes['batch']
    noise_levels = xarray_jax.DataArray(
        data=samplers_utils.rho_inverse_cdf(
            min_value=self._noise_config.training_min_noise_level,
            max_value=self._noise_config.training_max_noise_level,
            rho=self._noise_config.training_noise_level_rho,
            cdf=jax.random.uniform(key, shape=(batch_size,), dtype=dtype)),
        dims=('batch',))

    # Sample noise and apply it to targets:
    noise = regional_utils.spherical_white_noise_like(targets) * noise_levels
    noisy_targets = targets + noise

    denoised_predictions = self._preconditioned_denoiser(
        inputs, noisy_targets, noise_levels, forcings)
    loss, diagnostics = losses.weighted_mse_per_level_regional(
        denoised_predictions,
        targets,
        per_variable_weights=CONFIG["training_config"]["per_variable_weights"],
    )
    loss *= self._loss_weighting(noise_levels)
    return loss, diagnostics

  def __call__(self,
               inputs: xarray.Dataset,
               targets_template: xarray.Dataset,
               forcings: Optional[xarray.Dataset] = None,
               **kwargs) -> xarray.Dataset:
    if self._sampler_config is None:
      raise ValueError(
          'Sampler config must be specified to run inference on GenCast.'
      )
    if self._sampler is None:
      self._sampler = dpm_solver_plus_plus_2s.Sampler(
          self._preconditioned_denoiser, **self._sampler_config
      )
    return self._sampler(inputs, targets_template, forcings, **kwargs)
