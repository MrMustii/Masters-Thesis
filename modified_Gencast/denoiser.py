# Copyright 2024 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Derived from google-deepmind/graphcast at commit c037656, file graphcast/denoiser.py.
# Modifications copyright 2026 Mustafa Jawad El-Madani.
# Changed: the icosahedral mesh is cropped to the regional grid and rebuilt when that
# grid changes, and the denoiser is conditioned on forecast lead time.
# Code and docstrings outside the "MODIFIED FROM UPSTREAM" markers are DeepMind's.
"""The denoiser: a graph neural network over an icosahedral mesh cropped to the region.

Two changes to upstream's denoiser, both required for regional work:

    lead-time conditioning  ``ModifiedDenoiser`` adds Fourier features of the forecast
                            lead time beside the noise-level encoding, so one set of
                            weights serves every lead time (exp3 and exp4)
    regional mesh           ``ModifiedDenoiserArchitecture`` crops the global
                            icosahedral mesh to the grid's own extent and rebuilds the
                            graph whenever the grid changes, which is what lets an exp4
                            checkpoint run over NW America or N Africa unchanged

``ModifiedDenoiser`` and ``ModifiedDenoiserArchitecture`` subclass upstream's versions;
the mesh cropping in ``_crop_mesh`` and ``_maybe_init`` is the substantive addition.
"""

import dataclasses
import chex
import jax
import haiku as hk
import jax.numpy as jnp
import numpy as np
import xarray
from google.cloud import storage
from typing import Optional

from graphcast import denoiser
from graphcast import deep_typed_graph_net
from graphcast import icosahedral_mesh
from graphcast import model_utils
from graphcast import sparse_transformer
from graphcast import transformer
from graphcast import xarray_jax
from graphcast import gencast
from graphcast import checkpoint
from graphcast import nan_cleaning
from graphcast import normalization
from graphcast import data_utils
from graphcast import rollout
from modified_Gencast import regional_utils
jax.config.update("jax_default_matmul_precision", "highest")

# Fallback mesh-crop bounds, replaced by _maybe_init on first call; NOT the thesis domain.
MIN_LAT, MAX_LAT = 40., 70.
MIN_LON, MAX_LON = -10., 30.

##### MODIFIED FROM UPSTREAM: begin
@chex.dataclass(frozen=True, eq=True)
class LeadTimeEncoderConfig:
  """Fourier encoding of forecast lead time; absent from the config means no conditioning.

  Properties:
    max_horizon_hours: Lead times are divided by this before encoding, so it sets the
      horizon the encoding can represent - 168 h for the week-long rollouts.
    num_frequencies: Number of integer multiples of the base frequency to use.
    output_sizes: Layer sizes for the MLP applied to the Fourier features.
  """
  max_horizon_hours: float = 240.0
  num_frequencies: int = 32
  output_sizes: tuple[int, int] = (128, 128)
##### MODIFIED FROM UPSTREAM: end


@chex.dataclass(frozen=True, eq=True)
class NoiseEncoderConfig:
  """Configures the noise level encoding.

  Properties:
    apply_log_first: Whether to take the log of the inputs before computing
      Fourier features.
    base_period: The base period to use. This should be greater or equal to the
      range of the values, or to the period if the values have periodic
      semantics (e.g. 2pi if they represent angles). Frequencies used will be
      integer multiples of 1/base_period.
    num_frequencies: The number of frequencies to use, we will use integer
      multiples of 1/base_period from 1 up to num_frequencies inclusive. (We
      don't include a zero frequency as this would just give constant features
      which are redundant if a bias term is present).
    output_sizes: Layer sizes for the MLP.
  """
  apply_log_first: bool = True
  base_period: float = 16.0
  num_frequencies: int = 32
  # 2-layer MLP applied to Fourier features
  output_sizes: tuple[int, int] = (32, 16)


@chex.dataclass(eq=True)
class SparseTransformerConfig:
  """Sparse Transformer config."""
  # Neighbours to attend to.
  attention_k_hop: int
  # Primary width, the number of channels on the carrier path.
  d_model: int
  # Depth, or num transformer blocks. One 'layer' is attn + ffw.
  num_layers: int = 16
  # Number of heads for self-attention.
  num_heads: int = 4
  # Attention type.
  attention_type: str = "splash_mha"
  # mask type if splash attention being used.
  mask_type: str = "lazy"
  block_q: int = 1024
  block_kv: int = 512
  block_kv_compute: int = 256
  block_q_dkv: int = 512
  block_kv_dkv: int = 1024
  block_kv_dkv_compute: int = 1024
  # Init scale for final ffw layer (divided by depth)
  ffw_winit_final_mult: float = 0.0
  # Init scale for mha w (divided by depth).
  attn_winit_final_mult: float = 0.0
  # Number of hidden units in the MLP blocks. Defaults to 4 * d_model.
  ffw_hidden: int = 2048
  # Name for haiku module.
  name: Optional[str] = None


@chex.dataclass(eq=True)
class DenoiserArchitectureConfig:
  """Defines the GenCast architecture.

  Properties:
    sparse_transformer_config: Config for the mesh transformer.
    mesh_size: How many refinements to do on the multi-mesh.
    latent_size: How many latent features to include in the various MLPs.
    hidden_layers: How many hidden layers for each MLP.
    radius_query_fraction_edge_length: Scalar that will be multiplied by the
      length of the longest edge of the finest mesh to define the radius of
      connectivity to use in the Grid2Mesh graph. Reasonable values are
      between 0.6 and 1. 0.6 reduces the number of grid points feeding into
      multiple mesh nodes and therefore reduces edge count and memory use, but
      1 gives better predictions.
    norm_conditioning_features: List of feature names which will be used to
      condition the GNN via norm_conditioning, rather than as regular
      features. If this is provided, the GNN has to support the
      `global_norm_conditioning` argument. For now it only supports global
      norm conditioning (e.g. the same vector conditions all edges and nodes
      normalization), which means features passed here must not have "lat" or
      "lon" axes. In the future it may support node level norm conditioning
      too.
    grid2mesh_aggregate_normalization: Optional constant to normalize the output
      of aggregate_edges_for_nodes_fn in the mesh2grid GNN. This can be used to
        reduce the shock the model undergoes when switching resolution, which
        increases the number of edges connected to a node.
    node_output_size: Size of the output node representations for
        each node type. For node types not specified here, the latent node
        representation from the output of the processor will be returned.
  """

  sparse_transformer_config: SparseTransformerConfig
  mesh_size: int
  latent_size: int = 512
  hidden_layers: int = 1
  radius_query_fraction_edge_length: float = 0.6
  norm_conditioning_features: tuple[str, ...] = ("noise_level_encodings",)
  grid2mesh_aggregate_normalization: Optional[float] = None
  node_output_size: Optional[int] = None

class ModifiedDenoiser(denoiser.Denoiser):
  """Extends Denoiser with lead time conditioning."""

  def __init__(
      self,
      noise_encoder_config,
      denoiser_architecture_config,
      ##### MODIFIED FROM UPSTREAM: begin
      lead_time_encoder_config=None,
      ##### MODIFIED FROM UPSTREAM: end
  ):
    self._predictor = ModifiedDenoiserArchitecture(denoiser_architecture_config=denoiser_architecture_config)
    self._noise_level_encoder = denoiser.FourierFeaturesMLP(**noise_encoder_config)
    ##### MODIFIED FROM UPSTREAM: begin
    self._lead_time_encoder_config = lead_time_encoder_config
    if lead_time_encoder_config is not None:
      self._lead_time_encoder = denoiser.FourierFeaturesMLP(
          apply_log_first=False,
          base_period=1.0,
          num_frequencies=lead_time_encoder_config.num_frequencies,
          output_sizes=lead_time_encoder_config.output_sizes,
      )
    else:
      self._lead_time_encoder = None
    ##### MODIFIED FROM UPSTREAM: end

  def __call__(
      self,
      inputs,
      noisy_targets,
      noise_levels,
      forcings=None,
      ##### MODIFIED FROM UPSTREAM: begin
      lead_time_hours=None,
      ##### MODIFIED FROM UPSTREAM: end
      **kwargs,
  ):
    if forcings is None:
      forcings = xarray.Dataset()
    forcings = forcings.assign(noisy_targets)

    if noise_levels.dims != ("batch",):
      raise ValueError("noise_levels expected to be shape (batch,).")

    # --- Noise level encoding (same as parent) ---
    noise_level_encodings = self._noise_level_encoder(
        xarray_jax.unwrap_data(noise_levels)
    )
    inputs = inputs.assign(
        noise_level_encodings=xarray_jax.Variable(
            ("batch", "noise_level_encoding_channels"), noise_level_encodings
        )
    )

    ##### MODIFIED FROM UPSTREAM: begin
    # --- Lead time encoding  NEW ---
    if self._lead_time_encoder is not None:
      if lead_time_hours is not None:
        scaled = jnp.asarray(lead_time_hours, dtype=jnp.float32) / self._lead_time_encoder_config.max_horizon_hours
      else:
        time_vals = noisy_targets.time.values
        if np.issubdtype(time_vals.dtype, np.timedelta64):
          lead_h = float(time_vals[0] / np.timedelta64(1, 'h'))
        else:
          lead_h = 12.0
        scaled = jnp.full(
            (inputs.sizes['batch'],),
            lead_h / self._lead_time_encoder_config.max_horizon_hours,
            dtype=jnp.float32,
        )
      lt_encodings = self._lead_time_encoder(scaled)
      inputs = inputs.assign(
          lead_time_encodings=xarray_jax.Variable(
              ("batch", "lead_time_encoding_channels"), lt_encodings
          )
      )
    ##### MODIFIED FROM UPSTREAM: end

    return self._predictor(
        inputs=inputs,
        targets_template=noisy_targets,
        forcings=forcings,
        **kwargs,
    )


class ModifiedDenoiserArchitecture(denoiser._DenoiserArchitecture):
    """Upstream denoiser architecture with the mesh cropped to a regional grid.

    Upstream builds one global icosahedral mesh at construction and keeps it. A regional
    model needs the mesh restricted to its own extent, and a transferred model needs it
    rebuilt when the extent changes, so the mesh and the Grid2Mesh query radius are
    recomputed lazily in ``_maybe_init`` from whatever grid the first batch carries.
    The ``lat_min``/``lat_max``/``lon_min``/``lon_max`` arguments only seed the mesh
    built here; the first real batch replaces it.
    """

    def __init__(self, denoiser_architecture_config,
                 ##### MODIFIED FROM UPSTREAM: begin
                 lat_min=MIN_LAT, lat_max=MAX_LAT, lon_min=MIN_LON, lon_max=MAX_LON
                 ##### MODIFIED FROM UPSTREAM: end
                 ):
        self._spatial_features_kwargs = dict(
            add_node_positions=False,
            add_node_latitude=True,
            add_node_longitude=True,
            add_relative_positions=True,
            relative_longitude_local_coordinates=True,
            relative_latitude_local_coordinates=True,
        )

        ##### MODIFIED FROM UPSTREAM: begin
        # Kept so _maybe_init can rebuild the mesh and query radius when the region changes.
        self._mesh_size = denoiser_architecture_config.mesh_size
        self._radius_query_fraction = denoiser_architecture_config.radius_query_fraction_edge_length
        ##### MODIFIED FROM UPSTREAM: end

        mesh = icosahedral_mesh.get_last_triangular_mesh_for_sphere(
            splits=self._mesh_size
        )
        ##### MODIFIED FROM UPSTREAM: begin
        mesh = self._crop_mesh(mesh, lat_min, lat_max, lon_min, lon_max)
        ##### MODIFIED FROM UPSTREAM: end
        self._mesh = denoiser._permute_mesh_to_banded(mesh=mesh)

        self._grid2mesh_gnn = deep_typed_graph_net.DeepTypedGraphNet(
            activation="swish",
            aggregate_normalization=denoiser_architecture_config.grid2mesh_aggregate_normalization,
            edge_latent_size=dict(grid2mesh=denoiser_architecture_config.latent_size),
            embed_edges=True,
            embed_nodes=True,
            f32_aggregation=True,
            include_sent_messages_in_node_update=False,
            mlp_hidden_size=denoiser_architecture_config.latent_size,
            mlp_num_hidden_layers=denoiser_architecture_config.hidden_layers,
            name="grid2mesh_gnn",
            node_latent_size=dict(
                grid_nodes=denoiser_architecture_config.latent_size,
                mesh_nodes=denoiser_architecture_config.latent_size,
            ),
            node_output_size=None,
            num_message_passing_steps=1,
            use_layer_norm=True,
            use_norm_conditioning=True,
        )

        self._mesh_gnn = transformer.MeshTransformer(
            name="mesh_transformer",
            transformer_ctor=sparse_transformer.Transformer,
            transformer_kwargs=dataclasses.asdict(denoiser_architecture_config.sparse_transformer_config),
        )

        self._mesh2grid_gnn = deep_typed_graph_net.DeepTypedGraphNet(
            activation="swish",
            edge_latent_size=dict(mesh2grid=denoiser_architecture_config.latent_size),
            embed_nodes=False,
            f32_aggregation=False,
            include_sent_messages_in_node_update=False,
            mlp_hidden_size=denoiser_architecture_config.latent_size,
            mlp_num_hidden_layers=denoiser_architecture_config.hidden_layers,
            name="mesh2grid_gnn",
            node_latent_size=dict(
                grid_nodes=denoiser_architecture_config.latent_size,
                mesh_nodes=denoiser_architecture_config.latent_size,
            ),
            node_output_size={"grid_nodes": denoiser_architecture_config.node_output_size},
            num_message_passing_steps=1,
            use_layer_norm=True,
            use_norm_conditioning=True,
        )

        self._norm_conditioning_features = denoiser_architecture_config.norm_conditioning_features

        self._query_radius = (
            denoiser._get_max_edge_distance(self._mesh)
            * self._radius_query_fraction
        )

        self._initialized = False
        self._num_mesh_nodes = None
        self._mesh_nodes_lat = None
        self._mesh_nodes_lon = None
        self._grid_lat = None
        self._grid_lon = None
        self._num_grid_nodes = None
        self._grid_nodes_lat = None
        self._grid_nodes_lon = None
        self._grid2mesh_graph_structure = None
        self._mesh_graph_structure = None
        self._mesh2grid_graph_structure = None

    ##### MODIFIED FROM UPSTREAM: begin
    def _maybe_init(self, sample_inputs):
        """Rebuild mesh and graphs if this batch's grid differs from the last one.

        Replaces upstream's build-once behaviour. A no-op after the first batch for a
        fixed region; the rebuild path is what region transfer relies on.
        """
        new_lat = sample_inputs.lat.values
        new_lon = sample_inputs.lon.values
        region_changed = (
            self._grid_lat is None
            or not np.array_equal(new_lat, self._grid_lat)
            or not np.array_equal(new_lon, self._grid_lon)
        )

        if not region_changed:
            return

        lat_min, lat_max = float(new_lat.min()), float(new_lat.max())
        lon_min, lon_max = float(new_lon.min()), float(new_lon.max())

        base_mesh = icosahedral_mesh.get_last_triangular_mesh_for_sphere(
            splits=self._mesh_size
        )
        cropped = self._crop_mesh(base_mesh, lat_min, lat_max, lon_min, lon_max)
        self._mesh = denoiser._permute_mesh_to_banded(cropped)
        self._query_radius = (
            denoiser._get_max_edge_distance(self._mesh) * self._radius_query_fraction
        )

        self._init_mesh_properties()
        self._init_grid_properties(new_lat, new_lon)
        self._grid2mesh_graph_structure = self._init_grid2mesh_graph()
        self._mesh_graph_structure      = self._init_mesh_graph()
        self._mesh2grid_graph_structure = self._init_mesh2grid_graph()
        self._initialized = True
    ##### MODIFIED FROM UPSTREAM: end

    ##### MODIFIED FROM UPSTREAM: begin
    @staticmethod
    def _crop_mesh(mesh, lat_min, lat_max, lon_min, lon_max):
        """Keep only the mesh vertices inside the bounds, and the faces they fully span.

        Vertices are reindexed to stay contiguous, and a face survives only if all three
        of its vertices do, so the result is a valid TriangularMesh covering the region.
        Longitude is wrapped to [-180, 180] first to match the bound convention.
        """
        phi, theta = model_utils.cartesian_to_spherical(
            mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2])
        lat, lon = model_utils.spherical_to_lat_lon(phi, theta)

        # Wrap longitude to [-180, 180] so it matches the MIN_LON/MAX_LON bounds correctly.
        lon = (lon + 180) % 360 - 180

        keep = ((lat >= lat_min) & (lat <= lat_max) &
                (lon >= lon_min) & (lon <= lon_max))
        keep_idx = np.where(keep)[0]

        old_to_new = np.full(len(mesh.vertices), -1, dtype=np.int32)
        old_to_new[keep_idx] = np.arange(len(keep_idx), dtype=np.int32)

        valid_faces = np.all(old_to_new[mesh.faces] >= 0, axis=1)
        new_faces = old_to_new[mesh.faces[valid_faces]]

        return icosahedral_mesh.TriangularMesh(
            vertices=mesh.vertices[keep_idx],
            faces=new_faces,
        )
    ##### MODIFIED FROM UPSTREAM: end


# if __name__ == "__main__":
#     # Test
#     gcs_client = storage.Client.create_anonymous_client()
#     gcs_bucket = gcs_client.get_bucket("dm_graphcast")
#     dir_prefix = "gencast/"
#     dataset_file = "source-era5_date-2019-03-29_res-0.25_levels-13_steps-01.nc"
#     with gcs_bucket.blob(dir_prefix + f"dataset/{dataset_file}").open("rb") as f:
#         google = xarray.load_dataset(f)
# 
#     with gcs_bucket.blob(dir_prefix + "params/GenCast 0p25deg <2019.npz").open("rb") as f:
#         ckpt = checkpoint.load(f, gencast.CheckPoint)
# 
#     params = ckpt.params
#     task_config = ckpt.task_config
#     sampler_config = ckpt.sampler_config
#     noise_config = ckpt.noise_config
#     noise_encoder_config = ckpt.noise_encoder_config
#     denoiser_architecture_config = ckpt.denoiser_architecture_config
#     denoiser_architecture_config.sparse_transformer_config.attention_type = "triblockdiag_mha"
#     denoiser_architecture_config.sparse_transformer_config.mask_type = "full"
# 
#     with gcs_bucket.blob(dir_prefix + "stats/diffs_stddev_by_level.nc").open("rb") as f:
#         diffs_stddev_by_level = xarray.load_dataset(f).compute()
#     with gcs_bucket.blob(dir_prefix + "stats/mean_by_level.nc").open("rb") as f:
#         mean_by_level = xarray.load_dataset(f).compute()
#     with gcs_bucket.blob(dir_prefix + "stats/stddev_by_level.nc").open("rb") as f:
#         stddev_by_level = xarray.load_dataset(f).compute()
#     with gcs_bucket.blob(dir_prefix + "stats/min_by_level.nc").open("rb") as f:
#         min_by_level = xarray.load_dataset(f).compute()
# 
#     # Crop dataset to region
#     google_region = google.assign_coords(lon=(google.lon.values + 180) % 360 - 180)
#     google_region = google_region.sortby("lon")
#     google_region = google_region.sel(lat=slice(MIN_LAT, MAX_LAT), lon=slice(MIN_LON, MAX_LON))
# 
#     eval_inputs, eval_targets, eval_forcings = data_utils.extract_inputs_targets_forcings(
#         google_region,
#         target_lead_times=slice("12h", f"{(google_region.sizes['time'] - 2) * 12}h"),
#         **dataclasses.asdict(task_config),
#     )
#     eval_inputs   = eval_inputs.compute()
#     eval_targets  = eval_targets.compute()
#     eval_forcings = eval_forcings.compute()
# 
#     regional_utils.patch_samplers_utils()
#     denoiser._DenoiserArchitecture = ModifiedDenoiserArchitecture
# 
#     def construct_predictor():
#         predictor = gencast.GenCast(
#             sampler_config=sampler_config,
#             task_config=task_config,
#             denoiser_architecture_config=denoiser_architecture_config,
#             noise_config=noise_config,
#             noise_encoder_config=noise_encoder_config,
#         )
#         predictor = normalization.InputsAndResiduals(
#             predictor,
#             diffs_stddev_by_level=diffs_stddev_by_level,
#             mean_by_level=mean_by_level,
#             stddev_by_level=stddev_by_level,
#         )
#         predictor = nan_cleaning.NaNCleaner(
#             predictor=predictor,
#             reintroduce_nans=True,
#             fill_value=min_by_level,
#             var_to_clean="sea_surface_temperature",
#         )
#         return predictor
# 
#     @hk.transform_with_state
#     def run_forward(inputs, targets_template, forcings):
#         return construct_predictor()(inputs, targets_template=targets_template, forcings=forcings)
# 
#     state = {}
#     run_forward_jitted = jax.jit(
#         lambda rng, i, t, f: run_forward.apply(params, state, rng, i, t, f)[0]
#     )
#     run_forward_pmap = xarray_jax.pmap(run_forward_jitted, dim="sample")
# 
#     rng  = jax.random.PRNGKey(0)
#     rngs = np.stack([jax.random.fold_in(rng, i) for i in range(1)], axis=0)
# 
#     chunks = []
#     for chunk in rollout.chunked_prediction_generator_multiple_runs(
#         predictor_fn=run_forward_pmap,
#         rngs=rngs,
#         inputs=eval_inputs,
#         targets_template=eval_targets * np.nan,
#         forcings=eval_forcings,
#         num_steps_per_chunk=1,
#         num_samples=1,
#         pmap_devices=jax.local_devices(),
#     ):
#         chunks.append(chunk)
#     predictions = xarray.combine_by_coords(chunks)
# 
#     predictions.compute().to_zarr("regional_predictions.zarr", mode="w")
#     eval_targets.to_zarr("regional_ground_truth.zarr", mode="w")
#     print("Saved regional_predictions.zarr and regional_ground_truth.zarr")