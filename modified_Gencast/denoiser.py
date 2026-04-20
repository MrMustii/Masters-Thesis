import dataclasses
import chex
import sys
import jax
import haiku as hk
import jax.numpy as jnp
import numpy as np
from scipy import sparse
import xarray
from google.cloud import storage

sys.path.insert(0, '/project/project_465002687/Masters-Thesis/Googles_gencast')
sys.path.insert(0, '/project/project_465002687/Masters-Thesis')
from graphcast import denoiser
from graphcast import deep_typed_graph_net
from graphcast import denoisers_base as base
from graphcast import grid_mesh_connectivity
from graphcast import icosahedral_mesh
from graphcast import model_utils
from graphcast import sparse_transformer
from graphcast import transformer
from graphcast import typed_graph
from graphcast import xarray_jax
from graphcast import gencast
from graphcast import checkpoint
from graphcast import nan_cleaning
from graphcast import normalization
from graphcast import data_utils
from graphcast import rollout
from modified_Gencast import regional_utils

MIN_LAT, MAX_LAT = 25., 82.
MIN_LON, MAX_LON = -30., 60.
jax.config.update("jax_default_matmul_precision", "highest")

class ModifiedDenoiserArchitecture(denoiser._DenoiserArchitecture):
    def __init__(self, denoiser_architecture_config, lat_min=MIN_LAT, lat_max=MAX_LAT, lon_min=MIN_LON, lon_max=MAX_LON):
        self._spatial_features_kwargs = dict(
            add_node_positions=False,
            add_node_latitude=True,
            add_node_longitude=True,
            add_relative_positions=True,
            relative_longitude_local_coordinates=True,
            relative_latitude_local_coordinates=True,
        )

        # Store config needed to rebuild the mesh when the region changes
        self._mesh_size = denoiser_architecture_config.mesh_size
        self._radius_query_fraction = denoiser_architecture_config.radius_query_fraction_edge_length

        mesh = icosahedral_mesh.get_last_triangular_mesh_for_sphere(
            splits=self._mesh_size
        )
        mesh = self._crop_mesh(mesh, lat_min, lat_max, lon_min, lon_max)
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

    def _maybe_init(self, sample_inputs):
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

    @staticmethod
    def _crop_mesh(mesh, lat_min, lat_max, lon_min, lon_max):
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

if __name__ == "__main__":
    gcs_client = storage.Client.create_anonymous_client()
    gcs_bucket = gcs_client.get_bucket("dm_graphcast")
    dir_prefix = "gencast/"
    dataset_file = "source-era5_date-2019-03-29_res-0.25_levels-13_steps-01.nc"
    with gcs_bucket.blob(dir_prefix + f"dataset/{dataset_file}").open("rb") as f:
        google = xarray.load_dataset(f)

    with gcs_bucket.blob(dir_prefix + "params/GenCast 0p25deg <2019.npz").open("rb") as f:
        ckpt = checkpoint.load(f, gencast.CheckPoint)

    params = ckpt.params
    task_config = ckpt.task_config
    sampler_config = ckpt.sampler_config
    noise_config = ckpt.noise_config
    noise_encoder_config = ckpt.noise_encoder_config
    denoiser_architecture_config = ckpt.denoiser_architecture_config
    denoiser_architecture_config.sparse_transformer_config.attention_type = "triblockdiag_mha"
    denoiser_architecture_config.sparse_transformer_config.mask_type = "full"

    with gcs_bucket.blob(dir_prefix + "stats/diffs_stddev_by_level.nc").open("rb") as f:
        diffs_stddev_by_level = xarray.load_dataset(f).compute()
    with gcs_bucket.blob(dir_prefix + "stats/mean_by_level.nc").open("rb") as f:
        mean_by_level = xarray.load_dataset(f).compute()
    with gcs_bucket.blob(dir_prefix + "stats/stddev_by_level.nc").open("rb") as f:
        stddev_by_level = xarray.load_dataset(f).compute()
    with gcs_bucket.blob(dir_prefix + "stats/min_by_level.nc").open("rb") as f:
        min_by_level = xarray.load_dataset(f).compute()

    # Crop dataset to region
    google_region = google.assign_coords(lon=(google.lon.values + 180) % 360 - 180)
    google_region = google_region.sortby("lon")
    google_region = google_region.sel(lat=slice(MIN_LAT, MAX_LAT), lon=slice(MIN_LON, MAX_LON))

    eval_inputs, eval_targets, eval_forcings = data_utils.extract_inputs_targets_forcings(
        google_region,
        target_lead_times=slice("12h", f"{(google_region.sizes['time'] - 2) * 12}h"),
        **dataclasses.asdict(task_config),
    )
    eval_inputs   = eval_inputs.compute()
    eval_targets  = eval_targets.compute()
    eval_forcings = eval_forcings.compute()

    regional_utils.patch_samplers_utils()
    denoiser._DenoiserArchitecture = ModifiedDenoiserArchitecture

    def construct_predictor():
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

    @hk.transform_with_state
    def run_forward(inputs, targets_template, forcings):
        return construct_predictor()(inputs, targets_template=targets_template, forcings=forcings)

    state = {}
    run_forward_jitted = jax.jit(
        lambda rng, i, t, f: run_forward.apply(params, state, rng, i, t, f)[0]
    )
    run_forward_pmap = xarray_jax.pmap(run_forward_jitted, dim="sample")

    rng  = jax.random.PRNGKey(0)
    rngs = np.stack([jax.random.fold_in(rng, i) for i in range(1)], axis=0)

    chunks = []
    for chunk in rollout.chunked_prediction_generator_multiple_runs(
        predictor_fn=run_forward_pmap,
        rngs=rngs,
        inputs=eval_inputs,
        targets_template=eval_targets * np.nan,
        forcings=eval_forcings,
        num_steps_per_chunk=1,
        num_samples=1,
        pmap_devices=jax.local_devices(),
    ):
        chunks.append(chunk)
    predictions = xarray.combine_by_coords(chunks)

    predictions.compute().to_zarr("regional_predictions.zarr", mode="w")
    eval_targets.to_zarr("regional_ground_truth.zarr", mode="w")
    print("Saved regional_predictions.zarr and regional_ground_truth.zarr")