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

        mesh = icosahedral_mesh.get_last_triangular_mesh_for_sphere(
            splits=denoiser_architecture_config.mesh_size
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
            * denoiser_architecture_config.radius_query_fraction_edge_length
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
    dataset_file = "source-era5_date-2019-03-29_res-1.0_levels-13_steps-01.nc" 
    with gcs_bucket.blob(dir_prefix + f"dataset/{dataset_file}").open("rb") as f:
        google = xarray.load_dataset(f)

    with gcs_bucket.blob(dir_prefix + f"params/GenCast 1p0deg <2019.npz").open("rb") as f:
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

    @hk.transform_with_state
    def run_forward(inputs, targets_template, forcings):
        predictor = construct_wrapped_gencast()
        return predictor(inputs, targets_template=targets_template, forcings=forcings)

    state = {}
    run_forward_jitted = jax.jit(
        lambda rng, i, t, f: run_forward.apply(params, state, rng, i, t, f)[0]
    )
    run_forward_pmap = xarray_jax.pmap(run_forward_jitted, dim="sample")

    rng = jax.random.PRNGKey(0)
    rngs = np.stack([jax.random.fold_in(rng, i) for i in range(1)], axis=0)

    g_eval_inputs, g_eval_targets, g_eval_forcings = data_utils.extract_inputs_targets_forcings(
        google, target_lead_times=slice("12h", f"{(google.sizes['time']-2)*12}h"), **dataclasses.asdict(task_config))
    
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
    #################################
    google_shifted = google.assign_coords(lon=(google.lon.values + 180) % 360 - 180)
    google_shifted = google_shifted.sortby("lon")
    google_europe = google_shifted.sel(lat=slice(MIN_LAT, MAX_LAT), lon=slice(MIN_LON, MAX_LON))

    g_eval_inputs_eu, g_eval_targets_eu, g_eval_forcings_eu = \
    data_utils.extract_inputs_targets_forcings(
        google_europe,
        target_lead_times=slice("12h", f"{(google_europe.sizes['time']-2)*12}h"),
        **dataclasses.asdict(task_config)
    )
    g_eval_inputs_eu = g_eval_inputs_eu.compute()
    g_eval_targets_eu = g_eval_targets_eu.compute()
    g_eval_forcings_eu = g_eval_forcings_eu.compute()

    from graphcast import samplers_utils, xarray_tree
    import jax.numpy as jnp_patch

    def _gaussian_noise_like(template):
        def _noise_like_da(da):
            data = xarray_jax.unwrap_data(da)
            noise = jax.random.normal(hk.next_rng_key(), shape=data.shape, dtype=data.dtype)
            return xarray_jax.DataArray(noise, dims=da.dims, coords=da.coords)
        return xarray_tree.map_structure(_noise_like_da, template)

    def _patched_apply_stochastic_churn(x, noise_level, stochastic_churn_rate,
                                        noise_level_inflation_factor):
        new_noise_level = noise_level * (1.0 + stochastic_churn_rate)
        noise_diff = new_noise_level**2 - noise_level**2
        noise_diff = jnp_patch.maximum(noise_diff, 0)
        extra_noise_stddev = jnp_patch.sqrt(noise_diff) * noise_level_inflation_factor
        updated_x = x + _spherical_white_noise_like(x) * extra_noise_stddev
        return updated_x, new_noise_level
    
    def templetate_fn(patch, deg = 1):
            lat = np.arange(-90, 90 + deg, deg)
            lon = np.arange(0, 360, deg)
            global_arrays = {}
            for var_name, da in patch.items():
                non_spatial_dims   = [d for d in da.dims if d not in ('lat', 'lon')]
                non_spatial_shape  = [da.sizes[d] for d in non_spatial_dims]
                non_spatial_coords = {k: v for k, v in da.coords.items()
                                    if k not in ('lat', 'lon')}

                # Global shape: same leading dims, but 180 lat × 360 lon
                global_shape = non_spatial_shape + [len(lat), len(lon)]
                global_dims  = non_spatial_dims  + ['lat', 'lon']

                global_arrays[var_name] = xarray.DataArray(
                    data=np.zeros(global_shape, dtype=da.dtype),
                    dims=global_dims,
                    coords={**non_spatial_coords,
                            'lat': lat,
                            'lon': lon},)
            return xarray.Dataset(global_arrays)
    def _spherical_white_noise_like(temp):
        """Samples isotropic mean 0 variance 1 white noise on the sphere."""
        global_template = templetate_fn(temp)
        def spherical_white_noise_like_dataarray(data_array):
            num_wavenumbers = data_array.lon.shape[0] // 2
            key = hk.next_rng_key()
            return samplers_utils.sample(
                key=key,
                power_spectrum=xarray_jax.DataArray(
                    data=np.array([1/num_wavenumbers for _ in range(num_wavenumbers)]),
                    dims=['total_wavenumber']),
                template=data_array)
        full_noise = global_template.map(spherical_white_noise_like_dataarray)
        cropped = full_noise.sel(lat=temp.lat.values, lon=temp.lon.values % 360, method='nearest')
        return cropped.assign_coords(lon=temp.lon.values)


    samplers_utils.spherical_white_noise_like = _spherical_white_noise_like
    samplers_utils.apply_stochastic_churn = _patched_apply_stochastic_churn

    # Cleanly patch the architecture class before instantiating GenCast
    # to avoid Haiku registering a redundant grid2mesh_gnn and breaking parameter loading.
    denoiser._DenoiserArchitecture = ModifiedDenoiserArchitecture

    def construct_wrapped_gencast_modified():
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
    def run_forward_modified(inputs, targets_template, forcings):
        predictor = construct_wrapped_gencast_modified()
        return predictor(inputs, targets_template=targets_template, forcings=forcings)

    state = {}
    run_forward_jitted = jax.jit(
        lambda rng, i, t, f: run_forward_modified.apply(params, state, rng, i, t, f)[0]
    )
    run_forward_pmap = xarray_jax.pmap(run_forward_jitted, dim="sample")

    g_chunks = []
    for chunk in rollout.chunked_prediction_generator_multiple_runs(
        predictor_fn=run_forward_pmap,
        rngs=rngs,
        inputs=g_eval_inputs_eu,
        targets_template=g_eval_targets_eu * np.nan,
        forcings=g_eval_forcings_eu,
        num_steps_per_chunk=1,
        num_samples=1,
        pmap_devices=jax.local_devices()
    ):
        g_chunks.append(chunk)
    g_pred_modified = xarray.combine_by_coords(g_chunks)

    def check_mesh():
        test_arch = ModifiedDenoiserArchitecture(denoiser_architecture_config)
        return len(test_arch._mesh.vertices)

    mesh_nodes = hk.transform(check_mesh).apply({}, jax.random.PRNGKey(0))
    print("\n=== MESH SIZE CHECK ===")
    print(f"Regional mesh nodes : {mesh_nodes}")    

    g_pred_shifted = g_pred.assign_coords(
        lon=(g_pred.lon.values + 180) % 360 - 180
    ).sortby("lon")

    eu_lats = google_europe.lat.values
    eu_lons = google_europe.lon.values
    lat_indices = np.linspace(0, len(eu_lats) - 1, 10, dtype=int)
    lon_indices = np.linspace(0, len(eu_lons) - 1, 10, dtype=int)

    var = "2m_temperature"
    print(f"\n=== 10 POINT PREDICTION COMPARISON (inside Europe region) ===")
    print(f"NOTE: Differences expected — regional model has no global atmospheric context.")
    print(f"{'Point':<7} {'Lat':>6} {'Lon':>7} {'Global':>10} {'Regional':>10} {'Diff':>8}")
    print("-" * 55)
    for i in range(10):
        lat_val = eu_lats[lat_indices[i]]
        lon_val = eu_lons[lon_indices[i]]

        g_val = g_pred_shifted[var].sel(
            lat=lat_val, lon=lon_val, method="nearest"
        ).isel(batch=0, sample=0, time=0).values.item()

        r_val = g_pred_modified[var].isel(
            batch=0, sample=0, time=0,
            lat=lat_indices[i], lon=lon_indices[i]
        ).values.item()

        print(f"{i+1:<7} {lat_val:>6.1f} {lon_val:>7.1f} {g_val:>10.4f} {r_val:>10.4f} {abs(g_val-r_val):>8.4f}")

    google_europe.to_zarr("ground_truth_europe.zarr", mode="w")
    g_pred_global_europe = g_pred_shifted.sel(lat=slice(MIN_LAT, MAX_LAT), lon=slice(MIN_LON, MAX_LON))
    g_pred_global_europe.compute().to_zarr("global_predictions_europe.zarr", mode="w")
    g_pred_modified.compute().to_zarr("regional_predictions.zarr", mode="w")