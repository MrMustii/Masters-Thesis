"""Temporally correlated noise for the continuous hourly rollout (exp3, RQ2).

GenCast draws independent noise at each autoregressive step, which yields an hourly
trajectory that is implausible as a time series even when every single hour is
individually well calibrated. The constructions here differ only in how the noise is
correlated across lead times:

    fixed   one spherical draw reused at every lead time, the reference design
    ou      Ornstein-Uhlenbeck, z_i = alpha * z_{i-1} + beta * nu_i, with
            correlation length 1 / ou_rho hours and ou_rho = 0 collapsing to fixed

Marginal hourly scores cannot separate these - two designs with the same marginal
variance give the same hourly CRPS and different trajectories - so comparison 4 reads
the temporal variogram instead.

``build_noise_stacked`` is the batched implementation the rollout calls;
``fixed_noise_sequence`` and ``ou_noise_sequence`` are the readable per-step
equivalents and are kept as the reference the batched version is checked against.
"""

import functools
import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr

from graphcast import samplers_utils
from graphcast import xarray_jax
from modified_Gencast.regional_utils import GLOBAL_LON, build_global_template


sample_fn_cache: dict = {}
batched_sample_fn_cache: dict = {}


def _get_jitted_sample_fn(global_template: xr.DataArray):
    """Return a cached jax.jit-compiled spherical sampler for a given template structure."""
    cache_key = (global_template.dims, global_template.shape)
    if cache_key not in sample_fn_cache:
        num_wavenumbers = len(GLOBAL_LON) // 2
        power_spectrum = xarray_jax.DataArray(
            data=np.array([1.0 / num_wavenumbers] * num_wavenumbers, dtype=np.float32),
            dims=['total_wavenumber'],
        )
        grid = samplers_utils.Grid.for_nodal_data(global_template)
        sample_fn_cache[cache_key] = jax.jit(
            functools.partial(
                samplers_utils.sample,
                power_spectrum=power_spectrum,
                template=global_template,
                grid=grid,
            )
        )
    return sample_fn_cache[cache_key]


def _get_batched_sample_fn(global_template: xr.DataArray):
    """Return a cached jit+vmap sampler that takes (num_ensemble, 2) keys."""
    cache_key = (global_template.dims, global_template.shape)
    if cache_key not in batched_sample_fn_cache:
        sample_jitted = _get_jitted_sample_fn(global_template)
        batched_sample_fn_cache[cache_key] = jax.jit(
            jax.vmap(lambda key: xarray_jax.jax_data(sample_jitted(key=key)))
        )
    return batched_sample_fn_cache[cache_key]


def _sample_spherical_dataset(template: xr.Dataset, rng_key) -> xr.Dataset:
    """Sample one spherical white-noise Dataset using an explicit JAX key."""
    lat_vals = template.lat.values
    lon_query = template.lon.values % 360.0
    original_lon = template.lon.values

    variable_names = list(template.data_vars)
    cpu = jax.devices("cpu")[0]
    rng_key = jax.device_put(rng_key, cpu)
    variable_keys = jax.random.split(rng_key, len(variable_names))

    noise_vars = {}
    for var_name, var_key in zip(variable_names, variable_keys):
        data_array = template[var_name]
        global_template = build_global_template(data_array)
        sample_jitted = _get_jitted_sample_fn(global_template)
        global_noise_jax = sample_jitted(key=var_key)
        global_noise = xr.DataArray(
            np.asarray(global_noise_jax.data),
            dims=global_noise_jax.dims,
            coords={
                d: (np.asarray(data_array.coords[d]) if d not in ('lat', 'lon') and d in data_array.coords
                    else np.asarray(global_noise_jax.coords[d]))
                for d in global_noise_jax.dims
                if d in global_noise_jax.coords
            },
        )
        cropped = global_noise.sel(lat=lat_vals, lon=lon_query, method='nearest')
        noise_vars[var_name] = cropped.assign_coords(lat=lat_vals, lon=original_lon)

    return xr.Dataset(noise_vars)


def fixed_noise_sequence(
    templates: list,
    rng_key,
) -> list:
    """Method A: identical noise vector for every lead time.

    The noise values are the same across all lead times; only the time
    coordinate is updated to match each template.

    Args:
        templates: list of xr.Dataset, one per lead time.  Each must carry the
            correct timedelta64 time coordinate (used by the denoiser for lead-
            time conditioning).
        rng_key: JAX PRNGKey seeding the single spherical noise draw.

    Returns:
        List of len(templates) noise Datasets, identical values, correct coords.
    """
    z_0 = _sample_spherical_dataset(templates[0], rng_key)
    noise_sequence = [z_0]
    for template in templates[1:]:
        z_lead_vars = {
            var: z_0[var].assign_coords(time=template.time.values)
            for var in z_0.data_vars
        }
        noise_sequence.append(xr.Dataset(z_lead_vars))
    return noise_sequence


def ou_noise_sequence(
    templates: list,
    ou_rho: float,
    rng_key,
) -> list:
    """Method B: OU-correlated spherical noise sequence across lead times.

    Discrete OU update (Andrae et al. 2025, Algorithm 2):
        z_i = exp(-rho * dt) * z_{i-1} + sqrt(1 - exp(-2*rho*dt)) * nu_i

    The spherical covariance Sigma_sph is preserved at every step because
    alpha^2 + beta^2 = 1 by construction.  Setting ou_rho = 0 collapses to
    Method A (fixed noise).

    Args:
        templates: list of xr.Dataset, one per lead time.  Each must carry the
            correct timedelta64 time coordinate.  Lead times are extracted from
            these coords; they need not be equally spaced.
        ou_rho: OU decay rate in 1/hours.  Correlation length = 1/ou_rho hours.
        rng_key: JAX PRNGKey seeding the full sequence.

    Returns:
        List of len(templates) OU-correlated noise Datasets.
    """
    num_lead_times = len(templates)
    lead_time_keys = jax.random.split(rng_key, num_lead_times)

    z_prev = _sample_spherical_dataset(templates[0], lead_time_keys[0])
    noise_sequence = [z_prev]

    for i in range(1, num_lead_times):
        delta_t = (
            float(templates[i].time.values[0] / np.timedelta64(1, 'h'))
            - float(templates[i - 1].time.values[0] / np.timedelta64(1, 'h'))
        )
        alpha = float(jnp.exp(-ou_rho * delta_t))
        beta = float(jnp.sqrt(1.0 - jnp.exp(-2.0 * ou_rho * delta_t)))

        nu = _sample_spherical_dataset(templates[i], lead_time_keys[i])

        z_current_vars = {}
        for var in z_prev.data_vars:
            z_current_vals = alpha * z_prev[var].values + beta * nu[var].values
            z_current_vars[var] = xr.DataArray(
                z_current_vals,
                dims=nu[var].dims,
                coords=nu[var].coords,
            )
        z_prev = xr.Dataset(z_current_vars)
        noise_sequence.append(z_prev)

    return noise_sequence


def build_noise_stacked(
    templates: list,
    ou_rho: float,
    noise_method: str,
    noise_rng_keys: list,
) -> xr.Dataset:
    """Build the full noise_stacked Dataset for all ensemble members and lead times.

    Replaces the per-member loop + ou/fixed_noise_sequence calls with a single
    batched approach that eliminates most Python dispatch and device-transfer
    overhead:

      - All (num_ensemble x num_lead_times x num_vars) RNG keys are split in
        one vectorised JAX call so each (member, step, variable) gets a unique
        independent key — identical key hierarchy to the per-step sequential
        implementation.
      - Per lead-time step, a single vmap call generates all ensemble members'
        samples for one variable at once (one device transfer per variable per
        step instead of one per member).
      - OU recursion is applied in numpy on the already-cropped regional arrays,
        so there is no device traffic during the OU update.
      - noise_stacked is assembled directly from pre-allocated numpy arrays;
        no intermediate xr.concat chains.

    Correctness guarantees:
      - Unique starting noise: member m uses noise_rng_keys[m] as base; keys
        are split independently per member before splitting per lead time, so
        member noise sequences are mutually independent.
      - OU correctness: z_h = alpha*z_{h-1} + beta*nu_h applied sequentially
        in numpy; each member evolves along its own independent chain.
      - No noise leakage: noise_accumulated uses pre-allocated np.empty;
        every slice is independently written; no broadcast views are mutated.

    Args:
        templates: list[xr.Dataset] of length chunk_size, one per lead time,
            with time coords [np.timedelta64(h+1, 'h')] for h = 0..chunk_size-1.
        ou_rho: OU decay rate in 1/hours (ignored when noise_method='fixed').
        noise_method: 'ou' or 'fixed'.
        noise_rng_keys: list of num_ensemble distinct JAX PRNGKeys, one per member.

    Returns:
        xr.Dataset with dims (sample=num_ensemble, batch=chunk_size, ...) ready
        for pmap.  The time coordinate is set to [1h] for all batch elements to
        avoid coordinate conflicts during stacking.
    """
    num_ensemble = len(noise_rng_keys)
    num_lead_times = len(templates)
    placeholder_time = [np.timedelta64(1, "h")]
    variable_names = list(templates[0].data_vars)
    num_vars = len(variable_names)


    rng_keys_stacked = jnp.stack(noise_rng_keys) 

    def derive_var_keys_for_member(base_key):
        lead_keys = jax.random.split(base_key, num_lead_times)              
        return jax.vmap(lambda k: jax.random.split(k, num_vars))(lead_keys)  

    
    all_keys = jax.jit(jax.vmap(derive_var_keys_for_member))(rng_keys_stacked)
    
    lat_vals = templates[0].lat.values
    lon_query = templates[0].lon.values % 360.0
    original_lon = templates[0].lon.values

    global_templates_per_var = {}
    lat_crop_indices = {}
    lon_crop_indices = {}
    for var_name in variable_names:
        data_array = templates[0][var_name]
        global_template = build_global_template(data_array)
        global_templates_per_var[var_name] = global_template
        global_lat = global_template.coords["lat"].values
        global_lon = global_template.coords["lon"].values
        lat_crop_indices[var_name] = np.array(
            [np.argmin(np.abs(global_lat - lat)) for lat in lat_vals]
        )
        lon_crop_indices[var_name] = np.array(
            [np.argmin(np.abs(global_lon - lon)) for lon in lon_query]
        )

    noise_accumulated = {var_name: np.empty((num_ensemble, num_lead_times) + templates[0][var_name].shape,
                                            dtype=np.float32,)for var_name in variable_names}

    if noise_method == "fixed":
        for vi, var_name in enumerate(variable_names):
            step_var_keys = all_keys[:, 0, vi]  # (num_ensemble, 2) on GPU
            batched_sample = _get_batched_sample_fn(global_templates_per_var[var_name])
            all_global = np.asarray(batched_sample(step_var_keys))
            lat_idx = lat_crop_indices[var_name]
            lon_idx = lon_crop_indices[var_name]
            all_regional = all_global[..., lat_idx[:, None], lon_idx[None, :]]
            noise_accumulated[var_name][:] = all_regional[:, np.newaxis]

    else:
        delta_ts = np.array([
            float(templates[i].time.values[0] / np.timedelta64(1, "h"))
            - float(templates[i - 1].time.values[0] / np.timedelta64(1, "h")) for i in range(1, num_lead_times)])
        alphas = np.exp(-ou_rho * delta_ts)                          
        betas = np.sqrt(1.0 - np.exp(-2.0 * ou_rho * delta_ts))     

        for h in range(num_lead_times):
            for vi, var_name in enumerate(variable_names):
                step_var_keys = all_keys[:, h, vi]  # (num_ensemble, 2) on GPU
                batched_sample = _get_batched_sample_fn(global_templates_per_var[var_name])
                all_global = np.asarray(batched_sample(step_var_keys))
                lat_idx = lat_crop_indices[var_name]
                lon_idx = lon_crop_indices[var_name]
                all_regional = all_global[..., lat_idx[:, None], lon_idx[None, :]]

                if h == 0:
                    noise_accumulated[var_name][:, 0] = all_regional
                else:
                    noise_accumulated[var_name][:, h] = (alphas[h - 1] * noise_accumulated[var_name][:, h - 1] + betas[h - 1] * all_regional)

    dataset_vars = {}
    for var_name in variable_names:
        data_array_template = templates[0][var_name].assign_coords(time=placeholder_time)
        non_spatial_coords = {k: v for k, v in data_array_template.coords.items() if k not in ("lat", "lon")}
        dataset_vars[var_name] = xr.DataArray(
            noise_accumulated[var_name],
            dims=["sample", "batch"] + list(data_array_template.dims),
            coords={
                "sample": np.arange(num_ensemble),
                "batch": np.arange(num_lead_times),
                **non_spatial_coords,
                "lat": lat_vals,
                "lon": original_lon,
            },
        )

    return xr.Dataset(dataset_vars)
