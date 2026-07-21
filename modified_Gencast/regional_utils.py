import haiku as hk
import jax.numpy as jnp
import numpy as np
import xarray

from graphcast import samplers_utils
from graphcast import xarray_jax


RESOLUTION = 0.25  
GLOBAL_LAT = np.arange(-90.0, 90.0 + RESOLUTION * 0.5, RESOLUTION)
GLOBAL_LON = np.arange(0.0, 360.0, RESOLUTION)

assert len(GLOBAL_LON) == 2 * (len(GLOBAL_LAT) - 1), (f"Global grid constraint violated: {len(GLOBAL_LON)} lon != " f"2 * ({len(GLOBAL_LAT)} lat - 1)")


def build_global_template(regional_da):
    """Zero DataArray on the full global grid, matching non-spatial dims of da."""
    non_spatial_dims   = [d for d in regional_da.dims if d not in ('lat', 'lon')]
    non_spatial_shape  = [regional_da.sizes[d] for d in non_spatial_dims]
    non_spatial_coords = {k: v for k, v in regional_da.coords.items()
                          if k not in ('lat', 'lon')}

    return xarray.DataArray(
        data=np.zeros(non_spatial_shape + [len(GLOBAL_LAT), len(GLOBAL_LON)],
                      dtype=regional_da.dtype),
        dims=non_spatial_dims + ['lat', 'lon'],
        coords={**non_spatial_coords, 'lat': GLOBAL_LAT, 'lon': GLOBAL_LON},
    )


def boundary_margin_mask(template, margin_degrees):
    """Boolean (lat, lon) mask, True on the outer `margin_degrees` band of the grid.

    The band is the lateral-boundary belt of a regional grid: every cell within
    `margin_degrees` of any edge of the grid's lat/lon extent. Used at inference
    to overwrite the belt with host truth (lateral boundary forcing, see
    BoundaryForcedPredictor in gencast.py) when eval.py/eval_continuous.py is
    explicitly run with --boundary-margin-degrees > 0 (default is 0, off). NOT
    used to mask the training loss: weighted_mse_per_level_regional in
    losses.py scores the full grid, boundary belt included. The interior
    region-of-interest is everything the mask leaves False.
    """
    lat = template.coords["lat"]
    lon = template.coords["lon"]
    lat_min, lat_max = float(lat.min()), float(lat.max())
    lon_min, lon_max = float(lon.min()), float(lon.max())
    in_lat_margin = (lat < lat_min + margin_degrees) | (lat > lat_max - margin_degrees)
    in_lon_margin = (lon < lon_min + margin_degrees) | (lon > lon_max - margin_degrees)
    return in_lat_margin | in_lon_margin


def spherical_white_noise_like(template):
    """Isotropic mean-0 variance-1 noise for regional (non-global) grids.

    Samples on the full globe then crops to the regional extent.
    Preserves the lon convention of the input (e.g. [-180, 180]).
    """
    lat_vals  = template.lat.values
    lon_query = template.lon.values % 360.0  # index into [0, 360] global grid

    num_wavenumbers = len(GLOBAL_LON) // 2
    power_spectrum  = xarray_jax.DataArray(
        data=np.array([1.0 / num_wavenumbers] * num_wavenumbers, dtype=np.float32),
        dims=['total_wavenumber'],
    )

    def _sample_one(da):
        global_noise = samplers_utils.sample(
            key=hk.next_rng_key(),
            power_spectrum=power_spectrum,
            template=build_global_template(da),
        )
        cropped = global_noise.sel(lat=lat_vals, lon=lon_query, method='nearest')
        return cropped.assign_coords(lat=lat_vals, lon=template.lon.values)

    return template.map(_sample_one)


def apply_stochastic_churn(x, noise_level, stochastic_churn_rate,
                           noise_level_inflation_factor):
    """Drop-in replacement for samplers_utils.apply_stochastic_churn."""
    new_noise_level    = noise_level * (1.0 + stochastic_churn_rate)
    noise_diff         = jnp.maximum(new_noise_level ** 2 - noise_level ** 2, 0)
    extra_noise_stddev = jnp.sqrt(noise_diff) * noise_level_inflation_factor
    return x + spherical_white_noise_like(x) * extra_noise_stddev, new_noise_level


def patch_samplers_utils():
    """Patch graphcast.samplers_utils with regional-safe versions.

    Patches:
        samplers_utils.spherical_white_noise_like
        samplers_utils.apply_stochastic_churn
    """
    samplers_utils.spherical_white_noise_like = spherical_white_noise_like
    samplers_utils.apply_stochastic_churn     = apply_stochastic_churn
