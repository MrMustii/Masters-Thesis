import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cftr
import xarray as xr

def plot_map(dataset, variable, title=None):
    lambert_proj = ccrs.LambertConformal(central_longitude=8, central_latitude=50, standard_parallels=(50, 50))
    lambert_proj.threshold /= 1000
    lonlat_proj = ccrs.PlateCarree()

    def lonlat_to_grid(lon, lat):
        coords = lambert_proj.transform_point(lon, lat, lonlat_proj)
        return coords[0] / 5500, coords[1] / 5500

    c_i, c_j = lonlat_to_grid(10.4, 56.3)
    
    corners_i, corners_j = [], []
    for lon in [5.025, 16.98]:
        for lat in [52.03, 59.98]:
            i, j = lonlat_to_grid(lon, lat)
            corners_i.append(i)
            corners_j.append(j)

    width = max(corners_i) - min(corners_i)
    height = max(corners_j) - min(corners_j)
    powI = int(np.ceil(np.log2(width)))
    powJ = int(np.ceil(np.log2(height)))

    im = int(np.floor(c_i - 2**powI / 2))
    jm = int(np.floor(c_j - 2**powJ / 2))
    iM = im + 2**powI - 1
    jM = jm + 2**powJ - 1

    xm, xM = im * 5500, iM * 5500
    ym, yM = jm * 5500, jM * 5500

    data = dataset[variable]
    isel_kwargs = {dim: 0 for dim in data.dims if dim not in ("lat", "lon")}
    data = data.isel(**isel_kwargs)

    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=lambert_proj)
    
    ax.set_extent([xm, xM, ym, yM], crs=lambert_proj)

    pcolormesh_im = ax.pcolormesh(
        dataset.coords["lon"], dataset.coords["lat"], data,
        transform=lonlat_proj,
        cmap="RdBu_r",
        shading="nearest"
    )

    ax.add_feature(cftr.BORDERS, edgecolor="black", linewidth=0.8)
    ax.add_feature(cftr.COASTLINE, edgecolor="black", linewidth=0.8)
    
    cbar = plt.colorbar(pcolormesh_im, ax=ax, orientation="horizontal", pad=0.05, fraction=0.046)
    cbar.set_label(f"{variable} (K)")
    ax.set_title(title or variable)
    
    return fig


dataset = "gencast_infernce/predictions/predictions_1year_region.zarr"
ds = xr.open_zarr(dataset)
# get first time step and first level for all variables
ds = ds.isel(time=0)
fig = plot_map(ds, "2m_temperature", title="Predicted 2m Temperature")
plt.savefig("predicted_2m_temperature.png", dpi=300)