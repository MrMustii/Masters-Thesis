import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cftr
import xarray as xr


import matplotlib.gridspec as gridspec
import cmcrameri


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

class Plot:
    def __init__(self, dataset, projection='LambertConformal'):
        self.dataset = dataset
        if projection == 'LambertConformal':
            self.projection = ccrs.LambertConformal(
                central_longitude=8, 
                central_latitude=50, 
                standard_parallels=(50, 50)
            )
        else:
            self.projection = ccrs.PlateCarree()
        
        self.cmap = cmcrameri.cm.vik
        self.lon_dim = "lon"
        self.lat_dim = "lat"

    def individual_plot(self, map_data, ax=None, title="", vmin=None, vmax=None, gridlines=True, coastlines=True, cmap=None):
        if ax is None:
            ax = plt.figure().add_subplot(111, projection=self.projection)
        if cmap is None:
            cmap = self.cmap
            
        im = ax.pcolormesh(
            map_data[self.lon_dim], map_data[self.lat_dim], map_data,
            vmin=vmin, vmax=vmax, cmap=cmap,
            transform=ccrs.PlateCarree()
        )
        
        g = ax.gridlines(draw_labels=True, linestyle='--', color='grey', 
                         x_inline=False, y_inline=False, linewidth=0.3 * gridlines)
        g.right_labels = False
        g.bottom_labels = False
        g.xlabel_style = {'size': 10, 'color': 'grey'}
        g.ylabel_style = {'size': 10, 'color': 'grey'}
        ax.set_title(title)
        
        if coastlines:
            ax.coastlines(resolution='50m', color='black', lw=0.1)
            
        return im, ax, g

    def show_single_map(self, variable="2m_temperature", vmin=240, vmax=306):
        fig = plt.figure(figsize=(8, 6))
        gs = gridspec.GridSpec(1, 2, width_ratios=[16, 1])
        gs.update(wspace=0.1, hspace=0.1)

        ax1 = fig.add_subplot(gs[0, 0], projection=self.projection)
        
        map_data = self.dataset[variable]
        if "time" in map_data.dims:
            map_data = map_data.isel(time=0)
            
        im1, ax1, g1 = self.individual_plot(map_data, ax=ax1, title=variable, vmin=vmin, vmax=vmax)

        ax2 = fig.add_subplot(gs[0, 1])
        fig.colorbar(im1, cax=ax2, label=f"{variable} (K)", use_gridspec=True)

        return fig

# Execution:


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import xarray as xr
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    ds_truth = xr.open_zarr("gencast_infernce/predictions/crop_vs_uncroped/ground_truth_europe.zarr")
    ds_global = xr.open_zarr("gencast_infernce/predictions/crop_vs_uncroped/global_predictions_europe.zarr")
    ds_regional = xr.open_zarr("gencast_infernce/predictions/crop_vs_uncroped/regional_predictions.zarr")

    variable = "2m_temperature"

    def prep_data(ds, var):
        data = ds[var]
        isel_kwargs = {dim: 0 for dim in data.dims if dim not in ("lat", "lon")}
        data = data.isel(**isel_kwargs)
        data = data.sortby(["lat", "lon"])
        return data.sel(lon=slice(3.5, 18.25), lat=slice(50.5, 61.25)).compute()

    data_truth = prep_data(ds_truth, variable)
    data_global = prep_data(ds_global, variable)
    data_regional = prep_data(ds_regional, variable)
    data_diff = data_regional - data_global

    vmin = min(data_truth.min().item(), data_global.min().item(), data_regional.min().item())
    vmax = max(data_truth.max().item(), data_global.max().item(), data_regional.max().item())
    diff_max = max(abs(data_diff.min().item()), abs(data_diff.max().item()))

    plotter = Plot(ds_truth)

    fig = plt.figure(figsize=(28, 6))
    gs = gridspec.GridSpec(1, 6, width_ratios=[1, 1, 1, 0.05, 1, 0.05])
    gs.update(wspace=0.15, hspace=0.1)

    ax1 = fig.add_subplot(gs[0, 0], projection=plotter.projection)
    im1, _, _ = plotter.individual_plot(data_truth, ax=ax1, title="Ground Truth", vmin=vmin, vmax=vmax)

    ax2 = fig.add_subplot(gs[0, 1], projection=plotter.projection)
    plotter.individual_plot(data_global, ax=ax2, title="Global Model (Unsliced)", vmin=vmin, vmax=vmax)

    ax3 = fig.add_subplot(gs[0, 2], projection=plotter.projection)
    plotter.individual_plot(data_regional, ax=ax3, title="Regional Model (Sliced)", vmin=vmin, vmax=vmax)

    cax1 = fig.add_subplot(gs[0, 3])
    fig.colorbar(im1, cax=cax1, label=f"{variable} (K)")

    ax4 = fig.add_subplot(gs[0, 4], projection=plotter.projection)
    im4, _, _ = plotter.individual_plot(data_diff, ax=ax4, title="Difference (Regional - Global)", vmin=-diff_max, vmax=diff_max, cmap="RdBu_r")

    cax2 = fig.add_subplot(gs[0, 5])
    fig.colorbar(im4, cax=cax2, label="Difference (K)")

    plt.savefig("comparison_plot2.png", dpi=300, bbox_inches='tight')
    plt.close()

    # full EU domain
    eu_proj = ccrs.LambertConformal(central_longitude=15, central_latitude=50, standard_parallels=(50, 50))
    plotter_eu = Plot(ds_truth)
    plotter_eu.projection = eu_proj

    data_truth_eu = ds_truth[variable].isel(**{dim: 0 for dim in ds_truth[variable].dims if dim not in ("lat", "lon")}).sortby(["lat", "lon"]).sel(lat=slice(25, 82), lon=slice(-30, 60)).compute()
    data_global_eu = ds_global[variable].isel(**{dim: 0 for dim in ds_global[variable].dims if dim not in ("lat", "lon")}).sortby(["lat", "lon"]).sel(lat=slice(25, 82), lon=slice(-30, 60)).compute()
    data_regional_eu = ds_regional[variable].isel(**{dim: 0 for dim in ds_regional[variable].dims if dim not in ("lat", "lon")}).sortby(["lat", "lon"]).sel(lat=slice(25, 82), lon=slice(-30, 60)).compute()
    data_diff_eu = data_regional_eu - data_global_eu

    vmin_eu = min(data_truth_eu.min().item(), data_global_eu.min().item(), data_regional_eu.min().item())
    vmax_eu = max(data_truth_eu.max().item(), data_global_eu.max().item(), data_regional_eu.max().item())
    diff_max_eu = max(abs(data_diff_eu.min().item()), abs(data_diff_eu.max().item()))

    fig2 = plt.figure(figsize=(28, 6))
    gs2 = gridspec.GridSpec(1, 6, width_ratios=[1, 1, 1, 0.05, 1, 0.05])
    gs2.update(wspace=0.15, hspace=0.1)

    ax1e = fig2.add_subplot(gs2[0, 0], projection=eu_proj)
    im1e, _, _ = plotter_eu.individual_plot(data_truth_eu, ax=ax1e, title="Ground Truth (Full EU)", vmin=vmin_eu, vmax=vmax_eu)

    ax2e = fig2.add_subplot(gs2[0, 1], projection=eu_proj)
    plotter_eu.individual_plot(data_global_eu, ax=ax2e, title="Global Model (Full EU)", vmin=vmin_eu, vmax=vmax_eu)

    ax3e = fig2.add_subplot(gs2[0, 2], projection=eu_proj)
    plotter_eu.individual_plot(data_regional_eu, ax=ax3e, title="Regional Model (Full EU)", vmin=vmin_eu, vmax=vmax_eu)

    cax1e = fig2.add_subplot(gs2[0, 3])
    fig2.colorbar(im1e, cax=cax1e, label=f"{variable} (K)")

    ax4e = fig2.add_subplot(gs2[0, 4], projection=eu_proj)
    im4e, _, _ = plotter_eu.individual_plot(data_diff_eu, ax=ax4e, title="Difference (Regional - Global, Full EU)", vmin=-diff_max_eu, vmax=diff_max_eu, cmap="RdBu_r")

    cax2e = fig2.add_subplot(gs2[0, 5])
    fig2.colorbar(im4e, cax=cax2e, label="Difference (K)")

    plt.savefig("comparison_plot_full_eu.png", dpi=300, bbox_inches='tight')