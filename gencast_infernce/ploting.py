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

    ZARR_PATH = "/work3/s215225/Predictions/predictions_0.25_2019-03-29_2019-03-30_12h_0.25deg.zarr"

    WB2_PATH  = "/work3/s215225/data/era5/wb2_2019-03-29_2019-03-30_12h_0.25deg.zarr"
    PRED_PATH = "/work3/s215225/Predictions/predictions_0.25_2019-03-29_2019-03-30_12h_0.25deg.zarr"

    ds_wb2  = xr.open_zarr(WB2_PATH)
    ds_pred = xr.open_zarr(PRED_PATH)

    variable = "2m_temperature"

    # prediction: first (only) prediction step, collapse batch/sample dims
    da_pred = ds_pred[variable]
    isel_kwargs = {dim: 0 for dim in da_pred.dims if dim not in ("lat", "lon", "time")}
    isel_kwargs["time"] = 0
    data_pred = (
        da_pred
        .isel(**isel_kwargs)
        .sortby(["lat", "lon"])
        .compute()
    )

    # ground truth: 3rd time step of wb2, convert lon to -180/180 then slice same region as saved predictions
    data_gt = (
        ds_wb2[variable]
        .isel(time=2)
        .assign_coords(lon=(((ds_wb2.lon + 180) % 360) - 180))
        .sortby(["lat", "lon"])
        .sel(lat=slice(45.0, 65.0), lon=slice(-5.0, 25.0))
        .compute()
    )

    data_diff = data_pred - data_gt

    vmin = min(data_gt.min().item(), data_pred.min().item())
    vmax = max(data_gt.max().item(), data_pred.max().item())
    diff_max = max(abs(data_diff.min().item()), abs(data_diff.max().item()))

    plotter = Plot(ds_wb2)

    fig = plt.figure(figsize=(24, 8))
    gs = gridspec.GridSpec(1, 5, width_ratios=[1, 1, 0.05, 1, 0.05])
    gs.update(wspace=0.15, hspace=0.1)

    ax1 = fig.add_subplot(gs[0, 0], projection=plotter.projection)
    im1, _, _ = plotter.individual_plot(data_gt, ax=ax1, title="Ground Truth (2019-03-30 00:00)", vmin=vmin, vmax=vmax)

    ax2 = fig.add_subplot(gs[0, 1], projection=plotter.projection)
    plotter.individual_plot(data_pred, ax=ax2, title="Prediction (2019-03-30 00:00)", vmin=vmin, vmax=vmax)

    cax1 = fig.add_subplot(gs[0, 2])
    fig.colorbar(im1, cax=cax1, label=f"{variable} (K)")

    ax3 = fig.add_subplot(gs[0, 3], projection=plotter.projection)
    im3, _, _ = plotter.individual_plot(data_diff, ax=ax3, title="Difference (Pred - GT)", vmin=-diff_max, vmax=diff_max, cmap="RdBu_r")

    cax2 = fig.add_subplot(gs[0, 4])
    fig.colorbar(im3, cax=cax2, label="Difference (K)")

    plt.savefig("gt_vs_pred.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved to gt_vs_pred.png")