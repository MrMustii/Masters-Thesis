import os

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature

_TEMP_VAR = "2m_temperature"

# Consistent palette across every plot.
C_ERA5 = "steelblue"
C_MODEL = "tomato"
C_BAND = "tomato"
MODEL_LABEL = "Model"

# Map padding: metrics computed on interior, displayed with this much extra frame.
MAP_PAD_DEG = 5.0


def _robust_limits(values, symmetric=False):
    """Exact colorbar limits: full data range, symmetric about 0 for bias maps."""
    v = np.asarray(values).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return (0.0, 1.0)
    if symmetric:
        m = float(np.abs(v).max())
        return (-m, m)
    return (float(v.min()), float(v.max()))


def _ar_gridlines(ax, lead_hours, ar_window_hours):
    """Day boundaries (grey dotted), AR-window seams (bold dashed navy), x-ticks every 12h."""
    h0, h1 = float(lead_hours[0]), float(lead_hours[-1])
    ax.set_xticks(np.arange(np.ceil(h0 / 12) * 12, h1 + 1e-6, 12))
    day_marks = np.arange(np.ceil(h0 / 24) * 24, h1 + 1e-6, 24)
    for x in day_marks:
        ax.axvline(x, color="grey", linewidth=0.7, linestyle=":", alpha=0.35, zorder=0)
    if ar_window_hours and ar_window_hours < (h1 - h0 + 1):
        bnds = np.arange(np.ceil(h0 / ar_window_hours) * ar_window_hours, h1 + 1e-6, ar_window_hours)
        for x in bnds:
            ax.axvline(x, color="navy", linewidth=1.4, linestyle="--", alpha=0.8, zorder=1)


def _map_extent(region):
    """Display extent padded outward by MAP_PAD_DEG from the (interior) region."""
    if region is None:
        return None
    return (region[2] - MAP_PAD_DEG, region[3] + MAP_PAD_DEG,
            region[0] - MAP_PAD_DEG, region[1] + MAP_PAD_DEG)


def _proj(region):
    clon = (region[2] + region[3]) / 2 if region is not None else 15.0
    clat = (region[0] + region[1]) / 2 if region is not None else 53.0
    return ccrs.LambertConformal(central_longitude=clon, central_latitude=clat)


def _spatial_panel(fig, ax, data, lat, lon, title, cmap, vmin, vmax, cbar_label,
                   extent=None, cbar_extend="neither"):
    ax.patch.set_facecolor("lightgrey")
    lat_step = float(lat[1] - lat[0]) if len(lat) > 1 else 0.25
    lon_step = float(lon[1] - lon[0]) if len(lon) > 1 else 0.25
    lat_edges = np.concatenate([lat - lat_step / 2, [lat[-1] + lat_step / 2]])
    lon_edges = np.concatenate([lon - lon_step / 2, [lon[-1] + lon_step / 2]])
    im = ax.pcolormesh(lon_edges, lat_edges, data, cmap=cmap, vmin=vmin, vmax=vmax,
                       transform=ccrs.PlateCarree(), zorder=2)
    ax.add_feature(cfeature.OCEAN, zorder=1, facecolor="aliceblue")
    ax.coastlines(resolution="50m", color="black", linewidth=0.8, zorder=3)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor="black", zorder=3)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.3,
                      color="grey", linestyle="--", alpha=0.6, zorder=4)
    gl.top_labels = False
    gl.right_labels = False
    gl.x_inline = False
    gl.y_inline = False
    gl.xlabel_style = {"size": 9}
    gl.ylabel_style = {"size": 9}
    if extent is not None:
        ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.set_title(title, fontsize=11, pad=8)
    cbar = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.08, fraction=0.046, extend=cbar_extend)
    cbar.set_label(cbar_label, fontsize=10)
    cbar.ax.tick_params(labelsize=9)


# ── Lead-time curves ──────────────────────────────────────────────────────────

def plot_temp_lead_curve(pred_da, tgt_da, out_path, ar_window_hours=None):
    """Domain-mean temperature vs lead: one top subplot per init hour (ERA5 vs model), one shared bias panel below."""
    weights = np.cos(np.deg2rad(tgt_da.lat))
    w = xr.DataArray(weights, coords={"lat": tgt_da.lat})
    tgt_spatial = tgt_da.weighted(w).mean(["lat", "lon"])
    pred_spatial = pred_da.weighted(w).mean(["lat", "lon"])  # keeps ensemble for the ±1σ band
    has_ensemble = "ensemble" in pred_spatial.dims
    lead_hours = tgt_da.hour.values
    init_hours = pd.DatetimeIndex(tgt_da.forecast.values).hour
    unique_init_hours = np.unique(init_hours)
    num_groups = len(unique_init_hours)

    fig = plt.figure(figsize=(6 * num_groups, 6))
    grid = fig.add_gridspec(2, num_groups)
    bias_ax = fig.add_subplot(grid[1, :])
    first_abs_ax = None
    for group_index, init_hour in enumerate(unique_init_hours):
        abs_ax = fig.add_subplot(grid[0, group_index], sharey=first_abs_ax)
        first_abs_ax = first_abs_ax or abs_ax
        group = init_hours == init_hour
        tgt_curve = tgt_spatial.isel(forecast=group).mean("forecast")
        pred_group = pred_spatial.isel(forecast=group)
        pred_curve = pred_group.mean(["ensemble", "forecast"]) if has_ensemble else pred_group.mean("forecast")
        suffix = f" (init {init_hour:02d} UTC)" if num_groups > 1 else ""
        abs_ax.plot(lead_hours, tgt_curve.values, color=C_ERA5, linewidth=1.8, label="ERA5", zorder=3)
        if has_ensemble:
            spread = pred_group.std("ensemble").mean("forecast")
            abs_ax.fill_between(lead_hours, (pred_curve - spread).values, (pred_curve + spread).values,
                                color=C_MODEL, alpha=0.18, zorder=2)
        abs_ax.plot(lead_hours, pred_curve.values, color=C_MODEL, linewidth=1.8, linestyle="--",
                    label=MODEL_LABEL + (" ±1σ" if has_ensemble else ""), zorder=4)
        _ar_gridlines(abs_ax, lead_hours, ar_window_hours)
        abs_ax.set_title(f"Domain-mean 2m temperature{suffix}", fontsize=11)
        abs_ax.set_ylabel("T2m (K)")
        abs_ax.legend(fontsize=9)
        abs_ax.grid(True, alpha=0.3)
        abs_ax.set_xlim(lead_hours[0], lead_hours[-1])

        color = plt.get_cmap("tab10")(group_index)
        bias_ax.plot(lead_hours, (pred_curve - tgt_curve).values, color=color, linewidth=1.8,
                     label=f"init {init_hour:02d} UTC" if num_groups > 1 else None)

    _ar_gridlines(bias_ax, lead_hours, ar_window_hours)
    bias_ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    bias_ax.set_xlabel("Lead time (hours)")
    bias_ax.set_ylabel("Bias (K)")
    bias_ax.set_title(f"{MODEL_LABEL} − ERA5 bias", fontsize=11)
    if num_groups > 1:
        bias_ax.legend(fontsize=9, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    bias_ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


def plot_temporal_errors(rmse, bias, out_path, ar_window_hours=None):
    lead_hours = rmse.hour.values
    fig, (ax_err, ax_bias) = plt.subplots(1, 2, figsize=(13, 4))
    ax_err.plot(lead_hours, rmse.values, color=C_MODEL, linewidth=1.8)
    _ar_gridlines(ax_err, lead_hours, ar_window_hours)
    ax_err.set_xlabel("Lead time (hours)")
    ax_err.set_ylabel("Error (K)")
    ax_err.set_title("RMSE (ensemble mean)", fontsize=11)
    ax_err.grid(True, alpha=0.3)
    ax_err.set_xlim(lead_hours[0], lead_hours[-1])
    ax_bias.plot(lead_hours, bias.values, color=C_ERA5, linewidth=1.8)
    ax_bias.axhline(0, color="black", linewidth=0.8, linestyle="--")
    _ar_gridlines(ax_bias, lead_hours, ar_window_hours)
    ax_bias.set_xlabel("Lead time (hours)")
    ax_bias.set_ylabel("Bias (K)")
    ax_bias.set_title("Mean bias", fontsize=11)
    ax_bias.grid(True, alpha=0.3)
    ax_bias.set_xlim(lead_hours[0], lead_hours[-1])
    fig.suptitle("2m temperature error growth", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")


def plot_crps_growth(crps_growth, out_path, ar_window_hours=None, ylabel="CRPS (K)",
                     title="Continuous Ranked Probability Score"):
    """Shared by field CRPS and twCRPS growth curves; ylabel/title let the caller relabel."""
    lead_hours = crps_growth.hour.values
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(lead_hours, crps_growth.values, color=C_MODEL, linewidth=1.8)
    _ar_gridlines(ax, lead_hours, ar_window_hours)
    ax.set_xlabel("Lead time (hours)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    ax.set_xlim(lead_hours[0], lead_hours[-1])
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")


def plot_spread_skill(spread, skill, ratio, out_path, ar_window_hours=None):
    lead_hours = spread.hour.values
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(lead_hours, spread.values, color=C_ERA5, linewidth=1.8, label="Spread (std)")
    axes[0].plot(lead_hours, skill.values, color=C_MODEL, linewidth=1.8, label="RMSE (ens mean)")
    _ar_gridlines(axes[0], lead_hours, ar_window_hours)
    axes[0].set_xlabel("Lead time (hours)")
    axes[0].set_ylabel("K")
    axes[0].set_title("Spread vs skill", fontsize=10)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(lead_hours, ratio.values, color="seagreen", linewidth=1.8)
    axes[1].axhline(1.0, color="black", linewidth=0.8, linestyle="--", label="Ratio = 1 (ideal)")
    _ar_gridlines(axes[1], lead_hours, ar_window_hours)
    axes[1].set_xlabel("Lead time (hours)")
    axes[1].set_ylabel("Spread / RMSE")
    axes[1].set_title("Spread-skill ratio", fontsize=10)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)
    fig.suptitle("Ensemble calibration", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")


def plot_mean_forecast(pred, tgt, out_path, ar_window_hours=None):
    """Mean domain-mean temperature per lead, one line pair per init hour."""
    weights = np.cos(np.deg2rad(tgt.lat))
    w = xr.DataArray(weights, coords={"lat": tgt.lat})
    tgt_line = tgt.weighted(w).mean(["lat", "lon"])
    pred_line = pred.weighted(w).mean(["lat", "lon"])
    if "ensemble" in pred_line.dims:
        pred_line = pred_line.mean("ensemble")
    lead_hours = tgt.hour.values
    init_hours = pd.DatetimeIndex(tgt.forecast.values).hour
    unique_init_hours = np.unique(init_hours)
    fig, ax = plt.subplots(figsize=(10, 4))
    for group_index, init_hour in enumerate(unique_init_hours):
        group = init_hours == init_hour
        color = plt.get_cmap("tab10")(group_index)
        suffix = f" (init {init_hour:02d} UTC)" if len(unique_init_hours) > 1 else ""
        ax.plot(lead_hours, tgt_line.isel(forecast=group).mean("forecast").values,
                color=color, linewidth=1.8, label=f"ERA5{suffix}")
        ax.plot(lead_hours, pred_line.isel(forecast=group).mean("forecast").values,
                color=color, linewidth=1.8, linestyle="--", label=f"{MODEL_LABEL}{suffix}")
    _ar_gridlines(ax, lead_hours, ar_window_hours)
    ax.set_xlabel("Lead time (hours)")
    ax.set_ylabel("Domain-mean T2m (K)")
    ax.set_title("Mean forecast 2m temperature", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")


def plot_mean_temporal_difference(pred_curve, tgt_curve, out_path, ar_window_hours=None):
    lead_hours = tgt_curve.hour.values
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(lead_hours, tgt_curve.values, color=C_ERA5, linewidth=1.8, label="ERA5")
    ax.plot(lead_hours, pred_curve.values, color=C_MODEL, linewidth=1.8, label=MODEL_LABEL)
    _ar_gridlines(ax, lead_hours, ar_window_hours)
    ax.set_xlabel("Lead time (hours)")
    ax.set_ylabel("|ΔT| (K)")
    ax.set_title("Mean frame-to-frame temperature difference", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")


# ── Spatial maps ──────────────────────────────────────────────────────────────

def plot_spatial_error_maps(rmse_temp, bias_temp, rmse_tmax, lat, lon, out_dir, tag, region=None):
    rmse_values = [rmse_temp.values.ravel()]
    if rmse_tmax is not None:
        rmse_values.append(rmse_tmax.values.ravel())
    vmin_err, vmax_err = _robust_limits(np.concatenate(rmse_values))
    vmin_b, vmax_b = _robust_limits(bias_temp.values, symmetric=True)
    extent = _map_extent(region)
    subplot_kw = {"projection": _proj(region)}
    panels = [
        ("rmse_temp", rmse_temp.values, "RMSE — 2m temperature", "RdYlBu_r", 0, vmax_err, "RMSE (K)"),
        ("bias_temp", bias_temp.values, "Bias — 2m temperature", "RdBu_r", vmin_b, vmax_b, "Bias (K)"),
    ]
    if rmse_tmax is not None:
        panels.append(("rmse_tmax", rmse_tmax.values, "RMSE — daily Tmax", "RdYlBu_r", 0, vmax_err, "RMSE (K)"))
    for slug, data, title, cmap, vmin, vmax, cbar_label in panels:
        fig_s, ax_s = plt.subplots(1, 1, figsize=(9, 7), subplot_kw=subplot_kw)
        _spatial_panel(fig_s, ax_s, data, lat, lon, title, cmap, vmin, vmax, cbar_label,
                       extent=extent)
        out_single = os.path.join(out_dir, f"spatial_{slug}_{tag}.png")
        fig_s.savefig(out_single, dpi=150, bbox_inches="tight")
        plt.close(fig_s)
        print(f"  saved {out_single}")


def plot_tmax_bias_map(bias_map, lat, lon, out_dir, tag, region=None):
    bias_vals = bias_map.mean("day_idx").values if "day_idx" in bias_map.dims else bias_map.values
    vmin_b, vmax_b = _robust_limits(bias_vals, symmetric=True)
    fig_s, ax_s = plt.subplots(1, 1, figsize=(9, 7), subplot_kw={"projection": _proj(region)})
    _spatial_panel(fig_s, ax_s, bias_vals, lat, lon, "Tmax signed bias", "RdBu_r",
                   vmin_b, vmax_b, "Bias (K)", extent=_map_extent(region))
    out_path = os.path.join(out_dir, f"spatial_bias_tmax_{tag}.png")
    fig_s.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig_s)
    print(f"  saved {out_path}")


def plot_metric_map(data_map, lat, lon, title, cbar_label, out_path, region=None, center=None, cmap=None):
    """Generic single-panel spatial map. With center set (0 for biases, 1 for
    ratios) the colormap is diverging and symmetric about that value."""
    values = data_map.values if hasattr(data_map, "values") else np.asarray(data_map)
    if center is not None:
        vmin_c, vmax_c = _robust_limits(values - center, symmetric=True)
        vmin, vmax = vmin_c + center, vmax_c + center
        cmap = cmap or "RdBu_r"
    else:
        vmin, vmax = _robust_limits(values)
        cmap = cmap or "RdYlBu_r"
    fig, ax = plt.subplots(1, 1, figsize=(9, 7), subplot_kw={"projection": _proj(region)})
    _spatial_panel(fig, ax, values, lat, lon, title, cmap, vmin, vmax, cbar_label,
                   extent=_map_extent(region))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


# ── Heatwave ──────────────────────────────────────────────────────────────────

def plot_heatwave_metrics(precision, recall, f1, f1_map, lat, lon, out_dir, tag, region=None):
    lead_days = precision.day_idx.values
    day_labels = (lead_days + 1).astype(int)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(day_labels, precision.values, color=C_ERA5, linewidth=1.8, marker="o", label="Precision")
    ax.plot(day_labels, recall.values, color=C_MODEL, linewidth=1.8, marker="o", label="Recall")
    ax.plot(day_labels, f1.values, color="seagreen", linewidth=1.8, marker="o", label="F1")
    ax.set_xlabel("Lead day")
    ax.set_ylabel("Score")
    ax.set_title("Heatwave detection — precision / recall / F1", fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_xticks(day_labels)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_curves = os.path.join(out_dir, f"heatwave_prf_{tag}.png")
    fig.savefig(out_curves, dpi=150)
    plt.close(fig)
    print(f"  saved {out_curves}")

    fig_s, ax_s = plt.subplots(1, 1, figsize=(9, 7), subplot_kw={"projection": _proj(region)})
    _spatial_panel(fig_s, ax_s, f1_map.values, lat, lon, "Heatwave F1", "RdYlGn",
                   vmin=0, vmax=1, cbar_label="F1", extent=_map_extent(region), cbar_extend="neither")
    out_map = os.path.join(out_dir, f"heatwave_f1_map_{tag}.png")
    fig_s.savefig(out_map, dpi=150, bbox_inches="tight")
    plt.close(fig_s)
    print(f"  saved {out_map}")


def plot_spell_length_histogram(pred_lengths, tgt_lengths, out_path):
    if len(pred_lengths) == 0 and len(tgt_lengths) == 0:
        print("  skipping spell length histogram (no spells detected)")
        return
    max_len = max(max(pred_lengths, default=1), max(tgt_lengths, default=1))
    bins = np.arange(0.5, max_len + 1.5, 1)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(tgt_lengths, bins=bins, alpha=0.6, color=C_ERA5, label="ERA5",
            density=True, weights=None)
    ax.hist(pred_lengths, bins=bins, alpha=0.6, color=C_MODEL, label=MODEL_LABEL, density=True)
    ax.set_xlabel("Spell length (days)")
    ax.set_ylabel("Density")
    ax.set_title(f"Heatwave spell length distribution "
                 f"(N model={len(pred_lengths)}, ERA5={len(tgt_lengths)})", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")


def plot_rev(rev_curve, out_path):
    cl_ratios = rev_curve.cl_ratio.values
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(cl_ratios, rev_curve.values, color="seagreen", linewidth=1.8)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Cost-loss ratio (C/L)")
    ax.set_ylabel("REV")
    ax.set_title("Relative Economic Value of heatwave forecast", fontsize=11)
    # Log axis: the heatwave-relevant regime is small C/L.
    ax.set_xscale("log")
    ax.set_xlim(cl_ratios[0], 1)
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")


# ── Ensemble diagnostics ──────────────────────────────────────────────────────

def plot_rank_histogram(hist, out_path, title="Rank histogram (Talagrand)"):
    ranks = np.arange(len(hist))
    flat_ref = 1.0 / len(hist)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(ranks, hist, color=C_ERA5, alpha=0.7, width=0.8)
    ax.axhline(flat_ref, color="black", linewidth=1.0, linestyle="--", label="Flat (ideal)")
    ax.set_xlabel("Rank")
    ax.set_ylabel("Relative frequency")
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")


def plot_reliability_diagram(reliability_forecast, reliability_observed, reliability_count, out_path):
    """Reliability curve for probabilistic heatwave detection: observed exceedance
    frequency vs forecast probability, one point per discrete ensemble probability
    bin (M+1 of them). On the diagonal = calibrated; below = over-forecasting,
    above = under-forecasting. The lower panel shows how many cell-days populate
    each bin (a point on a near-empty bin is noise)."""
    forecast_probability = np.asarray(reliability_forecast, dtype=float)
    observed_frequency = np.asarray(reliability_observed, dtype=float)
    bin_count = np.asarray(reliability_count, dtype=float)
    populated = np.isfinite(observed_frequency)
    fig, (ax_reliability, ax_count) = plt.subplots(
        2, 1, figsize=(6, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    ax_reliability.plot([0, 1], [0, 1], color="black", linewidth=0.8, linestyle="--", label="Perfect")
    ax_reliability.plot(forecast_probability[populated], observed_frequency[populated], color=C_MODEL,
                        linewidth=1.6, marker="o", markersize=5, label=MODEL_LABEL)
    ax_reliability.set_ylabel("Observed exceedance frequency")
    ax_reliability.set_title("Reliability — probabilistic heatwave detection", fontsize=11)
    ax_reliability.set_xlim(0, 1)
    ax_reliability.set_ylim(0, 1)
    ax_reliability.legend(fontsize=9)
    ax_reliability.grid(True, alpha=0.3)
    bar_width = 0.8 / max(len(forecast_probability), 1)
    ax_count.bar(forecast_probability, np.maximum(bin_count, 0), width=bar_width, color=C_ERA5, alpha=0.7)
    ax_count.set_yscale("log")
    ax_count.set_xlabel("Forecast probability (member fraction > τ)")
    ax_count.set_ylabel("Count")
    ax_count.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")


def plot_temporal_autocorrelation(pred_ac, tgt_ac, out_path):
    lags = tgt_ac.lag.values
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(lags, tgt_ac.values, color=C_ERA5, linewidth=1.8, label="ERA5")
    ax.plot(lags, pred_ac.values, color=C_MODEL, linewidth=1.8, label=MODEL_LABEL)
    ax.set_xlabel("Lag (hours)")
    ax.set_ylabel("Autocorrelation")
    ax.set_title("Temporal autocorrelation of 2m temperature", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")


def plot_power_spectra(pred_periods, pred_psd, tgt_periods, tgt_psd, out_path):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.loglog(tgt_periods, tgt_psd, color=C_ERA5, linewidth=1.5, label="ERA5")
    ax.loglog(pred_periods, pred_psd, color=C_MODEL, linewidth=1.5, label=MODEL_LABEL)
    for period, period_label in [(24, "24 h"), (12, "12 h")]:
        ax.axvline(period, color="grey", linewidth=0.8, linestyle="--", alpha=0.7)
        ax.text(period * 1.04, 0.97, period_label, fontsize=8, color="grey",
                va="top", transform=ax.get_xaxis_transform())
    ax.set_xlabel("Period (hours)")
    ax.set_ylabel("PSD (K²·h)")
    ax.set_title("Power spectrum of domain-mean 2m temperature", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")


def plot_ensemble_power_spectra(periods, psd_mean, psd_std, tgt_periods, tgt_psd, out_path):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.loglog(tgt_periods, tgt_psd, color=C_ERA5, linewidth=1.5, label="ERA5")
    ax.loglog(periods, psd_mean, color=C_MODEL, linewidth=1.5, label=f"{MODEL_LABEL} (ens mean)")
    ax.fill_between(periods, np.maximum(psd_mean - psd_std, 1e-10), psd_mean + psd_std,
                    color=C_BAND, alpha=0.25, label="±1σ across members")
    for period, period_label in [(24, "24 h"), (12, "12 h")]:
        ax.axvline(period, color="grey", linewidth=0.8, linestyle="--", alpha=0.7)
        ax.text(period * 1.04, 0.97, period_label, fontsize=8, color="grey",
                va="top", transform=ax.get_xaxis_transform())
    ax.set_xlabel("Period (hours)")
    ax.set_ylabel("PSD (K²·h)")
    ax.set_title("Ensemble power spectrum of domain-mean 2m temperature", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")


# ── Day-block curves (integer day-index axis) ─────────────────────────────────

def plot_dtr_error(curve, out_path):
    if curve.sizes.get("day_idx", 0) < 2:
        print("  skipping DTR error plot: fewer than 2 day_idx values")
        return
    day_indices = curve.day_idx.values.astype(int)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(day_indices, curve.values, color="darkorange", linewidth=1.8, marker="o")
    ax.set_xlabel("Day index")
    ax.set_ylabel("DTR MAE (K)")
    ax.set_title("Diurnal Temperature Range error (per member vs ERA5)", fontsize=11)
    ax.set_xticks(day_indices)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")


def plot_diurnal_asymmetry(curve, out_path):
    if curve.sizes.get("day_idx", 0) < 2:
        print("  skipping diurnal asymmetry plot: fewer than 2 day_idx values")
        return
    day_indices = curve.day_idx.values.astype(int)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.plot(day_indices, curve.values, color="mediumpurple", linewidth=1.8, marker="o")
    ax.set_xlabel("Day index")
    ax.set_ylabel("Asymmetry index bias (–)")
    ax.set_title("Diurnal asymmetry index bias (model − ERA5)", fontsize=11)
    ax.set_xticks(day_indices)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")


def plot_over_smoothing(pred_curve, tgt_curve, out_path, xlabel="Lead time (hours)", ar_window_hours=None):
    """Curvature curves — mean |X_{h+1} − 2·X_h + X_{h-1}| by hour, model vs ERA5. The
    model curve sitting below ERA5's = the diurnal cycle is too smooth (the
    over-regularised reconstruction signature). `xlabel` switches lead-hour (exp3)
    vs hour-of-day (exp4); pass ar_window_hours for the lead-hour case so day
    boundaries are marked, like every other lead-hour plot."""
    hours = tgt_curve.hour.values
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(hours, tgt_curve.values, color=C_ERA5, linewidth=1.8, marker="o", markersize=3, label="ERA5")
    ax.plot(hours, pred_curve.values, color=C_MODEL, linewidth=1.8, marker="o", markersize=3, label=MODEL_LABEL)
    if ar_window_hours is not None:
        _ar_gridlines(ax, hours, ar_window_hours)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Mean |2nd difference| (K)")
    ax.set_title("Diurnal-cycle curvature (over-smoothing check)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")




def plot_hour_of_day_curves(curves, ylabel, title, out_path):
    """Per-hour-of-day curves, one line per named entry."""
    fig, ax = plt.subplots(figsize=(9, 4))
    for name, curve in curves.items():
        ax.plot(curve.hour.values, curve.values, linewidth=1.8, marker="o", markersize=3, label=name)
    ax.set_xlabel("Hour of day (UTC)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    ax.set_xticks(range(0, 24, 3))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")


def plot_lead_day_curves(curves, ylabel, title, out_path):
    """Per-lead-day curves (x = forecast lead in days), one line per named entry.

    Used for the 12h fixed-step rollout (exp2), where daily-Tmax detection skill
    is reported against forecast lead day rather than raw 12h sub-steps.
    """
    fig, ax = plt.subplots(figsize=(9, 4))
    for name, curve in curves.items():
        ax.plot(curve[curve.dims[0]].values, curve.values, linewidth=1.8, marker="o", markersize=4, label=name)
    ax.set_xlabel("Forecast lead (days)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")


def plot_qq_tmax(pred_tmax, tgt_tmax, out_path):
    """Q-Q plot of pooled per-member daily Tmax vs ERA5 with an upper-tail
    zoom; below the 1:1 line in the tail = too-cold extremes."""
    pred_values = pred_tmax.values.ravel()
    tgt_values = tgt_tmax.values.ravel()
    pred_values = pred_values[np.isfinite(pred_values)]
    tgt_values = tgt_values[np.isfinite(tgt_values)]
    probabilities = np.concatenate([np.linspace(0.01, 0.99, 99), [0.995, 0.999]])
    pred_quantiles = np.quantile(pred_values, probabilities)
    tgt_quantiles = np.quantile(tgt_values, probabilities)
    fig, (ax_full, ax_tail) = plt.subplots(1, 2, figsize=(12, 5))
    for ax, mask, title in ((ax_full, slice(None), "All quantiles"),
                            (ax_tail, probabilities >= 0.90, "Upper tail (q ≥ 0.90)")):
        tgt_line = tgt_quantiles[mask]
        pred_line = pred_quantiles[mask]
        ax.plot(tgt_line, pred_line, marker="o", markersize=3, linewidth=1.0, color=C_MODEL)
        low = min(tgt_line.min(), pred_line.min())
        high = max(tgt_line.max(), pred_line.max())
        ax.plot([low, high], [low, high], color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("ERA5 daily Tmax quantile (K)")
        ax.set_ylabel(f"{MODEL_LABEL} daily Tmax quantile (K)")
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Q-Q: pooled per-member daily Tmax vs ERA5", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")


def plot_rmse_by_window_position(rmse_curve, out_path):
    positions = rmse_curve.window_pos.values
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(positions, rmse_curve.values, color=C_MODEL, linewidth=1.8)
    six = np.arange(0, positions[-1] + 1e-6, 6)
    for x in six:
        ax.axvline(x, color="grey", linewidth=0.4, linestyle=":", alpha=0.35)
    ax.set_xlabel("Position within AR window (hours)")
    ax.set_ylabel("RMSE (K)")
    ax.set_title("RMSE by position within AR window", fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")


# ── Single-day and weekly ─────────────────────────────────────────────────────

def plot_single_day_profile(pred, target, date_str, out_path, ar_window_hours=None):
    """Diurnal profile for one calendar day using ONLY the day-1 forecast
    (the forecast initialised the day before), so each hour has a single model point.
    Skips a candidate whose masked hours straddle an AR-window reseed (would fake a jump)."""
    date_pd = pd.Timestamp(date_str).date()
    weights = np.cos(np.deg2rad(pred[_TEMP_VAR].lat))
    w = xr.DataArray(weights, coords={"lat": pred[_TEMP_VAR].lat})
    pred_da = pred[_TEMP_VAR]
    tgt_da = target[_TEMP_VAR]

    forecast_times = pd.DatetimeIndex(tgt_da.forecast.values)
    lead_hours_arr = tgt_da.hour.values
    # Prefer the day-1 forecast (init the day before, hours 1..24 land on date);
    # for hour-of-day layouts (daily eval, hours 0..23) the same-date init is
    # the one whose hours cover the date, so try candidates until one does.
    init_candidates = (list(np.where(forecast_times.date == (date_pd - pd.Timedelta(days=1)))[0])
                       + list(np.where(forecast_times.date == date_pd)[0]))
    fc = None
    for candidate in init_candidates:
        candidate_init = forecast_times[int(candidate)]
        candidate_abs_times = np.array([candidate_init + pd.Timedelta(hours=int(h)) for h in lead_hours_arr])
        candidate_mask = np.array([t.date() == date_pd for t in candidate_abs_times])
        if not candidate_mask.any():
            continue
        if ar_window_hours and len(set(lead_hours_arr[candidate_mask] // ar_window_hours)) > 1:
            continue  # straddles an AR-window reseed, would fake a jump
        fc = int(candidate)
        abs_times = candidate_abs_times
        mask = candidate_mask
        break
    if fc is None:
        print(f"  WARNING: no forecast covers {date_str} within one AR window, skipping day profile")
        return

    tgt_line = tgt_da.isel(forecast=fc).weighted(w).mean(["lat", "lon"])
    pred_line = pred_da.isel(forecast=fc).weighted(w).mean(["lat", "lon"])
    hours_of_day = [abs_times[i].hour + abs_times[i].minute / 60 for i in np.where(mask)[0]]
    order = np.argsort(hours_of_day)
    hours_of_day = np.array(hours_of_day)[order]
    tgt_vals = tgt_line.values[mask][order]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(hours_of_day, tgt_vals, color=C_ERA5, marker="o", markersize=5, linewidth=1.5, label="ERA5", zorder=3)
    if "ensemble" in pred_da.dims and pred_da.sizes["ensemble"] > 1:
        model_values = pred_line.mean("ensemble").values[mask][order]
        pstd = pred_line.std("ensemble").values[mask][order]
        ax.errorbar(hours_of_day, model_values, yerr=pstd, fmt="o-", color=C_MODEL, markersize=5,
                    linewidth=1.5, capsize=3, label=f"{MODEL_LABEL} (mean ± 1σ)", zorder=4)
    else:
        pscalar = pred_line.mean("ensemble") if "ensemble" in pred_da.dims else pred_line
        model_values = pscalar.values[mask][order]
        ax.plot(hours_of_day, model_values, color=C_MODEL, marker="o", markersize=5, linewidth=1.5,
               label=MODEL_LABEL, zorder=4)
    # Each hour comes from a different lead of the same init; make that visible.
    lead_values = lead_hours_arr[mask][order]
    for hour_value, lead_value, model_value in zip(hours_of_day, lead_values, model_values):
        ax.annotate(f"+{int(lead_value)}h", (hour_value, model_value), textcoords="offset points",
                    xytext=(8, 4), fontsize=8, color="dimgrey")
    ax.set_ylabel("Domain-mean 2m Temperature (K)")
    ax.set_xlabel("Hour (UTC)")
    ax.set_title(f"2m temperature diurnal profile — {date_str} (day-1 forecast)", fontsize=11)
    ax.set_xticks(range(0, 25, 3))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")


def plot_weekly_temp(pred, target, week_start, out_path):
    """Full 168 h rollout of a SINGLE initialisation (the init on week_start),
    so the curve never starts mid-forecast."""
    weights = np.cos(np.deg2rad(pred[_TEMP_VAR].lat))
    w = xr.DataArray(weights, coords={"lat": pred[_TEMP_VAR].lat})
    pred_da = pred[_TEMP_VAR]
    tgt_da = target[_TEMP_VAR]
    forecast_times = pd.DatetimeIndex(tgt_da.forecast.values)
    target_date = pd.Timestamp(week_start).date()
    idx = np.where(forecast_times.date == target_date)[0]
    if len(idx) == 0:
        idx = [len(forecast_times) // 2]
    fc = int(idx[0])
    init_time = forecast_times[fc]
    lead_hours_arr = tgt_da.hour.values
    abs_times = pd.DatetimeIndex([init_time + pd.Timedelta(hours=int(h)) for h in lead_hours_arr])

    tgt_line = tgt_da.isel(forecast=fc).weighted(w).mean(["lat", "lon"])
    pred_line = pred_da.isel(forecast=fc).weighted(w).mean(["lat", "lon"])

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(abs_times, tgt_line.values, color=C_ERA5, linewidth=1.5, label="ERA5", zorder=2)
    if "ensemble" in pred_da.dims and pred_da.sizes["ensemble"] > 1:
        pmean = pred_line.mean("ensemble").values
        pstd = pred_line.std("ensemble").values
        ax.fill_between(abs_times, pmean - pstd, pmean + pstd, color=C_BAND, alpha=0.25, zorder=1)
        ax.plot(abs_times, pmean, color=C_MODEL, linewidth=1.5, label=f"{MODEL_LABEL} (mean ± 1σ)", zorder=3)
    else:
        pscalar = pred_line.mean("ensemble") if "ensemble" in pred_da.dims else pred_line
        ax.plot(abs_times, pscalar.values, color=C_MODEL, linewidth=1.5, label=MODEL_LABEL, zorder=3)

    xform = ax.get_xaxis_transform()
    t0 = pd.Timestamp(init_time).normalize()
    for d in range(0, 8):
        x = t0 + pd.Timedelta(days=d)
        ax.axvline(x, color="grey", linewidth=0.6, linestyle=":", alpha=0.6)
        ax.text(x, 1.01, x.strftime("%a %-d"), transform=xform, ha="center", va="bottom",
                fontsize=7, color="dimgrey")
    ax.set_ylabel("Domain-mean 2m Temperature (K)")
    ax.set_title(f"7-day forecast from {init_time.strftime('%Y-%m-%d %H UTC')}", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")