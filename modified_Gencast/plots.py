"""
VERY BAD NEED FIXING


Figure builders for the comparison layer. Every function returns a figure and does no I/O.

Step 5 of the pipeline. The ``comparison_*.py`` scripts read the metric caches, decide what
to draw, call one of these and save the result themselves - nothing here opens a file, reads
a cache or knows where a figure ends up. Each builder raises rather than returning an empty
figure when handed no data.

Which comparison uses which builder, from the call sites:

    build_paired_forest_figure           1  2/3  4/5  6/7
    build_pairwise_significance_figure   1  2/3  4/5  6/7
    build_reliability_figure             1  2/3  4/5  6/7
    build_stacked_rev_figure             1  2/3  4/5  6/7
    build_map_grid_figure                1  2/3  4/5  6/7
    build_lead_curve_figure              0     2/3  4/5  6/7
    build_tmax_distribution_figure       0  1  2/3
    build_tmax_scatter_figure            1  2/3
    build_tmax_error_by_observed_figure  1  2/3
    build_calendar_skill_figure          1  2/3
    build_domain_mean_figure                      4/5  6/7
    build_rmse_map_grid_figure                         6/7
    build_monthly_temperature_qq_figure                6/7

Three arguments are accepted and never read, each noted in its own docstring:
``build_monthly_temperature_qq_figure(probabilities)``, and ``title`` and
``observed_share`` in ``build_calendar_skill_figure``.
"""

import textwrap

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import BoundaryNorm, ListedColormap
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Set once here so no builder carries its own sizes. The only per-figure override left is
# the significance-matrix cell text. savefig.dpi applies only when a caller omits dpi=,
# which most do not: the comparison scripts pass dpi=200.
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.titlesize": 13,
    "savefig.dpi": 300,
    "figure.constrained_layout.use": True,
})
# \textwidth is 448.0 pt = 6.199 in. Author each figure at the width it will occupy and
# LaTeX scales it by 1, so 12 pt here is 12 pt on the page. Include with
# \includegraphics[width=...], never scale=, and save without bbox_inches="tight": that
# retrims the canvas and puts back the scaling this is meant to remove.
FULL_WIDTH_FIGURE_INCHES = 448.0 / 72.27


def wrap_title(title: str, width: int = 58) -> str:
    """Hard-wrap a title at `width` characters, preserving any newlines already in it.

    Wrapping only - the font is never shrunk, so a title longer than the canvas is clipped
    by the canvas rather than scaled to fit.
    """
    return "\n".join(textwrap.fill(line, width) for line in str(title).split("\n"))

MODEL_LABEL = "Model"

# Map padding: metrics computed on interior, displayed with this much extra frame.
MAP_PAD_DEG = 5.0


def _map_extent(region):
    """Region widened by MAP_PAD_DEG, reordered for cartopy's set_extent.

    Takes (min_lat, max_lat, min_lon, max_lon) - the order used everywhere else in this
    project - and returns (min_lon, max_lon, min_lat, max_lat), which is what set_extent
    expects. Returns None for a global region.
    """
    if region is None:
        return None
    return (region[2] - MAP_PAD_DEG, region[3] + MAP_PAD_DEG,
            region[0] - MAP_PAD_DEG, region[1] + MAP_PAD_DEG)


def _proj(region):
    """Lambert conformal projection centred on the region, or on Europe when region is None."""
    clon = (region[2] + region[3]) / 2 if region is not None else 15.0
    clat = (region[0] + region[1]) / 2 if region is not None else 53.0
    return ccrs.LambertConformal(central_longitude=clon, central_latitude=clat)


# ── Ensemble calibration and value ──────────────────────────────────────────

def build_reliability_figure(curves, title: str):
    """Return a model-only daily p90-exceedance reliability figure without I/O.

    curves is an iterable of (label, forecast_probability, observed_frequency) tuples.
    The event is Tmax above the 1981-2010 windowed p90 for that day and cell.
    Deterministic baselines are intentionally excluded by the caller.
    """
    curves = list(curves)
    if not curves:
        raise ValueError("At least one reliability curve is required")

    # Taller than the other single-axis plots: a three-line title and a below-axes
    # legend leave too little axes height for the y-label to fit beside it otherwise.
    figure, axis = plt.subplots(figsize=(FULL_WIDTH_FIGURE_INCHES, 6.71))
    palette = plt.get_cmap("tab10")
    axis.plot([0.0, 1.0], [0.0, 1.0], color="black", linewidth=1.0, linestyle="--",
              label="Perfect calibration", zorder=1)
    for index, (label, forecast_probability, observed_frequency) in enumerate(curves):
        forecast_probability = np.asarray(forecast_probability, dtype=float)
        observed_frequency = np.asarray(observed_frequency, dtype=float)
        populated = np.isfinite(forecast_probability) & np.isfinite(observed_frequency)
        if not populated.any():
            continue
        axis.plot(
            forecast_probability[populated],
            observed_frequency[populated],
            color=palette(index % 10),
            marker="o",
            markersize=4.2,
            linewidth=1.65,
            label=label,
            zorder=2,
        )
    axis.set(xlim=(0.0, 1.0), ylim=(0.0, 1.0),
             xlabel="Forecast probability that Tmax exceeds the threshold",
             ylabel="Observed frequency of exceedance")
    axis.set_title(wrap_title(title), pad=11)
    axis.grid(True, alpha=0.28)
    figure.legend(*axis.get_legend_handles_labels(), loc="outside lower center",
                  ncol=2, frameon=False, handlelength=2.4)
    return figure


def build_stacked_rev_figure(curves, title: str):
    """Return a stacked ensemble-model REV figure without I/O.

    curves is an iterable of (label, cost_loss_ratio, relative_economic_value) tuples,
    on the heatwave definition. The zero line is the climatological-reference value by
    definition.
    """
    curves = list(curves)
    if not curves:
        raise ValueError("At least one REV curve is required")

    figure, axis = plt.subplots(figsize=(FULL_WIDTH_FIGURE_INCHES, 4.88))
    palette = plt.get_cmap("tab10")
    axis.axhline(0.0, color="black", linewidth=1.0, linestyle="--",
                 label="Climatology reference (REV = 0)", zorder=1)
    for index, (label, cost_loss_ratio, value) in enumerate(curves):
        cost_loss_ratio = np.asarray(cost_loss_ratio, dtype=float)
        value = np.asarray(value, dtype=float)
        finite = np.isfinite(cost_loss_ratio) & np.isfinite(value)
        if not finite.any():
            continue
        axis.plot(
            cost_loss_ratio[finite],
            value[finite],
            color=palette(index % 10),
            linewidth=1.65,
            label=label,
            zorder=2,
        )
    axis.set(
        xscale="log",
        xlim=(0.01, 1.0),
        ylim=(-0.02, 1.02),
        xlabel="Cost-loss ratio",
        ylabel="Relative economic value",
    )
    axis.set_title(wrap_title(title), pad=11)
    axis.grid(True, alpha=0.28, which="both")
    figure.legend(*axis.get_legend_handles_labels(), loc="outside lower center",
                  ncol=2, frameon=False, handlelength=2.4)
    return figure


# ── Spatial maps ────────────────────────────────────────────────────────────

def build_rmse_map_grid_figure(fields, region):
    """A two-column shared-scale map grid for non-negative hourly RMSE fields.

    ``fields`` is an ordered iterable of (model label, RMSE map).  All panels
    share a white-to-red scale that starts at physical zero, with one colourbar.
    """
    fields = [(label, array) for label, array in fields if array is not None]
    if not fields:
        raise ValueError("At least one RMSE field is required")
    columns = 2
    rows = int(np.ceil(len(fields) / columns))
    values = np.concatenate([np.asarray(array.values, dtype=float).ravel() for _, array in fields])
    vmax = float(np.nanpercentile(values, 99))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    figure, axes = plt.subplots(rows, columns,
                                figsize=(FULL_WIDTH_FIGURE_INCHES, 6.8 * rows),
                                subplot_kw={"projection": _proj(region)}, squeeze=False)
    mesh = None
    for axis, (label_text, array) in zip(axes.ravel(), fields, strict=False):
        axis.set_extent(_map_extent(region), crs=ccrs.PlateCarree())
        axis.add_feature(cfeature.BORDERS, linewidth=0.45, edgecolor="0.25")
        axis.add_feature(cfeature.COASTLINE, linewidth=0.55, edgecolor="0.15")
        mesh = axis.pcolormesh(array.lon, array.lat, array.values,
                               transform=ccrs.PlateCarree(), cmap="Reds",
                               vmin=0.0, vmax=vmax, shading="auto")
        axis.set_title(wrap_title(label_text, 24))
    for axis in axes.ravel()[len(fields):]:
        axis.set_visible(False)
    figure.colorbar(mesh, ax=axes.ravel().tolist(), orientation="horizontal",
                    fraction=0.035, pad=0.055, aspect=45).set_label("Hourly 2 m temperature RMSE (K)")
    return figure


def build_map_grid_figure(fields, truth_label: str, title: str, difference: bool = False,
                          truth_field=None):
    """Return one row-wrapped map grid without I/O.

    fields is an iterable of (label, data array on lat/lon). With difference=False every
    panel shares one temperature scale taken from the 1st and 99th percentiles of all
    fields; with difference=True they share one symmetric scale centred on zero.

    truth_field adds the truth as a first panel on its own colourbar. No caller uses it,
    and it currently raises inside matplotlib's constrained layout, so treat it as
    unsupported rather than as a working option.
    """
    fields = [(label, array) for label, array in fields if array is not None]
    if not fields:
        raise ValueError("At least one map field is required")

    stacked = np.concatenate([np.asarray(array.values, dtype=float).ravel() for _, array in fields])
    if difference:
        limit = float(np.nanpercentile(np.abs(stacked), 99))
        vmin, vmax, colourbar_label = -limit, limit, f"{MODEL_LABEL} - {truth_label} (K)"
    else:
        vmin, vmax = (float(np.nanpercentile(stacked, 1)), float(np.nanpercentile(stacked, 99)))
        colourbar_label = "2 m temperature (K)"

    panels = ([(truth_label, truth_field, False)] if truth_field is not None else [])
    panels += [(label, array, difference) for label, array in fields]
    # Two columns: four panel titles across the text width collide at 12 pt.
    columns = min(2, len(panels))
    rows = int(np.ceil(len(panels) / columns))
    region = (float(fields[0][1].lat.min()), float(fields[0][1].lat.max()),
              float(fields[0][1].lon.min()), float(fields[0][1].lon.max()))
    figure = plt.figure(figsize=(FULL_WIDTH_FIGURE_INCHES,
                                 max(2.8, (3.9 * rows + 1.0) * FULL_WIDTH_FIGURE_INCHES
                                     / (3.7 * columns))))
    difference_mesh = truth_mesh = None
    for index, (label_text, array, on_difference_scale) in enumerate(panels):
        axis = figure.add_subplot(rows, columns, index + 1, projection=_proj(region))
        axis.set_extent(_map_extent(region), crs=ccrs.PlateCarree())
        axis.add_feature(cfeature.BORDERS, linewidth=0.45, edgecolor="0.25")
        axis.add_feature(cfeature.COASTLINE, linewidth=0.55, edgecolor="0.15")
        limits = ((vmin, vmax) if on_difference_scale or truth_field is None
                  else (float(np.nanpercentile(array.values, 1)),
                        float(np.nanpercentile(array.values, 99))))
        mesh = axis.pcolormesh(array.lon, array.lat, array.values, transform=ccrs.PlateCarree(),
                               cmap="RdBu_r", vmin=limits[0], vmax=limits[1], shading="auto")
        axis.set_title(wrap_title(label_text, 22))
        if on_difference_scale or truth_field is None:
            difference_mesh = mesh
        else:
            truth_mesh = mesh
    figure.suptitle(wrap_title(title))
    if truth_mesh is not None:
        figure.colorbar(truth_mesh, ax=figure.axes, orientation="horizontal",
                        fraction=0.028, pad=0.04, aspect=45).set_label(
                            "2 m temperature (K)")
    figure.colorbar(difference_mesh, ax=figure.axes, orientation="horizontal",
                    fraction=0.028, pad=0.05, aspect=45).set_label(colourbar_label)
    return figure


# ── Distribution diagnostics ────────────────────────────────────────────────

def build_monthly_temperature_qq_figure(model_quantiles, target_quantiles, probabilities):
    """Monthly Q--Q diagnostic for reconstructed hourly temperature distributions.

    ``target_quantiles`` and every entry of ``model_quantiles`` have shape
    (12, n_quantiles). Values pool every reconstructed hourly grid-cell value within one
    calendar month; model values are the ensemble mean first, so this is a distributional
    diagnostic of the displayed deterministic reconstruction.

    ``probabilities`` is accepted but not read: the x axis is the target quantiles
    themselves, not the probabilities they were taken at.
    """
    month_names = [pd.Timestamp(2021, month, 1).strftime("%b") for month in range(1, 13)]
    figure, axes = plt.subplots(4, 3, figsize=(FULL_WIDTH_FIGURE_INCHES, 10.8))
    colours = plt.get_cmap("tab10")(np.arange(len(model_quantiles)))
    for month, axis in enumerate(axes.ravel()):
        target = np.asarray(target_quantiles[month], dtype=float)
        values = [target, *[np.asarray(quantiles[month], dtype=float)
                            for quantiles in model_quantiles.values()]]
        low = min(np.nanmin(value) for value in values)
        high = max(np.nanmax(value) for value in values)
        padding = max(0.25, 0.04 * (high - low))
        for (label, prediction), colour in zip(model_quantiles.items(), colours, strict=True):
            axis.plot(target, prediction[month], color=colour, linewidth=1.25,
                      marker="o", markersize=2.5, label=label)
        axis.plot([low - padding, high + padding], [low - padding, high + padding],
                  color="black", linewidth=0.8, linestyle="--")
        axis.set_xlim(low - padding, high + padding)
        axis.set_ylim(low - padding, high + padding)
        axis.set_title(month_names[month])
        axis.grid(True, alpha=0.28)
    figure.suptitle("Monthly temperature-distribution Q--Q diagnostic, 2021")
    figure.legend(*axes.flat[0].get_legend_handles_labels(), loc="outside lower center",
                  ncol=2, frameon=False, handlelength=2.0)
    return figure


# ── Significance: candidate against one reference ───────────────────────────

def build_paired_forest_figure(rows, metric_label: str, unit: str, title: str):
    """Return one paired-difference forest on a single axis, without I/O.

    Every row supplies label, difference, lower, upper and is_baseline. Rows flagged as
    baselines are dropped before anything is drawn: their intervals are often tens of times
    wider and would crush every model interval onto the zero line. The x axis is therefore
    scaled to the models only, and a row whose difference falls outside it is marked at the
    edge with its real value instead of being clipped away.
    """
    rows = [row for row in rows if not row.get("is_baseline")]
    if not rows:
        raise ValueError("At least one forest row is required")

    figure, axis = plt.subplots(figsize=(FULL_WIDTH_FIGURE_INCHES,
                                         max(2.8, (0.42 * len(rows) + 1.2)
                                             * FULL_WIDTH_FIGURE_INCHES / 11.0)))
    positions = np.arange(len(rows))[::-1]
    difference = np.array([float(row["difference"]) for row in rows])
    lower = np.array([float(row["lower"]) for row in rows])
    upper = np.array([float(row["upper"]) for row in rows])
    # Enough room for the marker and cap at the extreme rows, and no more.
    padding = max(0.06 * max(upper.max() - lower.min(), 0.02), 0.01)
    left = min(lower.min(), 0.0) - padding
    right = max(upper.max(), 0.0) + padding
    axis.set_xlim(left, right)
    # Half-row margins keep the outer rows off the axis edge.
    axis.set_ylim(-0.6, len(rows) - 0.4)
    inside = (difference >= left) & (difference <= right)
    if inside.any():
        axis.errorbar(difference[inside], positions[inside],
                      xerr=np.vstack([difference - lower, upper - difference])[:, inside],
                      fmt="o", markersize=5.8, capsize=3.2, linewidth=1.25,
                      color="#0f4c5c", label="Regional models")
    # Rows off the scale are marked at the edge with their real value.
    for index in np.flatnonzero(~inside):
        edge = right if difference[index] > right else left
        axis.plot(edge, positions[index], marker=">" if edge == right else "<",
                  markersize=8, color="#0f4c5c", clip_on=False)
        axis.annotate(f"{difference[index]:.3g}", (edge, positions[index]),
                      textcoords="offset points",
                      xytext=(-6 if edge == right else 6, 9),
                      ha="right" if edge == right else "left", color="#0f4c5c")
    axis.axvline(0.0, color="black", linewidth=0.9, zorder=0)
    axis.set_yticks(positions)
    axis.set_yticklabels([str(row["label"]) for row in rows])
    axis.set_xlabel(wrap_title(f"Candidate − reference {metric_label} ({unit}); "
                              f"left favours the candidate", 46))
    axis.grid(axis="x", alpha=0.25)
    figure.legend(*axis.get_legend_handles_labels(), loc="outside lower center",
                  ncol=2, frameon=False)
    axis.set_title(wrap_title(title), pad=12)
    return figure


# ── Lead time and time series ───────────────────────────────────────────────

def build_lead_curve_figure(curves, xlabel: str, ylabel: str, title: str):
    """Return one lead-time curve figure without I/O.

    curves is an iterable of (label, lead values, metric values). A curve counts as a
    baseline when its label contains "climatology" or "persistence"; those are drawn
    thicker and in a fixed shade so they read as references rather than candidates -
    climatology solid in greys, persistence dash-dotted in browns - and are plotted last
    so they sit at the end of the legend. Everything else takes the next tab10 colour.
    """
    curves = list(curves)
    if not curves:
        raise ValueError("At least one lead curve is required")

    # Keep the two-row legend inside the canvas.  The former outside-legend layout
    # was clipped when saved without a tight bounding box.
    figure, axis = plt.subplots(figsize=(FULL_WIDTH_FIGURE_INCHES, 5.25))
    figure.subplots_adjust(bottom=0.24)
    palette = plt.get_cmap("tab10")
    # One shade per region so six baselines stay distinguishable, not two colours for all.
    climatology_shades = ("0.10", "0.45", "0.68")
    persistence_shades = ("#6b3f2e", "#a9714f", "#d9a679")
    seen = {"climatology": [], "persistence": []}
    def baseline_key(label):
        lowered = label.lower()
        return next((key for key in seen if key in lowered), None)
    model_index = 0
    # Baselines are drawn last so they sit at the end of the legend.
    for label, lead, values in sorted(curves, key=lambda curve: baseline_key(curve[0]) is not None):
        key = baseline_key(label)
        if key is None:
            colour, style, width = palette(model_index % 10), "-", 1.6
            model_index += 1
        else:
            if label not in seen[key]:
                seen[key].append(label)
            shades = climatology_shades if key == "climatology" else persistence_shades
            colour = shades[seen[key].index(label) % 3]
            style = "-" if key == "climatology" else "-."
            width = 2.0
        axis.plot(np.asarray(lead, dtype=float), np.asarray(values, dtype=float),
                  label=label, linewidth=width, color=colour, linestyle=style)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.set_title(wrap_title(title), pad=10)
    axis.grid(alpha=0.28)
    figure.legend(*axis.get_legend_handles_labels(), loc="lower center",
                  bbox_to_anchor=(0.5, 0.02), ncol=2, frameon=False)
    return figure


def build_domain_mean_figure(curves, title: str, truth_label: str = "ERA5 truth"):
    """Return one domain-mean time-series figure without I/O.

    curves is an iterable of (label, times, values). The truth curve is drawn black and
    thicker so the models read against it.
    """
    curves = list(curves)
    if not curves:
        raise ValueError("At least one domain-mean curve is required")

    figure, axis = plt.subplots(figsize=(FULL_WIDTH_FIGURE_INCHES, 4.28))
    palette = plt.get_cmap("tab10")
    model_index = 0
    for label, times, values in curves:
        is_truth = label == truth_label
        axis.plot(times, values, label=label,
                  color="black" if is_truth else palette(model_index % 10),
                  linewidth=2.0 if is_truth else 1.4,
                  zorder=3 if is_truth else 2)
        model_index += 0 if is_truth else 1
    for day in pd.date_range(curves[0][1][0], curves[0][1][-1], freq="D"):
        axis.axvline(day, color="grey", linewidth=0.6, linestyle=":", alpha=0.4, zorder=0)
    axis.set_xlabel("Valid time (UTC)")
    axis.set_ylabel("2 m temperature (K)")
    axis.set_title(wrap_title(title), pad=10)
    axis.grid(alpha=0.25)
    figure.legend(*axis.get_legend_handles_labels(), loc="outside lower center",
                  ncol=2, frameon=False)
    # Rotate the dates by hand: figure.autofmt_xdate() calls subplots_adjust, which
    # fights constrained layout and shifts the axes off the y-label.
    for tick in axis.get_xticklabels():
        tick.set_rotation(30)
        tick.set_horizontalalignment("right")
    return figure


# ── Daily Tmax ──────────────────────────────────────────────────────────────

def build_tmax_distribution_figure(curves, bin_edges, title: str, truth_label: str = "ERA5 truth"):
    """Return a daily-Tmax distribution figure without I/O.

    curves is an iterable of (label, counts) histogram counts over the whole year,
    sharing bin_edges (metrics.tmax_histogram's fixed bins). The full-year
    distribution is bimodal — a cold winter mode and a warm summer mode — so runs are
    drawn as step outlines rather than filled bars: overlaid opaque bars hide every
    curve but the last one drawn and flatten the two modes into a single blob.
    Density-normalised so runs with different case counts overlay cleanly; ERA5 is
    drawn last, thicker and black, so the truth outline reads on top.
    """
    curves = list(curves)
    if not curves:
        raise ValueError("At least one distribution curve is required")
    bin_edges = np.asarray(bin_edges, dtype=float)
    centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    widths = np.diff(bin_edges)

    figure, axis = plt.subplots(figsize=(FULL_WIDTH_FIGURE_INCHES,
                                         FULL_WIDTH_FIGURE_INCHES / 1.618),
                                layout="constrained")
    palette = plt.get_cmap("tab10")
    model_index = 0
    truth_curve = None
    populated = np.zeros(len(centres), dtype=bool)
    for label, counts in curves:
        counts = np.asarray(counts, dtype=float)
        # Older caches stored one row per season; sum them back to the full year.
        if counts.ndim > 1:
            counts = counts.sum(axis=0)
        populated |= counts > 0
        total = counts.sum()
        density = counts / (total * widths) if total > 0 else counts
        if label == truth_label:
            truth_curve = density
            continue
        colour = palette(model_index % 10)
        axis.bar(centres, density, width=widths, align="center", alpha=0.25,
                 color=colour, zorder=2)
        axis.step(centres, density, where="mid", linewidth=1.7,
                  color=colour, label=label, zorder=3)
        model_index += 1
    if truth_curve is not None:
        axis.bar(centres, truth_curve, width=widths, align="center", facecolor="none",
                 edgecolor="black", linewidth=0.8, alpha=0.55, zorder=4)
        axis.step(centres, truth_curve, where="mid", linewidth=2.6, color="black",
                  label=truth_label, zorder=5)
    axis.set_xlabel("Daily Tmax (K)")
    axis.set_ylabel("Density")
    axis.set_title(wrap_title(title), pad=10)
    axis.grid(alpha=0.25)
    # The bins deliberately overshoot the data on both sides so nothing is discarded;
    # show only the range something actually lands in, plus a bin of margin.
    occupied = np.where(populated)[0]
    first, last = (occupied[0], occupied[-1]) if occupied.size else (0, len(centres) - 1)
    axis.set_xlim(bin_edges[max(first - 1, 0)], bin_edges[min(last + 2, len(bin_edges) - 1)])
    # Legend below the axes and inside the canvas: a side legend would be trimmed off a
    # figure authored at its final width, and constrained layout sizes the axes around it.
    figure.legend(*axis.get_legend_handles_labels(), loc="outside lower center",
                  ncol=2, frameon=False)
    return figure


def build_tmax_scatter_figure(curves, title: str, anomaly: bool = False):
    """Return a small-multiples predicted-vs-actual Tmax scatter, one panel per run.

    curves is an iterable of (label, prediction_values, target_values, category_values),
    category tagging each point by the OBSERVED definition: "heatwave" (>= 3 consecutive
    days, only day 3 onward), "threshold" (exceeds but fails the sieve), or "none". A
    black 1:1 line marks perfect prediction. anomaly=True labels the axes as anomalies
    against the average-year daily Tmax (the values must already be anomalies) and adds
    zero reference lines.
    """
    curves = list(curves)
    if not curves:
        raise ValueError("At least one scatter panel is required")
    category_colour = {"heatwave": "#c0392b", "threshold": "#f1c40f", "none": "#9099a2"}
    category_order = ("none", "threshold", "heatwave")  # grey drawn first, red drawn last
    category_label = {"heatwave": "Heatwave (≥3 days)",
                      "threshold": "Threshold only", "none": "Below threshold"}

    # Two columns, not three: at the text width a third column leaves each panel too
    # narrow for its own axis labels.
    columns = min(2, len(curves))
    rows = int(np.ceil(len(curves) / columns))
    figure, axes = plt.subplots(rows, columns, squeeze=False, sharex=True, sharey=True,
                                figsize=(FULL_WIDTH_FIGURE_INCHES,
                                         FULL_WIDTH_FIGURE_INCHES * rows / columns))
    all_values = np.concatenate([np.concatenate([np.asarray(p, dtype=float), np.asarray(t, dtype=float)])
                                 for _, p, t, _ in curves])
    low, high = float(np.nanmin(all_values)), float(np.nanmax(all_values))
    pad = 0.03 * (high - low)
    low, high = low - pad, high + pad

    for index, (label, prediction, target, category) in enumerate(curves):
        axis = axes[index // columns][index % columns]
        prediction = np.asarray(prediction, dtype=float)
        target = np.asarray(target, dtype=float)
        category = np.asarray(category)
        axis.plot([low, high], [low, high], color="black", linewidth=1.0, linestyle="--", zorder=1)
        for key in category_order:
            selected = category == key
            if not selected.any():
                continue
            axis.scatter(target[selected], prediction[selected], s=9, alpha=0.55,
                        color=category_colour[key], label=category_label[key], edgecolors="none",
                        zorder=2 if key == "none" else (3 if key == "threshold" else 4))
        axis.set_xlim(low, high)
        axis.set_ylim(low, high)
        axis.set_aspect("equal", adjustable="box")
        if anomaly:
            axis.axhline(0.0, color="black", linewidth=0.7, alpha=0.4, zorder=1)
            axis.axvline(0.0, color="black", linewidth=0.7, alpha=0.4, zorder=1)
        axis.set_title(label)
        axis.grid(True, alpha=0.25)
    for index in range(len(curves), rows * columns):
        axes[index // columns][index % columns].axis("off")
    figure.supxlabel("ERA5 Tmax anomaly (K)" if anomaly else "ERA5 daily Tmax (K)")
    figure.supylabel(f"{MODEL_LABEL} Tmax anomaly (K)" if anomaly
                     else f"{MODEL_LABEL} daily Tmax (K)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    spare = rows * columns - len(curves)
    if spare:
        axes[-1][-1].legend(handles, labels, loc="center", frameon=False)
    else:
        figure.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    figure.suptitle(wrap_title(title))
    return figure


def build_tmax_error_by_observed_figure(curves, threshold, title: str, num_bins: int = 20):
    """Return a model-error-versus-observed-Tmax figure without I/O.

    curves is an iterable of (label, target_values, prediction_values) drawn from the
    cached scatter sample, in absolute K. Observed Tmax is binned on the x axis and
    each curve is the mean signed error (prediction - target) in that bin, so a bias
    that only appears in the hot tail stays visible where a single pooled bias number
    would average it away. threshold marks the mean p90 — it must be on the same
    absolute scale as the samples, which is why the scatter is no longer stored as an
    anomaly.
    """
    curves = list(curves)
    if not curves:
        raise ValueError("At least one curve is required")
    all_targets = np.concatenate([np.asarray(target, dtype=float) for _, target, _ in curves])
    finite_targets = all_targets[np.isfinite(all_targets)]
    bin_edges = np.linspace(finite_targets.min(), finite_targets.max(), num_bins + 1)
    centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Taller than the other single-axis plots: the curves span a wide error range and a
    # 4.3 in canvas compresses them into an unreadable band.
    figure, axis = plt.subplots(figsize=(FULL_WIDTH_FIGURE_INCHES, 6.20))
    palette = plt.get_cmap("tab10")
    for index, (label, target_values, prediction_values) in enumerate(curves):
        target_values = np.asarray(target_values, dtype=float)
        prediction_values = np.asarray(prediction_values, dtype=float)
        finite = np.isfinite(target_values) & np.isfinite(prediction_values)
        error = prediction_values[finite] - target_values[finite]
        bin_index = np.digitize(target_values[finite], bin_edges[1:-1])
        mean_error = np.full(num_bins, np.nan)
        for bin_number in range(num_bins):
            in_bin = bin_index == bin_number
            # A handful of samples in a bin gives a meaningless mean; leave it blank.
            if in_bin.sum() >= 5:
                mean_error[bin_number] = error[in_bin].mean()
        axis.plot(centres, mean_error, marker="o", markersize=4, linewidth=1.65,
                  color=palette(index % 10), label=label)
    axis.axhline(0, color="black", linewidth=0.9, linestyle="--", zorder=1)
    if threshold is not None and np.isfinite(threshold):
        axis.axvline(threshold, color="0.35", linewidth=1.3, linestyle=":", zorder=1,
                     label=f"Mean p90 = {threshold:.1f} K")
    axis.set_xlabel("Observed daily Tmax (K)")
    axis.set_ylabel(f"{MODEL_LABEL} − ERA5 (K)")
    axis.set_title(wrap_title(title), pad=10)
    axis.grid(alpha=0.25)
    figure.legend(*axis.get_legend_handles_labels(), loc="outside lower center",
                  ncol=2, frameon=False)
    return figure


# ── Heatwave calendar ───────────────────────────────────────────────────────

def build_calendar_skill_figure(panels, title: str, num_members: int, region=None,
                                region_name: str = "Denmark", majority: float = 0.5, extra_cells=()):
    """Return one calendar per run without I/O: one cell per calendar day, rows are
    day-of-week and columns are week-of-year. One panel per row, stacked.

    panels is an iterable of (label, dates, observed_share, observed_heatwave,
    model_certainty) over the calendar region. Colour is how many of the num_members
    members put most of the region above p90 that day — plain exceedance, no spell.
    Observed heatwave days are outlined in black on top of that.

    Two arguments are accepted and not read. ``title`` is ignored because the suptitle
    slot carries the region instead, and ``observed_share`` is ignored because the dot
    that used to mark every observed hot day is no longer drawn.

    Dates are carried per panel because runs do not all cover the same days: the
    continuous exp1 store scores one more day than the fixed-lead runs. Days a run
    never forecast are grey, not white — white is a real zero.
    """
    panels = list(panels)
    if not panels:
        raise ValueError("At least one calendar panel is required")
    year = int(pd.DatetimeIndex(panels[0][1])[0].year)
    year_start = pd.Timestamp(f"{year}-01-01")
    full_year = pd.date_range(year_start, f"{year}-12-31", freq="D")
    week = (full_year.dayofyear - 1 + year_start.weekday()) // 7
    weekday = full_year.weekday
    month_starts = pd.date_range(year_start, f"{year}-12-31", freq="MS")
    # Real day boundaries: a line at the edge of the week holding the 1st is up to six
    # days early, which drew the 29-31 March event to the right of the April line.
    month_x = [(month_start.dayofyear - 1 + year_start.weekday()) // 7 + month_start.weekday() / 7
               for month_start in month_starts]
    # One colour block per member count. The value is an integer, and a continuous ramp
    # ticked at member positions asks the reader to interpolate it back to one. Days the
    # run never forecast come in as -1 and take the under colour; the cells of the first
    # and last week that fall outside the year stay masked and are not drawn at all.
    member_colours = plt.get_cmap("OrRd")(np.linspace(0.0, 1.0, num_members + 1))
    member_colormap = ListedColormap(member_colours)
    member_colormap.set_under("0.85")
    member_colormap.set_bad("white")
    member_norm = BoundaryNorm(np.arange(num_members + 2) - 0.5, num_members + 1)

    # One panel per row: the canvas is the text width and a year of days has no width
    # to give a second column.
    columns = 1
    rows = len(panels)
    figure_height = 1.9 * rows + 0.7
    figure, axes = plt.subplots(rows, columns, squeeze=False,
                                figsize=(FULL_WIDTH_FIGURE_INCHES, figure_height))
    flat_axes = axes.reshape(-1)
    mesh = None
    for axis, (label, dates, observed_share, observed_heatwave, model_certainty) in zip(flat_axes, panels):
        dates = pd.DatetimeIndex(dates)
        members = pd.Series(np.asarray(model_certainty, dtype=float) * num_members,
                            index=dates).reindex(full_year).values
        # Comparisons against NaN are False, so a day the run never covered is not a
        # heatwave and stays out of both the outlines and the counts.
        heatwave_day = pd.Series(np.asarray(observed_heatwave, dtype=float),
                                 index=dates).reindex(full_year).values > majority
        grid = np.full((7, int(week.max()) + 1), np.nan)
        grid[weekday, week] = np.where(np.isnan(members), -1.0, members)
        mesh = axis.pcolormesh(np.arange(grid.shape[1] + 1), np.arange(8),
                               np.ma.masked_invalid(grid), cmap=member_colormap, norm=member_norm,
                               edgecolors="white", linewidth=0.6)
        for day in np.where(heatwave_day)[0]:
            axis.add_patch(plt.Rectangle((week[day], weekday[day]), 1, 1, fill=False,
                                         edgecolor="black", linewidth=1.8, zorder=4))
        # A month boundary inside a week is a step, not a vertical cut: the days above it
        # in that column belong to the old month and the ones below to the new.
        for month_start in month_starts[1:]:
            column = (month_start.dayofyear - 1 + year_start.weekday()) // 7
            row = month_start.weekday()
            axis.plot([column, column], [row, 7], color="0.45", linewidth=1.1, zorder=5)
            axis.plot([column, column + 1], [row, row], color="0.45", linewidth=1.1, zorder=5)
            axis.plot([column + 1, column + 1], [0, row], color="0.45", linewidth=1.1, zorder=5)
        axis.set_yticks(np.arange(7) + 0.5)
        axis.set_yticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
        axis.invert_yaxis()
        axis.set_xticks(month_x)
        axis.set_xticklabels([d.strftime("%b") for d in month_starts])
        axis.set_title(label, pad=8, loc="left")
    for axis in flat_axes[len(panels):]:
        axis.axis("off")

    # No figure title, the caption carries it; the region goes in the suptitle slot so
    # constrained layout reserves room for it instead of it landing on the first panel.
    if region is not None:
        min_lat, max_lat, min_lon, max_lon = region
        # "Bornholm" is hard-coded: extra_cells is only ever the one Danish cell today.
        area = f"{region_name} + Bornholm" if extra_cells else region_name
        figure.suptitle(f"{area} ({min_lat:g}–{max_lat:g}°N, {min_lon:g}–{max_lon:g}°E)")
    colorbar = figure.colorbar(mesh, ax=flat_axes.tolist(), fraction=0.02,
                               ticks=np.arange(num_members + 1))
    colorbar.set_label("Members above p90")
    # "outside" placements are the ones constrained layout makes room for.
    figure.legend(handles=[Patch(facecolor="none", edgecolor="black", linewidth=1.8,
                                 label="Observed heatwave")],
                  loc="outside lower center", frameon=False)
    return figure


# ── Significance: every pair against every other ────────────────────────────

def build_pairwise_significance_figure(labels, difference, p_adjusted, alpha, metric_label: str,
                                       unit: str, title: str):
    """Return an every-model-against-every-model significance matrix without I/O.

    difference[row, column] is row's score minus column's, and p_adjusted[row, column]
    is the corrected p-value for that pair. Lower is better for every score here, so a
    negative difference means the row model wins: blue cells are row-better, red are
    row-worse, and the colour is drained on pairs that do not clear alpha, so only
    differences that survive correction read as strong. Each cell is annotated with
    the difference and its p-value.
    """
    labels = list(labels)
    difference = np.asarray(difference, dtype=float)
    p_adjusted = np.asarray(p_adjusted, dtype=float)
    count = len(labels)

    # Drain the colour of anything that fails the corrected threshold, so significance
    # is visible as saturation rather than having to be read off the annotation.
    limit = np.nanmax(np.abs(difference)) if np.isfinite(difference).any() else 1.0
    shading = np.where(np.isfinite(p_adjusted) & (p_adjusted <= alpha), difference, np.nan)

    figure, axis = plt.subplots(figsize=(FULL_WIDTH_FIGURE_INCHES,
                                         max(2.8, (1.35 * count + 2.6) * FULL_WIDTH_FIGURE_INCHES
                                             / (1.7 * count + 4.0))))
    axis.set_facecolor("white")
    mesh = axis.pcolormesh(np.arange(count + 1), np.arange(count + 1),
                           np.ma.masked_invalid(shading), cmap="RdBu_r",
                           vmin=-limit, vmax=limit, edgecolors="0.6", linewidth=0.8)
    for row in range(count):
        for column in range(count):
            if row == column:
                axis.add_patch(plt.Rectangle((column, row), 1, 1, facecolor="0.92",
                                             edgecolor="0.6", linewidth=0.8))
                continue
            value, probability = difference[row, column], p_adjusted[row, column]
            if not np.isfinite(value):
                continue
            significant = np.isfinite(probability) and probability <= alpha
            text = f"{value:+.3f}\np={probability:.3g}" if np.isfinite(probability) else f"{value:+.3f}"
            axis.text(column + 0.5, row + 0.5, text, ha="center", va="center",
                      fontsize=10, color="black",
                      fontweight="bold" if significant else "normal")
    axis.set_xticks(np.arange(count) + 0.5)
    axis.set_yticks(np.arange(count) + 0.5)
    axis.set_xticklabels(labels, rotation=30, ha="right")
    axis.set_yticklabels(labels)
    axis.invert_yaxis()
    axis.set_xlabel("compared against")
    axis.set_ylabel("row model")
    figure.suptitle(wrap_title(f"{title}\n{metric_label}: row minus column ({unit}); "
                               f"blue favours the row, bold clears p ≤ {alpha:g}"))
    figure.colorbar(mesh, ax=axis, fraction=0.03, pad=0.02).set_label(
        f"{metric_label} difference ({unit})")
    return figure
