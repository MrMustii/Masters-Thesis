import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..")))

from modified_Gencast.helper_fun import save_to_s3
from modified_Gencast.significance import (
    bootstrap_difference_interval, bootstrap_classification_interval, equivalence_test,
    estimate_block_length, stationary_bootstrap_indices)

CURVES = [
    ("rmse_ts_temp", "2m temperature RMSE (K)", "RMSE vs lead"),
    ("bias_ts_temp", "2m temperature bias (K)", "Bias vs lead"),
    ("crps_growth", "CRPS (K)", "CRPS vs lead"),
    ("twcrps_growth", "twCRPS (K)", "Tail-weighted CRPS vs lead"),
    ("spread_skill_ratio", "Spread / skill", "Spread-skill ratio vs lead"),
]

BASELINE_LABELS = {"climatology", "persistence"}
DECISIVE_BASELINE = {"exp1": "climatology", "exp2": "climatology",
                     "exp3": "climatology", "exp4": "climatology"}
RANKING_ATTRS = ["tmax_twcrps", "tmax_twcrps_day1", "twcrps_mean", "crps_mean"]
DIFFERENCE_SERIES = [
    ("tmax_twcrps_series", "Tmax-twCRPS"), ("tmax_twcrps_day1_series", "Tmax-twCRPS(d1)"),
    ("spatial_vs_series", "spatial VS"), ("twcrps_series", "field twCRPS"), ("crps_series", "CRPS"),
    ("twcrps_day1_series", "field twCRPS(d1)"), ("crps_day1_series", "CRPS(d1)"),
    ("variogram_series", "variogram"), ("variogram_day1_series", "variogram(d1)"),
    ("dtr_crps_series", "DTR-CRPS"), ("dtr_crps_day1_series", "DTR-CRPS(d1)"),
    ("asymmetry_crps_series", "ASY-CRPS"), ("asymmetry_crps_day1_series", "ASY-CRPS(d1)"),
    ("peak_time_ccrps_series", "PTE-cCRPS"),
]
SELECTION_METRICS_BY_EXPERIMENT = {
    "exp2": {"Tmax-twCRPS": "tmax_twcrps", "spatial VS": "spatial_vs"},
    "exp1": {"Tmax-twCRPS(d1)": "tmax_twcrps_day1"},
    "exp3": {"Tmax-twCRPS(d1)": "tmax_twcrps_day1"},
    "exp4": {"Tmax-twCRPS(d1)": "tmax_twcrps_day1"},
}
PRIMARY_SERIES = {
    "tmax_twcrps": "tmax_twcrps_series", "tmax_twcrps_day1": "tmax_twcrps_day1_series",
    "twcrps_mean": "twcrps_series", "crps_mean": "crps_series",
    "variogram_score": "variogram_series", "dtr_crps": "dtr_crps_series",
}

COMPARISONS = {
    "region":          {"experiment": "exp1", "decisive_baseline": "climatology",
                        "ranking_attrs": ["tmax_twcrps_day1"],
                        "selection_metrics": {"Tmax-twCRPS(d1)": "tmax_twcrps_day1"}},
    "var_margin":      {"experiment": "exp2", "decisive_baseline": "climatology",
                        "ranking_attrs": ["tmax_twcrps"],
                        "selection_metrics": {"Tmax-twCRPS": "tmax_twcrps", "spatial VS": "spatial_vs"}},
    "ou_rho":          {"experiment": "exp3", "decisive_baseline": "climatology",
                        "ranking_attrs": ["variogram_score_day1"],
                        "selection_metrics": {}},
    "lead_time":       {"experiment": "exp3", "decisive_baseline": "climatology",
                        "ranking_attrs": ["twcrps_mean", "crps_mean"],
                        "selection_metrics": {}},
    "day_select":      {"experiment": "exp4", "decisive_baseline": "climatology",
                        "ranking_attrs": ["tmax_twcrps", "tmax_twcrps_day1"],
                        "selection_metrics": {}},
    "deagg":           {"experiment": "exp4", "decisive_baseline": "climatology",
                        "ranking_attrs": ["dtr_crps", "tmax_twcrps"],
                        "selection_metrics": {}},
    "region_transfer": {"experiment": "exp4", "decisive_baseline": "climatology",  # RQ3, no code yet
                        "ranking_attrs": ["dtr_crps", "tmax_twcrps"],
                        "selection_metrics": {}},
}


METRIC_LABELS = {
    "crps_mean": ("Field CRPS (K)", "general"),
    "spatial_vs": ("Spatial VS", "general"),
    "tmax_rmse": ("Tmax RMSE (K)", "general"),
    "ssr_mean": ("Spread-skill ratio", "general"),
    "tmax_twcrps": ("Tmax-twCRPS (K)", "heatwave"),
    "tmax_twcrps_day1": ("Tmax-twCRPS d1 (K)", "heatwave"),
    "twcrps_mean": ("Field twCRPS (K)", "heatwave"),
    "hw_brier": ("Brier score", "heatwave"),
    "hw_brier_skill_score": ("Brier skill", "heatwave"),
    "hw_auc": ("Detection AUC", "heatwave"),
    "hw_f1": ("Detection F1", "heatwave"),
    "hw_precision": ("Detection precision", "heatwave"),
    "hw_recall": ("Detection recall", "heatwave"),
    "event_detection_rate": ("Event detection rate", "heatwave"),
    "tail_ratio": ("Tail ratio", "heatwave"),
    "tmax_qbias_q99": ("Q99 Tmax bias (K)", "heatwave"),
    "variogram_score": ("Temporal VS", "reconstruction"),
    "variogram_score_day1": ("Temporal VS (d1)", "reconstruction"),
    "dtr_crps": ("DTR-CRPS (K)", "reconstruction"),
    "peak_time_ccrps": ("PTE-cCRPS", "reconstruction"),
}
HIGHER_IS_BETTER = {"hw_f1", "hw_precision", "hw_recall", "hw_auc", "hw_brier_skill_score", "ssr_mean", "event_detection_rate"}
SERIES_FOR_ATTR = {
    "crps_mean": "crps_series", "twcrps_mean": "twcrps_series",
    "tmax_twcrps": "tmax_twcrps_series", "tmax_twcrps_day1": "tmax_twcrps_day1_series",
    "spatial_vs": "spatial_vs_series", "variogram_score": "variogram_series",
    "variogram_score_day1": "variogram_day1_series",
    "dtr_crps": "dtr_crps_series", "peak_time_ccrps": "peak_time_ccrps_series",
}
BAR_ATTRS = {
    "general": ["crps_mean", "spatial_vs", "tmax_rmse"],
    "heatwave": ["tmax_twcrps", "tmax_twcrps_day1", "twcrps_mean", "hw_brier", "hw_auc", "hw_f1", "event_detection_rate"],
    "reconstruction": ["variogram_score", "variogram_score_day1", "dtr_crps", "peak_time_ccrps"],
}


def short_label(label):
    """Strip exp2 run-name boilerplate so legends and tables stay readable."""
    return (label.replace("experiment_2_", "").replace("experiment_1", "exp1")
            .replace("_fixed_stage2", "").replace("_stage2", "").replace("experiment_", "exp"))


def bootstrap_mean_ci(series, resamples, seed=42):
    """Block-bootstrap 95% CI of a per-init series' mean, for the bar whiskers; NaN if too short."""
    values = np.asarray(series.values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 3:
        return float("nan"), float("nan")
    block_length = estimate_block_length(values)
    rng = np.random.default_rng(seed)
    means = np.array([values[stationary_bootstrap_indices(len(values), block_length, rng)].mean()
                      for _ in range(resamples)])
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def daily_series(metrics_ds, name, reduce):
    """Collapse a per-init series to one value per calendar day (scores average, counts sum) on "forecast"."""
    series = metrics_ds[name]
    dim_name = series.dims[0]
    dates = pd.DatetimeIndex(series[dim_name].values).floor("D").values
    series = series.assign_coords({dim_name: dates})
    grouped = series.groupby(dim_name)
    reduced = grouped.sum() if reduce == "sum" else grouped.mean()
    return reduced.rename({dim_name: "forecast"}) if dim_name != "forecast" else reduced


def pairwise_significance_matrix(labels, run_lookup, series_name, out_dir, run_id, resamples, no_s3):
    """Who-beats-whom matrix on series_name (lower=better): +/- row better/worse, ~ n.s., · diagonal."""
    present = [label for label in labels if series_name in run_lookup[label]]
    matrix = pd.DataFrame("", index=present, columns=present)
    for row_index, row_label in enumerate(present):
        matrix.loc[row_label, row_label] = "·"
        for col_label in present[row_index + 1:]:
            row_daily, col_daily = xr.align(
                daily_series(run_lookup[row_label], series_name, "mean"),
                daily_series(run_lookup[col_label], series_name, "mean"), join="inner")
            if row_daily.sizes["forecast"] < 10:
                matrix.loc[row_label, col_label] = matrix.loc[col_label, row_label] = "n/a"
                continue
            result = bootstrap_difference_interval(row_daily, col_daily, num_resamples=resamples)
            if result["significant"]:
                row_better = result["point_estimate"] < 0
                matrix.loc[row_label, col_label] = "+" if row_better else "-"
                matrix.loc[col_label, row_label] = "-" if row_better else "+"
            else:
                matrix.loc[row_label, col_label] = matrix.loc[col_label, row_label] = "~"

    csv_path = os.path.join(out_dir, f"significance_matrix_{series_name}.csv")
    matrix.to_csv(csv_path)
    print(f"  wrote {csv_path}")
    print(f"\nPAIRWISE MATRIX on {series_name}  (row vs col: + row better, − row worse, ~ n.s.):")
    print(matrix.to_string())
    cell_color = {"+": "#8fce8f", "-": "#e08f8f", "~": "0.85", "n/a": "0.7", "·": "0.95"}
    short_names = [short_label(label) for label in present]
    figure, axis = plt.subplots(figsize=(2 + 1.1 * len(present), 1.2 + 0.5 * len(present)))
    axis.axis("off")
    table = axis.table(cellText=matrix.values, rowLabels=short_names, colLabels=short_names,
                       cellColours=[[cell_color.get(value, "white") for value in row] for row in matrix.values],
                       loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.4)
    axis.set_title(f"Pairwise significance on {series_name}\n(green: row better, red: row worse, grey: n.s.)",
                   fontsize=10)
    figure.tight_layout()
    png_path = os.path.join(out_dir, f"significance_matrix_{series_name}.png")
    figure.savefig(png_path, dpi=140, bbox_inches="tight")
    plt.close(figure)
    print(f"  wrote {png_path}")
    if not no_s3:
        save_to_s3(csv_path, run_id)
        save_to_s3(png_path, run_id)


def main():
    parser = argparse.ArgumentParser(description="Rank runs and test the gaps for significance.")
    parser.add_argument("--runs", nargs="+", required=True,
                        help="metrics .nc paths with labels: /path/metrics.nc=label "
                             "(baselines must be labelled climatology/persistence)")
    parser.add_argument("--experiment", type=str, default=None, choices=["exp1", "exp2", "exp3", "exp4"],
                        help="Legacy selector (one rule per experiment). Prefer --comparison.")
    parser.add_argument("--comparison", type=str, default=None, choices=list(COMPARISONS),
                        help="The specific thesis comparison; sets the primary ranking metric, "
                             "selection metrics, and decisive baseline. Overrides --experiment.")
    parser.add_argument("--reference", type=str, default=None,
                        help="Run label every other run is difference-tested against. Default: the "
                             "experiment's decisive baseline if present (exp3/exp4), else the best model run.")
    parser.add_argument("--out", type=str, required=True, help="Output directory for the comparison figures/tables.")
    parser.add_argument("--run-id", type=str, default="comparison", help="S3 run_id folder for the uploaded tables.")
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--no-s3", action="store_true")
    args = parser.parse_args()
    if args.comparison:
        comparison = COMPARISONS[args.comparison]
        args.experiment = comparison["experiment"]
        ranking_attrs = comparison["ranking_attrs"]
        selection_metrics = comparison["selection_metrics"]
        decisive_baseline = comparison["decisive_baseline"]
    elif args.experiment:
        ranking_attrs = RANKING_ATTRS
        selection_metrics = SELECTION_METRICS_BY_EXPERIMENT[args.experiment]
        decisive_baseline = DECISIVE_BASELINE[args.experiment]
    else:
        parser.error("pass --comparison (preferred) or --experiment")

    runs = []
    for spec in args.runs:
        path, _, label = spec.partition("=")
        runs.append((label or os.path.splitext(os.path.basename(path))[0], xr.open_dataset(path)))
    run_lookup = dict(runs)
    model_labels = [label for label, _ in runs if label not in BASELINE_LABELS]
    os.makedirs(args.out, exist_ok=True)

    labels_order = model_labels + [label for label, _ in runs if label in BASELINE_LABELS]
    shorts = {label: short_label(label) for label in labels_order}
    color_cycle = plt.get_cmap("tab10")
    baseline_colors = {"climatology": "#7f7f7f", "persistence": "#8c564b"}
    run_colors, model_index = {}, 0
    for label in labels_order:
        if label in BASELINE_LABELS:
            run_colors[label] = baseline_colors.get(label, "0.6")
        else:
            run_colors[label] = color_cycle(model_index)
            model_index += 1

    # overlay curves (raw per-run values vs lead)
    for var, ylabel, title in CURVES:
        present = [(label, metrics_ds) for label, metrics_ds in runs if var in metrics_ds]
        if len(present) < 2:
            continue
        figure, axis = plt.subplots(figsize=(8, 4.5))
        for label, metrics_ds in present:
            curve = metrics_ds[var]
            axis.plot(curve.hour.values, curve.values, marker="o", markersize=3,
                      color=run_colors[label], label=shorts[label])
        axis.set_xlabel("Forecast lead (h)")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(True, alpha=0.3)
        axis.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=8)
        out_path = os.path.join(args.out, f"compare_{var}.png")
        figure.savefig(out_path, dpi=140, bbox_inches="tight")
        plt.close(figure)
        print(f"  wrote {out_path}")

    # rankings, reported SEPARATELY (never merged): decisive, then general (CRPS), then heatwave (Tmax-twCRPS)
    def shared_attr(candidates):
        return next((attr for attr in candidates
                     if attr and all(attr in run_lookup[label].attrs for label in model_labels)), None)

    decisive_attr = shared_attr(ranking_attrs)
    general_attr = shared_attr(["crps_mean"])
    heatwave_attr = shared_attr(["tmax_twcrps", "tmax_twcrps_day1"])
    ranking_plan = ([("DECISIVE", decisive_attr)] if decisive_attr not in (None, general_attr, heatwave_attr) else [])
    ranking_plan += [("GENERAL", general_attr), ("HEATWAVE", heatwave_attr)]
    ranking_winners = {}
    for kind, attr in ranking_plan:
        if attr is None:
            continue
        ordered = sorted(model_labels, key=lambda label: float(run_lookup[label].attrs[attr]))
        ranking_winners[kind] = ordered[0]
        print(f"\n{kind} ranking on {attr} (lower = better):")
        for rank, label in enumerate(ordered, start=1):
            print(f"  {rank}. {shorts[label]}: {float(run_lookup[label].attrs[attr]):.4f}"
                  + ("  <- winner" if rank == 1 else ""))
    if "GENERAL" in ranking_winners and "HEATWAVE" in ranking_winners \
            and ranking_winners["GENERAL"] != ranking_winners["HEATWAVE"]:
        print(f"\nNOTE: general winner ({shorts[ranking_winners['GENERAL']]}) != heatwave winner "
              f"({shorts[ranking_winners['HEATWAVE']]}) — report both; the split is the finding.")

    best_label = (ranking_winners.get("DECISIVE") or ranking_winners.get("HEATWAVE")
                  or ranking_winners.get("GENERAL") or model_labels[0])
    reference_label = args.reference or (decisive_baseline
                       if args.experiment in ("exp3", "exp4") and decisive_baseline in run_lookup else best_label)
    assert reference_label in run_lookup, f"--reference {reference_label} not among {list(run_lookup)}"
    reference_ds = run_lookup[reference_label]
    print(f"\nReference for significance tests: {shorts[reference_label]}")

    # headline bars, one file per skill group (bar per run + baselines, bootstrap CI whisker where a series exists)
    bars_paths = []
    for group, group_title in (("general", "General skill"), ("heatwave", "Heatwave skill"),
                               ("reconstruction", "Reconstruction skill")):
        bar_attrs = [attr for attr in BAR_ATTRS[group]
                     if any(attr in run_lookup[label].attrs for label in labels_order)]
        if not bar_attrs:
            continue
        figure, axes = plt.subplots(1, len(bar_attrs), squeeze=False, figsize=(3.1 * len(bar_attrs), 3.6))
        for column_index, attr in enumerate(bar_attrs):
            axis = axes[0][column_index]
            positions, heights, colors, low_error, high_error, tick_labels = [], [], [], [], [], []
            for position, label in enumerate(labels_order):
                if attr not in run_lookup[label].attrs:
                    continue
                value = float(run_lookup[label].attrs[attr])
                positions.append(position)
                heights.append(value)
                colors.append(run_colors[label])
                tick_labels.append(shorts[label])
                series_name = SERIES_FOR_ATTR.get(attr)
                if series_name and series_name in run_lookup[label]:
                    low, high = bootstrap_mean_ci(daily_series(run_lookup[label], series_name, "mean"), args.resamples)
                    low_error.append(value - low if np.isfinite(low) else 0.0)
                    high_error.append(high - value if np.isfinite(high) else 0.0)
                else:
                    low_error.append(0.0)
                    high_error.append(0.0)
            axis.bar(positions, heights, color=colors, capsize=3,
                     yerr=[low_error, high_error] if any(low_error) or any(high_error) else None,
                     error_kw={"elinewidth": 1})
            axis.set_xticks(positions)
            axis.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=7)
            axis.set_title(METRIC_LABELS[attr][0], fontsize=9)
            axis.grid(True, alpha=0.3, axis="y")
        figure.suptitle(f"{group_title} — {args.comparison or args.experiment}", fontsize=12)
        figure.tight_layout()
        bars_path = os.path.join(args.out, f"compare_bars_{group}.png")
        figure.savefig(bars_path, dpi=140, bbox_inches="tight")
        plt.close(figure)
        bars_paths.append(bars_path)
        print(f"  wrote {bars_path}")

    # reliability overlay (probabilistic heatwave detection)
    reliability_runs = [label for label in model_labels if "reliability_forecast" in run_lookup[label]]
    if reliability_runs:
        figure, axis = plt.subplots(figsize=(6.5, 6))
        axis.plot([0, 1], [0, 1], color="black", linewidth=0.8, linestyle="--", label="Perfect")
        for label in reliability_runs:
            forecast_probability = run_lookup[label]["reliability_forecast"].values
            observed = run_lookup[label]["reliability_observed"].values
            populated = np.isfinite(observed)
            axis.plot(forecast_probability[populated], observed[populated], marker="o", markersize=4,
                      color=run_colors[label], label=shorts[label])
        axis.set_xlabel("Forecast probability (member fraction > τ)")
        axis.set_ylabel("Observed exceedance frequency")
        axis.set_title("Reliability — probabilistic heatwave detection", fontsize=11)
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.grid(True, alpha=0.3)
        axis.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=8)
        out_path = os.path.join(args.out, "compare_reliability.png")
        figure.savefig(out_path, dpi=140, bbox_inches="tight")
        plt.close(figure)
        print(f"  wrote {out_path}")

    # curated summary table: labelled rows, runs as columns, best model per row starred, skill % vs both baselines
    present_attrs = [attr for attr in METRIC_LABELS if any(attr in run_lookup[label].attrs for label in labels_order)]
    row_labels = [METRIC_LABELS[attr][0] for attr in present_attrs]
    table = pd.DataFrame(index=row_labels, columns=[shorts[label] for label in labels_order], dtype=object)
    for attr, row_label in zip(present_attrs, row_labels):
        values = {label: (float(run_lookup[label].attrs[attr]) if attr in run_lookup[label].attrs else np.nan)
                  for label in labels_order}
        finite_models = {label: value for label, value in values.items() if label in model_labels and np.isfinite(value)}
        best = ((max if attr in HIGHER_IS_BETTER else min)(finite_models, key=finite_models.get)
                if finite_models else None)
        for label in labels_order:
            value = values[label]
            table.loc[row_label, shorts[label]] = (
                "" if not np.isfinite(value) else f"{value:.4g}" + (" *" if label == best else ""))
    for baseline in ("climatology", "persistence"):
        if heatwave_attr and baseline in run_lookup and heatwave_attr in run_lookup[baseline].attrs:
            baseline_score = float(run_lookup[baseline].attrs[heatwave_attr])
            row_label = f"Skill vs {baseline} (%)"
            for label in labels_order:
                cell = ""
                if heatwave_attr in run_lookup[label].attrs and baseline_score != 0:
                    cell = f"{100.0 * (baseline_score - float(run_lookup[label].attrs[heatwave_attr])) / baseline_score:.1f}"
                table.loc[row_label, shorts[label]] = cell
    csv_path = os.path.join(args.out, "summary_table.csv")
    table.to_csv(csv_path)
    print(f"  wrote {csv_path}")

    climatology_ds = run_lookup.get("climatology")
    selection_delta = {}
    for metric_name, attr in selection_metrics.items():
        if climatology_ds is not None and attr in climatology_ds.attrs and attr in reference_ds.attrs:
            selection_delta[metric_name] = 0.10 * (float(climatology_ds.attrs[attr])
                                                   - float(reference_ds.attrs[attr]))
        else:
            selection_delta[metric_name] = None
    if climatology_ds is None:
        print("\n⚠ No climatology run supplied → δ undefined; selection verdicts are 'no-δ'. "
              "Score the climatology baseline and re-run to conclude preserved/degraded.")

    significance_rows = []    # plain per-metric table (core columns, no stray NaNs)
    verdict_rows = []         # separate equivalence-verdict table (selection metrics only)
    selection_verdicts = {}   # label -> {metric_name: verdict}
    for label in labels_order:
        if label == reference_label:
            continue
        candidate_ds = run_lookup[label]
        for series_name, metric_name in DIFFERENCE_SERIES:
            if series_name not in candidate_ds or series_name not in reference_ds:
                continue
            candidate_daily, reference_daily = xr.align(
                daily_series(candidate_ds, series_name, "mean"),
                daily_series(reference_ds, series_name, "mean"), join="inner")
            if candidate_daily.sizes["forecast"] < 10:
                continue
            result = bootstrap_difference_interval(candidate_daily, reference_daily,
                                                   num_resamples=args.resamples)
            # "not significant" is inconclusive, never evidence of no difference.
            better = (label if result["significant"] and result["point_estimate"] < 0
                      else reference_label if result["significant"] else "inconclusive")
            significance_rows.append({
                "run": label, "reference": reference_label, "metric": metric_name,
                "delta": result["point_estimate"], "ci_low": result["lower_bound"],
                "ci_high": result["upper_bound"], "significant": result["significant"],
                "better": better, "block_length": result["block_length"],
                "paired_days": int(candidate_daily.sizes["forecast"]),
            })
            # selection metrics also get an equivalence verdict, in a separate table
            if metric_name in selection_metrics:
                margin = selection_delta[metric_name]
                verdict_row = {"run": label, "reference": reference_label, "metric": metric_name,
                               "delta": result["point_estimate"], "ci_low": result["lower_bound"],
                               "ci_high": result["upper_bound"], "delta_margin": margin}
                if margin is None:
                    verdict_row["verdict"] = "no-δ"
                else:
                    equivalence = equivalence_test(result["lower_bound"], result["upper_bound"], margin)
                    verdict_row["verdict"] = equivalence["verdict"]
                    selection_verdicts.setdefault(label, {})[metric_name] = equivalence["verdict"]
                if "twCRPS" in metric_name:  # ACF of the difference series → justifies the block length
                    centered = candidate_daily.values - reference_daily.values
                    centered = centered - centered.mean()
                    variance = float((centered ** 2).sum())
                    verdict_row["acf_lag1_2_3"] = "/".join(
                        f"{float((centered[:-lag] * centered[lag:]).sum() / variance):+.2f}"
                        if variance > 0 else "0" for lag in (1, 2, 3))
                verdict_rows.append(verdict_row)
        count_names = ("hw_tp_series", "hw_fp_series", "hw_fn_series")
        if all(name in candidate_ds and name in reference_ds for name in count_names):
            aligned = xr.align(*[daily_series(candidate_ds, name, "sum") for name in count_names],
                               *[daily_series(reference_ds, name, "sum") for name in count_names], join="inner")
            if aligned[0].sizes["forecast"] >= 10:
                result = bootstrap_classification_interval(*aligned, num_resamples=args.resamples)
                better = (label if result["significant"] and result["point_estimate"] > 0
                          else reference_label if result["significant"] else "inconclusive")
                significance_rows.append({
                    "run": label, "reference": reference_label, "metric": "HW F1",
                    "delta": result["point_estimate"], "ci_low": result["lower_bound"],
                    "ci_high": result["upper_bound"], "significant": result["significant"],
                    "better": better, "block_length": result["block_length"],
                    "paired_days": int(aligned[0].sizes["forecast"]),
                })

    significance_csv_path = os.path.join(args.out, "significance_table.csv")
    pd.DataFrame(significance_rows).to_csv(significance_csv_path, index=False)
    print(f"  wrote {significance_csv_path}")
    verdict_csv_path = os.path.join(args.out, "verdict_table.csv")
    pd.DataFrame(verdict_rows).to_csv(verdict_csv_path, index=False)
    print(f"  wrote {verdict_csv_path}")

    #  Preservation verdict: preserved needs ALL selection metrics preserved. 
    if selection_verdicts:
        print(f"\nPRESERVATION verdict (preserved ⇔ all of [{', '.join(selection_metrics)}] preserved):")
        for label in [candidate for candidate in model_labels if candidate in selection_verdicts]:
            per_metric = selection_verdicts[label]
            verdicts = list(per_metric.values())
            overall = ("preserved" if verdicts and all(verdict == "preserved" for verdict in verdicts)
                       else "degraded" if any(verdict == "degraded" for verdict in verdicts) else "inconclusive")
            print(f"  {shorts[label]:22s} {overall:12s}  "
                  f"({', '.join(f'{name}:{verdict}' for name, verdict in per_metric.items())})")

    # forest plot: delta vs reference +- 95% CI, one panel per metric (green better, red worse, grey n.s.)
    forest_metrics = [name for _, name in DIFFERENCE_SERIES if any(r["metric"] == name for r in significance_rows)]
    if any(r["metric"] == "HW F1" for r in significance_rows):
        forest_metrics.append("HW F1")
    if forest_metrics:
        num_columns = min(3, len(forest_metrics))
        num_rows = int(np.ceil(len(forest_metrics) / num_columns))
        figure, axes = plt.subplots(num_rows, num_columns, squeeze=False,
                                    figsize=(4.6 * num_columns, 1.2 + 1.7 * num_rows))
        for index, metric_name in enumerate(forest_metrics):
            axis = axes[index // num_columns][index % num_columns]
            metric_rows = [r for r in significance_rows if r["metric"] == metric_name]
            y_positions = np.arange(len(metric_rows))
            for y_position, row in zip(y_positions, metric_rows):
                color = ("seagreen" if row["significant"] and row["better"] == row["run"]
                         else "firebrick" if row["significant"] else "0.5")
                axis.errorbar(row["delta"], y_position,
                              xerr=[[row["delta"] - row["ci_low"]], [row["ci_high"] - row["delta"]]],
                              fmt="o", color=color, capsize=3)
            axis.axvline(0, color="black", linewidth=1.0, linestyle="--")
            axis.set_yticks(y_positions)
            axis.set_yticklabels([shorts[row["run"]] for row in metric_rows], fontsize=7)
            axis.set_title(metric_name, fontsize=9)
            axis.grid(True, alpha=0.3, axis="x")
        for index in range(len(forest_metrics), num_rows * num_columns):
            axes[index // num_columns][index % num_columns].axis("off")
        figure.suptitle(f"Δ vs {shorts[reference_label]} (95% CI; green=better, red=worse, grey=n.s.)", fontsize=11)
        figure.tight_layout()
        forest_path = os.path.join(args.out, "significance_forest.png")
        figure.savefig(forest_path, dpi=140, bbox_inches="tight")
        plt.close(figure)
        print(f"  wrote {forest_path}")

    # pairwise who-beats-whom matrix, once per decisive metric every model run carries a series for
    matrix_attrs = [attr for attr in (decisive_attr, "crps_mean", heatwave_attr, "spatial_vs")
                    if attr and SERIES_FOR_ATTR.get(attr)
                    and all(SERIES_FOR_ATTR[attr] in run_lookup[label] for label in model_labels)]
    matrix_attrs = list(dict.fromkeys(matrix_attrs))  # dedupe, keep order
    for attr in matrix_attrs:
        pairwise_significance_matrix(labels_order, run_lookup, SERIES_FOR_ATTR[attr],
                                     args.out, args.run_id, args.resamples, args.no_s3)

    #  Curated summary-table PNG. 
    figure, axis = plt.subplots(figsize=(1.8 + 1.3 * len(table.columns), 0.5 + 0.34 * len(table.index)))
    axis.axis("off")
    rendered = axis.table(cellText=table.values, rowLabels=list(table.index),
                          colLabels=list(table.columns), loc="center", cellLoc="center")
    rendered.auto_set_font_size(False)
    rendered.set_fontsize(8)
    rendered.scale(1, 1.3)
    summary_png = os.path.join(args.out, "summary_table.png")
    figure.savefig(summary_png, dpi=140, bbox_inches="tight")
    plt.close(figure)
    print(f"  wrote {summary_png}")

    if not args.no_s3:
        for uploaded in [csv_path, significance_csv_path, verdict_csv_path, summary_png, *bars_paths]:
            save_to_s3(uploaded, args.run_id)
    print(f"\nDone. Comparison in {args.out}")


if __name__ == "__main__":
    main()


