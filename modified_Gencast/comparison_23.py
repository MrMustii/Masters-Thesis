"""Comparisons 2 and 3 — regional input-variable sets and the boundary-margin ablation.

Comparison numbers run 0 to 7 across five files and do not line up with the experiment
numbers, because one experiment can raise more than one question. These are comparisons 2
and 3, answering the second half of RQ1 from the exp2 and baseline caches.

Reads compact report caches only, apart from the single valid hour the maps need.
Every run here shares one fixed-lead layout, so candidate and reference pair directly
on initialisation keys; no valid-time mapping is needed, unlike comparison 1.

Comparison 2 asks what the smallest input set is that keeps full-variable skill, so it
has two references: the all-variable model and the combined temperature/geopotential
model. Comparison 3 asks whether an 8.5 degree forced boundary helps, against the
otherwise identical no-margin model. The headline horizon is the full shared week;
day 1 is reported alongside it.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from modified_Gencast.comparison_1 import field_at
from modified_Gencast.lumio_buckets import get_loading_region, load
from modified_Gencast.metrics import (
    DENMARK_CALENDAR_EXTRA_CELLS, DENMARK_CALENDAR_REGION, TARGET_VAR, relative_economic_value, to_scoring_grid)
from modified_Gencast.plots import (
    build_calendar_skill_figure,
    build_lead_curve_figure,
    build_map_grid_figure,
    build_paired_forest_figure,
    build_pairwise_significance_figure,
    build_reliability_figure,
    build_stacked_rev_figure,
    build_tmax_distribution_figure,
    build_tmax_error_by_observed_figure,
    build_tmax_scatter_figure,
)
from modified_Gencast.significance import (
    apply_multiplicity_correction,
    bootstrap_paired_difference,
    bootstrap_paired_metric,
    family_block_length,
)


REGION = (45.0, 65.0, -5.0, 25.0)
# run key -> (store directory, cached-tag suffix, reader-facing label)
# Short display labels: they double as CSV row keys and as plot panel/legend text.
RUNS = {
    "retrained_all": ("experiment_2_all_gencast_fixed_stage2", "",
                      "All variables"),
    "surface": ("experiment_2_surface_fixed_stage2", "", "Surface only"),
    "temp1000": ("experiment_2_temp1000_fixed_stage2", "", "t1000"),
    "temp850": ("experiment_2_temp850_fixed_stage2", "", "t850"),
    "z500": ("experiment_2_z500_fixed_stage2", "", "z500"),
    "z700": ("experiment_2_z700_fixed_stage2", "", "z700"),
    "combined": ("experiment_2_combined_fixed_stage2", "", "combined"),
    "margin": ("experiment_2_combined_fixed_stage2", "_margin8.5", "combined + 8.5° forced boundary"),
}
BASELINES = {
    "climatology": "Climatology",
    "persistence": "Persistence",
}
# Both fixed-lead persistence caches share exactly the model case grid. The
# appropriate lag depends on the verification horizon, so load both but use
# exactly one persistence baseline in each reported horizon.
PERSISTENCE_BY_HORIZON = {
    "full": "persistence",
    "day1": "persistence_24h",
}
BASELINE_CACHE_RUNS = ("climatology", "persistence", "persistence_24h")
CLIMATOLOGY_LABEL = BASELINES["climatology"]
# (cache suffix, display label) — every heatwave-conditioned diagnostic is reported
# under both: a single day above threshold, and every day of a run of >= 3
# consecutive days above it.
HEATWAVE_DEFINITIONS = (
    ("threshold", "Threshold (single day)"),
    ("heatwave", "Heatwave (≥3 consecutive days)"),
)
VARIABLE_SETS = ("surface", "temp1000", "temp850", "z500", "z700")
# (question, reference run, candidates, reference label override)
QUESTIONS = (
    ("variables_versus_all", "retrained_all", VARIABLE_SETS + ("combined",), None),
    ("variables_versus_combined", "combined", VARIABLE_SETS + ("retrained_all",), None),
    ("margin_versus_no_margin", "combined", ("margin",), "combined, free boundary"),
)
# (cache variable, lead dimension, display label, unit, bootstrap kind)
# "linear": paired difference of per-case means (bootstrap_paired_difference).
# "rmse": nonlinear sqrt-of-mean-squared-error (bootstrap_paired_metric, GenCast Table E2 style).
# A None lead dimension is already pooled to one value per forecast in the cache (no lead
# axis left to select on), so it has no day-one variant and is reported at "full" only.
PRIMARY_METRICS = (
    ("tmax_crps_case", "lead_day", "Tmax CRPS", "K", "linear"),
    ("tmax_twcrps_case", "lead_day", "Tmax twCRPS", "K", "linear"),
    ("brier_score_heatwave", "lead_day", "Heatwave BS", "", "linear"),
)
SECONDARY_METRICS = (
    ("t2m_crps_case", "hour", "CRPS", "K", "linear"),
    ("t2m_twcrps_case", "hour", "twCRPS", "K", "linear"),
    ("squared_error_case", "hour", "RMSE", "K", "rmse"),
    ("conditional_tmax_squared_error_case_threshold", None, "Cond Tmax RMSE (thresh)", "K", "rmse"),
    ("conditional_tmax_squared_error_case_heatwave", None, "Cond Tmax RMSE (hw)", "K", "rmse"),
)
METRICS = PRIMARY_METRICS + SECONDARY_METRICS
METRIC_TIER = {label: "primary" for _, _, label, _, _ in PRIMARY_METRICS}
METRIC_TIER.update({label: "secondary" for _, _, label, _, _ in SECONDARY_METRICS})
HORIZONS = (("full", "full shared forecast"), ("day1", "day 1"))
# The two scientific comparisons own their own reliability, REV and lead curves.
GROUPS = (
    ("comparison_2", "Input variable sets",
     ("retrained_all", "surface", "temp1000", "temp850", "z500", "z700", "combined")),
    ("comparison_3", "Forced boundary ablation", ("combined", "margin")),
)
# (variable, lead dimension, axis label, figure label)
LEAD_CURVES = (
    ("t2m_crps_case", "hour", "Lead time (hours)", "2 m CRPS (K)"),
    ("tmax_twcrps_case", "lead_day", "Lead day", "Tmax twCRPS (K)"),
)


def case_series(dataset, metric, lead_dimension, horizon):
    """One score per initialisation for the requested horizon, keyed by initialisation.

    lead_dimension=None means the cache already holds one value per forecast (no lead
    axis to select or average); only the "full" horizon applies, and the caller skips
    "day1" for it.
    """
    values = dataset[metric]
    if lead_dimension is None:
        collapsed = values
    else:
        if horizon == "day1":
            if lead_dimension == "hour":
                # Day one is every lead inside the first 24 h: hours 12 and 24 of a 12 UTC start.
                values = values.isel(hour=values.hour.values < values.hour.values.min() + 24)
            else:
                values = values.sel(lead_day=[int(values.lead_day.values.min())])
        collapsed = values.mean(lead_dimension)
    return pd.Series(np.asarray(collapsed.values, dtype=float),
                     index=pd.DatetimeIndex(collapsed.forecast.values))


def event_window(dataset, horizon):
    """Event primitives restricted to the requested horizon."""
    if horizon == "day1":
        return dataset.sel(lead_day=[int(dataset.lead_day.values.min())])
    return dataset


def spread_and_skill(dataset, horizon):
    """Ensemble spread, RMSE and their ratio; RMSE is sqrt of the mean squared error."""
    if "ensemble_variance_case" not in dataset:
        return float("nan"), float("nan"), float("nan")
    selected = dataset
    if horizon == "day1":
        hours = dataset.hour.values
        selected = dataset.isel(hour=hours < hours.min() + 24)
    rmse = np.sqrt(float(selected["squared_error_case"].mean()))
    members = int(dataset.probability_bin.size) - 1
    if members < 2:
        return float("nan"), rmse, float("nan")
    spread = np.sqrt(float(selected["ensemble_variance_case"].mean()) * (members + 1) / (members - 1))
    return spread, rmse, spread / rmse


def main():
    """Write each comparison's CSVs and figures into its own directory.

    diagnostics.csv carries one row per run, heatwave definition and horizon: Brier and its
    skill score, event rate, AUC, F1, precision, recall, spread, RMSE and their ratio.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/scratch/project_465002687/verified_metrics"))
    parser.add_argument("--scratch", type=Path, default=Path("/scratch/project_465002687"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--map-date", type=str, default="2021-06-20")
    # Redraw the figures from the caches without repeating the bootstrap, for when
    # only a label or a layout changed. The significance CSVs are left untouched.
    parser.add_argument("--figures-only", action="store_true")
    args = parser.parse_args()
    output_directory = args.out or Path.cwd() / "comparison_23"
    output_directory.mkdir(parents=True, exist_ok=True)

    datasets = {}
    for run in RUNS:
        path = args.root / "reports" / "exp2" / run / "metric_cases.nc"
        if path.is_file():
            datasets[run] = xr.open_dataset(path)
    for run in BASELINE_CACHE_RUNS:
        path = args.root / "reports" / "baselines" / run / "metric_cases.nc"
        if path.is_file():
            datasets[run] = xr.open_dataset(path)
    missing = [run for run in RUNS if run not in datasets]
    if missing:
        print(f"Missing exp2 report caches, those rows are omitted: {missing}")

    label_of = {**{run: label for run, (_, _, label) in RUNS.items()}, **BASELINES,
                "persistence_24h": BASELINES["persistence"]}
    try:
        significance_rows = []
        score_rows = []
        for horizon, horizon_label in (() if args.figures_only else HORIZONS):
            baseline_runs = ("climatology", PERSISTENCE_BY_HORIZON[horizon])
            for metric, lead_dimension, metric_label, unit, kind in METRICS:
                if lead_dimension is None and horizon == "day1":
                    continue
                series = {run: case_series(datasets[run], metric, lead_dimension, horizon)
                          for run in (*RUNS, *baseline_runs)
                          if run in datasets}
                # One date set for the whole family. Scoring each pair on its own
                # intersection would let different pairs rest on different forecasts,
                # so their differences would not be comparable to each other.
                common_dates = None
                for values in series.values():
                    dates = values.dropna().index
                    common_dates = dates if common_dates is None else common_dates.intersection(dates)
                if common_dates is None or len(common_dates) < 3:
                    continue
                series = {run: values.reindex(common_dates) for run, values in series.items()}
                print(f"  {metric_label} / {horizon_label}: {len(common_dates)} shared initialisations",
                      flush=True)
                # One block length for the family, the median across its pairs. Per-pair
                # selection is unstable near the Politis-White critical band, and that
                # instability alone decided significance between comparable pairs. The
                # median rather than the maximum: the largest value in a family is
                # routinely the selector failing and falling back to its analytic cap,
                # which would let one broken estimate set the block length for every pair.
                family_differences = [
                    (series[candidate_run] - series[reference_run]).dropna().values
                    for _question, reference_run, candidate_runs, _label in QUESTIONS
                    if reference_run in series
                    for candidate_run in (*candidate_runs, *baseline_runs) if candidate_run in series]
                block_length = family_block_length(family_differences)
                climatology = series.get("climatology")
                for run, values in series.items():
                    mean_score = float(values.mean())
                    denominator_mean = (climatology.reindex(values.index).mean()
                                        if climatology is not None else None)
                    if kind == "rmse":
                        mean_score = float(np.sqrt(mean_score))
                        if denominator_mean is not None:
                            denominator_mean = np.sqrt(denominator_mean)
                    score_rows.append({
                        "run": label_of[run], "metric": metric_label, "horizon": horizon_label,
                        "mean_score": mean_score, "unit": unit,
                        "crpss_versus_climatology": (float(1.0 - mean_score / denominator_mean)
                                                     if denominator_mean is not None else float("nan")),
                        "n_initialisations": int(values.notna().sum()),
                    })
                for question, reference_run, candidate_runs, reference_label in QUESTIONS:
                    if reference_run not in series:
                        continue
                    for candidate_run in (*candidate_runs, *baseline_runs):
                        if candidate_run not in series:
                            continue
                        paired = pd.DataFrame({"candidate": series[candidate_run],
                                               "reference": series[reference_run]}).dropna()
                        if kind == "rmse":
                            result = bootstrap_paired_metric(
                                {"squared_error": paired["candidate"].values},
                                {"squared_error": paired["reference"].values},
                                lambda components: float(np.sqrt(components["squared_error"])),
                                num_resamples=args.resamples,
                                block_length=block_length)
                            estimate_candidate = float(np.sqrt(paired["candidate"].mean()))
                            estimate_reference = float(np.sqrt(paired["reference"].mean()))
                        else:
                            result = bootstrap_paired_difference(
                                (paired["candidate"] - paired["reference"]).values,
                                num_resamples=args.resamples,
                                block_length=block_length)
                            estimate_candidate = float(paired["candidate"].mean())
                            estimate_reference = float(paired["reference"].mean())
                        significance_rows.append({
                            "question": question,
                            "candidate": label_of[candidate_run],
                            "reference": reference_label or label_of[reference_run],
                            "metric": metric_label, "horizon": horizon_label,
                            "tier": METRIC_TIER[metric_label],
                            "family": question,
                            "estimate_candidate": estimate_candidate,
                            "estimate_reference": estimate_reference,
                            "delta_candidate_minus_reference": float(result["point_estimate"]),
                            "ci_low": float(result["lower_bound"]),
                            "ci_high": float(result["upper_bound"]),
                            "p_value": float(result["p_value"]),
                            "block_length": float(result["block_length"]),
                            "n_initialisations": int(len(paired)),
                            "resamples": args.resamples,
                            "direction": ("candidate better" if result["upper_bound"] < 0
                                          else "reference better" if result["lower_bound"] > 0
                                          else "not distinguishable"),
                            "is_baseline": candidate_run in BASELINE_CACHE_RUNS,
                        })

        apply_multiplicity_correction(significance_rows)

        # Every run against every other, so a reader can tell any two apart and not only
        # each against the two references. One family per (group, horizon, metric): the
        # groups are separate tables at separate lead windows, and pooling them would
        # correct over tests that never appear side by side.
        print("[comparison_23] pairwise matrices", flush=True)
        pairwise_rows = []
        pairwise_directory = output_directory / "pairwise"
        pairwise_directory.mkdir(parents=True, exist_ok=True)
        for group, group_title, group_runs in (() if args.figures_only else GROUPS):
            # Baselines join the matrix so the table can carry them as rows; that makes the
            # family larger, so Holm corrects over more pairs than the models alone.
            present = [run for run in (*group_runs, *BASELINES) if run in datasets]
            if len(present) < 3:
                continue
            for metric, lead_dimension, metric_label, unit, kind in METRICS:
                for horizon, horizon_label in HORIZONS:
                    if lead_dimension is None and horizon == "day1":
                        continue
                    present = [run for run in (*group_runs, "climatology", PERSISTENCE_BY_HORIZON[horizon])
                               if run in datasets]
                    series_of = {run: case_series(datasets[run], metric, lead_dimension, horizon)
                                 for run in present if metric in datasets[run]}
                    labels = [label_of[run] for run in series_of]
                    count = len(labels)
                    if count < 3:
                        continue
                    common_dates = None
                    for values in series_of.values():
                        dates = values.dropna().index
                        common_dates = dates if common_dates is None else common_dates.intersection(dates)
                    if common_dates is None or len(common_dates) < 3:
                        continue
                    series_of = {run: values.reindex(common_dates) for run, values in series_of.items()}
                    block_length = family_block_length([
                        (series_of[first_run] - series_of[second_run]).dropna().values
                        for first_run in series_of for second_run in series_of
                        if first_run < second_run])
                    print(f"  {group} {metric_label} / {horizon_label}: {len(common_dates)} shared "
                          f"initialisations, block length {block_length:.2f}", flush=True)
                    difference = np.full((count, count), np.nan)
                    probability = np.full((count, count), np.nan)
                    metric_rows = []
                    for row, row_run in enumerate(series_of):
                        for column, column_run in enumerate(series_of):
                            if row >= column:
                                continue
                            paired = pd.DataFrame({"row": series_of[row_run],
                                                   "column": series_of[column_run]}).dropna()
                            if len(paired) < 3:
                                continue
                            if kind == "rmse":
                                result = bootstrap_paired_metric(
                                    {"squared_error": paired["row"].values},
                                    {"squared_error": paired["column"].values},
                                    lambda components: float(np.sqrt(components["squared_error"])),
                                    num_resamples=args.resamples,
                                    block_length=block_length)
                            else:
                                result = bootstrap_paired_difference(
                                    (paired["row"] - paired["column"]).values,
                                    num_resamples=args.resamples,
                                    block_length=block_length)
                            metric_rows.append({
                                "group": group, "metric": metric_label, "unit": unit,
                                "horizon": horizon_label, "tier": METRIC_TIER[metric_label],
                                "family": f"pairwise_{group}_{horizon}_{metric_label}",
                                "row_model": label_of[row_run], "column_model": label_of[column_run],
                                "delta_row_minus_column": float(result["point_estimate"]),
                                "ci_low": float(result["lower_bound"]),
                                "ci_high": float(result["upper_bound"]),
                                "p_value": float(result["p_value"]),
                                "block_length": float(result["block_length"]),
                                "block_length_selected": float(result["block_length_selected"]),
                                "n_initialisations": int(len(paired)), "resamples": args.resamples,
                                "_row": row, "_column": column,
                            })
                    if not metric_rows:
                        continue
                    apply_multiplicity_correction(metric_rows, alpha=0.05)
                    for entry in metric_rows:
                        row, column = entry.pop("_row"), entry.pop("_column")
                        difference[row, column] = entry["delta_row_minus_column"]
                        difference[column, row] = -entry["delta_row_minus_column"]
                        probability[row, column] = probability[column, row] = entry["p_adjusted"]
                    pairwise_rows.extend(metric_rows)
                    figure = build_pairwise_significance_figure(
                        labels, difference, probability, metric_rows[0]["alpha_threshold"],
                        metric_label, unit, f"Pairwise differences, {horizon_label}")
                    figure.savefig(pairwise_directory / f"pairwise_{metric}_{group}_{horizon}.png")
                    plt.close(figure)

        # Heatwave diagnostics rebuilt from the stored per-case primitives, under both
        # heatwave definitions. Reliability is day one; REV is the full shared week,
        # per the agreed figure scope.
        diagnostic_rows = []
        reliability_curves = []
        rev_curves = []
        for run, dataset in datasets.items():
            for horizon, horizon_label in HORIZONS:
                if run not in (*RUNS, "climatology", PERSISTENCE_BY_HORIZON[horizon]):
                    continue
                window = event_window(dataset, horizon)
                for suffix, definition_label in HEATWAVE_DEFINITIONS:
                    case_dimensions = [dimension for dimension in ("forecast", "lead_day")
                                       if dimension in window[f"brier_score_{suffix}"].dims]
                    positives = window[f"true_positive_share_{suffix}"].mean(case_dimensions)
                    false_positives = window[f"false_positive_share_{suffix}"].mean(case_dimensions)
                    false_negatives = window[f"false_negative_share_{suffix}"].mean(case_dimensions)
                    # The forecast declares a heatwave when it predicts exceedance: the majority
                    # of members for an ensemble, the field itself for a deterministic baseline.
                    warned = int(np.argmin(np.abs(window.decision_probability_threshold.values - 0.5)))
                    event_rate = float(window[f"event_rate_{suffix}"].mean())
                    spread, rmse, ratio = spread_and_skill(dataset, horizon)
                    diagnostic_rows.append({
                        "run": label_of[run], "definition": definition_label, "horizon": horizon_label,
                        "brier_score": float(window[f"brier_score_{suffix}"].mean()),
                        "event_rate": event_rate,
                        "auc": float(window[f"auc_case_{suffix}"].mean(skipna=True)),
                        "f1": float((2 * positives / (2 * positives + false_positives
                                                      + false_negatives)).values[warned]),
                        "precision": float((positives / (positives + false_positives)).values[warned]),
                        "recall": float((positives / (positives + false_negatives)).values[warned]),
                        "ensemble_spread": spread, "rmse": rmse, "spread_skill_ratio": ratio,
                        "conditional_absolute_error": float(
                            window[f"conditional_abs_error_sum_{suffix}"].sum()
                            / window[f"conditional_error_support_{suffix}"].sum()),
                        "is_baseline": run in BASELINE_CACHE_RUNS,
                    })
                    # Reliability and REV are drawn on the threshold definition; both
                    # definitions stay in the CSVs.
                    if suffix != "threshold":
                        continue
                    if horizon == "day1":
                        # Reliability is ensemble-only: a deterministic baseline has no
                        # probability to calibrate.
                        if run in BASELINE_CACHE_RUNS:
                            continue
                        bin_count = window[f"probability_bin_count_{suffix}"].sum(case_dimensions)
                        bin_events = window[f"probability_bin_event_count_{suffix}"].sum(case_dimensions)
                        reliability_curves.append((
                            run, label_of[run], window.forecast_probability.values,
                            (bin_events / bin_count.where(bin_count > 0)).values))
                    elif run != "climatology":
                        # REV is a decision score, so persistence belongs on it;
                        # climatology never warns and would sit flat on zero.
                        rev_curves.append((run, label_of[run],
                                           np.linspace(0.01, 0.99, 99),
                                           relative_economic_value(
                                               event_rate, (positives + false_positives).values,
                                               false_negatives.values)))

        climatology_brier = {(row["horizon"], row["definition"]): row["brier_score"] for row in diagnostic_rows
                             if row["run"] == CLIMATOLOGY_LABEL}
        for row in diagnostic_rows:
            denominator = climatology_brier.get((row["horizon"], row["definition"]), float("nan"))
            # A zero climatology Brier is legitimate (no events and no warnings on the
            # window, e.g. the spell sieve can never fire on lead day 1): no skill baseline.
            row["brier_skill_score"] = (float(1.0 - row["brier_score"] / denominator)
                                        if denominator and np.isfinite(denominator) else float("nan"))

        for group, group_title, group_runs in GROUPS:
            selected = [(label, *rest) for run, label, *rest in reliability_curves if run in group_runs]
            if selected:
                figure = build_reliability_figure(
                    selected, "Day-one reliability")
                figure.savefig(output_directory / f"reliability_day1_{group}.png", dpi=200)
                plt.close(figure)
            selected = [(label, *rest) for run, label, *rest in rev_curves if run in group_runs]
            if selected:
                figure = build_stacked_rev_figure(
                    selected, "Relative economic value")
                figure.savefig(output_directory / f"rev_full_{group}.png", dpi=200)
                plt.close(figure)

            # Daily-Tmax distribution and predicted-vs-actual scatter, from the histogram
            # and sample already cached at report time (no raw fields, no recomputation).
            group_datasets = [(label_of[run], datasets[run]) for run in (*group_runs, *BASELINES) if run in datasets]
            if group_datasets:
                truth_dataset = datasets.get("climatology", group_datasets[0][1])
                distribution_curves = [("ERA5 truth", truth_dataset["tmax_histogram_target"].values)]
                distribution_curves += [(label, dataset["tmax_histogram_prediction"].values)
                                        for label, dataset in group_datasets]
                figure = build_tmax_distribution_figure(
                    distribution_curves, truth_dataset["tmax_bin_edges"].values,
                    "Daily Tmax distribution")
                figure.savefig(output_directory / f"tmax_distribution_{group}.png")
                plt.close(figure)

                scatter_curves = [(label, dataset["tmax_scatter_prediction"].values,
                                   dataset["tmax_scatter_target"].values, dataset["tmax_scatter_category"].values)
                                  for label, dataset in group_datasets]
                figure = build_tmax_scatter_figure(scatter_curves,
                                                   "Predicted versus actual daily Tmax")
                figure.savefig(output_directory / f"tmax_scatter_{group}.png")
                plt.close(figure)

                # One stacked calendar per run over the calendar region, coloured by how
                # many members put most of the region above p90 that day, with the observed
                # heatwave days outlined on top. observed_share is passed but not drawn.
                calendar_panels = [
                    (label_of[run], datasets[run]["date"].values,
                     datasets[run]["calendar_observed_share"].values,
                     datasets[run]["calendar_observed_heatwave"].values,
                     datasets[run]["calendar_model_certainty"].values)
                    for run in group_runs
                    if run in datasets and "calendar_model_certainty" in datasets[run]]
                if calendar_panels:
                    members_per_run = int(group_datasets[0][1].sizes["probability_bin"]) - 1
                    figure = build_calendar_skill_figure(
                        calendar_panels, "Daily hot-day certainty", members_per_run,
                        region=DENMARK_CALENDAR_REGION, extra_cells=DENMARK_CALENDAR_EXTRA_CELLS)
                    figure.savefig(output_directory / f"calendar_heatwave_{group}.png")
                    plt.close(figure)

                # Model error against the observed Tmax it was scored on.
                error_curves = [(label, dataset["tmax_scatter_target"].values,
                                 dataset["tmax_scatter_prediction"].values) for label, dataset in group_datasets]
                threshold_mean = (float(truth_dataset["tmax_threshold_mean"])
                                  if "tmax_threshold_mean" in truth_dataset else None)
                figure = build_tmax_error_by_observed_figure(
                    error_curves, threshold_mean, "Model error by observed Tmax")
                figure.savefig(output_directory / f"tmax_error_by_observed_{group}.png")
                plt.close(figure)

            # Lead-time curves: every score keeps its native lead axis in the caches, so a
            # curve is a mean over initialisations with nothing recomputed from raw fields.
            for variable, lead_dimension, xlabel, ylabel in LEAD_CURVES:
                curves = [(label_of[run], datasets[run][lead_dimension].values,
                           datasets[run][variable].mean("forecast").values)
                          for run in (*group_runs, *BASELINES) if run in datasets]
                figure = build_lead_curve_figure(curves, xlabel, ylabel,
                                                 f"{ylabel.split(' (')[0]} over lead time")
                figure.savefig(output_directory / f"lead_{variable}_{group}.png", dpi=200)
                plt.close(figure)

            curves = [(label_of[run], datasets[run].hour.values,
                       np.sqrt(datasets[run]["squared_error_case"].mean("forecast").values))
                      for run in (*group_runs, *BASELINES)
                      if run in datasets and "squared_error_case" in datasets[run]]
            if curves:
                figure = build_lead_curve_figure(curves, "Lead time (hours)", "RMSE (K)",
                                                 "Ensemble-mean RMSE over lead time")
                figure.savefig(output_directory / f"lead_rmse_{group}.png", dpi=200)
                plt.close(figure)

            # Deterministic baselines have no spread, so they are absent from the SSR curve.
            curves = []
            for run in group_runs:
                dataset = datasets.get(run)
                members = int(dataset.probability_bin.size) - 1 if dataset is not None else 0
                if dataset is None or members < 2 or "ensemble_variance_case" not in dataset:
                    continue
                spread = np.sqrt(dataset["ensemble_variance_case"].mean("forecast").values
                                 * (members + 1) / (members - 1))
                curves.append((label_of[run], dataset.hour.values,
                               spread / np.sqrt(dataset["squared_error_case"].mean("forecast").values)))
            if curves:
                figure = build_lead_curve_figure(curves, "Lead time (hours)", "Spread-skill ratio",
                                                 "Spread-skill ratio of RMSE over lead time")
                figure.savefig(output_directory / f"lead_spread_skill_{group}.png", dpi=200)
                plt.close(figure)

        for name, records in () if args.figures_only else (("paired_significance.csv", significance_rows),
                              ("pairwise_significance.csv", pairwise_rows),
                              ("primary_scores.csv", score_rows),
                              ("diagnostics.csv", diagnostic_rows)):
            if not records:
                continue
            fieldnames = [key for key in records[0] if key not in ("ci_low", "ci_high")]
            with (output_directory / name).open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(records)

        # One forest per question and metric, on the full shared forecast. Day-one rows stay
        # in the tables; a second set of forests for them adds nothing.
        forest_directory = output_directory / "forest"
        forest_directory.mkdir(parents=True, exist_ok=True)
        for question, _, _, _ in QUESTIONS:
            for metric, _, metric_label, unit, _ in METRICS:
                for horizon, horizon_label in HORIZONS[:1]:
                    rows = [row for row in significance_rows
                            if row["question"] == question and row["metric"] == metric_label
                            and row["horizon"] == horizon_label]
                    if not rows:
                        continue
                    figure = build_paired_forest_figure(
                        [{"label": row["candidate"], "difference": row["delta_candidate_minus_reference"],
                          "lower": row["ci_low"], "upper": row["ci_high"],
                          "is_baseline": row["is_baseline"]} for row in rows],
                        metric_label, unit,
                        f"{rows[0]['reference']} as reference\n{metric_label}, {horizon_label}")
                    figure.savefig(forest_directory / f"forest_{question}_{metric}_{horizon}.png", dpi=200)
                    plt.close(figure)

        # Maps need the raw fields: one valid hour from each prediction cache.
        map_time = pd.Timestamp(args.map_date) + pd.Timedelta(hours=12)
        loading_region = get_loading_region(json.loads(
            Path("modified_Gencast/traninig configs/experiment_2_all_gencast.json").read_text())["task_config"])
        truth = to_scoring_grid(load(str(map_time.date()), str(map_time.date()),
                                     variables=[TARGET_VAR], region=loading_region,
                                     freq="1h")[TARGET_VAR], REGION).sel(time=map_time).compute()
        stamp = map_time.strftime("%d %B %Y %H UTC")
        for group, group_title, group_runs in GROUPS:
            map_fields = []
            for run in group_runs:
                store, suffix, label = RUNS[run]
                tag = f"{store}_step_00016000_2021-01-01_2021-12-31_ar14{suffix}"
                path = args.scratch / "Predictions" / store / f"predictions_{tag}.nc"
                if path.is_file():
                    map_fields.append((label, field_at(path, map_time)))
            if not map_fields:
                continue
            figure = build_map_grid_figure([("ERA5 truth", truth), *map_fields], "ERA5",
                                           f"2 m temperature, {stamp}")
            figure.savefig(output_directory / f"maps_{args.map_date}_{group}.png", dpi=200)
            plt.close(figure)
            figure = build_map_grid_figure([(label, field - truth) for label, field in map_fields],
                                           "ERA5", f"Model minus ERA5, {stamp}",
                                           difference=True)
            figure.savefig(output_directory / f"maps_error_{args.map_date}_{group}.png", dpi=200)
            plt.close(figure)
    finally:
        for dataset in datasets.values():
            dataset.close()
    print(output_directory)


if __name__ == "__main__":
    main()
