"""Comparisons 6 and 7 — exp4 daily-statistic inputs and region transferability.

Comparison numbers run 0 to 7 across five files and do not line up with the experiment
numbers, because one experiment can raise more than one question. These are comparisons 6
and 7, answering RQ3 from the exp4 and baseline caches.

Reads compact caches only, apart from the raw hours the maps and week trajectories need.

Comparison 6 varies the input days over Europe: same region, same calendar days, so
candidate and reference pair directly. Two references — the richest input set and the
current-day-only set — so the table answers both "does less information cost anything"
and "does more information buy anything".

Comparison 7 varies the region. Raw scores across climates are not comparable, so the
transfer measure is each region's skill against **its own** climatology and persistence.
Every region therefore carries its own two baselines, its own truth and its own crop.

Re-aggregation error is the honesty check: take Tmax, Tmean and Tmin back out of the
reconstructed hours and score them against truth. These models are handed those daily
statistics, so a model that cannot reproduce its own input is failing the task whatever
its CRPS says.
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from modified_Gencast.comparison_1 import domain_mean_series, field_at
from modified_Gencast.lumio_buckets import load
from modified_Gencast.metrics import (
    DENMARK_CALENDAR_REGION, TARGET_VAR, relative_economic_value, to_scoring_grid,
)
from modified_Gencast.plots import (
    build_domain_mean_figure,
    build_lead_curve_figure,
    build_map_grid_figure,
    build_rmse_map_grid_figure,
    build_monthly_temperature_qq_figure,
    build_paired_forest_figure,
    build_pairwise_significance_figure,
    build_reliability_figure,
    build_stacked_rev_figure,
)
BOUNDARY_MARGIN_DEGREES = 8.5
STAGE = "step_00016000_2021-01-01_2021-12-31_daily_fixed"
EUROPE = (45.0, 65.0, -5.0, 25.0)
OFFSETS_M1_0_P1 = "experiment_4_offsets_m1_0_p1_stage2"
# run key -> (store directory, cached-tag suffix, reader-facing label, scoring region)
EXP4_RUNS = {
    "offsets_0": ("experiment_4_offsets_0_stage2", "_churn0.0", "Current day only", EUROPE),
    "offsets_m1_0_p1": (OFFSETS_M1_0_P1, "_churn0.0", "Previous, current and next day", EUROPE),
    "offsets_m2_m1_0": ("experiment_4_offsets_m2_m1_0_stage2", "_churn0.0", "Two preceding days and current", EUROPE),
    "offsets_m2_m1_0_p1_p2": ("experiment_4_offsets_m2_m1_0_p1_p2_stage2", "_churn0.0", "Two days either side", EUROPE),
    "nw_america": (OFFSETS_M1_0_P1, "_nw_america_churn0.0", "North-West America", (45.0, 65.0, -135.0, -105.0)),
    "n_africa": (OFFSETS_M1_0_P1, "_n_africa_churn0.0", "North Africa", (15.0, 35.0, 10.0, 40.0)),
}
from modified_Gencast.significance import (
    apply_multiplicity_correction,
    bootstrap_paired_difference,
    family_block_length,
)


# Each run's own two baselines, on its own grid and in its own climate.
BASELINE_FOR = {
    "offsets_0": ("climatology_daily", "persistence_daily"),
    "offsets_m1_0_p1": ("climatology_daily", "persistence_daily"),
    "offsets_m2_m1_0": ("climatology_daily", "persistence_daily"),
    "offsets_m2_m1_0_p1_p2": ("climatology_daily", "persistence_daily"),
    "nw_america": ("climatology_nw_america", "persistence_nw_america"),
    "n_africa": ("climatology_n_africa", "persistence_n_africa"),
}
BASELINE_LABELS = {"climatology": "ERA5 average-year climatology",
                   "persistence": "ERA5 persistence (1-day lag)"}
EUROPE_RUNS = ("offsets_0", "offsets_m1_0_p1", "offsets_m2_m1_0", "offsets_m2_m1_0_p1_p2")
QUESTIONS = (
    ("inputs_versus_richest", "offsets_m2_m1_0_p1_p2",
     ("offsets_0", "offsets_m1_0_p1", "offsets_m2_m1_0"), False),
    ("inputs_versus_current_day", "offsets_0",
     ("offsets_m1_0_p1", "offsets_m2_m1_0", "offsets_m2_m1_0_p1_p2"), False),
    ("regions_versus_europe", "offsets_m1_0_p1", ("nw_america", "n_africa"), True),
)
GROUPS = (
    ("comparison_6", "Daily statistic inputs", EUROPE_RUNS),
    ("comparison_7", "Region transfer",
     ("offsets_m1_0_p1", "nw_america", "n_africa")),
)
# In comparison 7 every row is a region, so the shared checkpoint is named for where it ran.
REGION_LABEL = {"offsets_m1_0_p1": "Europe"}
METRICS = (
    ("t2m_crps_case", "hour", "2 m CRPS", "K"),
    ("peak_time_ccrps_case", "day_idx", "Peak-time circular CRPS", "h"),
    ("daily_range_crps_case", "day_idx", "Daily range CRPS", "K"),
    ("asymmetry_crps_case", "day_idx", "Diurnal asymmetry CRPS", ""),
    ("hours_above_threshold_crps_case", "day_idx", "Hours-above-threshold CRPS", "h"),
    ("temporal_variogram_case", "day_idx", "Temporal variogram score", ""),
    ("tmax_reaggregation_crps_case", "day_idx", "Re-aggregated Tmax CRPS", "K"),
    ("tmean_reaggregation_crps_case", "day_idx", "Re-aggregated Tmean CRPS", "K"),
    ("tmin_reaggregation_crps_case", "day_idx", "Re-aggregated Tmin CRPS", "K"),
)
# Same split as comparison_45, so RQ2 and RQ3 decide on an identical set. Re-aggregated
# Tmax is primary as the only Tmax these runs produce; Tmean and Tmin stay supporting.
SECONDARY_METRICS = ("Re-aggregated Tmean CRPS", "Re-aggregated Tmin CRPS")
METRIC_TIER = {label: ("secondary" if label in SECONDARY_METRICS else "primary")
               for _, _, label, _ in METRICS}
HOURLY_CURVES =(("t2m_crps_case", "2 m CRPS (K)"), ("t2m_twcrps_case", "2 m twCRPS (K)"))
BIAS_COLUMNS = ("daily_range_bias_case", "asymmetry_bias_case", "tmax_reaggregation_bias_case",
                "tmean_reaggregation_bias_case", "tmin_reaggregation_bias_case")


def members_of(dataset):
    """Ensemble size from the probability-bin axis; xr.merge drops the attribute."""
    return int(dataset.sizes.get("probability_bin", 2)) - 1


def climatology_for(run):
    """The climatology cache a run is scored against: its own region's, baselines included."""
    if run in BASELINE_FOR:
        return BASELINE_FOR[run][0]
    return run.replace("persistence", "climatology")


def case_series(dataset, metric, case_dimension):
    """One score per calendar day, keyed by that day."""
    values = dataset[metric].mean(case_dimension)
    return pd.Series(np.asarray(values.values, dtype=float),
                     index=pd.DatetimeIndex(values.forecast.values))


DISTRIBUTION_PROBABILITIES = np.array(
    [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99], dtype=float)


def monthly_hourly_temperature_quantiles(path, region, ensemble_mean):
    """Quantiles of all hourly grid-cell temperatures, separately for each month.

    A month is loaded at a time so the diagnostic reads raw cached fields but never
    materialises an annual forecast cube.  Forecasts in exp4 are daily issue dates,
    therefore their month is also the valid calendar month of all 24 reconstructed
    hourly frames.
    """
    with xr.open_dataset(path, chunks={"forecast": 8}) as store:
        field = to_scoring_grid(store[TARGET_VAR], region)
        if ensemble_mean and "ensemble" in field.dims:
            field = field.mean("ensemble")
        forecast_month = pd.DatetimeIndex(field.forecast.values).month
        reduction_dims = list(field.dims)
        quantiles = []
        for month in range(1, 13):
            values = field.isel(forecast=np.flatnonzero(forecast_month == month)).load()
            quantiles.append(np.nanquantile(values.values, DISTRIBUTION_PROBABILITIES,
                                            axis=tuple(range(values.ndim))))
    return np.stack(quantiles)


def comparison_6_hourly_rmse_maps(scratch: Path):
    """Hourly ensemble-mean RMSE maps for the four Europe daily-statistic models."""
    fields = []
    for run in EUROPE_RUNS:
        store, suffix, label, region = EXP4_RUNS[run]
        tag = f"{store}_{STAGE}{suffix}"
        prediction_path = scratch / "Predictions" / store / f"predictions_{tag}.nc"
        target_path = scratch / "Predictions" / store / f"targets_{tag}.nc"
        with xr.open_dataset(prediction_path, chunks={"forecast": 8}) as prediction_ds, \
             xr.open_dataset(target_path, chunks={"forecast": 8}) as target_ds:
            prediction = to_scoring_grid(prediction_ds[TARGET_VAR], region).mean("ensemble")
            target = to_scoring_grid(target_ds[TARGET_VAR], region)
            fields.append((label, np.sqrt(((prediction - target) ** 2).mean(["forecast", "hour"])).compute()))
    return fields


def main():
    """Write each comparison's CSVs and figures into its own directory.

    baseline_audit.csv records which baseline cache each run was paired against.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/scratch/project_465002687/verified_metrics"))
    parser.add_argument("--scratch", type=Path, default=Path("/scratch/project_465002687"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--map-date", type=str, default="2021-06-20")
    parser.add_argument("--week-start", type=str, default="2021-06-16")
    # Redraw the figures from the caches without repeating the bootstrap, for when
    # only a label or a layout changed. The significance CSVs are left untouched.
    parser.add_argument("--figures-only", action="store_true")
    parser.add_argument("--rmse-maps-only", action="store_true",
                        help="Generate only the 2x2 Comparison 6 hourly RMSE map grid.")
    parser.add_argument("--hourly-twcrps-comparison-6-only", action="store_true",
                        help="Redraw only the Comparison 6 hourly 2-m twCRPS curve.")
    args = parser.parse_args()
    output_directory = args.out or Path.cwd() / "comparison_67"
    output_directory.mkdir(parents=True, exist_ok=True)

    if args.rmse_maps_only:
        figure = build_rmse_map_grid_figure(comparison_6_hourly_rmse_maps(args.scratch), EUROPE)
        figure.savefig(output_directory / "hourly_rmse_maps_comparison_6.png", dpi=200)
        plt.close(figure)
        print(output_directory / "hourly_rmse_maps_comparison_6.png")
        return

    if args.hourly_twcrps_comparison_6_only:
        # This fast path deliberately reads only the six compact caches needed for one
        # existing figure: four Comp6 models and their Europe climatology/persistence.
        selected = (*EUROPE_RUNS, "climatology_daily", "persistence_daily")
        datasets = {}
        for run in selected:
            report_group = "exp4" if run in EUROPE_RUNS else "baselines"
            path = args.root / "reports" / report_group / run / "metric_cases.nc"
            if not path.is_file():
                raise FileNotFoundError(f"Missing cache for targeted redraw: {path}")
            datasets[run] = xr.open_dataset(path)
        label_of = {run: label for run, (_, _, label, _) in EXP4_RUNS.items()}
        label_of["climatology_daily"] = "ERA5 average-year climatology (Europe)"
        label_of["persistence_daily"] = "ERA5 persistence (1-day lag) (Europe)"
        curves = [(label_of[run], datasets[run].hour.values,
                   datasets[run]["t2m_twcrps_case"].mean("forecast").values)
                  for run in selected]
        curves = [(label, *map(np.asarray, zip(*sorted(zip(hours, values)))))
                  for label, hours, values in curves]
        figure = build_lead_curve_figure(curves, "Hour of day (UTC)", "2 m twCRPS (K)",
                                         "2 m twCRPS over lead time")
        path = output_directory / "hourly_t2m_twcrps_case_comparison_6.png"
        figure.savefig(path, dpi=200)
        plt.close(figure)
        for dataset in datasets.values():
            dataset.close()
        print(path)
        return

    datasets, baseline_audit = {}, []
    for run in EXP4_RUNS:
        path = args.root / "reports" / "exp4" / run / "metric_cases.nc"
        if path.is_file():
            datasets[run] = xr.open_dataset(path)
    for name in sorted({name for pair in BASELINE_FOR.values() for name in pair}):
        path = args.root / "reports" / "baselines" / name / "metric_cases.nc"
        if path.is_file():
            datasets[name] = xr.open_dataset(path)
        baseline_audit.append({"baseline": name, "cache": str(path), "available": path.is_file(),
                               "reason": "" if path.is_file() else "run report_baselines for this grid and region"})
    label_of = {run: label for run, (_, _, label, _) in EXP4_RUNS.items()}
    for run, (climatology, persistence) in BASELINE_FOR.items():
        region_name = label_of[run] if run not in EUROPE_RUNS else "Europe"
        label_of.setdefault(climatology, f"{BASELINE_LABELS['climatology']} ({region_name})")
        label_of.setdefault(persistence, f"{BASELINE_LABELS['persistence']} ({region_name})")
    missing = [run for run in EXP4_RUNS if run not in datasets]
    if missing:
        print(f"Missing exp4 caches, those rows are omitted: {missing}")

    try:
        print("[comparison_67] paired bootstrap", flush=True)
        significance_rows, score_rows = [], []
        for metric, case_dimension, metric_label, unit in (() if args.figures_only else METRICS):
            series = {run: case_series(dataset, metric, case_dimension)
                      for run, dataset in datasets.items() if metric in dataset}
            # One date set for the whole family. Scoring each pair on its own intersection
            # would let different pairs rest on different days, so their differences would
            # not be comparable to each other.
            common_dates = None
            for values in series.values():
                dates = values.dropna().index
                common_dates = dates if common_dates is None else common_dates.intersection(dates)
            if common_dates is None or len(common_dates) < 3:
                continue
            series = {run: values.reindex(common_dates) for run, values in series.items()}
            # One block length for the family, the median across its pairs. Per-pair
            # Politis-White selection is unstable near its critical band, and that
            # instability alone decided significance between comparable pairs.
            family_differences = []
            for _question, reference_run, candidate_runs, _crosses in QUESTIONS:
                if reference_run not in series:
                    continue
                for candidate_run in (*candidate_runs, *dict.fromkeys(
                        name for run in (reference_run, *candidate_runs)
                        for name in BASELINE_FOR[run] if name in series)):
                    if candidate_run in series and candidate_run != reference_run:
                        family_differences.append(
                            (series[candidate_run] - series[reference_run]).dropna().values)
            block_length = family_block_length(family_differences)
            print(f"  {metric_label}: {len(common_dates)} shared days, "
                  f"block length {block_length:.2f}", flush=True)
            for run, values in series.items():
                # Every run is scored against its own region's climatology, never Europe's.
                denominator = series.get(climatology_for(run))
                score_rows.append({
                    "run": label_of[run], "metric": metric_label,
                    "mean_score": float(values.mean()), "unit": unit,
                    "skill_versus_own_climatology": (
                        float(1.0 - values.mean() / denominator.mean())
                        if denominator is not None else float("nan")),
                    "n_cases": int(values.notna().sum()),
                })
            for question, reference_run, candidate_runs, crosses_climate in QUESTIONS:
                if reference_run not in series:
                    continue
                # Each question carries the baselines of the runs it contains.
                baselines = [name for run in (reference_run, *candidate_runs)
                             for name in BASELINE_FOR[run] if name in series]
                for candidate_run in (*candidate_runs, *dict.fromkeys(baselines)):
                    if candidate_run not in series or candidate_run == reference_run:
                        continue
                    paired = pd.DataFrame({"candidate": series[candidate_run],
                                           "reference": series[reference_run]}).dropna()
                    result = bootstrap_paired_difference(
                        (paired["candidate"] - paired["reference"]).values,
                        num_resamples=args.resamples, block_length=block_length)
                    significance_rows.append({
                        "question": question, "candidate": label_of[candidate_run],
                        "reference": label_of[reference_run], "metric": metric_label,
                        "crosses_climate": crosses_climate,
                        "tier": METRIC_TIER[metric_label], "family": question,
                        "estimate_candidate": float(paired["candidate"].mean()),
                        "estimate_reference": float(paired["reference"].mean()),
                        "delta_candidate_minus_reference": float(result["point_estimate"]),
                        "ci_low": float(result["lower_bound"]),
                        "ci_high": float(result["upper_bound"]),
                        "p_value": float(result["p_value"]),
                        "block_length": float(result["block_length"]),
                        "block_length_method": result["block_length_method"],
                        "n_cases": int(len(paired)), "resamples": args.resamples,
                        "direction": ("candidate better" if result["upper_bound"] < 0
                                      else "reference better" if result["lower_bound"] > 0
                                      else "not distinguishable"),
                        "is_baseline": candidate_run in label_of and candidate_run not in EXP4_RUNS,
                    })

        apply_multiplicity_correction(significance_rows)

        # Every run against every other, so a reader can tell any two apart and not only
        # each against its reference. One family per (group, metric).
        print("[comparison_67] pairwise matrices", flush=True)
        pairwise_rows = []
        pairwise_directory = output_directory / "pairwise"
        pairwise_directory.mkdir(parents=True, exist_ok=True)
        for group, group_title, group_runs in (() if args.figures_only else GROUPS):
            # Comparison 7 crosses climates, where a raw difference between two regions is
            # mostly a difference of climate. Every run there is carried as its own
            # region's regret — its score minus that region's climatology, day by day —
            # which is comparable across climates. Comparison 6 is one region throughout,
            # so it stays on raw scores. Under the transform each climatology is
            # identically zero, so the climatologies are not rows in the comparison 7 matrix.
            relative = group == "comparison_7"
            baselines = list(dict.fromkeys(
                name for run in group_runs for name in BASELINE_FOR[run]
                if not (relative and name == climatology_for(run))))
            present = [run for run in (*group_runs, *baselines) if run in datasets]
            if len(present) < 3:
                continue
            for metric, case_dimension, metric_label, unit in METRICS:
                series_of = {}
                for run in present:
                    if metric not in datasets[run]:
                        continue
                    values = case_series(datasets[run], metric, case_dimension)
                    if relative:
                        own_climatology = datasets.get(climatology_for(run))
                        if own_climatology is None or metric not in own_climatology:
                            continue
                        values = values - case_series(own_climatology, metric, case_dimension)
                    series_of[run] = values
                # Only comparison 7 is a table of regions; in comparison 6 the shared
                # checkpoint is one input set among four and keeps its input-set label.
                labels = [REGION_LABEL.get(run, label_of[run]) if relative else label_of[run]
                          for run in series_of]
                count = len(labels)
                if count < 3:
                    continue
                common_dates = None
                for values in series_of.values():
                    dates = values.dropna().index
                    common_dates = (dates if common_dates is None
                                    else common_dates.intersection(dates))
                if common_dates is None or len(common_dates) < 3:
                    continue
                series_of = {run: values.reindex(common_dates) for run, values in series_of.items()}
                block_length = family_block_length([
                    (series_of[first_run] - series_of[second_run]).dropna().values
                    for first_run in series_of for second_run in series_of
                    if first_run < second_run])
                print(f"  {group} {metric_label}: {len(common_dates)} shared days, "
                      f"block length {block_length:.2f}", flush=True)
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
                        result = bootstrap_paired_difference(
                            (paired["row"] - paired["column"]).values,
                            num_resamples=args.resamples, block_length=block_length)
                        metric_rows.append({
                            "group": group, "metric": metric_label, "unit": unit,
                            "measure": "regret against own climatology" if relative else "raw score",
                            "tier": METRIC_TIER[metric_label],
                            "family": f"pairwise_{group}_{metric_label}",
                            "row_model": labels[row], "column_model": labels[column],
                            "delta_row_minus_column": float(result["point_estimate"]),
                            "ci_low": float(result["lower_bound"]),
                            "ci_high": float(result["upper_bound"]),
                            "p_value": float(result["p_value"]),
                            "block_length": float(result["block_length"]),
                            "block_length_selected": float(result["block_length_selected"]),
                            "n_cases": int(len(paired)), "resamples": args.resamples,
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
                    metric_label, unit,
                    "Pairwise regret differences" if relative else "Pairwise differences")
                figure.savefig(pairwise_directory / f"pairwise_{metric}_{group}.png")
                plt.close(figure)

        print("[comparison_67] heatwave diagnostics", flush=True)
        diagnostic_rows, reliability_curves, rev_curves = [], [], []
        for run, dataset in datasets.items():
            case_dimensions = [d for d in ("forecast", "day_idx") if d in dataset["brier_score_heatwave"].dims]
            positives = dataset["true_positive_share_heatwave"].mean(case_dimensions)
            false_positives = dataset["false_positive_share_heatwave"].mean(case_dimensions)
            false_negatives = dataset["false_negative_share_heatwave"].mean(case_dimensions)
            warned = int(np.argmin(np.abs(dataset.decision_probability_threshold.values - 0.5)))
            event_rate = float(dataset["event_rate_heatwave"].mean())
            members = members_of(dataset)
            rmse = np.sqrt(float(dataset["squared_error_case"].mean()))
            spread = (np.sqrt(float(dataset["ensemble_variance_case"].mean())
                              * (members + 1) / (members - 1)) if members > 1 else float("nan"))
            row = {"run": label_of[run], "brier_score": float(dataset["brier_score_heatwave"].mean()),
                   "event_rate": event_rate, "auc": float(dataset["auc_case_heatwave"].mean(skipna=True)),
                   "f1": float((2 * positives / (2 * positives + false_positives
                                                 + false_negatives)).values[warned]),
                   "precision": float((positives / (positives + false_positives)).values[warned]),
                   "recall": float((positives / (positives + false_negatives)).values[warned]),
                   "ensemble_spread": spread, "rmse": rmse,
                   "spread_skill_ratio": spread / rmse if members > 1 else float("nan"),
                   "observed_hours_above_threshold": float(dataset["target_hours_above_threshold_case"].mean()),
                   "is_baseline": run not in EXP4_RUNS}
            for column in BIAS_COLUMNS:
                row[column.replace("_case", "")] = (float(dataset[column].mean())
                                                    if column in dataset else float("nan"))
            diagnostic_rows.append(row)
            if run in EXP4_RUNS:
                bin_count = dataset["probability_bin_count_threshold"].sum(case_dimensions)
                bin_events = dataset["probability_bin_event_count_threshold"].sum(case_dimensions)
                reliability_curves.append((run, label_of[run], dataset.forecast_probability.values,
                                           (bin_events / bin_count.where(bin_count > 0)).values))
                threshold_positives = dataset["true_positive_share_threshold"].mean(case_dimensions)
                threshold_false_positives = dataset["false_positive_share_threshold"].mean(case_dimensions)
                threshold_false_negatives = dataset["false_negative_share_threshold"].mean(case_dimensions)
                threshold_event_rate = float(dataset["event_rate_threshold"].mean())
                rev_curves.append((run, label_of[run], np.linspace(0.01, 0.99, 99),
                                   relative_economic_value(
                                       threshold_event_rate,
                                       (threshold_positives + threshold_false_positives).values,
                                       threshold_false_negatives.values)))
        # BSS uses the climatology of the run's own region.
        brier_of = {row["run"]: row["brier_score"] for row in diagnostic_rows}
        for row, run in zip(diagnostic_rows, datasets):
            denominator = brier_of.get(label_of.get(climatology_for(run)))
            row["brier_skill_score"] = (float(1.0 - row["brier_score"] / denominator)
                                        if denominator else float("nan"))

        for name, records in (("paired_significance.csv", significance_rows),
                              ("pairwise_significance.csv", pairwise_rows),
                              ("primary_scores.csv", score_rows),
                              ("diagnostics.csv", diagnostic_rows),
                              ("baseline_audit.csv", baseline_audit)):
            if records:
                fieldnames = [key for key in records[0] if key not in ("ci_low", "ci_high")]
                with (output_directory / name).open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
                    writer.writeheader()
                    writer.writerows(records)

        print("[comparison_67] forests", flush=True)
        forest_directory = output_directory / "forest"
        forest_directory.mkdir(parents=True, exist_ok=True)
        for question, _, _, crosses_climate in QUESTIONS:
            for metric, _, metric_label, unit in METRICS:
                rows = [row for row in significance_rows
                        if row["question"] == question and row["metric"] == metric_label]
                if not rows:
                    continue
                note = "\ndifferent climates: read with the per-region skill table" if crosses_climate else ""
                figure = build_paired_forest_figure(
                    [{"label": row["candidate"], "difference": row["delta_candidate_minus_reference"],
                      "lower": row["ci_low"], "upper": row["ci_high"],
                      "is_baseline": row["is_baseline"]} for row in rows],
                    metric_label, unit, f"{rows[0]['reference']} as reference\n{metric_label}{note}")
                figure.savefig(forest_directory / f"forest_{question}_{metric}.png", dpi=200)
                plt.close(figure)

        print("[comparison_67] hourly curves, reliability and REV", flush=True)
        for group, group_title, group_runs in GROUPS:
            baselines = list(dict.fromkeys(name for run in group_runs for name in BASELINE_FOR[run]))
            # Regions sit in different time zones, so the diurnal cycle is drawn on local
            # solar hour; on a UTC axis the afternoon peaks would not line up.
            local_offset = {}
            for run in (*group_runs, *baselines):
                region = EXP4_RUNS[run][3] if run in EXP4_RUNS else None
                if region is None:
                    region = next((EXP4_RUNS[other][3] for other in group_runs
                                   if other in EXP4_RUNS and run in BASELINE_FOR[other]), None)
                local_offset[run] = 0.0 if region is None else (region[2] + region[3]) / 30.0
            xlabel = "Local solar hour" if group == "comparison_7" else "Hour of day (UTC)"
            for variable, ylabel in HOURLY_CURVES:
                curve_runs = group_runs if group == "comparison_7" else (*group_runs, *baselines)
                curves = [(REGION_LABEL.get(run, label_of[run]) if group == "comparison_7"
                           else label_of[run],
                           (datasets[run].hour.values
                            + (local_offset[run] if group == "comparison_7" else 0.0)) % 24,
                           datasets[run][variable].mean("forecast").values)
                          for run in curve_runs
                          if run in datasets and variable in datasets[run]]
                curves = [(label, *map(np.asarray, zip(*sorted(zip(hours, values)))))
                          for label, hours, values in curves]
                if curves:
                    figure = build_lead_curve_figure(curves, xlabel, ylabel,
                                                     f"{ylabel.split(' (')[0]} over lead time")
                    figure.savefig(output_directory / f"hourly_{variable}_{group}.png", dpi=200)
                    plt.close(figure)
            selected = [(label, *rest) for run, label, *rest in reliability_curves if run in group_runs]
            if selected:
                figure = build_reliability_figure(
                    selected, "Daily p90-exceedance reliability")
                figure.savefig(output_directory / f"reliability_{group}.png", dpi=200)
                plt.close(figure)
            selected = [(label, *rest) for run, label, *rest in rev_curves if run in group_runs]
            if selected:
                figure = build_stacked_rev_figure(
                    selected, "Heatwave relative economic value")
                figure.savefig(output_directory / f"rev_{group}.png", dpi=200)
                plt.close(figure)

        print("[comparison_67] monthly temperature-distribution diagnostic", flush=True)
        # This is deliberately a visual diagnostic, not a new selection metric.  It
        # tests whether matching daily Tmin/Tmean/Tmax merely recovers central values
        # or also reproduces the monthly cold and hot hourly tails.
        reference_run = "offsets_m1_0_p1"
        reference_store, reference_suffix, _, _ = EXP4_RUNS[reference_run]
        reference_tag = f"{reference_store}_{STAGE}{reference_suffix}"
        target_path = (args.scratch / "Predictions" / reference_store
                       / f"targets_{reference_tag}.nc")
        target_quantiles = monthly_hourly_temperature_quantiles(target_path, EUROPE, ensemble_mean=False)
        model_quantiles = {}
        for run in EUROPE_RUNS:
            store, suffix, label, region = EXP4_RUNS[run]
            prediction_path = args.scratch / "Predictions" / store / f"predictions_{store}_{STAGE}{suffix}.nc"
            if prediction_path.is_file():
                model_quantiles[label] = monthly_hourly_temperature_quantiles(
                    prediction_path, region, ensemble_mean=True)
        figure = build_monthly_temperature_qq_figure(
            model_quantiles, target_quantiles, DISTRIBUTION_PROBABILITIES)
        figure.savefig(output_directory / "monthly_temperature_distribution_qq_comparison_6.png", dpi=200)
        plt.close(figure)

        print("[comparison_67] reconstruction error curves", flush=True)
        for group, group_title, group_runs in GROUPS:
            for variable, ylabel in (("daily_range_bias_case", "Daily range bias (K)"),
                                     ("asymmetry_bias_case", "Diurnal asymmetry bias"),
                                     ("peak_time_ccrps_case", "Peak-time circular CRPS (h)"),
                                     ("tmax_reaggregation_bias_case", "Re-aggregated Tmax bias (K)"),
                                     ("tmean_reaggregation_bias_case", "Re-aggregated Tmean bias (K)"),
                                     ("tmin_reaggregation_bias_case", "Re-aggregated Tmin bias (K)")):
                curves = []
                for run in group_runs:
                    if run not in datasets or variable not in datasets[run]:
                        continue
                    series = case_series(datasets[run], variable, "day_idx")
                    monthly = series.groupby(series.index.month).mean()
                    curves.append((REGION_LABEL.get(run, label_of[run])
                                   if group == "comparison_7" else label_of[run],
                                   monthly.index.values, monthly.values))
                if curves:
                    figure = build_lead_curve_figure(curves, "Month of 2021", ylabel,
                                                     f"{ylabel.split(' (')[0]} by month")
                    figure.savefig(output_directory / f"reconstruction_{variable}_{group}.png",
                                   dpi=200)
                    plt.close(figure)

        print("[comparison_67] week trajectories and maps", flush=True)
        week_start = pd.Timestamp(args.week_start)
        week_end = week_start + pd.Timedelta(days=7)
        map_day = pd.Timestamp(args.map_date)
        for group, group_title, group_runs in GROUPS:
            for run in group_runs:
                store, suffix, label, region = EXP4_RUNS[run]
                path = args.scratch / "Predictions" / store / f"predictions_{store}_{STAGE}{suffix}.nc"
                if not path.is_file():
                    continue
                # Each region gets its own truth on its own crop: never Europe's.
                loading = (region[0] - BOUNDARY_MARGIN_DEGREES, region[1] + BOUNDARY_MARGIN_DEGREES,
                           region[2] - BOUNDARY_MARGIN_DEGREES, region[3] + BOUNDARY_MARGIN_DEGREES)
                hourly = to_scoring_grid(load(str(week_start.date()), str(week_end.date()),
                                              variables=[TARGET_VAR], region=loading,
                                              freq="1h")[TARGET_VAR], region)
                truth_series = hourly.weighted(np.cos(np.deg2rad(hourly.lat))).mean(
                    ["lat", "lon"]).sel(time=slice(week_start, week_end)).compute()
                series = domain_mean_series(path, week_start, week_end, region=region)
                figure = build_domain_mean_figure(
                    [("ERA5 truth", pd.DatetimeIndex(truth_series.time.values), truth_series.values),
                     (label, series.index, series.values)],
                    f"{label} - area-mean 2 m temperature, "
                    f"{week_start.strftime('%d')}-{week_end.strftime('%d %B %Y')}")
                figure.savefig(output_directory / f"week_domain_mean_{group}_{run}.png", dpi=200)
                plt.close(figure)

                # Keep the existing full-region map.  On the selected calendar date,
                # use Denmark's warmest area-mean hour for Europe and the region's
                # own warmest area-mean hour for each transfer map.
                selector = (hourly.sel(lat=slice(DENMARK_CALENDAR_REGION[0], DENMARK_CALENDAR_REGION[1]),
                                       lon=slice(DENMARK_CALENDAR_REGION[2], DENMARK_CALENDAR_REGION[3]))
                            if region == EUROPE else hourly)
                selector_weights = np.cos(np.deg2rad(selector.lat))
                selector_mean = selector.weighted(selector_weights).mean(["lat", "lon"])
                map_time = pd.Timestamp(selector_mean.sel(
                    time=slice(map_day, map_day + pd.Timedelta(hours=23))).idxmax("time").compute().item())
                stamp = map_time.strftime("%d %B %Y %H UTC")
                selected_area = "Denmark-area mean" if region == EUROPE else "regional mean"
                map_truth = hourly.sel(time=map_time).compute()
                field = field_at(path, map_time, region=region)
                figure = build_map_grid_figure([("", map_truth), (label, field)],
                                               "ERA5", f"2 m temperature, {stamp}")
                figure.savefig(output_directory / f"maps_{args.map_date}_{group}_{run}.png",
                               dpi=200)
                plt.close(figure)
                figure = build_map_grid_figure([(label, field - map_truth)],
                                               "ERA5", f"Model minus ERA5, {stamp}",
                                               difference=True)
                figure.savefig(output_directory / f"maps_error_{args.map_date}_{group}_{run}.png",
                               dpi=200)
                plt.close(figure)
    finally:
        for dataset in datasets.values():
            dataset.close()
    print(output_directory)


if __name__ == "__main__":
    main()
