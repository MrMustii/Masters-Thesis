"""Comparisons 4 and 5 — exp3 noise design and continuous-reconstruction horizon.

Comparison numbers run 0 to 7 across five files and do not line up with the experiment
numbers, because one experiment can raise more than one question. These are comparisons 4
and 5, answering RQ2 from the exp3 and baseline caches.

Reads compact caches only, apart from the raw hours the maps and week trajectory need.
All exp3 runs and both hourly baselines share one layout — 00 UTC daily starts, hourly
leads to 168 h — so everything pairs directly on initialisation keys.

Comparison 4 asks which noise construction is best, against fixed noise. Marginal hourly
scores cannot answer it: designs with the same marginal variance give the same hourly
CRPS and different trajectories. The temporal variogram is the discriminating metric and
calibration is read beside it.

Comparison 5 asks how long the reconstruction stays useful. That answer lives in the lead
curves, and the baselines define it: climatology is flat by construction, so the lead at
which a model's curve crosses it is the usable horizon.
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

from modified_Gencast.comparison_1 import domain_mean_series, field_at
from modified_Gencast.lumio_buckets import get_loading_region, load
from modified_Gencast.metrics import (
    DENMARK_CALENDAR_REGION, TARGET_VAR, relative_economic_value, to_scoring_grid,
)
from modified_Gencast.plots import (
    build_domain_mean_figure,
    build_lead_curve_figure,
    build_map_grid_figure,
    build_paired_forest_figure,
    build_pairwise_significance_figure,
    build_reliability_figure,
    build_stacked_rev_figure,
)
STAGE = "step_00016000_2021-01-01_2021-12-31_AR168h"
# run key -> (store directory, cached-tag suffix, reader-facing label)
EXP3_RUNS = {
    "fixed": ("experiment_3_1day_stage2", "fixed_churn0.0_margin8.5", "Fixed noise"),
    "fixed_churn": ("experiment_3_1day_stage2", "fixed_churn2.5_margin8.5", "Fixed noise with churn"),
    "ou_rho1": ("experiment_3_1day_stage2", "ou_rho1.0_churn0.0_margin8.5", "Correlated noise rho 1.0"),
    "ou_rho2.3": ("experiment_3_1day_stage2", "ou_rho2.302585093_churn0.0_margin8.5", "Correlated noise rho 2.3"),
    "ou_rho10000": ("experiment_3_1day_stage2", "ou_rho10000.0_churn0.0_margin8.5", "Correlated noise rho 10000"),
    "1day": ("experiment_3_1day_stage2", "fixed_churn0.0_margin8.5", "One-day model"),
    "3day": ("experiment_3_3day_stage2", "fixed_churn0.0_margin8.5", "Three-day model"),
    "7day": ("experiment_3_7day_stage2", "fixed_churn0.0_margin8.5", "Seven-day model"),
}
from modified_Gencast.significance import (
    apply_multiplicity_correction,
    bootstrap_paired_difference,
    bootstrap_paired_metric,
    family_block_length,
)


REGION = (45.0, 65.0, -5.0, 25.0)
BASELINES = {
    "climatology_hourly": "ERA5 average-year climatology",
    "persistence_hourly": "ERA5 persistence (7-day lag)",
}
CLIMATOLOGY = "climatology_hourly"
QUESTIONS = (
    ("noise_design", "fixed", ("fixed_churn", "ou_rho1", "ou_rho2.3", "ou_rho10000")),
    ("horizon", "1day", ("3day", "7day")),
)
GROUPS = (
    ("comparison_4", "Noise design", 
     ("fixed", "fixed_churn", "ou_rho1", "ou_rho2.3", "ou_rho10000")),
    ("comparison_5", "Reconstruction horizon", ("1day", "3day", "7day")),
)
# (variable, case dimension, reader-facing label, unit)
METRICS = (
    ("t2m_crps_case", "hour", "2 m CRPS", "K"),
    ("temporal_variogram_case", "day_idx", "Temporal variogram score", ""),
    ("t2m_twcrps_case", "hour", "2 m twCRPS", "K"),
    ("peak_time_ccrps_case", "day_idx", "Peak-time circular CRPS", "h"),
    ("daily_range_crps_case", "day_idx", "Daily range CRPS", "K"),
    ("asymmetry_crps_case", "day_idx", "Diurnal asymmetry CRPS", ""),
    ("hours_above_threshold_crps_case", "day_idx", "Hours-above-threshold CRPS", "h"),
    ("tmax_reaggregation_crps_case", "day_idx", "Re-aggregated Tmax CRPS", "K"),
    ("tmean_reaggregation_crps_case", "day_idx", "Re-aggregated Tmean CRPS", "K"),
    ("tmin_reaggregation_crps_case", "day_idx", "Re-aggregated Tmin CRPS", "K"),
)
# Both tiers run under Holm at alpha=0.05; the tier only decides which family a test is
# corrected in. Re-aggregated Tmax is primary because it is the only Tmax these runs
# produce, and Tmax is what the heatwave question rests on. Tmean and Tmin re-aggregation
# stay supporting: they are the honesty check on inputs the model was handed.
SECONDARY_METRICS = ("Re-aggregated Tmean CRPS", "Re-aggregated Tmin CRPS",
                     "Brier skill score", "F1", "Spread-skill calibration |SSR - 1|")
METRIC_TIER = {label: ("secondary" if label in SECONDARY_METRICS else "primary")
               for _, _, label, _ in METRICS}
METRIC_TIER.update({label: "secondary" for label in SECONDARY_METRICS})
LEAD_CURVES = (
    ("t2m_crps_case", "hour", "Lead time (hours)", "2 m CRPS (K)"),
    ("t2m_twcrps_case", "hour", "Lead time (hours)", "2 m twCRPS (K)"),
    ("temporal_variogram_case", "day_idx", "Lead day", "Temporal variogram score"),
    ("peak_time_ccrps_case", "day_idx", "Lead day", "Peak-time circular CRPS (h)"),
    ("daily_range_bias_case", "day_idx", "Lead day", "Daily range bias (K)"),
    ("asymmetry_bias_case", "day_idx", "Lead day", "Diurnal asymmetry bias"),
    ("hours_above_threshold_crps_case", "day_idx", "Lead day", "Hours-above-threshold CRPS (h)"),
    ("tmax_reaggregation_crps_case", "day_idx", "Lead day", "Re-aggregated Tmax CRPS (K)"),
    # Signed re-aggregation biases: which way the rebuilt daily statistics are wrong.
    ("tmax_reaggregation_bias_case", "day_idx", "Lead day", "Re-aggregated Tmax bias (K)"),
    ("tmean_reaggregation_bias_case", "day_idx", "Lead day", "Re-aggregated Tmean bias (K)"),
    ("tmin_reaggregation_bias_case", "day_idx", "Lead day", "Re-aggregated Tmin bias (K)"),
    ("target_hours_above_threshold_case", "day_idx", "Lead day", "Observed hours above p90 (h)"),
)
BIAS_COLUMNS = ("daily_range_bias_case", "asymmetry_bias_case", "tmax_reaggregation_bias_case",
                "tmean_reaggregation_bias_case", "tmin_reaggregation_bias_case")
HORIZONS = (("full", "full forecast"), ("day1", "day 1"))


def members_of(dataset):
    """Ensemble size from the probability-bin axis; xr.merge drops the attribute."""
    return int(dataset.sizes.get("probability_bin", 2)) - 1


def case_series(dataset, metric, case_dimension, horizon):
    """One score per initialisation for the requested window, keyed by initialisation."""
    values = dataset[metric]
    if horizon == "day1":
        if case_dimension == "hour":
            values = values.isel(hour=values.hour.values < values.hour.values.min() + 24)
        else:
            values = values.sel(day_idx=[int(values.day_idx.values.min())])
    collapsed = values.mean(case_dimension)
    return pd.Series(np.asarray(collapsed.values, dtype=float),
                     index=pd.DatetimeIndex(collapsed.forecast.values))


def event_window(dataset, horizon):
    """Event primitives restricted to the requested window."""
    if horizon == "day1":
        return dataset.sel(day_idx=[int(dataset.day_idx.values.min())])
    return dataset


def spread_and_skill(dataset, horizon):
    """Ensemble spread, RMSE and their ratio; RMSE is sqrt of the mean squared error."""
    selected = dataset
    if horizon == "day1":
        hours = dataset.hour.values
        selected = dataset.isel(hour=hours < hours.min() + 24)
    rmse = np.sqrt(float(selected["squared_error_case"].mean()))
    members = members_of(dataset)
    if members < 2:
        return float("nan"), rmse, float("nan")
    spread = np.sqrt(float(selected["ensemble_variance_case"].mean()) * (members + 1) / (members - 1))
    return spread, rmse, spread / rmse


def main():
    """Write each comparison's CSVs and figures into its own directory.

    baseline_audit.csv records which baseline cache each run was paired against.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/scratch/project_465002687/verified_metrics"))
    parser.add_argument("--scratch", type=Path, default=Path("/scratch/project_465002687"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--map-date", type=str, default="2021-06-20",
                        help="Calendar date (YYYY-MM-DD); map its warmest Denmark-area-mean hour.")
    parser.add_argument("--week-start", type=str, default="2021-06-16")
    # Redraw the figures from the caches without repeating the bootstrap, for when
    # only a label or a layout changed. The significance CSVs are left untouched.
    parser.add_argument("--figures-only", action="store_true")
    args = parser.parse_args()
    output_directory = args.out or Path.cwd() / "comparison_45"
    output_directory.mkdir(parents=True, exist_ok=True)

    datasets = {}
    baseline_audit = []
    for run in EXP3_RUNS:
        path = args.root / "reports" / "exp3" / run / "metric_cases.nc"
        if path.is_file():
            datasets[run] = xr.open_dataset(path)
    for run, label in BASELINES.items():
        path = args.root / "reports" / "baselines" / run / "metric_cases.nc"
        if path.is_file():
            datasets[run] = xr.open_dataset(path)
        baseline_audit.append({"baseline": label, "cache": str(path), "available": path.is_file(),
                               "reason": "" if path.is_file() else "run report_baselines for this grid"})
    missing = [run for run in (*EXP3_RUNS, *BASELINES) if run not in datasets]
    if missing:
        print(f"Missing caches, those rows are omitted: {missing}")
    label_of = {**{run: label for run, (_, _, label) in EXP3_RUNS.items()}, **BASELINES}

    try:
        print("[comparison_45] paired bootstrap", flush=True)
        significance_rows, score_rows = [], []
        for horizon, horizon_label in (() if args.figures_only else HORIZONS):
            for metric, case_dimension, metric_label, unit in METRICS:
                series = {run: case_series(dataset, metric, case_dimension, horizon)
                          for run, dataset in datasets.items() if metric in dataset}
                # One date set for the whole family. Scoring each pair on its own
                # intersection would let different pairs rest on different forecasts, so
                # their differences would not be comparable; exp3's three-day model carries
                # 355 cases against the one-day model's 357, so the pairs really do differ.
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
                family_differences = [
                    (series[candidate_run] - series[reference_run]).dropna().values
                    for _question, reference_run, candidate_runs in QUESTIONS
                    if reference_run in series
                    for candidate_run in (*candidate_runs, *BASELINES) if candidate_run in series]
                block_length = family_block_length(family_differences)
                print(f"  {metric_label} / {horizon_label}: {len(common_dates)} shared "
                      f"initialisations, block length {block_length:.2f}", flush=True)
                climatology = series.get(CLIMATOLOGY)
                for run, values in series.items():
                    score_rows.append({
                        "run": label_of[run], "metric": metric_label, "horizon": horizon_label,
                        "mean_score": float(values.mean()), "unit": unit,
                        "skill_versus_climatology": (
                            float(1.0 - values.mean() / climatology.mean())
                            if climatology is not None else float("nan")),
                        "n_initialisations": int(values.notna().sum()),
                    })
                for question, reference_run, candidate_runs in QUESTIONS:
                    if reference_run not in series:
                        continue
                    for candidate_run in (*candidate_runs, *BASELINES):
                        if candidate_run not in series:
                            continue
                        paired = pd.DataFrame({"candidate": series[candidate_run],
                                               "reference": series[reference_run]}).dropna()
                        result = bootstrap_paired_difference(
                            (paired["candidate"] - paired["reference"]).values,
                            num_resamples=args.resamples, block_length=block_length)
                        significance_rows.append({
                            "question": question, "candidate": label_of[candidate_run],
                            "reference": label_of[reference_run], "metric": metric_label,
                            "horizon": horizon_label,
                            "tier": METRIC_TIER[metric_label], "family": question,
                            "estimate_candidate": float(paired["candidate"].mean()),
                            "estimate_reference": float(paired["reference"].mean()),
                            "delta_candidate_minus_reference": float(result["point_estimate"]),
                            "ci_low": float(result["lower_bound"]),
                            "ci_high": float(result["upper_bound"]),
                            "p_value": float(result["p_value"]),
                            "block_length": float(result["block_length"]),
                            "block_length_method": result["block_length_method"],
                            "n_initialisations": int(len(paired)), "resamples": args.resamples,
                            "direction": ("candidate better" if result["upper_bound"] < 0
                                          else "reference better" if result["lower_bound"] > 0
                                          else "not distinguishable"),
                            "is_baseline": candidate_run in BASELINES,
                        })

        print("[comparison_45] heatwave diagnostics", flush=True)
        diagnostic_rows, reliability_curves, rev_curves = [], [], []
        for run, dataset in datasets.items():
            for horizon, horizon_label in HORIZONS:
                window = event_window(dataset, horizon)
                case_dimensions = [d for d in ("forecast", "day_idx") if d in window["brier_score_heatwave"].dims]
                positives = window["true_positive_share_heatwave"].mean(case_dimensions)
                false_positives = window["false_positive_share_heatwave"].mean(case_dimensions)
                false_negatives = window["false_negative_share_heatwave"].mean(case_dimensions)
                warned = int(np.argmin(np.abs(window.decision_probability_threshold.values - 0.5)))
                event_rate = float(window["event_rate_heatwave"].mean())
                spread, rmse, ratio = spread_and_skill(dataset, horizon)
                row = {"run": label_of[run], "horizon": horizon_label,
                       "brier_score": float(window["brier_score_heatwave"].mean()), "event_rate": event_rate,
                       "auc": float(window["auc_case_heatwave"].mean(skipna=True)),
                       "f1": float((2 * positives / (2 * positives + false_positives
                                                     + false_negatives)).values[warned]),
                       "precision": float((positives / (positives + false_positives)).values[warned]),
                       "recall": float((positives / (positives + false_negatives)).values[warned]),
                       "ensemble_spread": spread, "rmse": rmse, "spread_skill_ratio": ratio,
                       "conditional_absolute_error": float(
                           window["conditional_abs_error_sum_heatwave"].sum()
                           / window["conditional_error_support_heatwave"].sum()),
                       "is_baseline": run in BASELINES}
                for column in BIAS_COLUMNS:
                    row[column.replace("_case", "")] = (float(window[column].mean())
                                                        if column in window else float("nan"))
                diagnostic_rows.append(row)
                if run in BASELINES:
                    continue
                if horizon == "day1":
                    bin_count = window["probability_bin_count_threshold"].sum(case_dimensions)
                    bin_events = window["probability_bin_event_count_threshold"].sum(case_dimensions)
                    reliability_curves.append((run, label_of[run], window.forecast_probability.values,
                                               (bin_events / bin_count.where(bin_count > 0)).values))
                else:
                    threshold_positives = window["true_positive_share_threshold"].mean(case_dimensions)
                    threshold_false_positives = window["false_positive_share_threshold"].mean(case_dimensions)
                    threshold_false_negatives = window["false_negative_share_threshold"].mean(case_dimensions)
                    threshold_event_rate = float(window["event_rate_threshold"].mean())
                    rev_curves.append((run, label_of[run], np.linspace(0.01, 0.99, 99),
                                       relative_economic_value(
                                           threshold_event_rate,
                                           (threshold_positives + threshold_false_positives).values,
                                           threshold_false_negatives.values)))
        climatology_brier = {row["horizon"]: row["brier_score"] for row in diagnostic_rows
                             if row["run"] == BASELINES[CLIMATOLOGY]}
        for row in diagnostic_rows:
            row["brier_skill_score"] = float(
                1.0 - row["brier_score"] / climatology_brier.get(row["horizon"], float("nan")))

        print("[comparison_45] nonlinear metric intervals", flush=True)
        # BSS, F1 and SSR are nonlinear in their components, so each resample averages the
        # per-case components and recomputes the metric; averaging per-case metric values
        # would be wrong. GenCast Table E2.
        def components_of(run, horizon):
            window = event_window(datasets[run], horizon)
            # Climatology spans every case; take only the ones this run actually has.
            climatology_window = event_window(
                datasets[CLIMATOLOGY].sel(forecast=window.forecast), horizon)
            warned = int(np.argmin(np.abs(window.decision_probability_threshold.values - 0.5)))
            hourly = datasets[run]
            if horizon == "day1":
                hours = hourly.hour.values
                hourly = hourly.isel(hour=hours < hours.min() + 24)
            # Keyed by initialisation so candidate and reference can be intersected: the
            # three-day model carries 355 cases against the one-day model's 357.
            return pd.DataFrame({
                "brier": window["brier_score_heatwave"].mean("day_idx").values,
                "climatology_brier": climatology_window["brier_score_heatwave"].mean("day_idx").values,
                "true_positive": window["true_positive_share_heatwave"].isel(decision=warned).mean("day_idx").values,
                "false_positive": window["false_positive_share_heatwave"].isel(decision=warned).mean("day_idx").values,
                "false_negative": window["false_negative_share_heatwave"].isel(decision=warned).mean("day_idx").values,
                "variance": hourly["ensemble_variance_case"].mean("hour").values,
                "squared_error": hourly["squared_error_case"].mean("hour").values,
            }, index=pd.DatetimeIndex(window.forecast.values))

        def spread_skill(components, members):
            return (float("nan") if members < 2 else
                    float(np.sqrt(components["variance"] * (members + 1) / (members - 1))
                          / np.sqrt(components["squared_error"])))

        # Each metric carries the columns its own serial dependence lives in. Without that
        # the block length falls back to the first component alphabetically ("brier"), so
        # F1 and spread-skill would both be blocked on Brier-score differences, which is
        # unrelated to either. Spread-skill is tested as |SSR - 1|: the ideal is 1, not
        # zero, so a difference of raw ratios cannot say which run is better calibrated —
        # 0.8 and 1.2 are equally wrong but differ by -0.4.
        for horizon, horizon_label in (() if args.figures_only else HORIZONS):
            components = {run: components_of(run, horizon) for run in datasets}
            common_dates = None
            for frame in components.values():
                common_dates = (frame.index if common_dates is None
                                else common_dates.intersection(frame.index))
            if common_dates is None or len(common_dates) < 3:
                continue
            components = {run: frame.loc[common_dates] for run, frame in components.items()}
            for metric_label, metric_builder, proxy_columns in (
                    ("Brier skill score",
                     lambda members: lambda c: float(1.0 - c["brier"] / c["climatology_brier"]),
                     ("brier",)),
                    ("F1",
                     lambda members: lambda c: float(2 * c["true_positive"]
                                                     / (2 * c["true_positive"] + c["false_positive"]
                                                        + c["false_negative"] + 1e-12)),
                     ("true_positive", "false_positive", "false_negative")),
                    ("Spread-skill calibration |SSR - 1|",
                     lambda members: lambda c: abs(spread_skill(c, members) - 1.0),
                     ("variance", "squared_error"))):
                pairs = [(question, reference_run, candidate_run)
                         for question, reference_run, candidate_runs in QUESTIONS
                         if reference_run in components
                         for candidate_run in (*candidate_runs, *BASELINES)
                         if candidate_run in components
                         and not (metric_label.startswith("Spread-skill")
                                  and min(members_of(datasets[candidate_run]),
                                          members_of(datasets[reference_run])) < 2)]
                if not pairs:
                    continue
                block_length = family_block_length([
                    components[candidate_run][list(proxy_columns)].sum(axis=1).values
                    - components[reference_run][list(proxy_columns)].sum(axis=1).values
                    for _question, reference_run, candidate_run in pairs])
                print(f"  {metric_label} / {horizon_label}: {len(common_dates)} shared "
                      f"initialisations, block length {block_length:.2f}", flush=True)
                for question, reference_run, candidate_run in pairs:
                    proxy = (components[candidate_run][list(proxy_columns)].sum(axis=1).values
                             - components[reference_run][list(proxy_columns)].sum(axis=1).values)
                    result = bootstrap_paired_metric(
                        {name: components[candidate_run][name].values
                         for name in components[candidate_run]},
                        {name: components[reference_run][name].values
                         for name in components[reference_run]},
                        metric_builder(members_of(datasets[candidate_run])), proxy=proxy,
                        num_resamples=args.resamples, block_length=block_length)
                    significance_rows.append({
                        "question": question, "candidate": label_of[candidate_run],
                        "reference": label_of[reference_run], "metric": metric_label,
                        "horizon": horizon_label,
                        "tier": METRIC_TIER[metric_label], "family": question,
                        "estimate_candidate": float("nan"), "estimate_reference": float("nan"),
                        "delta_candidate_minus_reference": float(result["point_estimate"]),
                        "ci_low": float(result["lower_bound"]),
                        "ci_high": float(result["upper_bound"]),
                        "p_value": float(result["p_value"]),
                        "block_length": float(result["block_length"]),
                        "block_length_selected": float(result["block_length_selected"]),
                        "block_length_method": result["block_length_method"],
                        "n_initialisations": len(common_dates),
                        "resamples": args.resamples,
                        "direction": ("candidate better" if result["lower_bound"] > 0
                                      else "reference better" if result["upper_bound"] < 0
                                      else "not distinguishable"),
                        "is_baseline": candidate_run in BASELINES,
                    })

        apply_multiplicity_correction(significance_rows)

        # Every run against every other, so a reader can tell any two apart and not only
        # each against its reference. One family per (group, horizon, metric): the groups
        # are separate tables, and pooling them would correct over tests that never appear
        # side by side.
        print("[comparison_45] pairwise matrices", flush=True)
        pairwise_rows = []
        pairwise_directory = output_directory / "pairwise"
        pairwise_directory.mkdir(parents=True, exist_ok=True)
        for group, group_title, group_runs in (() if args.figures_only else GROUPS):
            # Baselines join the matrix so the table can carry them as rows; that makes the
            # family larger, so Holm corrects over more pairs than the models alone.
            present = [run for run in (*group_runs, *BASELINES) if run in datasets]
            if len(present) < 3:
                continue
            for metric, case_dimension, metric_label, unit in METRICS:
                for horizon, horizon_label in HORIZONS:
                    series_of = {run: case_series(datasets[run], metric, case_dimension, horizon)
                                 for run in present if metric in datasets[run]}
                    labels = [label_of[run] for run in series_of]
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
                    series_of = {run: values.reindex(common_dates)
                                 for run, values in series_of.items()}
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
                            result = bootstrap_paired_difference(
                                (paired["row"] - paired["column"]).values,
                                num_resamples=args.resamples, block_length=block_length)
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

        print("[comparison_45] forests", flush=True)
        forest_directory = output_directory / "forest"
        forest_directory.mkdir(parents=True, exist_ok=True)
        for question, _, _ in QUESTIONS:
            for metric, _, metric_label, unit in METRICS:
                for horizon, horizon_label in HORIZONS:
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

        print("[comparison_45] lead curves, reliability and REV", flush=True)
        for group, group_title, group_runs in GROUPS:
            for variable, case_dimension, xlabel, ylabel in LEAD_CURVES:
                curves = [(label_of[run], datasets[run][case_dimension].values,
                           datasets[run][variable].mean("forecast").values)
                          for run in (*group_runs, *BASELINES)
                          if run in datasets and variable in datasets[run]]
                if curves:
                    figure = build_lead_curve_figure(curves, xlabel, ylabel,
                                                     f"{ylabel.split(' (')[0]} over lead time")
                    figure.savefig(output_directory / f"lead_{variable}_{group}.png", dpi=200)
                    plt.close(figure)
            # Deterministic baselines have no spread, so SSR carries models only.
            curves = []
            for run in group_runs:
                dataset = datasets.get(run)
                if dataset is None or members_of(dataset) < 2:
                    continue
                members = members_of(dataset)
                spread = np.sqrt(dataset["ensemble_variance_case"].mean("forecast").values
                                 * (members + 1) / (members - 1))
                curves.append((label_of[run], dataset.hour.values,
                               spread / np.sqrt(dataset["squared_error_case"].mean("forecast").values)))
            if curves:
                figure = build_lead_curve_figure(curves, "Lead time (hours)", "Spread-skill ratio",
                                                 "Spread-skill ratio of RMSE over lead time")
                figure.savefig(output_directory / f"lead_spread_skill_{group}.png", dpi=200)
                plt.close(figure)

            selected = [(label, *rest) for run, label, *rest in reliability_curves if run in group_runs]
            if selected:
                figure = build_reliability_figure(
                    selected, "Day-one daily p90-exceedance reliability")
                figure.savefig(output_directory / f"reliability_day1_{group}.png", dpi=200)
                plt.close(figure)
            selected = [(label, *rest) for run, label, *rest in rev_curves if run in group_runs]
            if selected:
                figure = build_stacked_rev_figure(
                    selected, "Heatwave relative economic value")
                figure.savefig(output_directory / f"rev_full_{group}.png", dpi=200)
                plt.close(figure)

        print("[comparison_45] week trajectory and maps", flush=True)
        loading_region = get_loading_region(json.loads(
            Path("modified_Gencast/traninig configs/experiment_2_all_gencast.json").read_text())["task_config"])
        week_start = pd.Timestamp(args.week_start)
        week_end = week_start + pd.Timedelta(days=7)
        hourly = to_scoring_grid(load(str(week_start.date()), str(week_end.date()),
                                      variables=[TARGET_VAR], region=loading_region,
                                      freq="1h")[TARGET_VAR], REGION)
        truth_series = hourly.weighted(np.cos(np.deg2rad(hourly.lat))).mean(["lat", "lon"]).sel(
            time=slice(week_start, week_end)).compute()
        # Keep the original map date and full-Europe layout. Only the plotted hour
        # changes: it is Denmark's warmest area-mean hour on that same calendar day.
        map_day = pd.Timestamp(args.map_date)
        denmark_hourly = hourly.sel(
            lat=slice(DENMARK_CALENDAR_REGION[0], DENMARK_CALENDAR_REGION[1]),
            lon=slice(DENMARK_CALENDAR_REGION[2], DENMARK_CALENDAR_REGION[3]))
        denmark_weights = np.cos(np.deg2rad(denmark_hourly.lat))
        denmark_mean = denmark_hourly.weighted(denmark_weights).mean(["lat", "lon"])
        map_time = pd.Timestamp(denmark_mean.sel(
            time=slice(map_day, map_day + pd.Timedelta(hours=23))).idxmax("time").compute().item())
        map_truth = hourly.sel(time=map_time).compute()
        map_description = "warmest Denmark-area-mean hour on this date"
        stamp = map_time.strftime("%d %B %Y %H UTC")
        for group, group_title, group_runs in GROUPS:
            curves = [("ERA5 truth", pd.DatetimeIndex(truth_series.time.values), truth_series.values)]
            map_fields = []
            for run in group_runs:
                store, suffix, label = EXP3_RUNS[run]
                path = args.scratch / "Predictions" / store / f"predictions_{store}_{STAGE}_{suffix}.nc"
                if not path.is_file():
                    continue
                series = domain_mean_series(path, week_start, week_end)
                curves.append((label, series.index, series.values))
                map_fields.append((label, field_at(path, map_time)))
            if len(curves) > 1:
                figure = build_domain_mean_figure(
                    curves, f"Area-mean 2 m temperature, "
                            f"{week_start.strftime('%d')}-{week_end.strftime('%d %B %Y')}")
                figure.savefig(output_directory / f"week_domain_mean_{group}.png", dpi=200)
                plt.close(figure)
            if map_fields:
                figure = build_map_grid_figure([("", map_truth), *map_fields], "ERA5",
                                               f"2 m temperature, {stamp}")
                figure.savefig(output_directory / f"maps_{args.map_date}_{group}.png", dpi=200)
                plt.close(figure)
                figure = build_map_grid_figure(
                    [(label, field - map_truth) for label, field in map_fields], "ERA5",
                    f"Model minus ERA5, {stamp}", difference=True)
                figure.savefig(output_directory / f"maps_error_{args.map_date}_{group}.png", dpi=200)
                plt.close(figure)
    finally:
        for dataset in datasets.values():
            dataset.close()
    print(output_directory)


if __name__ == "__main__":
    main()
