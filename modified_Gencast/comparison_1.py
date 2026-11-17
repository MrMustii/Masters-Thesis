"""Comparison 1 — pretrained global GenCast versus the regional models, day one only.

Comparison numbers run 0 to 7 across five files and do not line up with the experiment
numbers, because one experiment can raise more than one question. This is comparison 1,
answering the first half of RQ1 from the exp1, exp2 and baseline caches.

Reads compact report caches only; it never opens a raw prediction grid. The
global reference is valid-time indexed and the regional runs are forecast/lead
indexed, so every pairing goes through the valid_time and valid_day variables
stored in each cache. Day one is the only honest shared window: the global
source supplies no matching regional forecast sequence beyond it.
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

from modified_Gencast.lumio_buckets import get_loading_region, load
from modified_Gencast.metrics import (
    DENMARK_CALENDAR_EXTRA_CELLS, DENMARK_CALENDAR_REGION, TARGET_VAR, relative_economic_value, to_scoring_grid)
from modified_Gencast.plots import (
    build_calendar_skill_figure,
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
# Short display labels: they double as CSV row keys and as plot panel/legend text.
REFERENCE = ("reports/exp1/exp1/metric_cases.nc", "GenCast")
MODELS = (
    ("reports/exp1/exp1_original_weights/metric_cases.nc", "Regional GenCast"),
    ("reports/exp2/retrained_all/metric_cases.nc", "Retrained model"),
)
BASELINES = (
    ("reports/baselines/climatology/metric_cases.nc", "Climatology"),
    # 1-day lag, not the 7-day one: comparison_1 scores day one, so the honest
    # persistence reference is yesterday's observation, not last week's.
    ("reports/baselines/persistence_24h/metric_cases.nc", "Persistence (1-day)"),
)
CLIMATOLOGY_LABEL = BASELINES[0][1]
# (cache suffix, display label) — every heatwave-conditioned diagnostic is reported
# under both: a single day above threshold, and every day of a run of >= 3
# consecutive days above it.
HEATWAVE_DEFINITIONS = (
    ("threshold", "Threshold (single day)"),
    ("heatwave", "Heatwave (≥3 consecutive days)"),
)
# (cache variable, lead dimension, display label, unit, bootstrap kind)
# "linear": paired difference of per-case means (bootstrap_paired_difference).
# "rmse": nonlinear sqrt-of-mean-squared-error (bootstrap_paired_metric, GenCast Table E2 style).
PRIMARY_METRICS = (
    ("tmax_crps_case", "lead_day", "Tmax CRPS", "K", "linear"),
    ("tmax_twcrps_case", "lead_day", "Tmax twCRPS", "K", "linear"),
    ("brier_score_heatwave", "lead_day", "Heatwave BS", "", "linear"),
)
SECONDARY_METRICS = (
    ("t2m_crps_case", "hour", "t2m CRPS", "K", "linear"),
    ("t2m_twcrps_case", "hour", "t2m twCRPS", "K", "linear"),
    ("squared_error_case", "hour", "t2m RMSE", "K", "rmse"),
)
METRICS = PRIMARY_METRICS + SECONDARY_METRICS
METRIC_TIER = {label: "primary" for _, _, label, _, _ in PRIMARY_METRICS}
METRIC_TIER.update({label: "secondary" for _, _, label, _, _ in SECONDARY_METRICS})


def day_one_cases(dataset, metric, lead_dimension):
    """Day-one score per initialisation, plus the [forecast, lead] stamps each case covers."""
    values = dataset[metric]
    if lead_dimension == "hour":
        # Day one is every lead inside the first 24 h, i.e. hours 12 and 24 of a 12 UTC start.
        within_day_one = values.hour.values < values.hour.values.min() + 24
        scores = values.isel(hour=within_day_one).mean("hour")
        stamps = dataset["valid_time"].isel(hour=within_day_one).values
    else:
        first_lead_day = int(values.lead_day.values.min())
        scores = values.sel(lead_day=first_lead_day)
        stamps = dataset["valid_day"].sel(lead_day=first_lead_day).values[:, None]
    forecast = pd.DatetimeIndex(scores.forecast.values)
    return pd.Series(np.asarray(scores.values, dtype=float), index=forecast), stamps


def global_cases(dataset, metric, stamps, forecast):
    """Global valid-time scores averaged onto the regional initialisation keys."""
    values = dataset[metric]
    coordinate = "time" if "time" in values.dims else "day"
    indexed = pd.Series(np.asarray(values.values, dtype=float),
                        index=pd.DatetimeIndex(values[coordinate].values))
    # One lookup for every stamp, then average the day-one leads of each case.
    mapped = indexed.reindex(pd.DatetimeIndex(stamps.ravel())).values.reshape(stamps.shape)
    return pd.DataFrame(mapped, index=forecast).mean(axis=1)


RAW_FIELDS = {
    "Global pretrained": "Predictions/exp1/predictions_2021-01-01_2021-12-31.zarr",
    "Regional pretrained":
        "Predictions/exp1_original_weights/predictions_exp1_original_weights_0.25_2021-01-01_2021-12-31_ar14_freerun.nc",
    "Regional retrained (all vars)":
        "Predictions/experiment_2_all_gencast_fixed_stage2/"
        "predictions_experiment_2_all_gencast_fixed_stage2_step_00016000_2021-01-01_2021-12-31_ar14.nc",
}


def field_at(path, valid_time, member=0, region=REGION):
    """One member's field valid at the requested time, from a raw prediction cache.

    A fixed-lead cache holds several (forecast, lead) cases with that validity; the
    shortest lead is taken so every run is shown at its day-one skill.
    """
    store = (xr.open_zarr(path, consolidated=True) if str(path).endswith(".zarr")
             else xr.open_dataset(path))
    try:
        field = to_scoring_grid(store[TARGET_VAR], region).isel(ensemble=member)
        if "time" in field.dims:
            return field.sel(time=valid_time).compute()
        forecast = pd.DatetimeIndex(field.forecast.values)
        hours = np.asarray(field.hour.values, dtype=int)
        valid = forecast.values[:, None] + hours[None, :] * np.timedelta64(1, "h")
        matches = np.argwhere(valid == np.datetime64(valid_time))
        case_index, lead_index = matches[int(np.argmin(hours[matches[:, 1]]))]
        return field.isel(forecast=int(case_index), hour=int(lead_index)).compute()
    finally:
        store.close()


def spread_and_skill(dataset):
    """Day-one ensemble spread, RMSE and their ratio from the stored components.

    RMSE is sqrt of the mean squared error, so the metric stays a function of the
    mean of a per-case statistic and bootstraps the way GenCast's Table E2 requires.
    """
    if "ensemble_variance_case" not in dataset:
        return float("nan"), float("nan"), float("nan")
    if "hour" in dataset["ensemble_variance_case"].dims:
        hours = dataset.hour.values
        selected = dataset.isel(hour=hours < hours.min() + 24)
    else:
        selected = dataset
    rmse = np.sqrt(float(selected["squared_error_case"].mean()))
    members = int(dataset.probability_bin.size) - 1
    if members < 2:
        return float("nan"), rmse, float("nan")
    spread = np.sqrt(float(selected["ensemble_variance_case"].mean()) * (members + 1) / (members - 1))
    return spread, rmse, spread / rmse


def main():
    """Write the CSVs and figures.

    paired_significance.csv tests each run against the reference, pairwise_significance.csv
    tests every run against every other, and diagnostics.csv carries one row per run,
    heatwave definition and horizon of the detection and calibration numbers.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/scratch/project_465002687/verified_metrics"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--scratch", type=Path, default=Path("/scratch/project_465002687"))
    parser.add_argument("--map-date", type=str, default="2021-06-20")
    args = parser.parse_args()
    output_directory = args.out or Path.cwd() / "comparison_1"
    output_directory.mkdir(parents=True, exist_ok=True)

    reference_dataset = xr.open_dataset(args.root / REFERENCE[0])
    candidates = []
    for relative_path, label in MODELS + BASELINES:
        path = args.root / relative_path
        is_baseline = (relative_path, label) in BASELINES
        if path.is_file():
            candidates.append((label, xr.open_dataset(path), is_baseline))

    try:
        # Climatology day-one series per metric: the only CRPSS denominator.
        climatology_dataset = next((dataset for label, dataset, _ in candidates
                                    if label == CLIMATOLOGY_LABEL), None)
        climatology_series = {} if climatology_dataset is None else {
            metric: day_one_cases(climatology_dataset, metric, lead_dimension)[0]
            for metric, lead_dimension, _, _, _ in METRICS}

        significance_rows = []
        score_rows = []
        for metric, lead_dimension, metric_label, unit, kind in METRICS:
            cases = {label: day_one_cases(dataset, metric, lead_dimension)
                     for label, dataset, _ in candidates}
            # One date set for the whole family. Scoring each pair on its own
            # intersection would let different pairs rest on different forecasts, so
            # their differences would not be comparable to each other.
            common_dates = None
            for values, _stamps in cases.values():
                dates = values.dropna().index
                common_dates = dates if common_dates is None else common_dates.intersection(dates)
            if common_dates is None or len(common_dates) < 3:
                continue
            candidate_of, reference_of = {}, {}
            for label, (values, stamps) in cases.items():
                # The stamps array is positional, so it has to follow the same
                # reordering the series does before the reference can be mapped.
                order = pd.Series(np.arange(len(values.index)),
                                  index=values.index).reindex(common_dates).values
                candidate_of[label] = values.reindex(common_dates)
                reference_of[label] = global_cases(reference_dataset, metric,
                                                   stamps[order], common_dates)
            # One block length for the family, the median across its pairs. Per-pair
            # Politis-White selection is unstable near its critical band, and that
            # instability alone decided significance between comparable pairs.
            block_length = family_block_length(
                [(candidate_of[label] - reference_of[label]).dropna().values
                 for label in candidate_of])
            print(f"  {metric_label}: {len(common_dates)} shared initialisations, "
                  f"block length {block_length:.2f}", flush=True)
            for label, dataset, is_baseline in candidates:
                paired = pd.DataFrame({"candidate": candidate_of[label],
                                       "reference": reference_of[label]}).dropna()
                if kind == "rmse":
                    result = bootstrap_paired_metric(
                        {"squared_error": paired["candidate"].values},
                        {"squared_error": paired["reference"].values},
                        lambda components: float(np.sqrt(components["squared_error"])),
                        num_resamples=args.resamples, block_length=block_length)
                    estimate_candidate = float(np.sqrt(paired["candidate"].mean()))
                    estimate_reference = float(np.sqrt(paired["reference"].mean()))
                else:
                    result = bootstrap_paired_difference(
                        (paired["candidate"] - paired["reference"]).values,
                        num_resamples=args.resamples, block_length=block_length)
                    estimate_candidate = float(paired["candidate"].mean())
                    estimate_reference = float(paired["reference"].mean())
                significance_rows.append({
                    "candidate": label,
                    "reference": REFERENCE[1],
                    "metric": metric_label,
                    "tier": METRIC_TIER[metric_label],
                    "family": "comparison_1",
                    "window": "day 1",
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
                    "is_baseline": is_baseline,
                })
                denominator = climatology_series.get(metric)
                denominator_mean = (denominator.reindex(paired.index).mean() if denominator is not None else None)
                if kind == "rmse" and denominator_mean is not None:
                    denominator_mean = np.sqrt(denominator_mean)
                score_rows.append({
                    "run": label,
                    "metric": metric_label,
                    "window": "day 1",
                    "mean_score": estimate_candidate,
                    "unit": unit,
                    "crpss_versus_climatology": (float(1.0 - estimate_candidate / denominator_mean)
                                                 if denominator_mean is not None else float("nan")),
                    "n_initialisations": int(len(paired)),
                })

        apply_multiplicity_correction(significance_rows)

        # Every model against every other, not just against the one reference: the
        # reference-only view cannot say whether two candidates differ from each other.
        # Each metric is its own correction family (one question, all its pairs).
        pairwise_rows = []
        pairwise_directory = output_directory / "pairwise"
        pairwise_directory.mkdir(parents=True, exist_ok=True)
        for metric, lead_dimension, metric_label, unit, kind in METRICS:
            # The reference is valid-time indexed, so it has to be mapped onto the
            # candidates' initialisation keys before any of them can be paired.
            anchor_series, anchor_times = day_one_cases(candidates[0][1], metric, lead_dimension)
            series_of = {REFERENCE[1]: global_cases(reference_dataset, metric, anchor_times,
                                                    anchor_series.index)}
            for label, dataset, _ in candidates:
                series_of[label] = day_one_cases(dataset, metric, lead_dimension)[0]
            # Same date set and same block length for every pair in the family, as in
            # the paired loop above and in comparisons 23, 45 and 67.
            common_dates = None
            for values in series_of.values():
                dates = values.dropna().index
                common_dates = dates if common_dates is None else common_dates.intersection(dates)
            if common_dates is None or len(common_dates) < 3:
                continue
            series_of = {label: values.reindex(common_dates)
                         for label, values in series_of.items()}
            block_length = family_block_length([
                (series_of[first_label] - series_of[second_label]).dropna().values
                for first_label in series_of for second_label in series_of
                if first_label < second_label])
            print(f"  pairwise {metric_label}: {len(common_dates)} shared initialisations, "
                  f"block length {block_length:.2f}", flush=True)
            labels = list(series_of)
            count = len(labels)
            difference = np.full((count, count), np.nan)
            probability = np.full((count, count), np.nan)
            metric_rows = []
            for row, row_label in enumerate(labels):
                for column, column_label in enumerate(labels):
                    if row >= column:
                        continue
                    paired = pd.DataFrame({"row": series_of[row_label],
                                           "column": series_of[column_label]}).dropna()
                    if len(paired) < 3:
                        continue
                    if kind == "rmse":
                        result = bootstrap_paired_metric(
                            {"squared_error": paired["row"].values},
                            {"squared_error": paired["column"].values},
                            lambda components: float(np.sqrt(components["squared_error"])),
                            num_resamples=args.resamples, block_length=block_length)
                    else:
                        result = bootstrap_paired_difference(
                            (paired["row"] - paired["column"]).values,
                            num_resamples=args.resamples, block_length=block_length)
                    metric_rows.append({
                        "metric": metric_label, "unit": unit, "window": "day 1",
                        "tier": METRIC_TIER[metric_label], "family": f"pairwise_{metric_label}",
                        "row_model": row_label, "column_model": column_label,
                        "delta_row_minus_column": float(result["point_estimate"]),
                        "ci_low": float(result["lower_bound"]),
                        "ci_high": float(result["upper_bound"]),
                        "p_value": float(result["p_value"]),
                        "block_length": float(result["block_length"]),
                        "block_length_selected": float(result["block_length_selected"]),
                        "n_initialisations": int(len(paired)), "resamples": args.resamples,
                        "_row": row, "_column": column,
                    })
            apply_multiplicity_correction(metric_rows, alpha=0.05)
            for entry in metric_rows:
                row, column = entry.pop("_row"), entry.pop("_column")
                difference[row, column] = entry["delta_row_minus_column"]
                difference[column, row] = -entry["delta_row_minus_column"]
                probability[row, column] = probability[column, row] = entry["p_adjusted"]
            pairwise_rows.extend(metric_rows)
            if metric_rows:
                figure = build_pairwise_significance_figure(
                    labels, difference, probability, metric_rows[0]["alpha_threshold"],
                    metric_label, unit, "Every model against every other")
                figure.savefig(pairwise_directory / f"pairwise_{metric}.png")
                plt.close(figure)

        # The reference carries no paired difference, only its absolute score on the
        # same day-one cases every candidate was paired against.
        for metric, _, metric_label, unit, kind in METRICS:
            paired_reference = [row for row in significance_rows if row["metric"] == metric_label]
            denominator = climatology_series.get(metric)
            denominator_mean = denominator.mean() if denominator is not None else None
            if kind == "rmse" and denominator_mean is not None:
                denominator_mean = np.sqrt(denominator_mean)
            score_rows.append({
                "run": REFERENCE[1],
                "metric": metric_label,
                "window": "day 1",
                "mean_score": paired_reference[0]["estimate_reference"],
                "unit": unit,
                "crpss_versus_climatology": (float(1.0 - paired_reference[0]["estimate_reference"]
                                                   / denominator_mean)
                                             if denominator_mean is not None else float("nan")),
                "n_initialisations": paired_reference[0]["n_initialisations"],
            })

        forest_directory = output_directory / "forest"
        forest_directory.mkdir(parents=True, exist_ok=True)
        for metric, _, metric_label, unit, _ in METRICS:
            metric_rows = [row for row in significance_rows if row["metric"] == metric_label]
            figure = build_paired_forest_figure(
                [{"label": row["candidate"], "difference": row["delta_candidate_minus_reference"],
                  "lower": row["ci_low"], "upper": row["ci_high"], "is_baseline": row["is_baseline"]}
                 for row in metric_rows],
                metric_label, unit,
                "Pretrained GenCast and regional retraining\n"
                f"{metric_label}, day 1 versus {REFERENCE[1]}")
            figure.savefig(forest_directory / f"forest_{metric}_day1.png", dpi=200)
            plt.close(figure)

        # Day-one heatwave diagnostics rebuilt from the stored per-case primitives,
        # under both heatwave definitions.
        diagnostic_rows = []
        reliability_curves = []
        rev_curves = []
        for label, dataset, is_baseline in [(REFERENCE[1], reference_dataset, False), *candidates]:
            lead = {"lead_day": int(dataset.lead_day.values.min())} if "lead_day" in dataset.dims else {}
            day_one = dataset.sel(lead) if lead else dataset.isel(day=slice(None))
            for suffix, definition_label in HEATWAVE_DEFINITIONS:
                case_dimension = "forecast" if "forecast" in day_one[f"brier_score_{suffix}"].dims else "day"
                bin_count = day_one[f"probability_bin_count_{suffix}"].sum(case_dimension)
                bin_events = day_one[f"probability_bin_event_count_{suffix}"].sum(case_dimension)
                positives = day_one[f"true_positive_share_{suffix}"].mean(case_dimension)
                false_positives = day_one[f"false_positive_share_{suffix}"].mean(case_dimension)
                false_negatives = day_one[f"false_negative_share_{suffix}"].mean(case_dimension)
                f1_by_decision = (2 * positives / (2 * positives + false_positives + false_negatives)).values
                # The forecast declares a heatwave when it predicts exceedance: the majority of
                # members for an ensemble, the single field for a deterministic baseline.
                warned = int(np.argmin(np.abs(day_one.decision_probability_threshold.values - 0.5)))
                event_rate = float(day_one[f"event_rate_{suffix}"].mean())
                diagnostic_rows.append({
                    "run": label,
                    "definition": definition_label,
                    "window": "day 1",
                    "brier_score": float(day_one[f"brier_score_{suffix}"].mean()),
                    "event_rate": event_rate,
                    "auc": float(day_one[f"auc_case_{suffix}"].mean(skipna=True)),
                    "f1": float(f1_by_decision[warned]),
                    "precision": float((positives / (positives + false_positives)).values[warned]),
                    "recall": float((positives / (positives + false_negatives)).values[warned]),
                    "ensemble_spread": spread_and_skill(dataset)[0],
                    "rmse": spread_and_skill(dataset)[1],
                    "spread_skill_ratio": spread_and_skill(dataset)[2],
                    "conditional_absolute_error": float(
                        day_one[f"conditional_abs_error_sum_{suffix}"].sum()
                        / day_one[f"conditional_error_support_{suffix}"].sum()),
                    "is_baseline": is_baseline,
                })
                # Reliability is ensemble-only on the threshold definition: a deterministic
                # baseline has no probability to calibrate.
                if not is_baseline and suffix == "threshold":
                    observed = (bin_events / bin_count.where(bin_count > 0)).values
                    reliability_curves.append((label, day_one.forecast_probability.values, observed))
                # REV is a decision score, so persistence belongs on it; climatology
                # never warns and would sit flat on zero.
                if label != CLIMATOLOGY_LABEL and suffix == "threshold":
                    rev_curves.append((label, np.linspace(0.01, 0.99, 99),
                                       relative_economic_value(
                                           event_rate, (positives + false_positives).values,
                                           false_negatives.values)))

        climatology_brier = {
            definition_label: next((row["brier_score"] for row in diagnostic_rows
                                    if row["run"] == CLIMATOLOGY_LABEL and row["definition"] == definition_label),
                                   float("nan"))
            for _, definition_label in HEATWAVE_DEFINITIONS}
        for row in diagnostic_rows:
            denominator = climatology_brier[row["definition"]]
            # A zero climatology Brier is legitimate (no events and no warnings on the
            # window, e.g. the spell sieve can never fire on lead day 1): no skill baseline.
            row["brier_skill_score"] = (float(1.0 - row["brier_score"] / denominator)
                                        if denominator and np.isfinite(denominator) else float("nan"))

        figure = build_reliability_figure(
            reliability_curves, "Day-one reliability")
        figure.savefig(output_directory / "reliability_day1.png", dpi=200)
        plt.close(figure)
        figure = build_stacked_rev_figure(
            rev_curves, "Day-one relative economic value")
        figure.savefig(output_directory / "rev_day1.png", dpi=200)
        plt.close(figure)

        # Daily-Tmax distribution and predicted-vs-actual scatter, from the histogram
        # and sample already cached at report time (no raw fields, no recomputation).
        all_runs = [(REFERENCE[1], reference_dataset, False), *candidates]
        truth_dataset = climatology_dataset if climatology_dataset is not None else all_runs[0][1]
        distribution_curves = [("ERA5 truth", truth_dataset["tmax_histogram_target"].values)]
        for label, dataset, _ in all_runs:
            distribution_curves.append((label, dataset["tmax_histogram_prediction"].values))
        figure = build_tmax_distribution_figure(
            distribution_curves, truth_dataset["tmax_bin_edges"].values,
            "Daily Tmax distribution")
        figure.savefig(output_directory / "tmax_distribution.png")
        plt.close(figure)

        scatter_curves = [(label, dataset["tmax_scatter_prediction"].values, dataset["tmax_scatter_target"].values,
                           dataset["tmax_scatter_category"].values) for label, dataset, _ in all_runs]
        figure = build_tmax_scatter_figure(scatter_curves, "Predicted versus actual daily Tmax")
        figure.savefig(output_directory / "tmax_scatter.png")
        plt.close(figure)

        # Ensemble size, so the calendar colourbar can be ticked in members rather than
        # in a fraction: the probability bins are 0/M, 1/M, ... M/M.
        members_per_run = int(reference_dataset.sizes["probability_bin"]) - 1
        # One stacked calendar per run over the calendar region, coloured by how many
        # members put most of the region above p90 that day, with the observed heatwave
        # days outlined on top. observed_share is passed but no longer drawn.
        calendar_panels = [
            (label, dataset["date"].values, dataset["calendar_observed_share"].values,
             dataset["calendar_observed_heatwave"].values, dataset["calendar_model_certainty"].values)
            for label, dataset, is_baseline in all_runs
            if not is_baseline and "calendar_model_certainty" in dataset]
        if calendar_panels:
            figure = build_calendar_skill_figure(
                calendar_panels, "Daily hot-day certainty", members_per_run,
                region=DENMARK_CALENDAR_REGION, extra_cells=DENMARK_CALENDAR_EXTRA_CELLS)
            figure.savefig(output_directory / "calendar_heatwave.png")
            plt.close(figure)

        # Model error against the observed Tmax it was scored on, from the same sample.
        error_curves = [(label, dataset["tmax_scatter_target"].values, dataset["tmax_scatter_prediction"].values)
                        for label, dataset, _ in all_runs]
        threshold_mean = float(truth_dataset["tmax_threshold_mean"]) if "tmax_threshold_mean" in truth_dataset else None
        figure = build_tmax_error_by_observed_figure(
            error_curves, threshold_mean, "Model error by observed Tmax")
        figure.savefig(output_directory / "tmax_error_by_observed.png")
        plt.close(figure)

        # Maps are the one product that needs the raw fields: load the single valid
        # hour from each prediction cache. 12 UTC is the warmest stamp these runs carry.
        map_time = pd.Timestamp(args.map_date) + pd.Timedelta(hours=12)
        truth = to_scoring_grid(load(
            str(map_time.date()), str(map_time.date()),
            variables=[TARGET_VAR], region=get_loading_region(json.loads(
                Path("modified_Gencast/traninig configs/experiment_2_all_gencast.json").read_text()
            )["task_config"]), freq="1h")[TARGET_VAR], REGION).sel(time=map_time).compute()
        map_fields = [("ERA5 truth", truth)]
        for label, relative_path in RAW_FIELDS.items():
            map_fields.append((label, field_at(args.scratch / relative_path, map_time)))
        stamp = map_time.strftime("%d %B %Y %H UTC")
        figure = build_map_grid_figure(map_fields, "ERA5", f"2 m temperature, {stamp}")
        figure.savefig(output_directory / f"maps_{args.map_date}.png", dpi=200)
        plt.close(figure)
        figure = build_map_grid_figure(
            [(label, field - truth) for label, field in map_fields[1:]],
            "ERA5", f"Model minus ERA5, {stamp}", difference=True)
        figure.savefig(output_directory / f"maps_error_{args.map_date}.png", dpi=200)
        plt.close(figure)

        for name, records in (("paired_significance.csv", significance_rows),
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

    finally:
        reference_dataset.close()
        for _, dataset, _ in candidates:
            dataset.close()
    print(output_directory)


if __name__ == "__main__":
    main()


def domain_mean_series(path, start, end, region=REGION):
    """Area-weighted domain-mean 2 m temperature over a window, as one hourly series.

    A continuous rollout contributes the single forecast that spans the window; a daily
    reconstruction contributes each day's own frames stitched in order. Only the needed
    forecasts are read, so this never touches the whole year.
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    with xr.open_dataset(path) as store:
        forecast = pd.DatetimeIndex(store.forecast.values)
        hours = np.asarray(store.hour.values, dtype=int)
        selection = ([int(np.argmin(np.abs(forecast - start)))] if len(hours) > 24
                     else np.flatnonzero((forecast >= start) & (forecast <= end)))
        field = to_scoring_grid(store[TARGET_VAR].isel(forecast=selection), region)
        weights = np.cos(np.deg2rad(field.lat))
        averaged = field.weighted(weights).mean(["lat", "lon"])
        if "ensemble" in averaged.dims:
            averaged = averaged.mean("ensemble")
        averaged = averaged.compute()
    valid = forecast[selection].values[:, None] + hours[None, :] * np.timedelta64(1, "h")
    series = pd.Series(np.asarray(averaged.transpose("forecast", "hour").values).ravel(),
                       index=pd.DatetimeIndex(valid.ravel())).sort_index()
    return series[(series.index >= start) & (series.index <= end)]
