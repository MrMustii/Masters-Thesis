"""Comparison 0 — the baselines against each other.

Comparison numbers run 0 to 7 across five files and do not line up with the experiment
numbers, because one experiment can raise more than one question. This is comparison 0,
from the baseline caches only.

Climatology and persistence appear everywhere else only as denominators, so nothing
states how they compare. That matters for reading every other comparison: a model that
beats climatology has cleared a different bar depending on whether climatology is itself
better or worse than yesterday's weather, and the answer changes with lead time.

Persistence is shown at both lags. The 1-day lag is the honest reference at day 1 and
the 7-day lag at day 7, so the pair brackets the horizon rather than favouring either.

All three share the fixed-lead case grid comparison_1 and comparison_23 score on, so
they pair directly on initialisation keys.
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import s3fs
import xarray as xr

from modified_Gencast.lumio_buckets import GENCAST_LEVELS, bucket_name, load, lumio
from modified_Gencast.metrics import (
    DENMARK_CALENDAR_REGION, TARGET_VAR, cos_weights, to_scoring_grid)
from modified_Gencast.plots import (
    build_domain_mean_figure, build_lead_curve_figure, build_tmax_distribution_figure)
from modified_Gencast.report_baselines import BOUNDARY_MARGIN_DEGREES, EUROPE
from modified_Gencast.significance import bootstrap_paired_difference, bootstrap_paired_metric


# cache directory -> reader-facing label. Climatology is the reference: it is the bar
# every other comparison measures against.
REFERENCE = ("climatology", "Climatology")
CANDIDATES = (
    ("persistence_24h", "Persistence (1-day)"),
    ("persistence", "Persistence (7-day)"),
)
# (cache variable, lead dimension, display label, unit, bootstrap kind)
# Split only to mirror comparison_1's ordering; no tiered correction is applied here.
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
HORIZONS = (("full", "full forecast"), ("day1", "day 1"), ("day7", "day 7"))
# The crossover figure: where persistence stops beating climatology.
LEAD_CURVES = (
    ("tmax_crps_case", "lead_day", "Lead day", "Tmax CRPS (K)"),
    ("tmax_twcrps_case", "lead_day", "Lead day", "Tmax twCRPS (K)"),
    ("t2m_crps_case", "hour", "Lead time (hours)", "2 m CRPS (K)"),
)
# The diurnal profile has to come from the daily reconstruction layout: the fixed-lead
# grid is 12-hourly, which cannot resolve a daily cycle at all. Built from raw fields
# because the caches store scores by hour, never temperatures.
DIURNAL_LAG_HOURS = 24


def case_series(dataset, metric, lead_dimension, horizon):
    """One score per initialisation for the requested horizon, keyed by initialisation."""
    values = dataset[metric]
    if horizon != "full":
        if lead_dimension == "hour":
            hours = values.hour.values
            within = hours < hours.min() + 24 if horizon == "day1" else hours >= hours.max() - 23
            values = values.isel(hour=within)
        else:
            days = values.lead_day.values
            values = values.sel(lead_day=[int(days.min() if horizon == "day1" else days.max())])
    collapsed = values.mean(lead_dimension)
    return pd.Series(np.asarray(collapsed.values, dtype=float),
                     index=pd.DatetimeIndex(collapsed.forecast.values))


def squared_error_components(dataset, horizon):
    """Per-case squared error for the RMSE bootstrap, restricted to the horizon."""
    values = dataset["squared_error_case"]
    if horizon != "full":
        hours = values.hour.values
        within = hours < hours.min() + 24 if horizon == "day1" else hours >= hours.max() - 23
        values = values.isel(hour=within)
    return pd.Series(np.asarray(values.mean("hour").values, dtype=float),
                     index=pd.DatetimeIndex(values.forecast.values))


def average_week(daily_targets, region):
    """One composite week: every week of the year folded onto a Monday-to-Sunday axis.

    Built off the daily reconstruction template, which holds 2021 at hourly resolution on
    scratch. Persistence is that series delayed by its own lag and climatology is the
    average-year field at each valid hour, both folded the same way.
    """
    with xr.open_dataset(daily_targets) as target_ds:
        cropped = to_scoring_grid(target_ds[TARGET_VAR], region)
        weights = cos_weights(cropped)
        domain_mean = cropped.weighted(weights).mean(["lat", "lon"]).compute()
    forecast = pd.DatetimeIndex(domain_mean.forecast.values)
    hours = np.asarray(domain_mean.hour.values, dtype=int)
    valid_time = pd.DatetimeIndex(
        (forecast.values[:, None] + hours[None, :] * np.timedelta64(1, "h")).ravel())
    truth = pd.Series(np.asarray(domain_mean.values, dtype=float).ravel(),
                      index=valid_time).sort_index()

    loading_region = (EUROPE[0] - BOUNDARY_MARGIN_DEGREES, EUROPE[1] + BOUNDARY_MARGIN_DEGREES,
                      EUROPE[2] - BOUNDARY_MARGIN_DEGREES, EUROPE[3] + BOUNDARY_MARGIN_DEGREES)
    store = s3fs.S3Map(root=bucket_name(GENCAST_LEVELS, loading_region) + "-climatology",
                       s3=lumio(), check=False)
    with xr.open_zarr(store, consolidated=True) as climatology_store:
        field = to_scoring_grid(climatology_store["climatology"], region).rename({"hour": "hour_of_day"})
        selected = field.sel(
            dayofyear=xr.DataArray(np.minimum(truth.index.dayofyear.values, 365),
                                   dims=["time"], coords={"time": truth.index}),
            hour_of_day=xr.DataArray(truth.index.hour.values,
                                     dims=["time"], coords={"time": truth.index}),
        ).drop_vars(["dayofyear", "hour_of_day"], errors="ignore")
        climatology = pd.Series(
            np.asarray(selected.weighted(weights).mean(["lat", "lon"]).compute().values, dtype=float),
            index=truth.index)

    def folded(series):
        """Average every week together: hour into the week, Monday 00 UTC to Sunday 23."""
        position = series.index.dayofweek * 24 + series.index.hour
        composite = series.groupby(position).mean().sort_index()
        return composite.index.values / 24.0, composite.values

    return [("ERA5 truth", *folded(truth)),
            ("Climatology", *folded(climatology)),
            ("Persistence (1-day)", *folded(truth.shift(freq=pd.Timedelta(hours=24)))),
            ("Persistence (7-day)", *folded(truth.shift(freq=pd.Timedelta(hours=168))))]


def main():
    """Write primary_scores.csv, paired_significance.csv and the figures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/scratch/project_465002687/verified_metrics"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--daily-targets", type=Path,
                        default=Path("/scratch/project_465002687/Predictions/experiment_4_offsets_0_stage2/"
                                     "targets_experiment_4_offsets_0_stage2_step_00016000_2021-01-01_2021-12-31_daily_fixed_churn0.0.nc"),
                        help="hourly targets file the average week is read from")
    parser.add_argument("--scratch", type=Path, default=Path("/scratch/project_465002687"))
    parser.add_argument("--skip-week", action="store_true")
    args = parser.parse_args()
    output_directory = args.out or Path.cwd() / "comparison_0"
    output_directory.mkdir(parents=True, exist_ok=True)

    runs = [REFERENCE, *CANDIDATES]
    datasets = {}
    for run, label in runs:
        path = args.root / "reports" / "baselines" / run / "metric_cases.nc"
        if not path.is_file():
            raise SystemExit(f"Missing baseline cache: {path}")
        datasets[run] = xr.open_dataset(path)
    label_of = dict(runs)

    try:
        print("[comparison_0] paired bootstrap", flush=True)
        significance_rows, score_rows = [], []
        for metric, lead_dimension, metric_label, unit, kind in METRICS:
            for horizon, horizon_label in HORIZONS:
                series = {run: case_series(datasets[run], metric, lead_dimension, horizon)
                          for run, _ in runs if metric in datasets[run]}
                for run, values in series.items():
                    score_rows.append({
                        "run": label_of[run], "metric": metric_label, "horizon": horizon_label,
                        "mean_score": (float(np.sqrt((values ** 2).mean())) if kind == "rmse"
                                       else float(values.mean())),
                        "unit": unit, "n_initialisations": int(len(values)),
                    })
                # Every pair, not only against climatology: the two persistence lags
                # against each other is the question of how fast the memory decays.
                for index, (candidate_run, _) in enumerate(runs):
                    for reference_run, _ in runs[index + 1:]:
                        if candidate_run not in series or reference_run not in series:
                            continue
                        paired = pd.concat({"candidate": series[candidate_run],
                                            "reference": series[reference_run]},
                                           axis=1).dropna()
                        if paired.empty:
                            continue
                        if kind == "rmse":
                            candidate_components = squared_error_components(
                                datasets[candidate_run], horizon).loc[paired.index]
                            reference_components = squared_error_components(
                                datasets[reference_run], horizon).loc[paired.index]
                            result = bootstrap_paired_metric(
                                {"squared_error": candidate_components.values},
                                {"squared_error": reference_components.values},
                                lambda components: float(np.sqrt(components["squared_error"])),
                                num_resamples=args.resamples)
                        else:
                            result = bootstrap_paired_difference(
                                (paired["candidate"] - paired["reference"]).values,
                                num_resamples=args.resamples)
                        significance_rows.append({
                            "candidate": label_of[candidate_run],
                            "reference": label_of[reference_run],
                            "metric": metric_label, "horizon": horizon_label, "unit": unit,
                            "estimate_candidate": float(paired["candidate"].mean()),
                            "estimate_reference": float(paired["reference"].mean()),
                            "delta_candidate_minus_reference": float(result["point_estimate"]),
                            "ci_low": float(result["lower_bound"]),
                            "ci_high": float(result["upper_bound"]),
                            "p_value": float(result["p_value"]),
                            "block_length": float(result["block_length"]),
                            "block_length_method": result["block_length_method"],
                            "n_initialisations": int(len(paired)),
                            "resamples": args.resamples,
                        })
        # No multiplicity correction here: this is a descriptive head-to-head of three
        # references, not a confirmatory family of hypotheses about a model.
        for row in significance_rows:
            distinguishable = row["p_value"] <= 0.05
            row["direction"] = ("candidate better" if distinguishable
                                and row["delta_candidate_minus_reference"] < 0
                                else "reference better" if distinguishable
                                else "not distinguishable")

        print("[comparison_0] figures", flush=True)
        for metric, lead_dimension, xlabel, ylabel in LEAD_CURVES:
            curves = []
            for run, label in runs:
                if metric not in datasets[run]:
                    continue
                values = datasets[run][metric].mean("forecast")
                curves.append((label, np.asarray(values[lead_dimension].values, dtype=float),
                               np.asarray(values.values, dtype=float)))
            if curves:
                figure = build_lead_curve_figure(curves, xlabel, ylabel,
                                                 f"{ylabel.split(' (')[0]} over lead time")
                figure.savefig(output_directory / f"lead_{metric}.png", dpi=200)
                plt.close(figure)

        truth = datasets[REFERENCE[0]]
        distribution_curves = [("ERA5 truth", truth["tmax_histogram_target"].values)]
        distribution_curves += [(label, datasets[run]["tmax_histogram_prediction"].values)
                                for run, label in runs]
        figure = build_tmax_distribution_figure(distribution_curves, truth["tmax_bin_edges"].values,
                                                "Baseline daily Tmax distribution")
        figure.savefig(output_directory / "tmax_distribution.png", dpi=200)
        plt.close(figure)

        if not args.skip_week:
            print("[comparison_0] average week", flush=True)
            # Denmark, not the full domain: a European mean washes out the signal the
            # rest of the thesis is about.
            figure = build_lead_curve_figure(
                average_week(args.daily_targets, DENMARK_CALENDAR_REGION),
                "Day of week (Monday 00 UTC to Sunday 23 UTC)", "2 m temperature (K)",
                "Average week over Denmark, 2021")
            figure.savefig(output_directory / "average_week.png", dpi=200)
            plt.close(figure)

        for name, records in (("paired_significance.csv", significance_rows),
                              ("primary_scores.csv", score_rows)):
            if not records:
                continue
            fieldnames = [key for key in records[0] if key not in ("ci_low", "ci_high")]
            with open(output_directory / name, "w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(records)
        print(output_directory)
    finally:
        for dataset in datasets.values():
            dataset.close()


if __name__ == "__main__":
    main()
