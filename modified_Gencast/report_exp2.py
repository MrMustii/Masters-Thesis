"""exp2 report: score one fixed-lead regional run and cache the result.

exp2 carries the rest of RQ1: what is the smallest input-variable set that keeps
full-variable skill, and does the 8.5 degree forced boundary help. Both are comparisons
between runs that differ only in their training config, so every run is scored here
identically and the comparison layer does the differencing. Also used with ``--group exp1``
for the regional model that exp1's pretrained reference is measured against.

Reads predictions_*.nc and its matching targets, and writes one metric_cases.nc holding:

    t2m_crps_case, t2m_twcrps_case     per (forecast, hour), hourly 2m temperature
    tmax_crps_case, tmax_twcrps_case   per (forecast, lead_day), daily maxima
    heatwave event metrics             Brier, reliability bins, detection shares and AUC,
                                       under both the day-over-p90 and the 3-day-spell
                                       definitions, suffixed _threshold and _heatwave
    spread and skill                   ensemble variance and squared error, kept as separate
                                       components so any resample can form the ratio itself
    valid_time, valid_day              the clock time each case is valid at, so runs on other
                                       layouts can be paired against this one

Nothing is averaged: one value per case, because significance.py resamples over the case
axis afterwards. The scoring region is the file's own grid shrunk by the boundary margin, so
the belt the model was trained on is excluded from every number written here.

    python -m modified_Gencast.report_exp2 --predictions <path> --name <run> [--group exp2]
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from modified_Gencast.lumio_buckets import load_hourly_p90_threshold, load_p90_threshold, load_true_daily_tmax
from modified_Gencast.metrics import (TARGET_VAR, area_weighted_crps, fixed_lead_hourly_threshold,
                                      fixed_lead_tmax_fields, heatwave_conditioned_metrics,
                                      spread_and_skill_cases, to_scoring_grid)


BOUNDARY_MARGIN_DEGREES = 8.5
def report_run(prediction_path: Path, output_root: Path, name: str, label: str,
               group: str, max_forecasts: int | None = None) -> Path:
    # reads the region off the forecast's own grid, so nothing about where it lives is hardcoded
    target_path = prediction_path.with_name(prediction_path.name.replace("predictions_", "targets_", 1))
    with xr.open_dataset(prediction_path) as probe:
        loading_region = (float(probe.lat.min()), float(probe.lat.max()),
                          float(probe.lon.min()), float(probe.lon.max()))
    region = (loading_region[0] + BOUNDARY_MARGIN_DEGREES, loading_region[1] - BOUNDARY_MARGIN_DEGREES,
              loading_region[2] + BOUNDARY_MARGIN_DEGREES, loading_region[3] - BOUNDARY_MARGIN_DEGREES)
    # the evaluated date range is only recorded in the filename, so it is read back from there
    start_date, end_date = re.search(r"_(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})_", prediction_path.name).groups()
    true_daily = to_scoring_grid(load_true_daily_tmax(start_date, end_date, loading_region, region), region)
    p90 = to_scoring_grid(load_p90_threshold(), region)
    hourly_p90 = to_scoring_grid(load_hourly_p90_threshold(), region)

    with xr.open_dataset(prediction_path, chunks={"forecast": 24}) as prediction_ds, \
         xr.open_dataset(target_path, chunks={"forecast": 24}) as target_ds:
        prediction = to_scoring_grid(prediction_ds[TARGET_VAR], region)
        target = to_scoring_grid(target_ds[TARGET_VAR], region)
        if max_forecasts is not None:
            prediction = prediction.isel(forecast=slice(0, max_forecasts))
            target = target.sel(forecast=prediction.forecast)

        # scores the hourly forecast and its daily maxima, plain and weighted towards hot values
        hourly_threshold = fixed_lead_hourly_threshold(prediction, hourly_p90)
        t2m_crps = area_weighted_crps(prediction, target).transpose("forecast", "hour")
        t2m_twcrps = area_weighted_crps(prediction, target, hourly_threshold).transpose("forecast", "hour")
        tmax_members, tmax_target, tmax_threshold, tmax_target_spell, tmax_valid_day, tmax_antecedent = fixed_lead_tmax_fields(prediction, true_daily, p90)
        tmax_crps = area_weighted_crps(tmax_members, tmax_target).transpose("forecast", "lead_day")
        tmax_twcrps = area_weighted_crps(tmax_members, tmax_target, tmax_threshold).transpose("forecast", "lead_day")
        events = heatwave_conditioned_metrics(tmax_members, tmax_target, tmax_threshold, day_dim="lead_day",
                                              target_spell=tmax_target_spell, valid_day=tmax_valid_day,
                                              antecedent_exceeds=tmax_antecedent)

        # stamps every case with the clock time it is valid at, so runs on other layouts can be paired
        forecast = pd.DatetimeIndex(t2m_crps.forecast.values)
        hours = np.asarray(t2m_crps.hour.values, dtype=int)
        lead_days = np.asarray(tmax_crps.lead_day.values, dtype=int)
        valid_time = forecast.values[:, None] + hours[None, :] * np.timedelta64(1, "h")
        valid_day = forecast.floor("D").values[:, None] + lead_days[None, :] * np.timedelta64(1, "D")
        metric_cases = xr.merge((xr.Dataset({
            "t2m_crps_case": t2m_crps.astype(np.float32),
            "t2m_twcrps_case": t2m_twcrps.astype(np.float32),
            "tmax_crps_case": tmax_crps.astype(np.float32),
            "tmax_twcrps_case": tmax_twcrps.astype(np.float32),
            "valid_time": xr.DataArray(valid_time, dims=("forecast", "hour"), coords={"forecast": t2m_crps.forecast, "hour": t2m_crps.hour}),
            "valid_day": xr.DataArray(valid_day, dims=("forecast", "lead_day"), coords={"forecast": tmax_crps.forecast, "lead_day": tmax_crps.lead_day}),
        }), events, spread_and_skill_cases(prediction, target)), compat="equals", join="exact").compute()
    # records how the numbers were made, since the file is read far from here
    metric_cases.attrs.update({
        "run_id": name,
        "display_name": label,
        "prediction_path": str(prediction_path),
        "target_path": str(target_path),
        "layout": "fixed_lead_regional",
        "spatial_weighting": "cosine latitude",
        "tmax_sampling": "maximum of matched 12-hourly forecast and target samples per complete valid day",
        "threshold": "1981-2010 p90 by valid day/hour and grid cell",
    })
    output_path = output_root / "reports" / group / name / "metric_cases.nc"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metric_cases.to_netcdf(output_path, mode="w")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True, help="cached predictions file for this run")
    parser.add_argument("--name", required=True, help="output directory under reports/<group>/")
    parser.add_argument("--label", default=None, help="reader-facing name in figures and tables")
    parser.add_argument("--group", default="exp2", help="report group the output is filed under")
    parser.add_argument("--output-root", type=Path, default=Path("/scratch/project_465002687/verified_metrics"))
    parser.add_argument("--max-forecasts", type=int, default=None)
    args = parser.parse_args()
    print(report_run(args.predictions, args.output_root, args.name, args.label or args.name,
                     args.group, args.max_forecasts))


if __name__ == "__main__":
    main()
