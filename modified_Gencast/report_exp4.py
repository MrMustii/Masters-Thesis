"""exp4 report: score one daily-statistics reconstruction and cache the result.

exp4 is RQ3: can the 24 hours of a day be reconstructed from that day's mean, max and min,
does giving neighbouring days help, and does a Denmark-trained checkpoint transfer to another
climate. The same script scores the European runs and the two transfer regions.

One case per calendar day: there is no lead axis, and ``hour`` means hour of day rather than
hours since initialisation. Writes one metric_cases.nc holding:

    t2m_crps_case, t2m_twcrps_case   per (forecast, hour), each reconstructed hour
    peak_time_ccrps_case             error in the hour of the daily maximum
    daily_range_crps_case            amplitude of the diurnal cycle
    asymmetry_crps_case              where in its own range the cycle sits, 0.5 symmetric
    hours_above_threshold_crps_case  how long the day spent above p90
    temporal_variogram_case          short-lag temporal coherence
    re-aggregation CRPS and bias     Tmax, Tmin and Tmean taken back out of the
                                     reconstructed hours - the honesty check that the inputs
                                     were actually reproduced
    heatwave event metrics           under both definitions

Daily Tmax is an *input* to this model, so Tmax CRPS is recorded but is not a primary score;
that is what the tmax_note attribute on the output says.

Threshold buckets are per region: Europe reads the unsuffixed bucket, the transfer regions
read their own, so each region is scored against its own climate rather than Europe's.

    python -m modified_Gencast.report_exp4 --predictions <path> --name <run>
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from modified_Gencast.lumio_buckets import load_p90_threshold, load_hourly_p90_threshold
from modified_Gencast.metrics import (TARGET_VAR, area_weighted_crps, fixed_lead_hourly_threshold,
                                      heatwave_conditioned_metrics, peak_time_ccrps, reconstruction_cases,
                                      reconstruction_tmax_fields, spread_and_skill_cases, to_scoring_grid,
                                      variogram_score)


EUROPE = (45.0, 65.0, -5.0, 25.0)
BOUNDARY_MARGIN_DEGREES = 8.5


def report_run(prediction_path: Path, output_root: Path, name: str, label: str,
               group: str, max_forecasts: int | None = None) -> Path:
    # reads the region off the forecast's own grid, and its threshold bucket from that region
    target_path = prediction_path.with_name(prediction_path.name.replace("predictions_", "targets_", 1))
    with xr.open_dataset(prediction_path) as probe:
        loading_region = (float(probe.lat.min()), float(probe.lat.max()),
                          float(probe.lon.min()), float(probe.lon.max()))
    region = (loading_region[0] + BOUNDARY_MARGIN_DEGREES, loading_region[1] - BOUNDARY_MARGIN_DEGREES,
              loading_region[2] + BOUNDARY_MARGIN_DEGREES, loading_region[3] - BOUNDARY_MARGIN_DEGREES)
    # Europe reads the unsuffixed threshold bucket; a transfer region reads its own
    threshold_region = None if np.allclose(region, EUROPE) else loading_region
    hourly_p90 = to_scoring_grid(load_hourly_p90_threshold(threshold_region), region)

    with xr.open_dataset(prediction_path, chunks={"forecast": 24}) as prediction_ds, \
         xr.open_dataset(target_path, chunks={"forecast": 24}) as target_ds:
        prediction = to_scoring_grid(prediction_ds[TARGET_VAR], region)
        target = to_scoring_grid(target_ds[TARGET_VAR], region)
        if max_forecasts is not None:
            prediction = prediction.isel(forecast=slice(0, max_forecasts))
            target = target.sel(forecast=prediction.forecast)

        # scores every reconstructed hour, plain and weighted towards hot values, with the ensemble spread
        hourly_threshold = fixed_lead_hourly_threshold(prediction, hourly_p90)
        t2m_crps = area_weighted_crps(prediction, target).transpose("forecast", "hour")
        t2m_twcrps = area_weighted_crps(prediction, target, hourly_threshold).transpose("forecast", "hour")
        hourly = xr.Dataset({
            "t2m_crps_case": t2m_crps.astype(np.float32),
            "t2m_twcrps_case": t2m_twcrps.astype(np.float32),
        }).merge(spread_and_skill_cases(prediction, target)).compute()

        # scores the shape of each reconstructed day and the heatwaves its daily maxima imply
        day_blocks = reconstruction_cases(prediction, target, hourly_threshold).compute()
        day_blocks["peak_time_ccrps_case"] = peak_time_ccrps(prediction, target)["series_by_day_idx"].compute().astype(np.float32)
        day_blocks["temporal_variogram_case"] = variogram_score(prediction, target)["series_by_day_idx"].astype(np.float32)
        tmax_members, tmax_target, tmax_threshold, tmax_target_spell, tmax_valid_day, tmax_antecedent = reconstruction_tmax_fields(
            prediction, target, to_scoring_grid(load_p90_threshold(threshold_region), region))
        day_blocks = day_blocks.merge(
            heatwave_conditioned_metrics(tmax_members, tmax_target, tmax_threshold, day_dim="day_idx",
                                         target_spell=tmax_target_spell, valid_day=tmax_valid_day,
                                         antecedent_exceeds=tmax_antecedent).compute())

    # stamps every case with the clock time it is valid at, so runs on other layouts can be paired
    forecast = pd.DatetimeIndex(hourly.forecast.values)
    hours = np.asarray(hourly.hour.values, dtype=int)
    metric_cases = xr.merge((hourly, day_blocks, xr.Dataset({
        "valid_time": xr.DataArray(forecast.values[:, None] + hours[None, :] * np.timedelta64(1, "h"),
                                   dims=("forecast", "hour"), coords={"forecast": hourly.forecast, "hour": hourly.hour}),
        "valid_day": xr.DataArray(forecast.floor("D").values, dims=("forecast",), coords={"forecast": hourly.forecast}),
    })), compat="equals", join="exact")
    # records how the numbers were made, since the file is read far from here
    metric_cases.attrs.update({
        "run_id": name,
        "display_name": label,
        "prediction_path": str(prediction_path),
        "target_path": str(target_path),
        "layout": "daily_statistics_reconstruction",
        "case_axes": "one case per calendar day; hour is hour-of-day, there is no lead axis",
        "scoring_region": str(region),
        "threshold_region": str(threshold_region or "Europe, unsuffixed bucket"),
        "spatial_weighting": "cosine latitude",
        "threshold": "1981-2010 p90 by valid day/hour and grid cell, for this run's own region",
        "primary_metrics": ("t2m_crps_case, peak_time_ccrps_case, daily_range_crps_case, "
                            "hours_above_threshold_crps_case"),
        "tmax_note": "daily Tmax is an input to this model, so Tmax CRPS is not a primary score",
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
    parser.add_argument("--group", default="exp4", help="report group the output is filed under")
    parser.add_argument("--output-root", type=Path, default=Path("/scratch/project_465002687/verified_metrics"))
    parser.add_argument("--max-forecasts", type=int, default=None)
    args = parser.parse_args()
    print(report_run(args.predictions, args.output_root, args.name, args.label or args.name,
                     args.group, args.max_forecasts))


if __name__ == "__main__":
    main()
