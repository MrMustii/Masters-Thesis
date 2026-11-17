"""exp3 report: score one continuous hourly rollout and cache the result.

exp3 is RQ2: can a lead-time-conditioned GenCast reconstruct the hourly cycle continuously,
which noise design does it best, and how far out does it stay useful. Marginal hourly CRPS
cannot answer the noise question - two designs with the same marginal variance score the
same hourly CRPS and produce different trajectories - so the temporal variogram is the
discriminating metric and is computed here alongside the ordinary scores.

Writes one metric_cases.nc on two case axes, because the questions live at two timescales:

    (forecast, hour)      t2m_crps_case, t2m_twcrps_case, and the spread and skill
                          components - how good each individual hour is
    (forecast, day_idx)   temporal_variogram_case  short-lag temporal coherence
                          peak_time_ccrps_case     error in the hour of the daily maximum
                          daily range, asymmetry, hours above threshold, re-aggregation
                          heatwave event metrics under both definitions

Nothing is averaged: one value per case, because significance.py resamples over the case
axis afterwards. The scoring region is the file's own grid shrunk by the boundary margin.

    python -m modified_Gencast.report_exp3 --predictions <path> --name <run>
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from modified_Gencast.lumio_buckets import load_p90_threshold, get_loading_region, load_hourly_p90_threshold
from modified_Gencast.metrics import (TARGET_VAR, area_weighted_crps, fixed_lead_hourly_threshold,
                                      heatwave_conditioned_metrics, peak_time_ccrps, reconstruction_cases,
                                      reconstruction_tmax_fields, spread_and_skill_cases, to_scoring_grid,
                                      variogram_score)


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
    hourly_p90 = to_scoring_grid(load_hourly_p90_threshold(), region)

    with xr.open_dataset(prediction_path, chunks={"forecast": 8}) as prediction_ds, \
         xr.open_dataset(target_path, chunks={"forecast": 8}) as target_ds:
        prediction = to_scoring_grid(prediction_ds[TARGET_VAR], region)
        target = to_scoring_grid(target_ds[TARGET_VAR], region)
        if max_forecasts is not None:
            prediction = prediction.isel(forecast=slice(0, max_forecasts))
            target = target.sel(forecast=prediction.forecast)

        # scores every forecast hour, plain and weighted towards hot values, with the ensemble spread
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
            prediction, target, to_scoring_grid(load_p90_threshold(), region))
        day_blocks = day_blocks.merge(
            heatwave_conditioned_metrics(tmax_members, tmax_target, tmax_threshold, day_dim="day_idx",
                                         target_spell=tmax_target_spell, valid_day=tmax_valid_day,
                                         antecedent_exceeds=tmax_antecedent).compute())

    # stamps every case with the clock time it is valid at, so runs on other layouts can be paired
    forecast = pd.DatetimeIndex(hourly.forecast.values)
    hours = np.asarray(hourly.hour.values, dtype=int)
    day_index = np.asarray(day_blocks.day_idx.values, dtype=int)
    metric_cases = xr.merge((hourly, day_blocks, xr.Dataset({
        "valid_time": xr.DataArray(forecast.values[:, None] + hours[None, :] * np.timedelta64(1, "h"),
                                   dims=("forecast", "hour"), coords={"forecast": hourly.forecast, "hour": hourly.hour}),
        "valid_day": xr.DataArray(forecast.floor("D").values[:, None] + day_index[None, :] * np.timedelta64(1, "D"),
                                  dims=("forecast", "day_idx"), coords={"forecast": hourly.forecast, "day_idx": day_blocks.day_idx}),
    })), compat="equals", join="exact")
    # records how the numbers were made, since the file is read far from here
    metric_cases.attrs.update({
        "run_id": name,
        "display_name": label,
        "prediction_path": str(prediction_path),
        "target_path": str(target_path),
        "layout": "continuous_hourly_rollout",
        "case_axes": "hourly scores on (forecast, hour); day-block scores on (forecast, day_idx)",
        "initialisation": "00 UTC daily",
        "spatial_weighting": "cosine latitude",
        "threshold": "1981-2010 p90 by valid day/hour and grid cell",
        "primary_metrics": "t2m_crps_case, temporal_variogram_case, spread and skill components",
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
    parser.add_argument("--group", default="exp3", help="report group the output is filed under")
    parser.add_argument("--output-root", type=Path, default=Path("/scratch/project_465002687/verified_metrics"))
    parser.add_argument("--max-forecasts", type=int, default=None)
    args = parser.parse_args()
    print(report_run(args.predictions, args.output_root, args.name, args.label or args.name,
                     args.group, args.max_forecasts))


if __name__ == "__main__":
    main()
