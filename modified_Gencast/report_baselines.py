"""Baseline report: score climatology and persistence on another run's case grid.

Every skill claim in the thesis is relative, so each model run needs two references scored
on exactly the same cases:

    climatology  the 2005-2019 ERA5 average year, read at each case's valid time. Flat by
                 construction, so the lead at which a model's curve crosses it is that
                 model's usable horizon.
    persistence  ERA5 delayed by the run's own forecast length, which is the honest
                 same-age reference rather than an arbitrary lag.

The case grid is *borrowed* from the targets file passed in - region, initialisations, lead
hours and all - so a baseline is never scored on cases the model was not scored on. That is
also why this runs once per model run rather than once overall, and why there are eleven
baseline reports rather than two.

The layout is inferred from that file: hourly leads mean a reconstruction run, so the
day-block scores are produced; anything coarser means the fixed-lead daily-Tmax grid. The
threshold bucket follows the region, so a transfer region is scored against its own climate.

Both baselines are deterministic. With no ensemble dimension, CRPS degenerates to absolute
error and AUC is undefined - the output records this in its ``deterministic`` attribute so a
reader does not mistake the numbers for ensemble scores.

    python -m modified_Gencast.report_baselines --targets <path> --baseline climatology --name <run>
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import s3fs
import xarray as xr

from modified_Gencast.lumio_buckets import (GENCAST_LEVELS, bucket_name, load, load_hourly_p90_threshold, load_p90_threshold, load_true_daily_tmax, lumio)
from modified_Gencast.metrics import (TARGET_VAR, area_weighted_crps, fixed_lead_hourly_threshold,
                                      fixed_lead_tmax_fields, heatwave_conditioned_metrics, reconstruction_cases,
                                      reconstruction_tmax_fields, spread_and_skill_cases, to_scoring_grid, variogram_score)


EUROPE = (45.0, 65.0, -5.0, 25.0)
NW_AMERICA = (45.0, 65.0, -135.0, -105.0)
N_AFRICA = (15.0, 35.0, 10.0, 40.0)
BOUNDARY_MARGIN_DEGREES = 8.5
LABELS = {"climatology": "ERA5 average-year climatology", "persistence": "ERA5 persistence baseline"}
def report_run(target_path: Path, output_root: Path, baseline: str, name: str, lag_hours: int, max_forecasts: int | None = None) -> Path:
    # reads the region and layout off the run's own targets file, so nothing about it is hardcoded
    with xr.open_dataset(target_path, chunks={"forecast": 8}) as probe:
        loading_region = (float(probe.lat.min()), float(probe.lat.max()),
                          float(probe.lon.min()), float(probe.lon.max()))
        hour_values = np.asarray(probe.hour.values, dtype=int)
    region = (loading_region[0] + BOUNDARY_MARGIN_DEGREES, loading_region[1] - BOUNDARY_MARGIN_DEGREES,
              loading_region[2] + BOUNDARY_MARGIN_DEGREES, loading_region[3] - BOUNDARY_MARGIN_DEGREES)
    # hourly leads mean a reconstruction grid; anything coarser is the fixed-lead daily-Tmax grid
    tmax_adapter = not (hour_values.size > 1 and np.all(np.diff(hour_values) == 1))
    threshold_region = None if np.allclose(region, EUROPE) else loading_region
    # persistence defaults to the run's own forecast length, the honest same-age reference
    lag_hours = lag_hours or int(hour_values.max()) + (1 if hour_values.min() == 0 else 0)
    label = LABELS[baseline] + (f" ({lag_hours} h lag)" if baseline == "persistence" else "")
    hourly_p90 = to_scoring_grid(load_hourly_p90_threshold(threshold_region), region)

    with xr.open_dataset(target_path, chunks={"forecast": 8}) as target_ds:
        target = to_scoring_grid(target_ds[TARGET_VAR], region)
        if max_forecasts is not None:
            target = target.isel(forecast=slice(0, max_forecasts))
        # works out the clock time of every case so the baseline can be sampled at those exact times
        forecast = pd.DatetimeIndex(target.forecast.values)
        hours = np.asarray(target.hour.values, dtype=int)
        valid_time = forecast.values[:, None] + hours[None, :] * np.timedelta64(1, "h")
        flat_valid = pd.DatetimeIndex(valid_time.ravel())
        case_coords = {"forecast": target.forecast, "hour": target.hour}

        # builds the baseline forecast: the average year read at each valid time, or ERA5 delayed by the lag
        if baseline == "climatology":
            store = s3fs.S3Map(root=bucket_name(GENCAST_LEVELS, loading_region) + "-climatology", s3=lumio(), check=False)
            with xr.open_zarr(store, consolidated=True) as climatology_store:
                field = to_scoring_grid(climatology_store["climatology"], region).rename({"hour": "hour_of_day"})
                prediction = field.sel(
                    dayofyear=xr.DataArray(np.minimum(flat_valid.dayofyear.values, 365).reshape(valid_time.shape), dims=("forecast", "hour"), coords=case_coords),
                    hour_of_day=xr.DataArray(flat_valid.hour.values.reshape(valid_time.shape), dims=("forecast", "hour"), coords=case_coords),
                ).drop_vars(["dayofyear", "hour_of_day"], errors="ignore").compute()
            source = "2005-2019 ERA5 average-year field, evaluated at each case valid time"
        else:
            source_time = flat_valid - pd.Timedelta(hours=lag_hours)
            hourly = to_scoring_grid(load(str(source_time.min().date()), str(flat_valid.max().date()),
                                          variables=[TARGET_VAR], region=loading_region, freq="1h")[TARGET_VAR], region)
           
           
            prediction = hourly.sel(time=xr.DataArray(source_time.values.reshape(valid_time.shape),
                                                      dims=("forecast", "hour"), coords=case_coords)).drop_vars("time", errors="ignore").compute()
            source = f"ERA5 ground truth delayed by the full {lag_hours} h forecast length"

        # scores the baseline hour by hour, plain and weighted towards hot values
        hourly_threshold = fixed_lead_hourly_threshold(target, hourly_p90)
        products = [xr.Dataset({
            "t2m_crps_case": area_weighted_crps(prediction, target).transpose("forecast", "hour").astype(np.float32),
            "t2m_twcrps_case": area_weighted_crps(prediction, target, hourly_threshold).transpose("forecast", "hour").astype(np.float32),
            "valid_time": xr.DataArray(valid_time, dims=("forecast", "hour"), coords=case_coords),
        }), spread_and_skill_cases(prediction, target)]

        # adds the daily Tmax and heatwave scores, on whichever day grid this run uses
        if tmax_adapter:
            p90 = to_scoring_grid(load_p90_threshold(threshold_region), region)
            true_daily = to_scoring_grid(load_true_daily_tmax("2021-01-01", "2021-12-31", loading_region, region), region)
            tmax_members, tmax_target, tmax_threshold, tmax_target_spell, tmax_valid_day, _ = fixed_lead_tmax_fields(prediction, true_daily, p90)
            lead_days = np.asarray(tmax_members.lead_day.values, dtype=int)
            products.append(xr.Dataset({
                "tmax_crps_case": area_weighted_crps(tmax_members, tmax_target).transpose("forecast", "lead_day").astype(np.float32),
                "tmax_twcrps_case": area_weighted_crps(tmax_members, tmax_target, tmax_threshold).transpose("forecast", "lead_day").astype(np.float32),
                "valid_day": xr.DataArray(forecast.floor("D").values[:, None] + lead_days[None, :] * np.timedelta64(1, "D"),
                                          dims=("forecast", "lead_day"), coords={"forecast": tmax_members.forecast, "lead_day": tmax_members.lead_day})}))
            products.append(heatwave_conditioned_metrics(tmax_members, tmax_target, tmax_threshold, day_dim="lead_day",
                                                        target_spell=tmax_target_spell, valid_day=tmax_valid_day))
        else:
            day_blocks = reconstruction_cases(prediction, target, hourly_threshold).compute()
            day_blocks["temporal_variogram_case"] = variogram_score(prediction, target)["series_by_day_idx"].astype(np.float32)
            p90 = to_scoring_grid(load_p90_threshold(threshold_region), region)
            tmax_members, tmax_target, tmax_threshold, tmax_target_spell, tmax_valid_day, tmax_antecedent = reconstruction_tmax_fields(prediction, target, p90)
            day_blocks = day_blocks.merge(
                heatwave_conditioned_metrics(tmax_members, tmax_target, tmax_threshold, day_dim="day_idx",
                                             target_spell=tmax_target_spell, valid_day=tmax_valid_day,
                                             antecedent_exceeds=tmax_antecedent).compute())
            day_index = np.asarray(day_blocks.day_idx.values, dtype=int)
            products.append(day_blocks)
            products.append(xr.Dataset({"valid_day": xr.DataArray(forecast.floor("D").values[:, None] + day_index[None, :] * np.timedelta64(1, "D"),
                                                                 dims=("forecast", "day_idx"), coords={"forecast": target.forecast, "day_idx": day_blocks.day_idx})}))

        metric_cases = xr.merge(products, compat="equals", join="exact").compute()

    # records how the numbers were made, since the file is read far from here
    metric_cases.attrs.update({
        "run_id": name,
        "display_name": label,
        "prediction_path": source,
        "target_path": str(target_path),
        "scoring_region": str(region),
        "threshold_region": str(threshold_region or "Europe, unsuffixed bucket"),
        "deterministic": "yes; no ensemble dimension, so CRPS is the absolute error and AUC is undefined",
        "spatial_weighting": "cosine latitude",
        "threshold": "1981-2010 p90 by valid day/hour and grid cell, for this run's own region",
    })
    output_path = output_root / "reports" / "baselines" / name / "metric_cases.nc"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metric_cases.to_netcdf(output_path, mode="w")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, required=True, help="targets file of the run this baseline is scored against; its case grid is borrowed")
    parser.add_argument("--baseline", choices=tuple(LABELS), required=True)
    parser.add_argument("--name", required=True, help="output directory under reports/baselines/")
    parser.add_argument("--lag-hours", type=int, default=None, help="persistence lag; defaults to the run's full forecast length")
    parser.add_argument("--output-root", type=Path, default=Path("/scratch/project_465002687/verified_metrics"))
    parser.add_argument("--max-forecasts", type=int, default=None)
    args = parser.parse_args()
    print(report_run(args.targets, args.output_root, args.baseline, args.name, args.lag_hours, args.max_forecasts))


if __name__ == "__main__":
    main()
