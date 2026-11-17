"""exp1 report: score Google's pretrained GenCast over the regional domain.

exp1 is the reference RQ1 is asked against - does a regionally trained model beat the global
model it was derived from, on the same region and the same year. Only the pretrained run is
scored here; the regional model it is compared with goes through report_exp2 with
``--group exp1``.

It needs its own script because its predictions are a zarr on a plain absolute-time axis
with no forecast or lead dimension, so daily maxima are formed by grouping hours into
calendar days rather than read off a lead axis. The variable names written are the same as
every other report, which is what lets comparison_1 pair the two on valid_time and valid_day.

Writes one metric_cases.nc holding t2m CRPS and twCRPS per hour, Tmax CRPS and twCRPS per
day, the heatwave event metrics under both definitions, and the spread and skill components.
Nothing is averaged - one value per case, so significance.py can resample over cases.

Note that REGION below states the interior directly, where the other four reports derive it
by shrinking the file's own grid by the 8.5 degree margin. Same domain, two mechanisms.

    python -m modified_Gencast.report_exp1 --run exp1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from modified_Gencast.lumio_buckets import (get_loading_region, load, load_hourly_p90_threshold, load_p90_threshold, load_true_daily_tmax)
from modified_Gencast.metrics import (TARGET_VAR, area_weighted_crps, fixed_lead_hourly_threshold,
                                      heatwave_conditioned_metrics, spread_and_skill_cases, to_scoring_grid)


REGION = (45.0, 65.0, -5.0, 25.0)

RUNS = {"exp1": "Pretrained global GenCast"}
def report_run(scratch: Path, output_root: Path, run: str, max_forecasts: int | None = None) -> Path:
    config = json.loads(Path("modified_Gencast/traninig configs/experiment_2_all_gencast.json").read_text())
    loading_region = get_loading_region(config["task_config"])
    true_daily = to_scoring_grid(load_true_daily_tmax("2021-01-01", "2021-12-31", loading_region, REGION), REGION)
    p90 = to_scoring_grid(load_p90_threshold(), REGION)
    hourly_p90 = to_scoring_grid(load_hourly_p90_threshold(), REGION)

    prediction_path = scratch / "Predictions" / "exp1" / "predictions_2021-01-01_2021-12-31.zarr"
    global_store = xr.open_zarr(prediction_path, consolidated=True)
    try:
        prediction = to_scoring_grid(global_store[TARGET_VAR], REGION)
        times = pd.DatetimeIndex(prediction.time.values)
        grid_p90 = p90.interp(lat=prediction.lat, lon=prediction.lon)
        grid_hourly_p90 = hourly_p90.interp(lat=prediction.lat, lon=prediction.lon)
        grid_true_daily = true_daily.interp(lat=prediction.lat, lon=prediction.lon)
        target = to_scoring_grid(load(str(times.min().date()), str(times.max().date()), variables=[TARGET_VAR], region=loading_region, freq="1h",
        )[TARGET_VAR], REGION).sel(time=times).interp(lat=prediction.lat, lon=prediction.lon)
        t2m_crps = area_weighted_crps(prediction, target)
        hourly_threshold = grid_hourly_p90.sel(
            dayofyear=xr.DataArray(np.minimum(times.dayofyear, 365), dims=("time",), coords={"time": prediction.time}),
            hour_of_day=xr.DataArray(times.hour, dims=("time",), coords={"time": prediction.time}),
        )
        t2m_twcrps = area_weighted_crps(prediction, target, hourly_threshold)
        days = times.floor("D")
        tmax_members = prediction.assign_coords(day=("time", days)).groupby("day").max()
        tmax_members = tmax_members.sel(day=pd.DatetimeIndex(tmax_members.day.values).intersection(pd.DatetimeIndex(grid_true_daily.time.values)))
        tmax_target = grid_true_daily.sel(time=tmax_members.day)
        tmax_threshold = grid_p90.sel(dayofyear=xr.DataArray(np.minimum(pd.DatetimeIndex(tmax_members.day.values).dayofyear, 365), dims=("day",), coords={"day": tmax_members.day}))
        tmax_crps = area_weighted_crps(tmax_members, tmax_target)
        tmax_twcrps = area_weighted_crps(tmax_members, tmax_target, tmax_threshold)
        # collects the two observed days before each one-day forecast so it can flag a heatwave too
        continuous_exceeds = grid_true_daily > grid_p90.sel(
            dayofyear=xr.DataArray(np.minimum(pd.DatetimeIndex(grid_true_daily.time.values).dayofyear, 365), dims=("time",), coords={"time": grid_true_daily.time}))
        antecedent_days = xr.DataArray(tmax_members.day.values[:, None] + np.array([-2, -1])[None, :] * np.timedelta64(1, "D"), dims=["day", "antecedent"], coords={"day": tmax_members.day, "antecedent": [-2, -1]})
        tmax_antecedent = continuous_exceeds.reindex(
            time=np.union1d(continuous_exceeds.time.values, np.unique(antecedent_days.values)),
            fill_value=False).sel(time=antecedent_days).drop_vars("time")
        events = heatwave_conditioned_metrics(tmax_members, tmax_target, tmax_threshold, day_dim="day", valid_day=tmax_members["day"], antecedent_exceeds=tmax_antecedent)
        metric_cases = xr.merge((xr.Dataset({
            "t2m_crps_case": t2m_crps.astype(np.float32),
            "t2m_twcrps_case": t2m_twcrps.astype(np.float32),
            "tmax_crps_case": tmax_crps.astype(np.float32),
            "tmax_twcrps_case": tmax_twcrps.astype(np.float32),
            "valid_time": xr.DataArray(t2m_crps.time.values, dims=("time",), coords={"time": t2m_crps.time}),
            "valid_day": xr.DataArray(tmax_crps.day.values, dims=("day",), coords={"day": tmax_crps.day}),
        }), events, spread_and_skill_cases(prediction, target)), compat="equals", join="exact").compute()
    finally:
        global_store.close()
    target_path = "ERA5 hourly target selected by global valid_time"
    layout = "global_valid_time"

    # records how the numbers were made, since the file is read far from here
    metric_cases.attrs.update({
        "run_id": run,
        "display_name": RUNS[run],
        "prediction_path": str(prediction_path),
        "target_path": str(target_path),
        "layout": layout,
        "spatial_weighting": "cosine latitude",
        "tmax_sampling": "maximum of matched 12-hourly prediction and target samples per complete valid day",
        "threshold": "1981-2010 p90 by valid day/hour and grid cell",
    })
    output_path = output_root / "reports" / "exp1" / run / "metric_cases.nc"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metric_cases.to_netcdf(output_path, mode="w")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", choices=tuple(RUNS), required=True)
    parser.add_argument("--scratch", type=Path, default=Path("/scratch/project_465002687"))
    parser.add_argument("--output-root", type=Path, default=Path("/scratch/project_465002687/verified_metrics"))
    parser.add_argument("--max-forecasts", type=int, default=None)
    args = parser.parse_args()
    print(report_run(args.scratch, args.output_root, args.run, args.max_forecasts))


if __name__ == "__main__":
    main()
