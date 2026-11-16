"""Build the -tp12 store: standard variables from WeatherBench2, precipitation from GenCast.

Needed only by exp1, which runs Google's pretrained checkpoint. That checkpoint lists
total_precipitation_12hr among its target variables, and the WeatherBench2 ERA5 dataset does
not carry it, so the two sources are fetched separately and merged into one bucket.

Unlike the other builders this one rewrites from scratch: it removes the existing store
before writing, because it exists to repair a store that is wrong rather than to extend one.

    python modified_Gencast/build_tp12_store.py --start-date 2021-01-01 --end-date 2021-12-31
"""

import argparse
import json
import os
import sys

import s3fs
import xarray as xr
from dask.distributed import LocalCluster, Client

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..")))

from gencast_infernce.config import CONFIGS_DIR
from modified_Gencast.lumio_buckets import (
    lumio, bucket_name, download, get_loading_region,
    GENCAST_LEVELS, GENERATED_VARS, DOWNLOAD_TIME_CHUNK,
)

TP12_VAR = "total_precipitation_12hr"


def main():
    parser = argparse.ArgumentParser(description="Rebuild the tp12 store: standard vars from WB2, precip from GenCast.")
    parser.add_argument("--start-date", type=str, default="2021-01-01")
    parser.add_argument("--end-date", type=str, default="2021-12-31")
    parser.add_argument("--data-step", type=str, default="6h")
    args = parser.parse_args()

    config = json.load(open(os.path.join(CONFIGS_DIR, "experiment_2_all_gencast.json")))
    task_config = config["task_config"]
    region = get_loading_region(task_config)

    all_variables = (
        list(task_config["input_variables"])
        + list(task_config.get("target_surface_variables", []))
        + list(task_config.get("target_atmospheric_variables", []))
    )
    fetch_variables = list(dict.fromkeys(
        variable for variable in all_variables
        if variable not in GENERATED_VARS
        and not variable.endswith(("_daily_mean", "_daily_max", "_daily_min"))
    ))
    standard_variables = [variable for variable in fetch_variables if variable != TP12_VAR]

    store_bucket = bucket_name(GENCAST_LEVELS, region, tp12=True)
    print(f"[build_tp12] store: {store_bucket}")
    print(f"[build_tp12] {args.start_date} → {args.end_date}  step={args.data_step}  region={region}")
    print(f"[build_tp12] standard (from WB2): {standard_variables}")
    print(f"[build_tp12] precip (from GenCast): [{TP12_VAR}]")

    with LocalCluster(n_workers=24, threads_per_worker=2, processes=True,
                      dashboard_address=None, memory_limit="11GB") as cluster, Client(cluster):
        print("[build_tp12] fetching standard variables from WB2 source ...")
        standard = download(args.start_date, args.end_date, standard_variables, GENCAST_LEVELS, region, args.data_step)
        print("[build_tp12] fetching total_precipitation_12hr from GenCast source ...")
        precip = download(args.start_date, args.end_date, [TP12_VAR], GENCAST_LEVELS, region, args.data_step)

        merged = xr.merge([standard, precip]).chunk({"time": DOWNLOAD_TIME_CHUNK, "lat": -1, "lon": -1})
        for name in merged.variables:
            merged[name].encoding.pop("chunks", None)

        fs = lumio()
        if fs.exists(store_bucket):
            print(f"[build_tp12] removing broken store {store_bucket} ...")
            fs.rm(store_bucket, recursive=True)
        if not fs.exists(store_bucket):
            fs.mkdir(store_bucket)
        store = s3fs.S3Map(root=store_bucket, s3=fs, check=False)
        print(f"[build_tp12] writing {dict(merged.sizes)} to {store_bucket} (mode='w-') ...")
        merged.to_zarr(store, mode="w-", consolidated=True)

    print("[build_tp12] done. Now run: sbatch --array=6 job_scripts/lumi_eval_exp2.sh")


if __name__ == "__main__":
    main()
