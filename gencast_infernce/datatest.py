from time import time
import xarray as xr
import os
import argparse
SCRATCH_PATH = "../../../scratch/project_465002687/data/era5/"

# scratch_path = f"../../../scratch/project_465002687/data/era5/wb2_{from_date}_{to_date}_12h_1deg.zarr"
# os.makedirs(os.path.dirname(scratch_path), exist_ok=True)
# t0 = time()
# print("Opening remote WB2 data lazily...")
# wb2 = xr.open_zarr("gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr", consolidated=True)
# print(f"Data opened in {time() - t0:.2f} seconds.")
# t0 = time()
# print("Slicing exactly 1 year of required data...")
# wb2 = wb2.sel(time=wb2.time.dt.hour.isin([0, 12]))
# wb2 = wb2.sel(time=slice(from_date, to_date))
# wb2 = wb2.sel(level=[50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000])
# wb2 = wb2[["2m_temperature", "mean_sea_level_pressure", "10m_u_component_of_wind",
#            "10m_v_component_of_wind", "sea_surface_temperature", "total_precipitation_12hr",
#            "geopotential_at_surface", "land_sea_mask", "geopotential", "specific_humidity",
#            "temperature", "u_component_of_wind", "v_component_of_wind", "vertical_velocity"]]

# wb2 = wb2.isel(latitude=slice(None, None, 4), longitude=slice(None, None, 4))
# wb2 = wb2.rename({"latitude": "lat", "longitude": "lon"})
# wb2 = wb2.sortby("lat")

# for var in wb2.variables:
#     wb2[var].encoding.clear()
# wb2.encoding.clear()
# print(f"Slicing complete in {time() - t0:.2f} seconds.")

# t0 = time()
# print(f"Downloading and saving to {scratch_path}...")
# wb2.to_zarr(scratch_path, mode="w")
# print(f"Download and save complete in {time() - t0:.2f} seconds.")


# print("NaNs in 2m_temperature:", xr.open_zarr(scratch_path)["2m_temperature"].isnull().sum().compute().item())

def prepare_data(location, deg=0.25, time_slice=slice("2019-03-28T12:00", "2019-03-30T00:00:00")):
    wb2 = xr.open_zarr(location)
    wb2 = wb2.sel(time=time_slice)
    wb2 = wb2[["2m_temperature", "mean_sea_level_pressure", "10m_u_component_of_wind",
            "10m_v_component_of_wind", "sea_surface_temperature", "total_precipitation_12hr",
            "geopotential_at_surface", "land_sea_mask", "geopotential", "specific_humidity",
            "temperature", "u_component_of_wind", "v_component_of_wind", "vertical_velocity"]]
    wb2 = wb2.sel(time=wb2.time.dt.hour.isin([0, 12]))
    if "latitude" in wb2.dims and "longitude" in wb2.dims:
        wb2 = wb2.rename({"latitude": "lat", "longitude": "lon"})
    if deg == 1 and wb2.sizes.get("lat") == 721 and wb2.sizes.get("lon") == 1440:
        wb2 = wb2.isel(lat=slice(None, None, 4), lon=slice(None, None, 4)).compute()
    wb2 = wb2.sortby("lat") 
    return wb2


def download_data(location, deg=0.25, time_slice=slice("2019-03-28T12:00", "2019-03-30T00:00:00")):
    wb2 = xr.open_zarr("gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr")
    wb2 = wb2.sel(time=wb2.time.dt.hour.isin([0, 12]))
    wb2 = wb2.sel(time=time_slice)
    wb2 = wb2.sel(level=[50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000])
    wb2 = wb2[["2m_temperature", "mean_sea_level_pressure", "10m_u_component_of_wind",
            "10m_v_component_of_wind", "sea_surface_temperature", "total_precipitation_12hr",
            "geopotential_at_surface", "land_sea_mask", "geopotential", "specific_humidity",
            "temperature", "u_component_of_wind", "v_component_of_wind", "vertical_velocity"]]
    if "latitude" in wb2.dims and "longitude" in wb2.dims:
        wb2 = wb2.rename({"latitude": "lat", "longitude": "lon"})
    if deg == 1:
        wb2 = wb2.isel(lat=slice(None, None, 4), lon=slice(None, None, 4))    
    wb2 = wb2.sortby("lat")
    for var in wb2.variables:
        wb2[var].encoding.clear()
    wb2.encoding.clear()

    wb2.to_zarr(location, mode="w")
    return wb2


def data_sanity_check(dataset):
    print("------- sanity check -------")
    print("Data variables:", list(dataset.data_vars))
    print("Data levels:", dataset.level.values)
    print("Data time range:", dataset.time.min().values, "to", dataset.time.max().values)
    print("Data time steps:", len(dataset.time))
    print("Data lat range:", dataset.lat.min().values, "to", dataset.lat.max().values)
    print("Data lon range:", dataset.lon.min().values, "to", dataset.lon.max().values)
    print("NaNs in 2m_temperature:", dataset["2m_temperature"].isnull().sum().compute().item())
    print("Data degree:", 1 if dataset.lat.size == 181 else 0.25)
    print("NaNs in each variable:")
    for var in dataset.data_vars:
        print(f"  {var}: {dataset[var].isnull().sum().compute().item()} NaNs")
    print("------- sanity check DONE-------")

def convert_025_to_1deg(path_025, path_1deg):
    print(f"Loading 0.25 data from {path_025}...")
    wb2_025 = xr.open_zarr(path_025)
    
    print("Slicing to 1.0 degree...")
    wb2_1deg = wb2_025.isel(lat=slice(None, None, 4), lon=slice(None, None, 4))
    
    # Fix the fragmented chunks
    wb2_1deg = wb2_1deg.chunk({"time": 1, "lat": -1, "lon": -1})
    
    for var in wb2_1deg.variables:
        wb2_1deg[var].encoding.clear()
    wb2_1deg.encoding.clear()
    
    print(f"Saving 1.0 data to {path_1deg}...")
    wb2_1deg.to_zarr(path_1deg, mode="w")
    print("Conversion complete.")



def main(from_date, to_date, d):
    path = SCRATCH_PATH + f"wb2_{from_date}_{to_date}_12h_{d}deg.zarr"
    print(f"getting data for degree {d} from {from_date} to {to_date}...")
    t0 = time()
    wb2 = download_data(path, deg=d, time_slice=slice(from_date+"T00:00:00", to_date+"T00:00:00"))
    print(f"Data {path} downloaded and saved in {time() - t0:.2f} seconds.")
    print(f"Storage used for {path}: {wb2.nbytes / 1e6:.2f} MB")
    print("==================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=str, required=True, help="Start date (e.g. 2019-03-28)")
    parser.add_argument('--end', type=str, required=True, help="End date (e.g. 2019-03-29)")
    parser.add_argument('--degree', type=float, required=True, help="Resolution degree (e.g. 1 or 0.25)")
    args = parser.parse_args()
    
    # main(args.start, args.end, args.degree)
    convert_025_to_1deg(SCRATCH_PATH + f"wb2_{args.start}_{args.end}_12h_0.25deg.zarr", SCRATCH_PATH + f"wb2_{args.start}_{args.end}_12h_1.0deg.zarr")
    wb2 = prepare_data(SCRATCH_PATH + f"wb2_{args.start}_{args.end}_12h_{args.degree}deg.zarr", deg=args.degree, time_slice=slice(args.start+"T00:00:00", args.end+"T00:00:00"))
    data_sanity_check(wb2)