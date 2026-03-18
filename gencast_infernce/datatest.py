from time import time

import xarray as xr
import os
from_date = "2021-12-30"
to_date = "2023-01-01"

scratch_path = f"../../../scratch/project_465002687/data/era5/wb2_{from_date}_{to_date}_12h_1deg.zarr"
os.makedirs(os.path.dirname(scratch_path), exist_ok=True)
t0 = time()
print("Opening remote WB2 data lazily...")
wb2 = xr.open_zarr("gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr", consolidated=True)
print(f"Data opened in {time() - t0:.2f} seconds.")
t0 = time()
print("Slicing exactly 1 year of required data...")
wb2 = wb2.sel(time=wb2.time.dt.hour.isin([0, 12]))
wb2 = wb2.sel(time=slice(from_date, to_date))
wb2 = wb2.sel(level=[50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000])
wb2 = wb2[["2m_temperature", "mean_sea_level_pressure", "10m_u_component_of_wind",
           "10m_v_component_of_wind", "sea_surface_temperature", "total_precipitation_12hr",
           "geopotential_at_surface", "land_sea_mask", "geopotential", "specific_humidity",
           "temperature", "u_component_of_wind", "v_component_of_wind", "vertical_velocity"]]

wb2 = wb2.isel(latitude=slice(None, None, 4), longitude=slice(None, None, 4))
wb2 = wb2.rename({"latitude": "lat", "longitude": "lon"})
wb2 = wb2.sortby("lat")

for var in wb2.variables:
    wb2[var].encoding.clear()
wb2.encoding.clear()
print(f"Slicing complete in {time() - t0:.2f} seconds.")

t0 = time()
print(f"Downloading and saving to {scratch_path}...")
wb2.to_zarr(scratch_path, mode="w")
print(f"Download and save complete in {time() - t0:.2f} seconds.")


print("NaNs in 2m_temperature:", xr.open_zarr(scratch_path)["2m_temperature"].isnull().sum().compute().item())








