import numpy as np
import pandas as pd
import xarray as xr

# ------------------------------------------------------------------
# Read CMEMS SST data
# ------------------------------------------------------------------

# file = "cmems_sst.nc"
file = "cmems_mod_bal_phy_anfc_P1M-m_WestBaltic_2021_2026.nc"

# Open NetCDF file
ds = xr.open_dataset(file)

# Inspect variables (similar to MATLAB ncdisp)
print(ds)

# ------------------------------------------------------------------
# Read coordinates
# ------------------------------------------------------------------

time = ds["time"].values
lon = ds["longitude"].values.astype(float)
lat = ds["latitude"].values.astype(float)

# Map limits
lonlim = [np.nanmin(lon), np.nanmax(lon)]
latlim = [np.nanmin(lat), np.nanmax(lat)]

# ------------------------------------------------------------------
# Adjust time to month/day/year
# ------------------------------------------------------------------

time_dt = pd.to_datetime(time)
monthn = time_dt.month
idx_june = monthn == 6

# ------------------------------------------------------------------
# Read SST and salinity
# ------------------------------------------------------------------

thetao = ds["thetao"].values
sal = ds["so"].values

sst = thetao.copy()

# ------------------------------------------------------------------
# Remove singleton dimensions
# Equivalent to:
# squeeze -> rot90(...,3) -> fliplr
# ------------------------------------------------------------------

sst = np.squeeze(sst)
sal = np.squeeze(sal)

sst = np.fliplr(np.rot90(sst, k=3))
sal = np.fliplr(np.rot90(sal, k=3))

# ------------------------------------------------------------------
# Monthly climatology
# ------------------------------------------------------------------

nlon = len(lon)
nlat = len(lat)

sst_monthly = np.full((nlat, nlon, 12), np.nan)
sal_monthly = np.full((nlat, nlon, 12), np.nan)

for ii in range(1, 13):
    idx_month = monthn == ii

    sst_monthly[:, :, ii - 1] = np.nanmean(
        sst[:, :, idx_month], axis=2
    )

    sal_monthly[:, :, ii - 1] = np.nanmean(
        sal[:, :, idx_month], axis=2
    )

# ------------------------------------------------------------------
# Convert Kelvin to Celsius if needed
# ------------------------------------------------------------------

if np.nanmean(sst) > 100:
    sst = sst - 273.15
    sst_monthly = sst_monthly - 273.15

print("Done.")
